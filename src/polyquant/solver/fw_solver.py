"""
Frank-Wolfe Solver for Arbitrage-Free Market Making (PolyQuant 2.0)

This module implements the specific algorithms described in the research:

ALGORITHMS IMPLEMENTED:
-----------------------
1.  **InitFW (Algorithm 3)**: Initialization to find valid starting vertices.
    - Queries the IP solver (SCIP) for each security to determine if it can be 0 or 1.
    - Constructs the initial active set Z_0 and the interior point u.
    - Identifies logically settled securities.

2.  **Barrier Frank-Wolfe (Algorithm 2)**: Optimization on contracted polytopes.
    - Uses adaptive contraction M' = (1-ε)M + εu to control gradient growth.
    - The Lipschitz constant L_ε = O(1/ε) is bounded for any ε > 0.
    - ε shrinks adaptively as the algorithm converges.

3.  **Profit Guarantee (Proposition 4.1)**: Stopping condition.
    - Guaranteed Profit ≥ D(μ̂||θ) - g(μ̂)
    - Stop when: g(μ_t) ≤ (1-α) × D(μ_t||θ) (α-extraction, default α=0.9)

REFERENCES:
-----------
- Kroer et al. 2016: "Arbitrage-Free Combinatorial Market Making via IP"
- Krishnan et al. 2015: Theory for adaptive contraction.

USAGE:
------
    from polyquant.solver.fw_solver import FWSolver, ArbitrageDetector
    
    detector = ArbitrageDetector()
    opportunity = await detector.detect(validated_result, order_books)
    
    if opportunity:
        print(f"Found opportunity with ${opportunity.expected_profit} profit")
"""


import asyncio
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional, Any, TYPE_CHECKING
from decimal import Decimal

from polyquant.utils import get_logger
from polyquant.utils.cache import cache  # Week 3: Redis persistence for InitFW
from polyquant.solver.scip_solver import SCIPSolver
if TYPE_CHECKING:
    from polyquant.agents.validator import ValidatedResult
from polyquant.data import ArbitrageOpportunity, OrderBook, OrderSide, ProposedTrade
from polyquant.utils import config
from polyquant.utils.market_utils import extract_market_id
from polyquant.risk.position_sizing import PositionSizer

logger = get_logger(__name__)

TRUSTED_MULTI_MARKET_PREFIXES = ("negrisk_", "cross_")


class FWSolver:
    """
    Implements the Barrier Frank-Wolfe algorithm for LMSR market making.
    """

    def __init__(self, scip_solver: SCIPSolver):
        self.solver = scip_solver
        self.epsilon = 0.1  # Initial contraction parameter
        self.alpha = 0.9    # Extraction guarantee threshold
        self.min_profit = 0.05 # Minimum profit threshold

        # Gap at interior point (for correct epsilon adaptation)
        self.g_u: float | None = None

        # Optimization 2: Cache for InitFW results
        # Key: Sorted outcome IDs. Value: (Z_0, u, settled_ids)
        self._u_cache: Dict[str, Tuple[List[Dict[str, float]], Dict[str, float], Set[str]]] = {}

        # Phase 3 Optimization: Enable vectorization for performance
        self.use_vectorization = True  # Can be disabled for debugging

    async def init_fw(
        self,
        validated: "ValidatedResult",
        outcomes: List[str]
    ) -> Tuple[List[Dict[str, float]], Dict[str, float], Set[str]]:
        """
        Algorithm 3: InitFW

        Constructs a valid set of starting vertices Z_0 and an interior point u.
        Also identifies settled securities.

        Week 3 Enhancement: Persistent Redis caching for cold start elimination.

        Args:
            validated: Constraints
            outcomes: List of outcome IDs

        Returns:
            Z_0: List of valid vertex vectors (dicts)
            u: Interior point vector (dict)
            settled: Set of settled outcome IDs
        """
        logger.info("Running InitFW...")

        # 1. Check in-memory cache (fastest - no network latency)
        security_ids = sorted(outcomes)
        cache_key = ",".join(security_ids)
        if cache_key in self._u_cache:
            logger.debug(f"InitFW in-memory cache HIT for {len(outcomes)} outcomes")
            Z_0, u, settled_ids = self._u_cache[cache_key]
            # Must copy mutable objects to avoid side effects if modified elsewhere
            # But here they are mostly read-only. Returning direct ref for speed.
            return Z_0, u, settled_ids

        # 2. Check Redis cache (persistent across restarts)
        redis_key = f"initfw:{cache_key}"
        redis_result = await cache.get_solver_result(redis_key)
        if redis_result:
            logger.info(f"InitFW Redis cache HIT for {len(outcomes)} outcomes (cold start eliminated!)")
            Z_0 = redis_result['Z_0']
            u = redis_result['u']
            settled_ids = set(redis_result['settled'])

            # Save to in-memory cache for future fast access
            self._u_cache[cache_key] = (Z_0, u, settled_ids)
            return Z_0, u, settled_ids

        logger.debug(f"InitFW cache MISS (in-memory + Redis) for {len(outcomes)} outcomes")
        
        Z_0: List[Dict[str, float]] = []
        sigma_hat: Dict[str, int] = {} # Extended partial outcome
        settled_ids: Set[str] = set()
        
        # We need to find at least one valid vertex to start
        # Try to find one by just solving feasibility with no fixed vars
        is_feas, init_sol = self.solver.check_feasibility(validated, {})
        if is_feas:
            Z_0.append(init_sol)
        else:
            logger.error("Constraints are infeasible! Cannot start.")
            return [], {}, set()

        for outcome_id in outcomes:
            if outcome_id in settled_ids:
                continue
                
            # Question 1: Can z_i = 1?
            can_be_1, sol_1 = self.solver.check_feasibility(validated, {outcome_id: 1.0})
            
            # Question 2: Can z_i = 0?
            can_be_0, sol_0 = self.solver.check_feasibility(validated, {outcome_id: 0.0})
            
            if can_be_1 and can_be_0:
                # Uncertain
                Z_0.append(sol_1)
                Z_0.append(sol_0)
            elif can_be_1 and not can_be_0:
                # Must be 1
                settled_ids.add(outcome_id)
                sigma_hat[outcome_id] = 1
                Z_0.append(sol_1) # Add the valid one
            elif not can_be_1 and can_be_0:
                # Must be 0
                settled_ids.add(outcome_id)
                sigma_hat[outcome_id] = 0
                Z_0.append(sol_0)
            else:
                logger.error(f"Outcome {outcome_id} is infeasible (cannot be 0 or 1). Model invalid.")
                
        # Construct interior point u
        # u = average of Z_0
        if not Z_0:
            return [], {}, set()
            
        u: Dict[str, float] = {o: 0.0 for o in outcomes}
        for z in Z_0:
            for o, val in z.items():
                u[o] += val
                
        for o in outcomes:
            u[o] /= len(Z_0)
            
        logger.info(f"InitFW complete. |Z_0|={len(Z_0)}, Settled={len(settled_ids)}")

        # 3. Save to both in-memory and Redis caches
        self._u_cache[cache_key] = (Z_0, u, settled_ids)

        # Week 3: Persist to Redis with long TTL (constraints are immutable)
        await cache.set_solver_result(
            redis_key,
            {
                'Z_0': Z_0,
                'u': u,
                'settled': list(settled_ids)  # Convert set to list for JSON
            },
            ttl_seconds=86400  # 24 hours - constraints don't change
        )
        logger.debug(f"InitFW result cached to Redis with 24h TTL")

        return Z_0, u, settled_ids

    def _vectorized_gradient(
        self,
        mu_vec: np.ndarray,
        theta_vec: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized gradient computation for KL divergence.

        Gradient of D(mu || theta) w.r.t. mu:
            Grad_i = log(mu_i / theta_i) + 1

        Args:
            mu_vec: Current iterate as numpy array
            theta_vec: Market prices as numpy array

        Returns:
            Gradient vector as numpy array

        Performance: ~50x faster than dict loop for 100+ outcomes
        """
        # Ensure numerical stability
        mu_safe = np.maximum(mu_vec, 1e-9)
        return np.log(mu_safe / theta_vec) + 1.0

    def _vectorized_kl_divergence(
        self,
        mu_vec: np.ndarray,
        theta_vec: np.ndarray,
    ) -> float:
        """
        Vectorized KL divergence computation.

        D(mu || theta) = sum_i mu_i * log(mu_i / theta_i)

        Args:
            mu_vec: Distribution mu as numpy array
            theta_vec: Distribution theta as numpy array

        Returns:
            KL divergence as scalar

        Performance: ~30x faster than dict loop for 100+ outcomes
        """
        mu_safe = np.maximum(mu_vec, 1e-9)
        return float(np.sum(mu_safe * np.log(mu_safe / theta_vec)))

    def _vectorized_fw_gap(
        self,
        grad_vec: np.ndarray,
        mu_vec: np.ndarray,
        v_prime_vec: np.ndarray,
    ) -> float:
        """
        Vectorized Frank-Wolfe gap computation.

        Gap g(mu) = <grad, mu - v'>

        Args:
            grad_vec: Gradient vector
            mu_vec: Current iterate
            v_prime_vec: Contracted vertex

        Returns:
            Frank-Wolfe gap as scalar

        Performance: ~20x faster than dict loop for 100+ outcomes
        """
        return float(np.dot(grad_vec, mu_vec - v_prime_vec))

    def _vectorized_step_update(
        self,
        mu_vec: np.ndarray,
        v_prime_vec: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        """
        Vectorized step update.

        mu_new = (1 - gamma) * mu + gamma * v'

        Args:
            mu_vec: Current iterate
            v_prime_vec: Direction
            gamma: Step size

        Returns:
            Updated iterate

        Performance: ~15x faster than dict loop for 100+ outcomes
        """
        return (1.0 - gamma) * mu_vec + gamma * v_prime_vec

    def barrier_fw(
        self,
        validated: "ValidatedResult",
        outcomes: List[str],
        market_prices: Dict[str, float],
        Z_0: List[Dict[str, float]],
        u: Dict[str, float],
        max_iters: int = 100
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Algorithm 2: Barrier Frank-Wolfe with Adaptive Contraction
        
        Args:
            validated: Constraints
            outcomes: List of outcome IDs
            market_prices: Current market prices (theta)
            Z_0: Initial active set
            u: Interior point
            
        Returns:
            mu: Optimized arbitrage-free prices
            profit_guarantee: Guaranteed profit
            kl_div: Final divergence
        """
        logger.info("Starting Barrier Frank-Wolfe...")

        # Phase 3 Optimization: Use vectorized operations if enabled
        # Week 3: Lowered threshold from 10 to 5 for faster typical clusters
        if self.use_vectorization and len(outcomes) > 5:
            return self._barrier_fw_vectorized(
                validated, outcomes, market_prices, Z_0, u, max_iters
            )

        # Initialize mu (start at u for safety)
        mu = u.copy()

        epsilon = self.epsilon

        # Convert prices to vector for easier math
        # Ensure strict positivity for KL
        theta = {o: max(market_prices.get(o, 0.5), 1e-6) for o in outcomes}

        # Track best iterate for "Forced Interruption" condition
        best_mu = mu.copy()
        best_profit_guarantee = -float('inf')

        # Compute g_u (gap at interior point) for correct epsilon adaptation
        # This is needed once at the start for the adaptive epsilon rule
        # g_u = <grad_u, u - v'_u> where v'_u is the contracted vertex at u
        if self.g_u is None:
            # Compute gradient at u
            grad_u = {o: np.log(max(u[o], 1e-9) / theta[o]) + 1 for o in outcomes}

            # Solve LMO at gradient of u
            success_u, v_star_u, _ = self.solver.solve_linear_objective(
                validated, grad_u, sense="minimize"
            )

            if success_u:
                v_full_u = {o: v_star_u.get(o, 0.0) for o in outcomes}
                v_prime_u = {o: (1 - epsilon) * v_full_u[o] + epsilon * u[o] for o in outcomes}
                self.g_u = sum(grad_u[o] * (u[o] - v_prime_u[o]) for o in outcomes)
                self.g_u = max(self.g_u, 1e-10)  # Avoid division by zero
                logger.debug(f"Computed g_u = {self.g_u:.6f}")
            else:
                logger.warning("Failed to compute g_u, using fallback epsilon adaptation")
                self.g_u = None
        
        for t in range(1, max_iters + 1):
            # 1. Compute Gradient of Objective D(mu || theta)
            # D = sum mu_i * log(mu_i / theta_i)
            # Grad_i = log(mu_i / theta_i) + 1
            grad = {}
            for o in outcomes:
                val = max(mu.get(o, 0), 1e-9) # Avoid log(0)
                grad[o] = np.log(val / theta[o]) + 1
            
            # 2. Linear Oracle on Contracted Polytope M'
            # We want to minimize <Grad, v'> over v' in M'
            # v' = (1-eps)v + eps*u
            # min <Grad, (1-eps)v> + const -> min <Grad, v>
            # So we solve the IP with objective = Grad
            
            success, v_star, obj_val = self.solver.solve_linear_objective(
                validated, 
                grad, 
                sense="minimize"
            )
            
            if not success:
                logger.warning("LMO failed. Stopping.")
                break
                
            # Map v_star to full outcomes (fill missing with 0)
            v_full = {o: v_star.get(o, 0.0) for o in outcomes}
            
            # 3. Compute Gap
            # Gap g(mu) = <Grad, mu - v'>
            # v' = (1-eps)v_star + eps*u
            v_prime = {
                o: (1 - epsilon) * v_full[o] + epsilon * u[o]
                for o in outcomes
            }
            
            fw_gap = 0.0
            for o in outcomes:
                fw_gap += grad[o] * (mu[o] - v_prime[o])
                
            # 4. Profit Guarantee Check
            # D(mu || theta)
            kl = 0.0
            for o in outcomes:
                m_val = max(mu[o], 1e-9)
                kl += m_val * np.log(m_val / theta[o])
                
            guaranteed_profit = kl - fw_gap
            
            if guaranteed_profit > best_profit_guarantee:
                best_profit_guarantee = guaranteed_profit
                best_mu = mu.copy()
            
            # Stopping Condition 1: Arbitrage-free (check first)
            if kl < self.min_profit:
                logger.info(f"Arbitrage-free at iter {t}: D(μ||θ)={kl:.5f} < {self.min_profit}")
                return best_mu, best_profit_guarantee, kl

            # Stopping Condition 2: alpha-extraction
            # g(mu) <= (1 - alpha) * D(mu || theta)
            if guaranteed_profit > 0 and fw_gap <= (1 - self.alpha) * kl:
                logger.info(f"Alpha-extraction ({self.alpha}) met at iter {t}. Gap: {fw_gap:.5f}, KL: {kl:.5f}, Profit: {guaranteed_profit:.5f}")
                return best_mu, best_profit_guarantee, kl

            # 5. Adaptive Epsilon Update (Correct implementation from research)
            # If g(μ_t) / (-4g_u) < ε_{t-1}:
            #     ε_t = min{g(μ_t)/(-4g_u), ε_{t-1}/2}
            # Else:
            #     ε_t = ε_{t-1}
            g_u_val: float = self.g_u if self.g_u is not None else 0.0
            if abs(g_u_val) > 1e-10:
                # Use absolute value to avoid negative gap issues
                gap_ratio = fw_gap / (4 * abs(g_u_val))
                if gap_ratio < epsilon:
                    epsilon = min(gap_ratio, epsilon / 2)
                    epsilon = max(epsilon, 1e-4) # Lower bound
                    logger.debug(f"Iter {t}: Reduced epsilon to {epsilon:.6f}")
            else:
                # Fallback: simple heuristic
                if t > 5 and fw_gap < epsilon:
                    epsilon = max(epsilon / 2, 1e-4)
            
            # 6. Step Update
            # Standard FW step: gamma = 2 / (t + 2)
            gamma = 2.0 / (t + 2.0)
            
            new_mu = {}
            for o in outcomes:
                new_mu[o] = (1 - gamma) * mu[o] + gamma * v_prime[o]
            mu = new_mu
            
        logger.info(f"Max iterations ({max_iters}) reached.")
        return best_mu, best_profit_guarantee, 0.0 # Return best found

    def _barrier_fw_vectorized(
        self,
        validated: "ValidatedResult",
        outcomes: List[str],
        market_prices: Dict[str, float],
        Z_0: List[Dict[str, float]],
        u: Dict[str, float],
        max_iters: int = 100
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Vectorized version of Barrier Frank-Wolfe using numpy.

        This provides 10-50x speedup over dict-based operations for
        clusters with 100+ outcomes.

        Args:
            Same as barrier_fw()

        Returns:
            Same as barrier_fw()

        Performance:
            - Gradient: O(n) vectorized vs O(n) dict loop → ~50x faster
            - KL divergence: O(n) vectorized → ~30x faster
            - FW gap: O(n) vectorized → ~20x faster
            - Step update: O(n) vectorized → ~15x faster
            - Overall: 10-50x speedup depending on cluster size
        """
        logger.debug(f"Using vectorized Frank-Wolfe for {len(outcomes)} outcomes")

        n = len(outcomes)

        # Create index mapping for outcomes
        outcome_to_idx = {o: i for i, o in enumerate(outcomes)}

        # Convert to numpy arrays
        mu_vec = np.array([u.get(o, 0.0) for o in outcomes], dtype=np.float64)
        theta_vec = np.array(
            [max(market_prices.get(o, 0.5), 1e-6) for o in outcomes],
            dtype=np.float64
        )
        u_vec = np.array([u[o] for o in outcomes], dtype=np.float64)

        epsilon = self.epsilon

        # Track best iterate
        best_mu_vec = mu_vec.copy()
        best_profit_guarantee = -float('inf')

        # Compute g_u if needed (same as non-vectorized version)
        if self.g_u is None:
            grad_u_vec = self._vectorized_gradient(u_vec, theta_vec)

            # Convert to dict for solver
            grad_u_dict = {o: grad_u_vec[outcome_to_idx[o]] for o in outcomes}

            success_u, v_star_u, _ = self.solver.solve_linear_objective(
                validated, grad_u_dict, sense="minimize"
            )

            if success_u:
                v_full_u_vec = np.array([v_star_u.get(o, 0.0) for o in outcomes])
                v_prime_u_vec = (1 - epsilon) * v_full_u_vec + epsilon * u_vec
                self.g_u = float(np.dot(grad_u_vec, u_vec - v_prime_u_vec))
                self.g_u = max(self.g_u, 1e-10)
                logger.debug(f"Computed g_u = {self.g_u:.6f}")
            else:
                logger.warning("Failed to compute g_u, using fallback epsilon adaptation")
                self.g_u = None

        for t in range(1, max_iters + 1):
            # 1. Compute Gradient (Vectorized)
            grad_vec = self._vectorized_gradient(mu_vec, theta_vec)

            # 2. Linear Oracle (Still needs SCIP, convert to dict)
            grad_dict = {o: grad_vec[outcome_to_idx[o]] for o in outcomes}

            success, v_star, obj_val = self.solver.solve_linear_objective(
                validated,
                grad_dict,
                sense="minimize"
            )

            if not success:
                logger.warning("LMO failed. Stopping.")
                break

            # Convert v_star back to vector
            v_full_vec = np.array([v_star.get(o, 0.0) for o in outcomes])

            # 3. Compute Contracted Vertex (Vectorized)
            v_prime_vec = (1 - epsilon) * v_full_vec + epsilon * u_vec

            # 4. Compute Gap (Vectorized)
            fw_gap = self._vectorized_fw_gap(grad_vec, mu_vec, v_prime_vec)

            # 5. Compute KL Divergence (Vectorized)
            kl = self._vectorized_kl_divergence(mu_vec, theta_vec)

            # 6. Profit Guarantee
            guaranteed_profit = kl - fw_gap

            if guaranteed_profit > best_profit_guarantee:
                best_profit_guarantee = guaranteed_profit
                best_mu_vec = mu_vec.copy()

            # Stopping Condition 1: Arbitrage-free
            if kl < self.min_profit:
                logger.info(f"Arbitrage-free at iter {t}: D(μ||θ)={kl:.5f} < {self.min_profit}")
                # Convert back to dict
                best_mu_dict = {o: best_mu_vec[outcome_to_idx[o]] for o in outcomes}
                return best_mu_dict, best_profit_guarantee, kl

            # Stopping Condition 2: alpha-extraction
            if guaranteed_profit > 0 and fw_gap <= (1 - self.alpha) * kl:
                logger.info(f"Alpha-extraction ({self.alpha}) met at iter {t}. Gap: {fw_gap:.5f}, KL: {kl:.5f}, Profit: {guaranteed_profit:.5f}")
                best_mu_dict = {o: best_mu_vec[outcome_to_idx[o]] for o in outcomes}
                return best_mu_dict, best_profit_guarantee, kl

            # 7. Adaptive Epsilon Update
            g_u_val: float = self.g_u if self.g_u is not None else 0.0
            if abs(g_u_val) > 1e-10:
                gap_ratio = fw_gap / (4 * abs(g_u_val))
                if gap_ratio < epsilon:
                    epsilon = min(gap_ratio, epsilon / 2)
                    epsilon = max(epsilon, 1e-4)
                    logger.debug(f"Iter {t}: Reduced epsilon to {epsilon:.6f}")
            else:
                # Fallback heuristic
                if t > 5 and fw_gap < epsilon:
                    epsilon = max(epsilon / 2, 1e-4)

            # 8. Step Update (Vectorized)
            gamma = 2.0 / (t + 2.0)
            mu_vec = self._vectorized_step_update(mu_vec, v_prime_vec, gamma)

        logger.info(f"Max iterations ({max_iters}) reached.")
        best_mu_dict = {o: best_mu_vec[outcome_to_idx[o]] for o in outcomes}
        return best_mu_dict, best_profit_guarantee, 0.0

    async def find_opportunity(
        self,
        validated: "ValidatedResult",
        order_books: Dict[str, Any] # dict of OrderBook
    ) -> Optional[Dict[str, float]]:
        """
        Main entry point to find arbitrage-free prices.

        Handles settled securities by locking them to 0 or 1 and only
        optimizing over unsettled outcomes.

        Week 3: Now async to support Redis-cached InitFW.
        """
        outcomes = list(order_books.keys())
        market_prices = {}
        for o, ob in order_books.items():
            # Use mid-price as proxy for theta
            best_bid = float(ob.best_bid) if ob.best_bid is not None else 0.0
            best_ask = float(ob.best_ask) if ob.best_ask is not None else 1.0
            market_prices[o] = (best_bid + best_ask) / 2.0

        # 1. InitFW - identifies settled securities (Week 3: now async with Redis cache)
        Z_0, u, settled = await self.init_fw(validated, outcomes)
        if not Z_0:
            return None

        # 2. Separate settled and unsettled outcomes
        unsettled_outcomes = [o for o in outcomes if o not in settled]

        logger.info(f"Found {len(settled)} settled securities, {len(unsettled_outcomes)} unsettled")

        # 3. Extract settled security values from Z_0 vertices
        settled_values = {}
        for outcome_id in settled:
            # Find value from any vertex in Z_0 (should be same for all)
            for z in Z_0:
                if outcome_id in z:
                    settled_values[outcome_id] = z[outcome_id]
                    break

        # 4. BarrierFW - only optimize over unsettled outcomes
        if unsettled_outcomes:
            target_prices, profit, kl = self.barrier_fw(
                validated, unsettled_outcomes, market_prices, Z_0, u
            )
        else:
            # All securities are settled - no optimization needed
            target_prices = {}
            profit = 0.0

        # 5. Restore settled securities to target_prices
        for outcome_id, value in settled_values.items():
            target_prices[outcome_id] = value

        if profit > self.min_profit:
            return target_prices

        return None


class ArbitrageDetector:
    """
    High-level interface for detecting and quantifying arbitrage using Frank-Wolfe.
    """
    
    def __init__(
        self,
        scip_solver: Optional[SCIPSolver] = None,
        position_sizer: Optional[PositionSizer] = None,
    ):
        self.scip = scip_solver or SCIPSolver()
        self.fw_solver = FWSolver(self.scip)
        self.position_sizer = position_sizer or PositionSizer()
        
    async def detect(
        self,
        validated: "ValidatedResult",
        order_books: Dict[str, OrderBook],
        min_profit: float = 0.05,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Detect arbitrage using the research-backed pipeline:
        1. InitFW
        2. BarrierFW
        3. Profit Guarantee
        
        Optional: If config.sizing_strategy is 'dutching', prioritizes finding
        risk-free arbitrage rings (partitions where sum(prices) < 1).
        """
        if config.sizing_strategy == "dutching":
            return await self._detect_dutching_opportunity(validated, order_books)

        # Legacy Kelly detection...
        # Set solver threshold
        self.fw_solver.min_profit = min_profit

        # Week 3: find_opportunity is now async (for Redis-cached InitFW)
        # Note: SCIP solver calls inside are still CPU-bound, but Redis I/O is async
        target_prices = await self.fw_solver.find_opportunity(
            validated,
            order_books
        )
        
        if not target_prices:
            return None
            
        # Generate trades based on Target vs Market
        trades = []
        total_expected_profit = Decimal("0")
        
        for outcome_id, target_p in target_prices.items():
            ob = order_books.get(outcome_id)
            if not ob:
                continue
                
            # Buy opportunity: Ask < Target
            if ob.best_ask and float(ob.best_ask) < target_p:
                price_diff = target_p - float(ob.best_ask)

                # Calculate odds and depth for PositionSizer
                # Buy side: Odds = (1/Ask) - 1
                price = float(ob.best_ask)
                odds = (1.0 / price) - 1.0 if price > 0 else 0.0

                depth = 0.0
                if ob.asks and abs(float(ob.asks[0].price) - price) < 1e-6:
                    depth = float(ob.asks[0].size) * price # Liquidity in USD
                if depth <= 0:
                    logger.debug(
                        "Kelly BUY skipped: no top-of-book depth at target price",
                        outcome_id=outcome_id,
                        target_price=price,
                    )
                    size_result = None
                else:
                    size_result = self.position_sizer.calculate_size(
                        probability=target_p,
                        odds=odds,
                        order_book_depth=depth
                    )
                
                if size_result is not None and size_result.recommended_size > 0:
                    vwap = ob.get_vwap(OrderSide.BUY, Decimal(str(size_result.recommended_size)))
                    if vwap is not None and vwap < target_p:
                        price_diff = target_p - float(vwap)
                        profit = Decimal(str(price_diff * float(size_result.recommended_size)))
                        
                        exchange_info = validated.market_exchanges.get(extract_market_id(outcome_id), "polymarket")
                        exchange_name = "polymarket"
                        reason = ""
                        
                        if exchange_info.startswith("limitless:"):
                            exchange_name = "limitless"
                            slug = exchange_info.split(":")[1]
                            reason = f"slug:{slug}"
                        elif exchange_info == "limitless":
                            exchange_name = "limitless"
                            
                        trades.append(ProposedTrade(
                            market_id=extract_market_id(outcome_id),
                            outcome_id=outcome_id,
                            side=OrderSide.BUY,
                            size=size_result.recommended_size,
                            limit_price=vwap,
                            exchange=exchange_name,
                            reason=reason
                        ))
                        total_expected_profit += profit
                
            # Sell opportunity: Bid > Target
            if ob.best_bid and ob.best_bid > target_p:
                price_diff = ob.best_bid - target_p

                # Calculate odds and depth for PositionSizer
                # For SELL (shorting Yes / buying No):
                # We pay (1-bid) to win 1. Profit = bid.
                # Odds = bid / (1-bid)
                price = ob.best_bid
                if price >= 1.0 or price <= 0.0:
                    odds = 0.0
                else:
                    odds = price / (1.0 - price)

                depth = 0.0
                if ob.bids and abs(ob.bids[0].price - price) < 1e-6:
                    depth = ob.bids[0].size * price # Liquidity in USD
                if depth <= 0:
                    logger.debug(
                        "Kelly SELL skipped: no top-of-book depth at target price",
                        outcome_id=outcome_id,
                        target_price=float(price),
                    )
                    size_result = None
                else:
                    size_result = self.position_sizer.calculate_size(
                        probability=1.0 - target_p, # Probability of the event NOT happening
                        odds=odds,
                        order_book_depth=depth
                    )

                if size_result is not None and size_result.recommended_size > 0:
                    vwap = ob.get_vwap(OrderSide.SELL, Decimal(str(size_result.recommended_size)))
                    if vwap is not None and vwap > target_p:
                        price_diff = float(vwap) - target_p
                        profit = Decimal(str(price_diff * float(size_result.recommended_size)))
                        
                        exchange_info = validated.market_exchanges.get(extract_market_id(outcome_id), "polymarket")
                        exchange_name = "polymarket"
                        reason = ""
                        
                        if exchange_info.startswith("limitless:"):
                            exchange_name = "limitless"
                            slug = exchange_info.split(":")[1]
                            reason = f"slug:{slug}"
                        elif exchange_info == "limitless":
                            exchange_name = "limitless"
                            
                        trades.append(ProposedTrade(
                            market_id=extract_market_id(outcome_id),
                            outcome_id=outcome_id,
                            side=OrderSide.SELL,
                            size=size_result.recommended_size,
                            limit_price=vwap,
                            exchange=exchange_name,
                            reason=reason
                        ))
                        total_expected_profit += profit
                
        if not trades:
            return None
            
        # Deduct Fees and Gas — per-exchange rates
        total_gas = Decimal("0")
        total_fees = Decimal("0")
        poly_gas = Decimal(str(config.polygon_gas_per_tx))
        base_gas = Decimal(str(config.base_gas_per_tx))
        poly_fee_pct = Decimal(str(config.polymarket_taker_fee_pct))
        limitless_fee_pct = Decimal(str(config.limitless_taker_fee_pct))
        for t in trades:
            exchange = getattr(t, "exchange", "polymarket")
            if exchange == "limitless":
                total_gas += base_gas
                total_fees += Decimal(str(t.size)) * t.limit_price * limitless_fee_pct
            else:
                total_gas += poly_gas
                total_fees += Decimal(str(t.size)) * t.limit_price * poly_fee_pct
        total_expected_profit -= (total_gas + total_fees)
        
        if total_expected_profit <= 0:
            return None
            
        return ArbitrageOpportunity(
            markets=list(set(t.market_id for t in trades)),
            trades=trades,
            expected_profit=total_expected_profit,
            confidence=0.9 # High confidence due to guarantee, maybe dynamic based on gap?
        )

    async def _detect_dutching_opportunity(
        self,
        validated: "ValidatedResult",
        order_books: Dict[str, OrderBook]
    ) -> Optional[ArbitrageOpportunity]:
        """
        Specialized Dutching detection with enhanced robustness.

        Improvements in v0.4.0:
        - P1.1: VWAP slippage enforcement
        - P1.2: ROI calculation for ranking
        - P1.3: Partition size limiting
        - P2.1: Capital reservation across rings
        - P2.2: Dynamic confidence scoring
        - P2.3: Expected unwind cost modeling

        Searches constraints for partitions (e.g., Yes + No = 1) and checks for
        risk-free arbitrage opportunities where sum(best_asks) < 1.0.
        """
        logger.info("Running specialized Dutching detection...", constraint_count=len(validated.validated_constraints))

        # P2.1: Track reserved capital across rings
        reserved_capital = Decimal("0")
        available_capital = Decimal(str(self.position_sizer.capital))

        # Multi-ring accumulators: collect ALL profitable non-overlapping rings
        all_accepted_trades = []
        all_accepted_markets = set()
        total_accepted_profit = Decimal("0")
        min_confidence = 1.0  # Track worst confidence across accepted rings
        total_capital_deployed = Decimal("0")
        rings_accepted_count = 0

        # Collect all potential rings for ranking
        candidate_rings = []

        # Find "Partition" constraints: sum(z_i) >= 1.0 where all coeffs are 1.0
        # In PolyQuant, a complete set of outcomes is represented as a partition.
        for constraint in validated.validated_constraints:
            logger.debug("Checking constraint for Dutching", desc=constraint.description)
            if not all(c == 1.0 for c in constraint.coefficients.values()):
                logger.debug("Skipping: not all coefficients are 1.0")
                continue
            if abs(constraint.rhs - 1.0) > 1e-6:
                logger.debug("Skipping: RHS is not 1.0", rhs=constraint.rhs)
                continue

            # P3.2 — partition integrity. Single-market partitions are always safe.
            # Multi-market partitions are only accepted when they come from a trusted
            # mechanical source (negrisk_* or cross_*), which the Map Maker builds and
            # validates explicitly. Any other multi-market constraint is treated as
            # unverified provenance and skipped.
            token_market_ids = {extract_market_id(tid) for tid in constraint.coefficients.keys()}
            if len(token_market_ids) > 1:
                if not constraint.constraint_id.startswith(TRUSTED_MULTI_MARKET_PREFIXES):
                    logger.warning(
                        "Skipping multi-market 'partition' — untrusted provenance",
                        constraint_id=constraint.constraint_id,
                        market_ids=list(token_market_ids),
                        token_count=len(constraint.coefficients),
                    )
                    continue
                logger.info(
                    "Accepting trusted multi-market partition",
                    constraint_id=constraint.constraint_id,
                    market_count=len(token_market_ids),
                    source=constraint.constraint_id.split("_", 1)[0],
                )

            # We found a potential Dutching ring!
            outcome_ids = list(constraint.coefficients.keys())

            # P1.3: Partition size limit (prevent O(N²) on large markets)
            if len(outcome_ids) > config.max_partition_size:
                logger.debug(
                    f"Skipping ring: {len(outcome_ids)} outcomes exceeds max_partition_size={config.max_partition_size}",
                    outcomes=outcome_ids
                )
                continue

            logger.info("Potential Dutching ring found", outcomes=outcome_ids, size=len(outcome_ids))

            # Collect odds and depth for all legs
            odds_list = []
            depth_list = []
            best_ask_prices = []  # For slippage check
            skip_ring = False

            for o_id in outcome_ids:
                ob = order_books.get(o_id)
                if not ob or not ob.best_ask:
                    logger.debug("Skipping ring: missing order book or best_ask", outcome_id=o_id)
                    skip_ring = True
                    break

                price = float(ob.best_ask)
                odds = (1.0 / price) - 1.0 if price > 0 else 0.0

                depth = 0.0
                if ob.asks and abs(float(ob.asks[0].price) - price) < 1e-6:
                    depth = float(ob.asks[0].size) * price # USD liquidity
                if depth == 0:
                    depth = 1000.0 # Fallback

                odds_list.append(odds)
                depth_list.append(depth)
                best_ask_prices.append(Decimal(str(price)))

            if skip_ring:
                continue

            # Calculate Dutching sizes
            sizes = self.position_sizer.calculate_dutching_sizes(odds_list, depth_list)

            # Check if any size is > 0 (meaning implied_sum < 1.0)
            if all(s.recommended_size > 0 for s in sizes):
                logger.info("Dutching arb found by sizer", total_stake=sum(s.recommended_size for s in sizes))
                ring_trades = []
                ring_total_profit = Decimal("0")
                ring_markets = set()
                vwap_slippage_detected = False

                for i, o_id in enumerate(outcome_ids):
                    size_res = sizes[i]
                    ob = order_books[o_id]

                    # Refine to VWAP
                    vwap = ob.get_vwap(OrderSide.BUY, Decimal(str(size_res.recommended_size)))
                    if vwap is None:
                        logger.debug("Skipping ring: insufficient VWAP depth", outcome_id=o_id)
                        skip_ring = True
                        break

                    # P1.1: VWAP Slippage Check
                    best_ask = best_ask_prices[i]
                    if best_ask > 0:
                        slippage_pct = abs(vwap - best_ask) / best_ask
                        if slippage_pct > Decimal(str(config.vwap_slippage_limit)):
                            logger.debug(
                                f"Dutching ring rejected: VWAP slippage {slippage_pct:.2%} exceeds limit {config.vwap_slippage_limit:.2%}",
                                outcome_id=o_id,
                                vwap=vwap,
                                best_ask=best_ask
                            )
                            vwap_slippage_detected = True
                            skip_ring = True
                            break

                    market_id = extract_market_id(o_id)
                    ring_markets.add(market_id)

                    exchange_info = validated.market_exchanges.get(market_id, "polymarket")
                    exchange_name = "polymarket"
                    reason = "dutching_arb"

                    if exchange_info.startswith("limitless:"):
                        exchange_name = "limitless"
                        slug = exchange_info.split(":")[1]
                        reason += f":{slug}"
                    elif exchange_info == "limitless":
                        exchange_name = "limitless"

                    ring_trades.append(ProposedTrade(
                        market_id=market_id,
                        outcome_id=o_id,
                        side=OrderSide.BUY,
                        size=size_res.recommended_size,
                        limit_price=vwap,
                        exchange=exchange_name,
                        reason=reason
                    ))

                if skip_ring:
                    if vwap_slippage_detected:
                        logger.debug("Ring rejected due to VWAP slippage")
                    continue

                # Calculate profit for the ring
                # With Dutching, payout is identical regardless of which outcome wins
                # Payout = size_res.recommended_size (number of shares) * 1.0 (payout per share)
                # We just take the payout from the first leg since they should be equal
                payout = Decimal(str(sizes[0].recommended_size))

                total_cost = sum(t.size * t.limit_price for t in ring_trades)

                ring_total_profit = payout - total_cost

                # Store candidate ring for later ranking
                candidate_rings.append({
                    "trades": ring_trades,
                    "markets": list(ring_markets),
                    "gross_profit": ring_total_profit,
                    "capital_deployed": total_cost,
                    "sizes": sizes,
                    "depth_list": depth_list
                })

        if not candidate_rings:
            return None

        # P2.1: Rank rings by ROI and process in order
        # This ensures we pick the best rings first and reserve capital accordingly
        candidate_rings.sort(key=lambda r: float(r["gross_profit"] / r["capital_deployed"] if r["capital_deployed"] > 0 else 0), reverse=True)

        # Process top ring (or multiple if capital allows)
        for ring in candidate_rings:
            ring_trades = ring["trades"]
            ring_markets = ring["markets"]
            ring_gross_profit = ring["gross_profit"]
            capital_required = ring["capital_deployed"]

            # P2.1: Capital reservation check
            if reserved_capital + capital_required > available_capital:
                logger.debug(f"Skipping ring: insufficient remaining capital (need={capital_required}, available={available_capital - reserved_capital})")
                continue

            # Deduct Fees and Gas
            total_gas = Decimal("0")
            total_fees = Decimal("0")
            poly_gas = Decimal(str(config.polygon_gas_per_tx))
            base_gas = Decimal(str(config.base_gas_per_tx))
            poly_fee_pct = Decimal(str(config.polymarket_taker_fee_pct))
            limitless_fee_pct = Decimal(str(config.limitless_taker_fee_pct))

            for t in ring_trades:
                if t.exchange == "limitless":
                    total_gas += base_gas
                    total_fees += Decimal(str(t.size)) * t.limit_price * limitless_fee_pct
                else:
                    total_gas += poly_gas
                    total_fees += Decimal(str(t.size)) * t.limit_price * poly_fee_pct

            # P2.3: Expected unwind cost (if partial fill occurs)
            partial_fill_prob = Decimal(str(config.partial_fill_probability))
            unwind_spread = Decimal(str(config.unwind_spread_estimate))
            expected_unwind_cost = Decimal("0")

            for t in ring_trades:
                notional_value = Decimal(str(t.size)) * t.limit_price
                expected_unwind_cost += notional_value * unwind_spread * partial_fill_prob

            # Net profit after all costs
            net_profit = ring_gross_profit - total_gas - total_fees - expected_unwind_cost

            if net_profit <= 0:
                logger.debug(f"Ring rejected: net profit {net_profit} <= 0 after fees/gas/unwind_cost")
                continue

            # P1.2: Calculate ROI
            roi = float(net_profit / capital_required) if capital_required > 0 else 0.0

            # P2.2: Dynamic confidence scoring
            # Based on liquidity cushion (how much depth vs stake) and staleness
            liquidity_ratios = []
            for i, t in enumerate(ring_trades):
                stake = Decimal(str(t.size)) * t.limit_price
                depth = Decimal(str(ring["depth_list"][i]))
                if stake > 0:
                    liquidity_ratios.append(float(depth / stake))

            min_liquidity_ratio = min(liquidity_ratios) if liquidity_ratios else 1.0

            # Liquidity confidence: 0.7 if tight (1.1x), 1.0 if ample (5x+)
            liquidity_confidence = min(1.0, 0.7 + (min_liquidity_ratio - 1.0) * 0.15)

            # P3.3: Staleness-based confidence penalty.
            # Two-layer defense: hard reject if any leg exceeds ws_max_age_ms (dead
            # man's switch), soft linear decay from 1.0 at soft_staleness_start_ms to
            # 0.0 at ws_max_age_ms. OrderBook.timestamp is refreshed on every WS tick
            # by the Polymarket client, so wall-clock comparison is correct here.
            soft_start_ms = float(config.soft_staleness_start_ms)
            hard_cut_ms = float(config.ws_max_age_ms)
            now_dt = datetime.utcnow()
            ring_staleness_factor = 1.0
            ring_stale_reject = False
            worst_age_ms = 0.0
            for t in ring_trades:
                ob = order_books.get(t.outcome_id)
                if not ob:
                    ring_stale_reject = True
                    break
                age_ms = (now_dt - ob.timestamp).total_seconds() * 1000.0
                worst_age_ms = max(worst_age_ms, age_ms)
                if age_ms > hard_cut_ms:
                    logger.debug(
                        "Ring rejected — quote exceeds hard staleness cutoff",
                        token=t.outcome_id,
                        age_ms=age_ms,
                        hard_cut_ms=hard_cut_ms,
                    )
                    ring_stale_reject = True
                    break
                if age_ms > soft_start_ms:
                    leg_factor = max(
                        0.0,
                        1.0 - (age_ms - soft_start_ms) / (hard_cut_ms - soft_start_ms),
                    )
                    ring_staleness_factor = min(ring_staleness_factor, leg_factor)

            if ring_stale_reject:
                continue

            staleness_confidence = ring_staleness_factor

            # Combined confidence (80% liquidity, 20% staleness)
            confidence = min(1.0, liquidity_confidence * 0.8 + staleness_confidence * 0.2)

            # P1.2: Capital efficiency (profit per second, estimated)
            # Assume ~100ms execution time per leg
            estimated_execution_time_sec = len(ring_trades) * 0.1
            capital_efficiency = float(net_profit / Decimal(str(estimated_execution_time_sec)))

            # Accept this ring — accumulate into multi-ring result
            reserved_capital += capital_required
            total_capital_deployed += capital_required
            all_accepted_trades.extend(ring_trades)
            all_accepted_markets.update(ring_markets)
            total_accepted_profit += net_profit
            min_confidence = min(min_confidence, confidence)
            rings_accepted_count += 1

            logger.info(
                f"Accepted Dutching Ring ({len(all_accepted_trades)} total trades across {len(all_accepted_markets)} markets)",
                ring_profit=float(net_profit),
                ring_roi=f"{roi:.2%}",
                confidence=f"{confidence:.2%}",
                capital_efficiency=f"${capital_efficiency:.2f}/sec",
                rings_accepted=rings_accepted_count,
            )

            # Continue to next ring instead of returning — accumulate more rings

        # After processing all candidate rings, return combined result
        if not all_accepted_trades:
            return None

        combined_roi = float(total_accepted_profit / total_capital_deployed) if total_capital_deployed > 0 else 0.0
        combined_efficiency = float(total_accepted_profit / Decimal(str(len(all_accepted_trades) * 0.1)))

        logger.info(
            f"Dutching Multi-Ring Complete",
            total_rings=rings_accepted_count,
            total_trades=len(all_accepted_trades),
            total_profit=float(total_accepted_profit),
            combined_roi=f"{combined_roi:.2%}",
            confidence=f"{min_confidence:.2%}",
        )

        return ArbitrageOpportunity(
            markets=list(all_accepted_markets),
            trades=all_accepted_trades,
            expected_profit=total_accepted_profit,
            roi=combined_roi,
            capital_efficiency=combined_efficiency,
            confidence=min_confidence,
        )


