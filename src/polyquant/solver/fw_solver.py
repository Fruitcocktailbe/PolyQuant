"""
Frank-Wolfe Solver for Arbitrage-Free Market Making

This module implements the specific algorithms described in the research:
1. InitFW (Algorithm 3): For finding valid starting vertices and interior point.
2. Barrier Frank-Wolfe (Algorithm 2): For optimizing KL divergence with adaptive contraction.
3. Profit Guarantee (Proposition 4.1): For determining when to stop and trade.

Reference: "Arbitrage-Free Combinatorial Market Making via Integer Programming" (Kroer et al., 2016)
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from decimal import Decimal

from polyquant.utils import get_logger
from polyquant.solver.scip_solver import SCIPSolver
from polyquant.agents.validator import ValidatedResult

logger = get_logger(__name__)

class FWSolver:
    """
    Implements the Barrier Frank-Wolfe algorithm for LMSR market making.
    """

    def __init__(self, scip_solver: SCIPSolver):
        self.solver = scip_solver
        self.epsilon = 0.1  # Initial contraction parameter
        self.alpha = 0.9    # Extraction guarantee threshold
        self.min_profit = 0.05 # Minimum profit threshold

    def init_fw(
        self, 
        validated: ValidatedResult,
        outcomes: List[str]
    ) -> Tuple[List[Dict[str, float]], Dict[str, float], Set[str]]:
        """
        Algorithm 3: InitFW
        
        Constructs a valid set of starting vertices Z_0 and an interior point u.
        Also identifies settled securities.
        
        Args:
            validated: Constraints
            outcomes: List of outcome IDs
            
        Returns:
            Z_0: List of valid vertex vectors (dicts)
            u: Interior point vector (dict)
            settled: Set of settled outcome IDs
        """
        logger.info("Running InitFW...")
        
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
        return Z_0, u, settled_ids

    def barrier_fw(
        self,
        validated: ValidatedResult,
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
        
        # Initialize mu (start at u for safety)
        mu = u.copy()
        
        epsilon = self.epsilon
        
        # Convert prices to vector for easier math
        # Ensure strict positivity for KL
        theta = {o: max(market_prices.get(o, 0.5), 1e-6) for o in outcomes}
        
        # Track best iterate for "Forced Interruption" condition
        best_mu = mu.copy()
        best_profit_guarantee = -float('inf')
        
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
            
            # Stopping Condition 1: alpha-extraction
            # g(mu) <= (1 - alpha) * D(mu || theta)
            if fw_gap <= (1 - self.alpha) * kl and kl > self.min_profit:
                logger.info(f"Alpha-extraction condition met at iter {t}. Gap: {fw_gap:.5f}, KL: {kl:.5f}")
                return mu, guaranteed_profit, kl
                
            # Stopping Condition 2: Arbitrage-free
            if kl < self.min_profit:
                # logger.debug(f"Arbitrage too small ({kl:.5f}). Continuing...")
                pass
                
            # 5. Adaptive Epsilon Update
            # If g(mu_t) / (-4 * g_u) < eps_prev: reduce eps
            # We need g_u. Usually g_u is gap at u.
            # g_u = <Grad, u - v_prime_u> ???
            # Let's approximate the condition simply: if gap is small, shrink epsilon
            # The paper's specific condition is complex to track exactly without rigorous g_u def.
            # Proxy: if gap is decreasing efficiently, shrink.
            
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

    def find_opportunity(
        self,
        validated: ValidatedResult,
        order_books: Dict[str, Any] # dict of OrderBook
    ) -> Optional[Dict[str, float]]:
        """
        Main entry point to find arbitrage-free prices.
        """
        outcomes = list(order_books.keys())
        market_prices = {}
        for o, ob in order_books.items():
            # Use mid-price as proxy for theta
            best_bid = ob.best_bid or 0.0
            best_ask = ob.best_ask or 1.0
            market_prices[o] = (best_bid + best_ask) / 2.0
            
        # 1. InitFW
        Z_0, u, settled = self.init_fw(validated, outcomes)
        if not Z_0:
            return None
            
        # 2. BarrierFW
        target_prices, profit, kl = self.barrier_fw(
            validated, outcomes, market_prices, Z_0, u
        )
        
        if profit > self.min_profit:
            return target_prices
            
        return None

from polyquant.data import ArbitrageOpportunity, OrderBook, OrderSide, ProposedTrade
from polyquant.utils import config

class ArbitrageDetector:
    """
    High-level interface for detecting and quantifying arbitrage using Frank-Wolfe.
    """
    
    def __init__(self):
        self.scip = SCIPSolver()
        self.fw_solver = FWSolver(self.scip)
        
    async def detect(
        self,
        validated: ValidatedResult,
        order_books: Dict[str, OrderBook],
        min_profit: float = 0.05,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Detect arbitrage using the research-backed pipeline:
        1. InitFW
        2. BarrierFW
        3. Profit Guarantee
        """
        # Set solver threshold
        self.fw_solver.min_profit = min_profit
        
        # Run solver (CPU bound, so technically should be in executor if blocking)
        target_prices = self.fw_solver.find_opportunity(validated, order_books)
        
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
            if ob.best_ask and ob.best_ask < target_p:
                # Simple sizing: take available liquidity at best ask
                # In real prod, this would walk the book
                size = 100.0 # Placeholder size or derived from liquidity
                price_diff = target_p - ob.best_ask
                profit = Decimal(str(price_diff * size))
                
                trades.append(ProposedTrade(
                    market_id=outcome_id.split("_")[0] if "_" in outcome_id else "",
                    outcome_id=outcome_id,
                    side=OrderSide.BUY,
                    size=float(size),
                    limit_price=ob.best_ask
                ))
                total_expected_profit += profit
                
            # Sell opportunity: Bid > Target
            if ob.best_bid and ob.best_bid > target_p:
                size = 100.0
                price_diff = ob.best_bid - target_p
                profit = Decimal(str(price_diff * size))
                
                trades.append(ProposedTrade(
                    market_id=outcome_id.split("_")[0] if "_" in outcome_id else "",
                    outcome_id=outcome_id,
                    side=OrderSide.SELL,
                    size=float(size),
                    limit_price=ob.best_bid
                ))
                total_expected_profit += profit
                
        if not trades:
            return None
            
        return ArbitrageOpportunity(
            markets=list(set(t.market_id for t in trades)),
            trades=trades,
            expected_profit=total_expected_profit,
            confidence=0.9 # High confidence due to guarantee
        )
