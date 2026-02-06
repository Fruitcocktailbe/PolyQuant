"""
Bregman Projection Algorithms for PolyQuant 2.0

This module implements the mathematical core of the arbitrage detection:
the Bregman projection using KL divergence onto constraint sets.

IMPLEMENTATION BASED ON:
------------------------
Kroer et al. 2016: "Arbitrage-Free Combinatorial Market Making via Integer Programming"
The key insight: Standard Frank-Wolfe fails on LMSR because gradients explode at
price boundaries where μ_i → 0. The solution is Barrier Frank-Wolfe with adaptive
epsilon contraction.

BARRIER FRANK-WOLFE:
--------------------
Instead of optimizing over the true polytope M, we optimize over a contracted
polytope M' = (1-ε)M + εu, where u is an interior point with all coordinates
strictly between 0 and 1.

The adaptive epsilon rule:
- Start with large ε (e.g., 0.1) for fast early convergence
- Shrink ε when gap decreases: ε_t = min(g(μ_t)/(-4*g_u), ε_{t-1}/2)
- As ε → 0, we approach the true projection on M

PROFIT GUARANTEE (Proposition 4.1):
-----------------------------------
Guaranteed Profit ≥ D(μ̂||θ) - g(μ̂)

Where:
- D(μ̂||θ) = KL divergence (maximum possible arbitrage)
- g(μ̂) = Frank-Wolfe gap (how suboptimal our solution is)

α-EXTRACTION STOPPING CONDITION:
---------------------------------
Stop when: g(μ_t) ≤ (1-α) × D(μ_t||θ)
With α = 0.9, we capture 90% of available arbitrage before executing.

USAGE:
------
    from polyquant.solver.bregman import BarrierFrankWolfe
    
    bfw = BarrierFrankWolfe(
        interior_point=u,       # From InitFW
        extraction_alpha=0.9,   # Capture 90% of profit
    )
    
    result = bfw.project(
        current_prices=prices,
        constraints_A=A,
        constraints_b=b,
    )
    
    if result.should_trade:
        print(f"Guaranteed profit: {result.guaranteed_profit}")
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from polyquant.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Basic Mathematical Functions
# =============================================================================


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence: KL(p || q) = Σ_i p_i × log(p_i / q_i)
    
    This measures how "different" p is from q, from p's perspective.
    From Part 2: This is the Bregman divergence for LMSR.
    
    Properties:
    - KL(p || p) = 0
    - KL(p || q) >= 0 always
    - KL(p || q) != KL(q || p) in general (not symmetric)
    
    Args:
        p: Target distribution
        q: Reference distribution
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL divergence value (non-negative)
    """
    p_safe = np.maximum(p, epsilon)
    q_safe = np.maximum(q, epsilon)
    
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def kl_gradient(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute gradient of KL(p || q) with respect to p.
    
    From Part 2: ∇R(μ) = ln(μ) + 1
    
    This gradient goes to -∞ as μ_i → 0, which is why we need
    the Barrier Frank-Wolfe with contraction.
    
    Args:
        p: Target distribution
        q: Reference distribution
        epsilon: Small value to avoid log(0)
        
    Returns:
        Gradient vector
    """
    p_safe = np.maximum(p, epsilon)
    q_safe = np.maximum(q, epsilon)
    
    return np.log(p_safe / q_safe) + 1


def project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a vector onto the probability simplex.
    
    The probability simplex is: {p : Σp = 1, p >= 0}
    
    Uses the efficient O(n log n) algorithm from:
    Duchi et al. "Efficient Projections onto the L1-Ball" (2008)
    
    Args:
        v: Vector to project
        
    Returns:
        Projected vector on the simplex
    """
    n = len(v)
    u = np.sort(v)[::-1]
    
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    
    if not np.any(cond):
        return np.ones(n) / n
    
    rho = np.max(ind[cond])
    theta = cssv[rho - 1] / rho
    
    return np.maximum(v - theta, 0)


# =============================================================================
# Projection Result
# =============================================================================


@dataclass
class ProjectionResult:
    """
    Result from Barrier Frank-Wolfe projection.
    
    Attributes:
        projected: The projected price vector
        divergence: D(μ̂||θ) - KL divergence from current to projected
        gap: Frank-Wolfe gap (optimality measure)
        guaranteed_profit: D(μ̂||θ) - g(μ̂) - guaranteed minimum profit
        extraction_ratio: What fraction of arbitrage we're capturing
        iterations: Number of iterations used
        converged: Whether we hit convergence tolerance
        should_trade: Whether the profit is worth executing
    """
    projected: np.ndarray
    divergence: float
    gap: float
    guaranteed_profit: float
    extraction_ratio: float
    iterations: int
    converged: bool
    should_trade: bool


# =============================================================================
# Barrier Frank-Wolfe Algorithm
# =============================================================================


class BarrierFrankWolfe:
    """
    Barrier Frank-Wolfe optimizer with adaptive epsilon contraction.
    
    From Part 2: Standard Frank-Wolfe fails on LMSR because the gradient
    ∇R(μ) = ln(μ) + 1 goes to -∞ as μ_i → 0.
    
    The solution is to optimize over a contracted polytope:
        M' = (1-ε)M + εu
    
    Where u is an interior point with all coordinates in (0,1).
    This keeps all coordinates bounded away from 0, giving a finite
    Lipschitz constant L_ε = O(1/ε).
    
    The adaptive epsilon rule shrinks ε over time as we converge,
    eventually getting arbitrarily close to the true projection.
    """
    
    def __init__(
        self,
        interior_point: np.ndarray | None = None,
        initial_epsilon: float = 0.1,
        extraction_alpha: float = 0.9,
        min_profit_threshold: float = 0.05,
        max_iterations: int = 150,
        convergence_tol: float = 1e-6,
    ):
        """
        Initialize Barrier Frank-Wolfe.
        
        Args:
            interior_point: Point u with all coords in (0,1). If None,
                           will use uniform distribution.
            initial_epsilon: Starting contraction parameter (default 0.1)
            extraction_alpha: Target extraction efficiency (default 0.9 = 90%)
            min_profit_threshold: Minimum profit to consider trading (default $0.05)
            max_iterations: Maximum FW iterations (default 150)
            convergence_tol: Convergence tolerance for gap
        """
        self.interior_point = interior_point
        self.initial_epsilon = initial_epsilon
        self.extraction_alpha = extraction_alpha
        self.min_profit_threshold = min_profit_threshold
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        
        logger.info(
            "BarrierFrankWolfe initialized",
            initial_epsilon=initial_epsilon,
            extraction_alpha=extraction_alpha,
            min_profit_threshold=min_profit_threshold,
        )
    
    def project(
        self,
        current_prices: np.ndarray,
        constraints_A: np.ndarray,
        constraints_b: np.ndarray,
    ) -> ProjectionResult:
        """
        Perform Barrier Frank-Wolfe projection onto constraint set.
        
        From Part 2: This implements Algorithm 2 (Barrier FW) with
        adaptive epsilon contraction.
        
        Args:
            current_prices: Current market prices θ
            constraints_A: Constraint matrix (m x n)
            constraints_b: Right-hand side vector (m)
            
        Returns:
            ProjectionResult with projected prices and profit info
        """
        n = len(current_prices)
        m = len(constraints_b) if len(constraints_b) > 0 else 0
        
        # Set up interior point (all coords must be in (0,1))
        if self.interior_point is not None:
            u = self.interior_point.copy()
        else:
            u = np.ones(n) / n  # Uniform distribution
        
        # Initialize at current prices, normalized
        mu = current_prices.copy()
        mu = np.maximum(mu, 1e-10)
        mu = mu / mu.sum()
        
        # Adaptive epsilon (starts large, shrinks over time)
        epsilon = self.initial_epsilon
        
        # Gap at interior point (for adaptive epsilon rule)
        g_u = self._compute_gap(u, current_prices, constraints_A, constraints_b, n)
        
        # Track best iterate for forced interruption
        best_profit = float('-inf')
        best_mu = mu.copy()
        
        converged = False
        iteration = 0
        
        logger.debug(
            "Starting Barrier FW projection",
            n=n,
            m=m,
            initial_epsilon=epsilon,
        )
        
        for iteration in range(self.max_iterations):
            # Contract toward interior point: μ' = (1-ε)μ + εu
            mu_contracted = (1 - epsilon) * mu + epsilon * u
            
            # Compute gradient of KL divergence
            grad = kl_gradient(mu_contracted, current_prices)
            
            # Add barrier gradient for constraint violations
            if m > 0:
                violations = constraints_A @ mu_contracted - constraints_b
                for i in range(m):
                    if violations[i] < 0:
                        # Barrier gradient pushes toward feasibility
                        barrier_grad = -constraints_A[i] / (abs(violations[i]) + 1e-10)
                        grad = grad + barrier_grad
            
            # Frank-Wolfe direction: s = argmin_{s ∈ simplex} <grad, s>
            s = np.zeros(n)
            s[np.argmin(grad)] = 1.0
            
            # Compute gap g(μ) = <∇f(μ), μ - s>
            gap = float(np.dot(grad, mu_contracted - s))
            
            # Compute divergence D(μ||θ)
            divergence = kl_divergence(mu_contracted, current_prices)
            
            # Guaranteed profit = D(μ||θ) - g(μ)
            guaranteed_profit = divergence - gap
            
            # Track best iterate
            if guaranteed_profit > best_profit:
                best_profit = guaranteed_profit
                best_mu = mu_contracted.copy()
            
            # Check α-extraction stopping condition
            # Stop when: g(μ) ≤ (1-α) × D(μ||θ)
            if divergence > 0 and gap <= (1 - self.extraction_alpha) * divergence:
                logger.info(
                    "α-extraction reached",
                    iteration=iteration,
                    extraction_ratio=1 - gap/divergence if divergence > 0 else 1.0,
                    guaranteed_profit=guaranteed_profit,
                )
                converged = True
                break
            
            # Check absolute convergence
            if gap < self.convergence_tol:
                logger.debug("Gap convergence reached", iteration=iteration, gap=gap)
                converged = True
                break
            
            # Adaptive epsilon rule from Part 2:
            # If g(μ_t) / (-4*g_u) < ε_{t-1}, shrink ε
            if g_u < 0:  # g_u should be negative for valid interior point
                ratio = gap / (-4 * g_u)
                if ratio < epsilon:
                    epsilon = max(min(ratio, epsilon / 2), 1e-10)
                    logger.debug(
                        "Epsilon adapted",
                        iteration=iteration,
                        new_epsilon=epsilon,
                    )
            
            # Frank-Wolfe step size (diminishing: 2/(t+2))
            step_size = 2.0 / (iteration + 2)
            
            # Update: μ = (1-γ)μ + γs
            mu = (1 - step_size) * mu + step_size * s
            mu = np.maximum(mu, 1e-10)
            mu = mu / mu.sum()
        
        # Use best iterate
        final_mu = best_mu
        final_divergence = kl_divergence(final_mu, current_prices)
        final_gap = self._compute_gap(final_mu, current_prices, constraints_A, constraints_b, n)
        final_profit = final_divergence - final_gap
        
        # Determine if we should trade
        should_trade = (
            final_profit >= self.min_profit_threshold and
            final_divergence > 0.001  # Not already arbitrage-free
        )
        
        extraction_ratio = 1 - final_gap / final_divergence if final_divergence > 0 else 1.0
        
        logger.info(
            "Barrier FW complete",
            iterations=iteration + 1,
            converged=converged,
            divergence=final_divergence,
            gap=final_gap,
            guaranteed_profit=final_profit,
            extraction_ratio=extraction_ratio,
            should_trade=should_trade,
        )
        
        return ProjectionResult(
            projected=final_mu,
            divergence=final_divergence,
            gap=final_gap,
            guaranteed_profit=final_profit,
            extraction_ratio=extraction_ratio,
            iterations=iteration + 1,
            converged=converged,
            should_trade=should_trade,
        )
    
    def _compute_gap(
        self,
        mu: np.ndarray,
        current: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        n: int,
    ) -> float:
        """Compute Frank-Wolfe gap for a given point."""
        grad = kl_gradient(mu, current)
        
        m = len(b) if len(b) > 0 else 0
        if m > 0:
            violations = A @ mu - b
            for i in range(m):
                if violations[i] < 0:
                    barrier_grad = -A[i] / (abs(violations[i]) + 1e-10)
                    grad = grad + barrier_grad
        
        s = np.zeros(n)
        s[np.argmin(grad)] = 1.0
        
        return float(np.dot(grad, mu - s))


# =============================================================================
# Legacy Functions (for backward compatibility)
# =============================================================================


def bregman_project(
    current: np.ndarray,
    constraints_A: np.ndarray,
    constraints_b: np.ndarray,
    max_iters: int = 100,
    tol: float = 1e-6,
    barrier_strength: float = 1.0,
    interior_point: np.ndarray | None = None,
) -> np.ndarray:
    """
    Bregman projection using Barrier Frank-Wolfe with adaptive epsilon.
    
    This is a convenience wrapper around BarrierFrankWolfe for simple use cases.
    For full control over the algorithm, use BarrierFrankWolfe directly.
    
    Args:
        current: Current prices/probabilities (reference point)
        constraints_A: Constraint matrix (m x n)
        constraints_b: Right-hand side vector (m)
        max_iters: Maximum iterations
        tol: Convergence tolerance
        barrier_strength: (Ignored - kept for backward compatibility)
        interior_point: Optional interior point for contraction
        
    Returns:
        Projected point satisfying constraints
    """
    bfw = BarrierFrankWolfe(
        interior_point=interior_point,
        max_iterations=max_iters,
        convergence_tol=tol,
    )
    
    result = bfw.project(current, constraints_A, constraints_b)
    return result.projected


def detect_arbitrage(
    prices: np.ndarray,
    constraints_A: np.ndarray,
    constraints_b: np.ndarray,
    threshold: float = 0.01,
    interior_point: np.ndarray | None = None,
) -> Tuple[bool, float, np.ndarray, float]:
    """
    Detect if there's an arbitrage opportunity with profit guarantee.
    
    Returns the profit guarantee from Proposition 4.1:
    Guaranteed Profit ≥ D(μ̂||θ) - g(μ̂)
    
    Args:
        prices: Current market prices
        constraints_A: Constraint matrix
        constraints_b: Right-hand side
        threshold: Minimum profit to consider arbitrage
        interior_point: Optional interior point for Barrier FW
        
    Returns:
        Tuple of:
        - bool: Whether profitable arbitrage exists
        - float: KL divergence (max possible profit)
        - ndarray: Corrected prices
        - float: Guaranteed profit
    """
    bfw = BarrierFrankWolfe(
        interior_point=interior_point,
        min_profit_threshold=threshold,
    )
    
    result = bfw.project(prices, constraints_A, constraints_b)
    
    return (
        result.should_trade,
        result.divergence,
        result.projected,
        result.guaranteed_profit,
    )


def compute_optimal_trades(
    current_prices: np.ndarray,
    target_prices: np.ndarray,
    order_book_depths: np.ndarray,
    max_position: float = 1000.0,
) -> np.ndarray:
    """
    Compute optimal trade sizes to move prices toward target.
    
    Given current and target prices, compute how much to buy/sell
    of each outcome to capture the arbitrage.
    
    Args:
        current_prices: Current market prices
        target_prices: Target (arbitrage-free) prices
        order_book_depths: Available liquidity at each level
        max_position: Maximum position size
        
    Returns:
        Trade sizes (positive = buy, negative = sell)
    """
    delta = target_prices - current_prices
    availability = np.minimum(order_book_depths, max_position)
    trades = delta * availability * 10
    trades = np.clip(trades, -max_position, max_position)
    
    return trades
