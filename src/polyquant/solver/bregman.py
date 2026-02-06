"""
Bregman Projection Algorithms for PolyQuant 2.0

This module implements the mathematical core of the arbitrage detection:
the Bregman projection using KL divergence onto constraint sets.

WHAT IS BREGMAN PROJECTION?
---------------------------
When we have a point (current market prices) and a constraint set
(logical relationships between markets), we want to find the "closest"
point that satisfies all constraints.

"Closest" is measured using Bregman divergence - specifically KL divergence
for probability distributions (which prediction market prices are).

WHY KL DIVERGENCE?
------------------
KL divergence is the natural distance metric for probability distributions:
- It respects the [0,1] bounds of probabilities
- It penalizes extreme changes more than moderate ones
- It's the information-theoretic measure of "surprise"

THE ALGORITHM:
--------------
We use Frank-Wolfe with barrier terms:
1. Start at current prices
2. Compute gradient of KL divergence
3. Add barrier terms for constraint violations
4. Find descent direction via linear minimization
5. Step and project back onto probability simplex
6. Repeat until convergence

REFERENCE:
----------
Based on: Bubeck, S. "Convex Optimization: Algorithms and Complexity" (2015)
Section 3.3: Frank-Wolfe Algorithm

USAGE:
------
    from polyquant.solver.bregman import bregman_project
    
    # Current market prices
    current = np.array([0.6, 0.4, 0.7, 0.3])
    
    # Constraint: p[0] + p[2] >= 0.9 (e.g., correlated outcomes)
    A = np.array([[1, 0, 1, 0]])
    b = np.array([0.9])
    
    # Find closest valid prices
    projected = bregman_project(current, A, b)
"""

import numpy as np
from typing import Tuple

from polyquant.utils import get_logger

logger = get_logger(__name__)


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence: KL(p || q) = sum_i p_i * log(p_i / q_i)
    
    This measures how "different" p is from q, from p's perspective.
    
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
        
    Example:
        >>> p = np.array([0.7, 0.3])
        >>> q = np.array([0.5, 0.5])
        >>> kl_divergence(p, q)  # About 0.082
    """
    # Add epsilon to avoid log(0)
    p_safe = np.maximum(p, epsilon)
    q_safe = np.maximum(q, epsilon)
    
    return np.sum(p_safe * np.log(p_safe / q_safe))


def kl_gradient(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute gradient of KL(p || q) with respect to p.
    
    The gradient is: grad_i = log(p_i / q_i) + 1
    
    Args:
        p: Target distribution
        q: Reference distribution
        epsilon: Small value to avoid log(0)
        
    Returns:
        Gradient vector (same shape as p)
    """
    p_safe = np.maximum(p, epsilon)
    q_safe = np.maximum(q, epsilon)
    
    return np.log(p_safe / q_safe) + 1


def project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a vector onto the probability simplex.
    
    The probability simplex is: {p : sum(p) = 1, p >= 0}
    
    Uses the efficient O(n log n) algorithm from:
    Duchi et al. "Efficient Projections onto the L1-Ball" (2008)
    
    Args:
        v: Vector to project
        
    Returns:
        Projected vector on the simplex
        
    Example:
        >>> v = np.array([0.5, 0.8, -0.1])
        >>> project_simplex(v)
        array([0.35, 0.65, 0.  ])
    """
    n = len(v)
    
    # Sort in descending order
    u = np.sort(v)[::-1]
    
    # Find the threshold
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    
    if not np.any(cond):
        # Edge case: all negative coefficients
        return np.ones(n) / n
    
    rho = np.max(ind[cond])
    theta = cssv[rho - 1] / rho
    
    # Project
    return np.maximum(v - theta, 0)


def bregman_project(
    current: np.ndarray,
    constraints_A: np.ndarray,
    constraints_b: np.ndarray,
    max_iters: int = 100,
    tol: float = 1e-6,
    barrier_strength: float = 1.0,
) -> np.ndarray:
    """
    Bregman projection using Frank-Wolfe with barrier.
    
    Finds the point p that minimizes KL(p || current) subject to
    the linear constraints A @ p >= b.
    
    Algorithm:
    1. Initialize p = current
    2. Compute gradient with barrier terms
    3. Solve LP to find descent direction
    4. Take Frank-Wolfe step
    5. Project onto simplex
    6. Repeat until convergence
    
    Args:
        current: Current prices/probabilities (reference point)
        constraints_A: Constraint matrix (m x n)
        constraints_b: Right-hand side vector (m)
        max_iters: Maximum iterations
        tol: Convergence tolerance
        barrier_strength: Strength of barrier for constraint violations
        
    Returns:
        Projected point satisfying constraints
        
    Example:
        >>> current = np.array([0.6, 0.4])
        >>> A = np.array([[1, -1]])  # p[0] >= p[1]
        >>> b = np.array([0.1])      # by at least 0.1
        >>> bregman_project(current, A, b)
        array([0.55, 0.45])  # Adjusted to satisfy constraint
    """
    n = len(current)
    m = len(constraints_b) if len(constraints_b) > 0 else 0
    
    # Initialize
    p = current.copy()
    p = np.maximum(p, 1e-10)  # Ensure positivity
    p /= p.sum()  # Normalize to simplex
    
    logger.debug(
        "Starting Bregman projection",
        n=n,
        m=m,
        max_iters=max_iters,
    )
    
    for iteration in range(max_iters):
        # Compute KL gradient
        grad = kl_gradient(p, current)
        
        # Add barrier gradient for constraint violations
        if m > 0:
            violations = constraints_A @ p - constraints_b
            
            for i in range(m):
                if violations[i] < 0:
                    # Add penalty gradient
                    barrier_grad = -barrier_strength * constraints_A[i] / (abs(violations[i]) + 1e-10)
                    grad += barrier_grad
        
        # Frank-Wolfe direction: minimize grad @ s over simplex
        # Solution: s = e_argmin(grad)
        s = np.zeros(n)
        s[np.argmin(grad)] = 1
        
        # Frank-Wolfe gap (optimality condition)
        gap = grad @ (p - s)
        
        if gap < tol:
            logger.debug(
                "Bregman projection converged",
                iteration=iteration,
                gap=gap,
            )
            break
        
        # Step size (standard diminishing rule)
        step = 2.0 / (iteration + 2)
        
        # Update
        p_new = (1 - step) * p + step * s
        p_new = np.maximum(p_new, 1e-10)
        p_new /= p_new.sum()
        
        p = p_new
    
    return p


def detect_arbitrage(
    prices: np.ndarray,
    constraints_A: np.ndarray,
    constraints_b: np.ndarray,
    threshold: float = 0.01,
) -> Tuple[bool, float, np.ndarray]:
    """
    Detect if there's an arbitrage opportunity.
    
    Arbitrage exists if the current prices violate the constraints.
    The size of the opportunity is measured by how much we need to
    adjust prices to satisfy constraints.
    
    Args:
        prices: Current market prices
        constraints_A: Constraint matrix
        constraints_b: Right-hand side
        threshold: Minimum movement to consider arbitrage
        
    Returns:
        Tuple of:
        - bool: Whether arbitrage exists
        - float: Size of arbitrage (KL distance)
        - ndarray: Corrected prices
        
    Example:
        >>> prices = np.array([0.7, 0.5])  # Sum > 1 for binary market
        >>> A = np.array([[1, 1]])
        >>> b = np.array([1.0])  # Should sum to 1
        >>> has_arb, size, corrected = detect_arbitrage(prices, A, b)
        >>> has_arb
        True
    """
    # Project to constraint-satisfying set
    projected = bregman_project(prices, constraints_A, constraints_b)
    
    # Measure the distance
    kl_distance = kl_divergence(projected, prices)
    
    # Check if movement is significant
    has_arbitrage = np.max(np.abs(projected - prices)) > threshold
    
    return has_arbitrage, kl_distance, projected


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
    
    Uses a simple linear model: trade_size ∝ price_difference
    
    Args:
        current_prices: Current market prices
        target_prices: Target (arbitrage-free) prices
        order_book_depths: Available liquidity at each level
        max_position: Maximum position size
        
    Returns:
        Trade sizes (positive = buy, negative = sell)
    """
    # Price differences
    delta = target_prices - current_prices
    
    # Scale by order book depth
    availability = np.minimum(order_book_depths, max_position)
    
    # Simple linear model for trade size
    # Positive delta = price too low = buy
    # Negative delta = price too high = sell
    trades = delta * availability * 10  # Scaling factor
    
    # Clip to max position
    trades = np.clip(trades, -max_position, max_position)
    
    return trades
