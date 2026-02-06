"""
Solver module exports.
"""

from polyquant.solver.bregman import (
    bregman_project,
    compute_optimal_trades,
    detect_arbitrage,
    kl_divergence,
    kl_gradient,
    project_simplex,
)
from polyquant.solver.scip_solver import (
    ArbitrageDetector,
    OptimizationResult,
    SCIPSolver,
)

__all__ = [
    # SCIP Solver
    "SCIPSolver",
    "OptimizationResult",
    "ArbitrageDetector",
    # Bregman algorithms
    "kl_divergence",
    "kl_gradient",
    "project_simplex",
    "bregman_project",
    "detect_arbitrage",
    "compute_optimal_trades",
]
