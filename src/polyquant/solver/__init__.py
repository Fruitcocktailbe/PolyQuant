"""
Solver module exports.
"""

from polyquant.solver.bregman import (
    BarrierFrankWolfe,
    ProjectionResult,
    bregman_project,
    compute_optimal_trades,
    detect_arbitrage,
    kl_divergence,
    kl_gradient,
    project_simplex,
)
from polyquant.solver.scip_solver import (
    InitFWResult,
    OptimizationResult,
    SCIPSolver,
    init_frank_wolfe,
)
from polyquant.solver.fw_solver import (
    FWSolver,
    ArbitrageDetector,
)

__all__ = [
    # SCIP Solver
    "SCIPSolver",
    "OptimizationResult",
    "init_frank_wolfe",
    "InitFWResult",
    # Frank-Wolfe Solver
    "FWSolver",
    "ArbitrageDetector",
    # Barrier Frank-Wolfe (NEW)
    "BarrierFrankWolfe",
    "ProjectionResult",
    # Bregman algorithms (Legacy)
    "kl_divergence",
    "kl_gradient",
    "project_simplex",
    "bregman_project",
    "detect_arbitrage",
    "compute_optimal_trades",
]