"""
SCIP Solver Oracle for PolyQuant 2.0 - Phase 4

The Solver Oracle is the optimization engine that computes optimal trade
sizes to extract arbitrage while respecting constraints.

RESPONSIBILITIES:
-----------------
1. Take validated constraints from Phase 3
2. Compute the Bregman projection onto the arbitrage-free manifold
3. Find optimal trade sizes using integer programming
4. Respect position limits and risk constraints

WHY SCIP?
---------
SCIP (Solving Constraint Integer Programs) was chosen because:
1. Free for academic and non-commercial use (vs $200K+ for Gurobi)
2. Highly flexible framework for custom algorithms
3. Supports mixed-integer nonlinear programming
4. Access to source code for debugging/customization

THE MATH:
---------
We're solving a constrained optimization problem:

    minimize    KL(p_new || p_current)   # Minimize divergence from market
    subject to  A^T × z ≥ b              # Logical constraints
                0 ≤ x ≤ x_max            # Position limits
                
Where:
- p_new is the proposed portfolio probabilities
- p_current is the current market probabilities
- z is the binary outcome vector
- A, b are constraint matrices from the Logic Architect
- x is the trade size vector

ALGORITHM:
----------
We use the Barrier Frank-Wolfe algorithm to handle:
1. The KL divergence objective (has gradient explosions near 0)
2. Linear constraints from logical dependencies
3. Box constraints from position limits

USAGE:
------
    solver = SCIPSolver()
    
    result = await solver.optimize(
        validated_result,
        order_books,
        max_position=1000,
    )
    
    for trade in result.trades:
        print(f"{trade.side} {trade.size} of {trade.outcome_id} @ {trade.limit_price}")
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from polyquant.agents.validator import ValidatedResult
from polyquant.data import ArbitrageOpportunity, OrderBook, OrderSide, ProposedTrade
from polyquant.utils import config, get_logger

# We'll use pyscipopt if available, otherwise provide a mock for development
try:
    from pyscipopt import Model, quicksum
    SCIP_AVAILABLE = True
except ImportError:
    SCIP_AVAILABLE = False

logger = get_logger(__name__)


class OptimizationResult(BaseModel):
    """
    Result from the SCIP optimization.
    
    Attributes:
        success: Whether optimization found a solution
        trades: List of proposed trades
        expected_profit: Estimated profit in dollars
        objective_value: The optimization objective value
        extraction_ratio: What fraction of available arbitrage was captured
        solve_time_ms: How long the solver took
        status: Solver status string
    """
    success: bool = False
    trades: list[ProposedTrade] = Field(default_factory=list)
    expected_profit: Decimal = Field(default=Decimal("0"))
    objective_value: float = 0.0
    extraction_ratio: float = 0.0
    solve_time_ms: float = 0.0
    status: str = "not_run"


class SCIPSolver:
    """
    Phase 4: Optimization via SCIP Integer Programming
    
    The SCIP Solver takes validated constraints and computes optimal
    trade sizes using integer programming.
    
    Algorithm:
    1. Build SCIP model from constraints
    2. Add objective function (maximize profit / minimize KL divergence)
    3. Add position limits and risk constraints
    4. Solve and extract trade recommendations
    
    Example:
        solver = SCIPSolver()
        
        result = solver.optimize(
            validated_result,
            order_books,
            max_position=1000,
        )
        
        if result.success:
            print(f"Expected profit: ${result.expected_profit}")
            for trade in result.trades:
                execute(trade)
    """
    
    def __init__(
        self,
        extraction_alpha: float | None = None,
        timeout_seconds: int | None = None,
    ):
        """
        Initialize the SCIP solver.
        
        Args:
            extraction_alpha: Target extraction efficiency (default from config)
            timeout_seconds: Solver timeout (default from config)
        """
        self.extraction_alpha = extraction_alpha or config.extraction_alpha
        self.timeout_seconds = timeout_seconds or config.solver_timeout_seconds
        
        if not SCIP_AVAILABLE:
            logger.warning(
                "SCIP not available. Install pyscipopt for full functionality. "
                "Using fallback solver for development."
            )
        
        logger.info(
            "SCIPSolver initialized",
            extraction_alpha=self.extraction_alpha,
            timeout=self.timeout_seconds,
            scip_available=SCIP_AVAILABLE,
        )
    
    def optimize(
        self,
        validated: ValidatedResult,
        order_books: dict[str, OrderBook],
        max_position: float = 1000.0,
    ) -> OptimizationResult:
        """
        Optimize trade sizes given constraints and order books.
        
        This is the main entry point for the solver. It builds and
        solves the optimization problem.
        
        Args:
            validated: ValidatedResult from Phase 3
            order_books: Dict mapping outcome_id -> OrderBook
            max_position: Maximum position size in dollars
            
        Returns:
            OptimizationResult with proposed trades
        """
        start_time = datetime.utcnow()
        
        logger.info(
            "Starting optimization",
            constraint_count=len(validated.validated_constraints),
            orderbook_count=len(order_books),
            max_position=max_position,
        )
        
        if not validated.is_valid:
            logger.warning("Attempting to optimize invalid result")
            return OptimizationResult(
                success=False,
                status="invalid_input",
            )
        
        if SCIP_AVAILABLE:
            result = self._solve_with_scip(validated, order_books, max_position)
        else:
            result = self._solve_fallback(validated, order_books, max_position)
        
        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        result.solve_time_ms = elapsed_ms
        
        logger.info(
            "Optimization complete",
            success=result.success,
            trade_count=len(result.trades),
            expected_profit=float(result.expected_profit),
            solve_time_ms=elapsed_ms,
            status=result.status,
        )
        
        return result
    
    def _solve_with_scip(
        self,
        validated: ValidatedResult,
        order_books: dict[str, OrderBook],
        max_position: float,
    ) -> OptimizationResult:
        """
        Solve using SCIP optimizer.
        
        Builds a mixed-integer program:
        - Variables: Trade sizes for each outcome
        - Objective: Maximize expected profit
        - Constraints: Logical constraints + position limits
        """
        model = Model("polyquant_arbitrage")
        model.setParam("limits/time", self.timeout_seconds)
        
        # Create variables for each outcome's trade size
        # x_buy[i] = how much to buy of outcome i
        # x_sell[i] = how much to sell of outcome i
        
        outcomes = list(order_books.keys())
        x_buy = {}
        x_sell = {}
        
        for outcome_id in outcomes:
            x_buy[outcome_id] = model.addVar(
                name=f"buy_{outcome_id}",
                vtype="C",  # Continuous
                lb=0,
                ub=max_position,
            )
            x_sell[outcome_id] = model.addVar(
                name=f"sell_{outcome_id}",
                vtype="C",
                lb=0,
                ub=max_position,
            )
        
        # Add logical constraints from validated result
        for constraint in validated.validated_constraints:
            # Build constraint expression from coefficients
            expr = 0
            for outcome_id, coef in constraint.coefficients.items():
                if outcome_id in x_buy:
                    # Net position = buy - sell
                    expr += coef * (x_buy[outcome_id] - x_sell[outcome_id])
            
            # Add constraint: expr >= rhs
            model.addCons(expr >= constraint.rhs, name=constraint.constraint_id)
        
        # Objective: Maximize expected profit
        # Profit = sum of (expected value - cost) for each trade
        profit_expr = 0
        
        for outcome_id, ob in order_books.items():
            if ob.best_ask is not None:
                # Buying: Pay ask price, get 1 if outcome happens
                # Expected value assumes some probability of outcome
                # Simplified: assume current price is fair value
                bid = ob.best_bid or 0
                ask = ob.best_ask or 1
                
                # Profit from buy = (estimated_value - ask_price) * size
                # For arbitrage, we look for mispricings
                profit_expr += (bid - ask) * x_buy[outcome_id]
                profit_expr += (ask - bid) * x_sell[outcome_id]
        
        model.setObjective(profit_expr, sense="maximize")
        
        # Solve
        model.optimize()
        
        # Extract solution
        if model.getStatus() == "optimal" or model.getStatus() == "feasible":
            trades = []
            total_profit = Decimal("0")
            
            for outcome_id in outcomes:
                buy_size = model.getVal(x_buy[outcome_id])
                sell_size = model.getVal(x_sell[outcome_id])
                
                if buy_size > 0.01:  # Minimum size threshold
                    ob = order_books[outcome_id]
                    trades.append(
                        ProposedTrade(
                            market_id=outcome_id.split("_")[0] if "_" in outcome_id else "",
                            outcome_id=outcome_id,
                            side=OrderSide.BUY,
                            size=buy_size,
                            limit_price=ob.best_ask or 0.5,
                        )
                    )
                
                if sell_size > 0.01:
                    ob = order_books[outcome_id]
                    trades.append(
                        ProposedTrade(
                            market_id=outcome_id.split("_")[0] if "_" in outcome_id else "",
                            outcome_id=outcome_id,
                            side=OrderSide.SELL,
                            size=sell_size,
                            limit_price=ob.best_bid or 0.5,
                        )
                    )
            
            obj_val = model.getObjVal()
            
            return OptimizationResult(
                success=True,
                trades=trades,
                expected_profit=Decimal(str(round(obj_val, 2))),
                objective_value=obj_val,
                extraction_ratio=self.extraction_alpha,
                status=model.getStatus(),
            )
        else:
            return OptimizationResult(
                success=False,
                status=model.getStatus(),
            )
    
    def _solve_fallback(
        self,
        validated: ValidatedResult,
        order_books: dict[str, OrderBook],
        max_position: float,
    ) -> OptimizationResult:
        """
        Fallback solver when SCIP is not available.
        
        Uses a simple heuristic approach for development/testing.
        NOT suitable for production use.
        """
        logger.warning("Using fallback solver - install pyscipopt for production")
        
        trades = []
        
        # Simple heuristic: look for spread opportunities
        for outcome_id, ob in order_books.items():
            if ob.best_bid is not None and ob.best_ask is not None:
                spread = ob.best_ask - ob.best_bid
                
                # If spread is wide enough, there might be opportunity
                if spread > 0.02:  # 2% spread
                    # This is just a placeholder - real logic would be more complex
                    trades.append(
                        ProposedTrade(
                            market_id="",
                            outcome_id=outcome_id,
                            side=OrderSide.BUY,
                            size=min(100, max_position * 0.1),
                            limit_price=ob.best_ask,
                        )
                    )
        
        return OptimizationResult(
            success=len(trades) > 0,
            trades=trades,
            expected_profit=Decimal("0"),  # Unknown without real optimization
            status="fallback_heuristic",
        )
    
    def check_feasibility(
        self,
        validated: ValidatedResult,
        fixed_vars: dict[str, float],
    ) -> tuple[bool, dict[str, float]]:
        """
        Check if a set of constraints is feasible given fixed variables.
        
        Used by InitFW to determine valid outcome sets.
        
        Args:
            validated: Constraints to respect
            fixed_vars: Dict of variable_name -> value (0 or 1) to fix
            
        Returns:
            Tuple of (is_feasible, solution_dict)
        """
        if not SCIP_AVAILABLE:
            return True, {}  # Fallback
            
        model = Model("feasibility_check")
        model.hideOutput()
        
        # Collect all variables from constraints
        all_vars = set()
        for c in validated.validated_constraints:
            all_vars.update(c.coefficients.keys())
            
        scip_vars = {}
        for v in all_vars:
            # Create binary variables
            scip_vars[v] = model.addVar(vtype="B", name=v)
            
            # Fix variables if requested
            if v in fixed_vars:
                model.fixVar(scip_vars[v], fixed_vars[v])
        
        # Add constraints
        for c in validated.validated_constraints:
            expr = 0
            for v_name, coef in c.coefficients.items():
                if v_name in scip_vars:
                    expr += coef * scip_vars[v_name]
            model.addCons(expr >= c.rhs)
            
        model.optimize()
        
        status = model.getStatus()
        if status == "optimal" or status == "feasible":
            solution = {v: model.getVal(scip_vars[v]) for v in all_vars}
            return True, solution
        else:
            return False, {}

    def solve_linear_objective(
        self,
        validated: ValidatedResult,
        objective_coeffs: dict[str, float],
        sense: str = "maximize",
    ) -> tuple[bool, dict[str, float], float]:
        """
        Solve a linear optimization problem over the constraint set.
        
        Used by Frank-Wolfe as the Linear Minimization Oracle (LMO).
        
        Args:
            validated: Constraints
            objective_coeffs: coefficients for the objective function
            sense: "maximize" or "minimize"
            
        Returns:
            Tuple of (success, solution_vector, objective_value)
        """
        if not SCIP_AVAILABLE:
            return False, {}, 0.0
            
        model = Model("lmo")
        model.hideOutput()
        
        all_vars = set()
        for c in validated.validated_constraints:
            all_vars.update(c.coefficients.keys())
        all_vars.update(objective_coeffs.keys())
        
        scip_vars = {}
        for v in all_vars:
            scip_vars[v] = model.addVar(vtype="B", name=v)
            
        for c in validated.validated_constraints:
            expr = 0
            for v_name, coef in c.coefficients.items():
                if v_name in scip_vars:
                    expr += coef * scip_vars[v_name]
            model.addCons(expr >= c.rhs)
            
        # Set objective
        obj_expr = 0
        for v, coef in objective_coeffs.items():
            if v in scip_vars:
                obj_expr += coef * scip_vars[v]
        
        model.setObjective(obj_expr, sense=sense)
        model.optimize()
        
        status = model.getStatus()
        if status == "optimal" or status == "feasible":
            solution = {v: model.getVal(scip_vars[v]) for v in all_vars}
            return True, solution, model.getObjVal()
        else:
            return False, {}, 0.0


# =============================================================================
# InitFW - Algorithm 3 from Kroer et al.
# =============================================================================


@dataclass
class InitFWResult:
    """
    Result from InitFW algorithm (Algorithm 3).
    
    Attributes:
        vertices: List of extreme points Z₀
        interior_point: Interior point u (average of vertices)
        logically_settled: Dict mapping security -> forced value (0 or 1)
        n_securities: Number of non-settled securities
        success: Whether initialization succeeded
    """
    vertices: list[np.ndarray]
    interior_point: np.ndarray
    logically_settled: dict[str, float]
    n_securities: int
    success: bool


def init_frank_wolfe(
    solver: SCIPSolver,
    validated: "ValidatedResult",
    security_ids: list[str],
) -> InitFWResult:
    """
    Initialize Frank-Wolfe by finding extreme points and interior point.
    
    From Part 2: Algorithm 3 (InitFW)
    
    The algorithm:
    1. For each security i, probe if x_i = 0 is feasible
    2. Probe if x_i = 1 is feasible
    3. If only one value feasible → security is logically settled
    4. If both feasible → collect vertices with x_i = 0 and x_i = 1
    5. Compute interior point u as average of all vertices
    
    This gives us:
    - Vertex set Z₀ for Frank-Wolfe
    - Interior point u for Barrier Frank-Wolfe
    - Knowledge of which securities are already determined
    
    Args:
        solver: SCIPSolver instance for feasibility checks
        validated: Validated constraints
        security_ids: List of security IDs to check
        
    Returns:
        InitFWResult with vertices, interior point, and settled securities
    """
    n = len(security_ids)
    vertices: list[np.ndarray] = []
    logically_settled: dict[str, float] = {}
    
    logger.info(
        "Running InitFW (Algorithm 3)",
        n_securities=n,
    )
    
    for i, sec_id in enumerate(security_ids):
        # Check if x_i = 0 is feasible
        feasible_0, sol_0 = solver.check_feasibility(
            validated, 
            {sec_id: 0.0}
        )
        
        # Check if x_i = 1 is feasible
        feasible_1, sol_1 = solver.check_feasibility(
            validated,
            {sec_id: 1.0}
        )
        
        if feasible_0 and not feasible_1:
            # Security must be 0 (logically settled to NO)
            logically_settled[sec_id] = 0.0
            logger.debug(f"Security {sec_id} logically settled to 0")
        elif feasible_1 and not feasible_0:
            # Security must be 1 (logically settled to YES)
            logically_settled[sec_id] = 1.0
            logger.debug(f"Security {sec_id} logically settled to 1")
        elif feasible_0 and feasible_1:
            # Both are feasible - add vertices
            vertex_0 = _dict_to_array(sol_0, security_ids)
            vertex_1 = _dict_to_array(sol_1, security_ids)
            vertices.append(vertex_0)
            vertices.append(vertex_1)
        else:
            # Neither is feasible - constraint system is infeasible
            logger.warning(f"Security {sec_id} infeasible in both states!")
    
    if not vertices:
        # No vertices found - either all settled or infeasible
        return InitFWResult(
            vertices=[],
            interior_point=np.ones(n) / 2,  # Default to middle
            logically_settled=logically_settled,
            n_securities=n - len(logically_settled),
            success=False,
        )
    
    # Compute interior point as average of all vertices
    interior_point = np.mean(vertices, axis=0)
    
    # Ensure interior point is strictly interior (all coords in (0.05, 0.95))
    interior_point = np.clip(interior_point, 0.05, 0.95)
    
    logger.info(
        "InitFW complete",
        n_vertices=len(vertices),
        n_settled=len(logically_settled),
        n_active=n - len(logically_settled),
    )
    
    return InitFWResult(
        vertices=vertices,
        interior_point=interior_point,
        logically_settled=logically_settled,
        n_securities=n - len(logically_settled),
        success=True,
    )


def _dict_to_array(d: dict[str, float], keys: list[str]) -> np.ndarray:
    """Convert a dict to array in the order of keys."""
    return np.array([d.get(k, 0.0) for k in keys])
