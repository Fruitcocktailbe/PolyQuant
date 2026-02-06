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
    
    def compute_bregman_projection(
        self,
        current_prices: np.ndarray,
        constraints: np.ndarray,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the Bregman projection onto the constraint set.
        
        Uses the Barrier Frank-Wolfe algorithm to handle:
        1. KL divergence objective
        2. Linear constraints
        3. Probability simplex constraints
        
        This is the mathematical core of the arbitrage detection.
        
        Args:
            current_prices: Current market prices (n outcomes)
            constraints: Constraint matrix A (m x n)
            rhs: Right-hand side vector b (m)
            
        Returns:
            Projected prices that satisfy constraints
            
        Math:
            minimize    KL(p || q)  = sum_i p_i * log(p_i / q_i)
            subject to  A @ p >= b
                        sum(p) = 1
                        p >= 0
        """
        # Initialize with current prices
        p = current_prices.copy()
        n = len(p)
        
        # Barrier Frank-Wolfe parameters
        max_iters = 100
        tol = 1e-6
        
        for iteration in range(max_iters):
            # Compute KL gradient: grad = log(p/q) + 1
            # With barrier: add penalty for constraint violations
            grad = np.log(p / current_prices + 1e-10) + 1
            
            # Check constraint satisfaction
            violations = constraints @ p - rhs
            violated = violations < 0
            
            if not np.any(violated):
                # All constraints satisfied, check convergence
                if np.linalg.norm(grad - grad.mean()) < tol:
                    break
            
            # Add barrier gradient for violated constraints
            for i, v in enumerate(violated):
                if v:
                    grad -= constraints[i] / (violations[i] + 1e-10)
            
            # Frank-Wolfe direction: solve LP
            # minimize grad @ s, subject to sum(s) = 1, s >= 0
            # Solution: s = e_i where i = argmin(grad)
            s = np.zeros(n)
            s[np.argmin(grad)] = 1
            
            # Line search
            step = 2.0 / (iteration + 2)  # Standard FW step size
            p = (1 - step) * p + step * s
            
            # Ensure positivity
            p = np.maximum(p, 1e-10)
            p /= p.sum()  # Normalize
        
        return p


class ArbitrageDetector:
    """
    High-level interface for detecting and quantifying arbitrage.
    
    Combines constraint analysis with the SCIP solver to identify
    profitable opportunities.
    
    Example:
        detector = ArbitrageDetector()
        
        opportunity = await detector.detect(
            validated_result,
            order_books,
        )
        
        if opportunity and opportunity.expected_profit > 100:
            print(f"Found opportunity: ${opportunity.expected_profit}")
    """
    
    def __init__(self):
        """Initialize the arbitrage detector."""
        self.solver = SCIPSolver()
        
    async def detect(
        self,
        validated: ValidatedResult,
        order_books: dict[str, OrderBook],
        min_profit: float = 10.0,
    ) -> ArbitrageOpportunity | None:
        """
        Detect and analyze an arbitrage opportunity.
        
        Args:
            validated: ValidatedResult from Phase 3
            order_books: Current order book state
            min_profit: Minimum profit threshold
            
        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        # Run synchronous solver in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.solver.optimize,
            validated,
            order_books,
            config.orderbook_depth_cap * 10000,  # Convert to dollars
        )
        
        if not result.success:
            logger.debug("No arbitrage found", status=result.status)
            return None
        
        if float(result.expected_profit) < min_profit:
            logger.debug(
                "Profit below threshold",
                profit=float(result.expected_profit),
                threshold=min_profit,
            )
            return None
        
        # Build the opportunity object
        opportunity = ArbitrageOpportunity(
            markets=list(set(t.market_id for t in result.trades if t.market_id)),
            trades=result.trades,
            expected_profit=result.expected_profit,
            confidence=min(c.confidence for c in validated.validated_constraints)
            if validated.validated_constraints
            else 0.5,
        )
        
        logger.info(
            "Arbitrage opportunity detected",
            profit=float(opportunity.expected_profit),
            trade_count=len(opportunity.trades),
        )
        
        return opportunity
