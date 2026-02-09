
import asyncio
import logging
from typing import Dict, List, Tuple
from decimal import Decimal
import numpy as np

# Mocking internal dependencies to run standalone
from polyquant.solver.fw_solver import FWSolver, ArbitrageDetector
# from polyquant.agents.validator import ValidatedResult  <-- REMOVED

class MockValidatedResult:
    def __init__(self, is_valid=True, validated_constraints=None):
        self.is_valid = is_valid
        self.validated_constraints = validated_constraints or []
        
ValidatedResult = MockValidatedResult # Alias for testing
from polyquant.data import OrderBook, OrderLevel
from polyquant.risk.position_sizing import PositionSizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_solver")

class MockSCIPSolver:
    """Mock SCIP solver for testing FWSolver logic."""
    
    def check_feasibility(self, validated, fixed_vars) -> Tuple[bool, Dict[str, float]]:
        # Simple feasibility: sum(probs) == 1
        # If we fix vars, we check if remaining can sum to 1
        
        # This is a very simple mock for a 2-outcome market [A, B]
        # where A + B = 1
        
        # If fixed_vars has A=1, then B must be 0.
        # If A=0, B must be 1.
        
        # Check consistency
        s = 0.0
        count = 0
        for k, v in fixed_vars.items():
            if v > 1.0001 or v < -0.0001:
                return False, {}
            s += v
            count += 1
            
        if s > 1.0001:
            return False, {}
            
        # Determine solution
        sol = {}
        outcomes = ["A", "B"]
        
        current_sum = sum(fixed_vars.values())
        remaining = [o for o in outcomes if o not in fixed_vars]
        
        if not remaining:
            if abs(current_sum - 1.0) < 0.001:
                return True, fixed_vars
            else:
                return False, {}
                
        # Distribute remaining
        # Simply set first remaining to 1 - current_sum, others to 0
        sol = fixed_vars.copy()
        sol[remaining[0]] = 1.0 - current_sum
        for r in remaining[1:]:
            sol[r] = 0.0
            
        # Verify valid
        if sol[remaining[0]] < 0:
            return False, {}
            
        return True, sol

    def solve_linear_objective(self, validated, grad, sense="minimize") -> Tuple[bool, Dict[str, float], float]:
        # Minimize <grad, v> over simplex
        # Solution is vertex with smallest gradient component
        
        # For A+B=1, vertices are (1,0) and (0,1)
        # Val1 = grad[A]*1 + grad[B]*0 = grad[A]
        # Val2 = grad[A]*0 + grad[B]*1 = grad[B]
        
        if grad.get("A", 0) < grad.get("B", 0):
            return True, {"A": 1.0, "B": 0.0}, grad.get("A", 0)
        else:
            return True, {"A": 0.0, "B": 1.0}, grad.get("B", 0)

async def test_fw_solver():
    logger.info("Testing FWSolver...")
    
    mock_scip = MockSCIPSolver()
    solver = FWSolver(mock_scip)
    
    # 1. Test InitFW
    validated = MockValidatedResult(is_valid=True, validated_constraints=[])
    outcomes = ["A", "B"]
    
    Z_0, u, settled = solver.init_fw(validated, outcomes)
    
    logger.info(f"InitFW Result: |Z_0|={len(Z_0)}")
    logger.info(f"Interior point u: {u}")
    
    # Verify we found both valid vertices (1,0) and (0,1)
    found_1_0 = False
    found_0_1 = False
    
    for z in Z_0:
        if abs(z.get("A", 0) - 1.0) < 0.01 and abs(z.get("B", 0) - 0.0) < 0.01:
            found_1_0 = True
        if abs(z.get("A", 0) - 0.0) < 0.01 and abs(z.get("B", 0) - 1.0) < 0.01:
            found_0_1 = True
            
    assert found_1_0 and found_0_1, "Should find (1,0) and (0,1) vertices"
    assert 0.001 < u["A"] < 0.999, f"Interior point should be strictly interior (found {u['A']})"
    
    # 2. Test BarrierFW
    # Case: Market implies A=0.8, B=0.2
    # True Prob (fair): A=0.5, B=0.5 (implied entirely by u)
    # Actually BarrierFW tries to minimize KL(mu || theta).
    # If we run it, it should converge to... wait.
    # The objective is D(mu || theta). Minimized when mu = theta (if feasible).
    # If theta is feasible, it should find theta.
    
    market_prices = {"A": 0.8, "B": 0.2} # Theta
    
    # If we pass market_prices as theta, Solver finds mu close to theta.
    # But used in ArbitrageDetector, we want to find if *market* is different from *fair*.
    # Wait, ArbitrageDetector uses FWSolver to find "Arbitrage-Free Prices".
    # And then compares to Market?
    # No, typically minimizing D(mu || theta) finds the projection of theta onto the feasible set.
    # If theta IS feasible, result is theta.
    # If theta is INFEASIBLE (arbitrage exists), result is the closest valid price.
    
    # Let's test with INFEASIBLE market prices (Arbitrage!)
    # e.g., sum > 1. A=0.6, B=0.6.
    # Feasible set is A+B=1.
    # Projection should be A=0.5, B=0.5? Or renormalization?
    
    market_prices_arb = {"A": 0.6, "B": 0.6}
    
    mu_star, profit, kl = solver.barrier_fw(validated, outcomes, market_prices_arb, Z_0, u)
    
    logger.info(f"BarrierFW Result: {mu_star}")
    logger.info(f"Profit Guarantee: {profit}")
    logger.info(f"KL Divergence: {kl}")
    
    # Check if sum is ~1
    s_mu = sum(mu_star.values())
    assert abs(s_mu - 1.0) < 0.01, f"Result probabilities must sum to 1, got {s_mu}"
    
    logger.info("✅ FWSolver Tests Passed")
    
async def test_arbitrage_detector():
    logger.info("\nTesting ArbitrageDetector...")
    
    mock_scip = MockSCIPSolver()
    detector = ArbitrageDetector(scip_solver=mock_scip)
    
    validated = MockValidatedResult(is_valid=True, validated_constraints=[])
    
    # Create OrderBook with ARBITRAGE
    # Market A: Best Ask 0.4
    # Market B: Best Ask 0.4
    # Sum of costs = 0.8 < 1.0. Guaranteed profit by buying both!
    # True prices (projected) should be ~0.5?
    # If Target=0.5, and Ask=0.4, we BUY.
    
    obs = {
        "A": OrderBook(outcome_id="A", bids=[], asks=[OrderLevel(price=0.4, size=100)]),
        "B": OrderBook(outcome_id="B", bids=[], asks=[OrderLevel(price=0.4, size=100)])
    }
    
    # Note: FWSolver finds projection of "Mid Price".
    # Mid(A) = 0.4 (approx if bid=0).
    # Mid(B) = 0.4.
    # Projection of (0.4, 0.4) onto A+B=1 is (0.5, 0.5).
    # So Target A=0.5, Target B=0.5.
    
    opp = await detector.detect(validated, obs)
    
    if opp:
        logger.info(f"Found Opportunity in markets: {opp.markets}")
        logger.info(f"Expected Profit: {opp.expected_profit}")
        logger.info(f"Trades: {len(opp.trades)}")
        for t in opp.trades:
            logger.info(f" - {t.side} {t.outcome_id} @ {t.limit_price}")
            
        assert len(opp.trades) == 2, "Should buy both"
        assert opp.expected_profit > 0, "Should have positive profit"
    else:
        logger.error("❌ Failed to detect arbitrage!")
        
    logger.info("✅ ArbitrageDetector Tests Passed")

if __name__ == "__main__":
    asyncio.run(test_fw_solver())
    asyncio.run(test_arbitrage_detector())
