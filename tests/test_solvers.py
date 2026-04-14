
import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from decimal import Decimal
import sys
import os

from polyquant.solver.scip_solver import SCIPSolver, OptimizationResult
from polyquant.solver.fw_solver import FWSolver, ArbitrageDetector
from polyquant.utils.market_utils import extract_market_id
from polyquant.agents.validator import ValidatedResult
from polyquant.agents.logic_architect import LogicalConstraint, AnalysisResult
from polyquant.data import OrderBook, OrderSide, ProposedTrade, OrderLevel
from polyquant.risk.position_sizing import PositionSizer
from polyquant.utils.config import config

class TestUtils(unittest.TestCase):
    def test_extract_market_id(self):
        self.assertEqual(extract_market_id("12345_Yes"), "12345")
        self.assertEqual(extract_market_id("12345_No"), "12345")
        self.assertEqual(extract_market_id("abc-def_Outcome1"), "abc-def")
        self.assertEqual(extract_market_id("simpleID"), "simpleID")
        self.assertEqual(extract_market_id(""), "")

class TestSolvers(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.validated = ValidatedResult(
            original=AnalysisResult(cluster_id="test_cluster"),
            is_valid=True,
            validated_constraints=[
                LogicalConstraint(
                    id="c1",
                    constraint_id="c1",
                    description="test",
                    coefficients={"m1_yes": Decimal("1.0"), "m1_no": Decimal("1.0")},
                    rhs=Decimal("1.0"),
                    operator=">="
                )
            ],
            issues=[],
            market_exchanges={"m1": "polymarket"}
        )
        self.order_books = {
            "m1_yes": OrderBook(
                outcome_id="m1_yes",
                asks=[OrderLevel(price=Decimal("0.6"), size=Decimal("10000.0"))],
                bids=[OrderLevel(price=Decimal("0.55"), size=Decimal("10000.0"))]
            ),
            "m1_no": OrderBook(
                outcome_id="m1_no",
                asks=[OrderLevel(price=Decimal("0.3"), size=Decimal("10000.0"))],
                bids=[OrderLevel(price=Decimal("0.25"), size=Decimal("10000.0"))]
            ),
        }

    async def test_scip_optimize_async(self):
        solver = SCIPSolver()
        solver._solve_with_scip = MagicMock()
        
        mock_result = OptimizationResult(
            success=True,
            trades=[
                ProposedTrade(market_id="m1", outcome_id="m1_yes", side=OrderSide.BUY, size=10.0, limit_price=0.6),
                ProposedTrade(market_id="m1", outcome_id="m1_no", side=OrderSide.BUY, size=10.0, limit_price=0.3)
            ],
            expected_profit=Decimal("1.0")
        )
        solver._solve_with_scip.return_value = mock_result
        
        result = await solver.optimize(self.validated, self.order_books)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.trades), 2)
        solver._solve_with_scip.assert_called_once()

    async def test_arbitrage_detector_async_call(self):
        # This test mocks fw_solver.find_opportunity (the Kelly path). Override the
        # default sizing_strategy ("dutching") so detect() routes to the Kelly code path
        # and actually hits the mock. Restore after to avoid leaking into other tests.
        orig_sizing = config.sizing_strategy
        config.sizing_strategy = "kelly"
        try:
            scip = SCIPSolver()
            sizer = PositionSizer()
            detector = ArbitrageDetector(scip_solver=scip, position_sizer=sizer)

            # Mock fw_solver.find_opportunity
            detector.fw_solver.find_opportunity = AsyncMock(return_value={
                "m1_yes": 0.7,
                "m1_no": 0.35,
            })

            opp = await detector.detect(self.validated, self.order_books)

            self.assertIsNotNone(opp)
            self.assertGreater(len(opp.trades), 0)
            detector.fw_solver.find_opportunity.assert_called_once()

            trades = {t.outcome_id: t for t in opp.trades}
            print(f"TRADES FOUND: {trades}")
            self.assertIn("m1_yes", trades)
            self.assertEqual(trades["m1_yes"].side, OrderSide.BUY)
            self.assertEqual(trades["m1_yes"].limit_price, Decimal("0.6"))

            self.assertIn("m1_no", trades)
            self.assertEqual(trades["m1_no"].side, OrderSide.BUY)
            self.assertEqual(trades["m1_no"].limit_price, Decimal("0.3"))
        finally:
            config.sizing_strategy = orig_sizing

if __name__ == "__main__":
    unittest.main()
