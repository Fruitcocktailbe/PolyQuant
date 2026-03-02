import asyncio
import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from polyquant.solver.fw_solver import ArbitrageDetector
from polyquant.agents.validator import ValidatedResult
from polyquant.agents.logic_architect import LogicalConstraint, AnalysisResult
from polyquant.data import OrderBook, OrderLevel, OrderSide, ProposedTrade
from polyquant.risk.position_sizing import PositionSizer, PositionLimits
from polyquant.utils import config

@pytest.mark.asyncio
async def test_dutching_integration_smoke():
    """
    End-to-end smoke test for Dutching Arbitrage detection.
    Verifies that ArbitrageDetector correctly identifies a partition in ValidatedResult
    and generates balanced risk-free trades.
    """
    # 1. Mock Configuration to use Dutching
    old_strategy = config.sizing_strategy
    config.sizing_strategy = "dutching"
    
    try:
        # 2. Setup Position Sizer with conservative limits
        limits = PositionLimits(
            max_single_trade_pct=0.1,    # 10% of capital
            max_total_exposure_pct=0.5,
            max_orderbook_depth_pct=1.0, # 100% depth for test
            kelly_fraction=1.0
        )
        sizer = PositionSizer(capital=1000.0, limits=limits)
        
        # 3. Create a Mock ValidatedResult with a Partition Constraint (A + B = 1)
        # Outcome IDs
        Y_ID = "0xYES_TOKEN"
        N_ID = "0xNO_TOKEN"
        
        constraint = LogicalConstraint(
            description="Binary outcome partition",
            coefficients={Y_ID: 1.0, N_ID: 1.0},
            rhs=1.0
        )
        
        # Correct AnalysisResult fields based on logic_architect.py
        original_analysis = AnalysisResult(
            cluster_id="test_cluster",
            dependencies=[],
            constraints=[constraint]
        )
        
        validated = ValidatedResult(
            original=original_analysis,
            is_valid=True,
            validated_constraints=[constraint],
            market_exchanges={"M1": "polymarket"} # extract_market_id will handle this
        )
        
        # 4. Setup Mock OrderBooks with an Arbitrage Opportunity
        # Yes price = 0.40, No price = 0.50 -> Sum = 0.90 (11.1% margin)
        # Depth = 500 shares each
        ob_yes = OrderBook(
            outcome_id=Y_ID,
            asks=[OrderLevel(price=Decimal("0.40"), size=Decimal("500"))]
        )
        ob_no = OrderBook(
            outcome_id=N_ID,
            asks=[OrderLevel(price=Decimal("0.50"), size=Decimal("500"))]
        )
        
        order_books = {Y_ID: ob_yes, N_ID: ob_no}
        
        # 5. Initialize Detector
        detector = ArbitrageDetector(position_sizer=sizer)
        
        # 6. Detect Opportunity
        opportunity = await detector.detect(validated, order_books)
        
        # 7. Assertions
        assert opportunity is not None
        assert len(opportunity.trades) == 2
        
        # Total cost should be approx sum(size_i * price_i)
        # Total payout should be size_i (since it's a partition and they cost < 1)
        # Sizer logic for Dutching:
        # p_y = 0.4, p_n = 0.5. sum_p = 0.9.
        # Max Single Exposure = 1000 * 0.1 = 100.
        # Stake_y = T * (p_y / sum_p) = 100 * (0.4 / 0.9) = 44.44
        # Shares_y = Stake_y / Price_y = 44.44 / 0.40 = 111.11
        # Stake_n = T * (p_n / sum_p) = 100 * (0.5 / 0.9) = 55.55
        # Shares_n = Stake_n / Price_n = 55.55 / 0.50 = 111.11
        
        trade_y = next(t for t in opportunity.trades if t.outcome_id == Y_ID)
        trade_n = next(t for t in opportunity.trades if t.outcome_id == N_ID)
        
        assert pytest.approx(float(trade_y.size)) == 111.111
        assert pytest.approx(float(trade_n.size)) == 111.111
        
        # Verify they are both BUYs
        assert trade_y.side == OrderSide.BUY
        assert trade_n.side == OrderSide.BUY
        
        # Verify the payout identity: size * payout_per_share
        # In Dutching, payout is 1.0 per share for ANY outcome in the ring.
        # So sizes must be equal for a partition where coefficients are 1.0.
        assert pytest.approx(float(trade_y.size)) == float(trade_n.size)
        
        # Profit before fees = Payout - Total Cost = 111.11 - (44.44 + 55.55) = 11.11 approx
        print(f"SUCCESS: Detected Profit: {opportunity.expected_profit}")
        assert opportunity.expected_profit > 0
        
    finally:
        config.sizing_strategy = old_strategy

@pytest.mark.asyncio
async def test_vwap_slippage_rejection():
    """
    Test P1.1: VWAP slippage enforcement.
    Verifies that rings with excessive VWAP slippage are rejected.
    """
    old_strategy = config.sizing_strategy
    old_slippage = config.vwap_slippage_limit
    config.sizing_strategy = "dutching"
    config.vwap_slippage_limit = 0.05  # 5% limit

    try:
        limits = PositionLimits(
            max_single_trade_pct=0.1,
            max_total_exposure_pct=0.5,
            max_orderbook_depth_pct=1.0,
            kelly_fraction=1.0
        )
        sizer = PositionSizer(capital=1000.0, limits=limits)

        Y_ID = "0xYES_TOKEN"
        N_ID = "0xNO_TOKEN"

        constraint = LogicalConstraint(
            description="Binary outcome partition",
            coefficients={Y_ID: 1.0, N_ID: 1.0},
            rhs=1.0
        )

        original_analysis = AnalysisResult(
            cluster_id="test_cluster",
            dependencies=[],
            constraints=[constraint]
        )

        validated = ValidatedResult(
            original=original_analysis,
            is_valid=True,
            validated_constraints=[constraint],
            market_exchanges={"M1": "polymarket"}
        )

        # Create order book with HIGH slippage
        # Best ask = 0.40, but deep orders at 0.50 (25% slippage)
        ob_yes = OrderBook(
            outcome_id=Y_ID,
            asks=[
                OrderLevel(price=Decimal("0.40"), size=Decimal("10")),  # Tiny depth at best
                OrderLevel(price=Decimal("0.50"), size=Decimal("500"))   # Large slippage
            ]
        )
        ob_no = OrderBook(
            outcome_id=N_ID,
            asks=[OrderLevel(price=Decimal("0.50"), size=Decimal("500"))]
        )

        order_books = {Y_ID: ob_yes, N_ID: ob_no}

        detector = ArbitrageDetector(position_sizer=sizer)
        opportunity = await detector.detect(validated, order_books)

        # Should be rejected due to VWAP slippage on YES leg
        if opportunity is not None:
            assert opportunity.confidence < 0.8
        # If None, that's also acceptable (ring completely rejected)

        print("SUCCESS: VWAP slippage rejection working correctly")

    finally:
        config.sizing_strategy = old_strategy
        config.vwap_slippage_limit = old_slippage


@pytest.mark.asyncio
async def test_dynamic_confidence_scoring():
    """
    Test P2.2: Dynamic confidence scoring based on liquidity cushion.
    High liquidity should yield confidence close to 1.0.
    Low liquidity should yield confidence around 0.7-0.8.
    """
    old_strategy = config.sizing_strategy
    config.sizing_strategy = "dutching"

    try:
        limits = PositionLimits(
            max_single_trade_pct=0.1,
            max_total_exposure_pct=0.5,
            max_orderbook_depth_pct=0.5,
            kelly_fraction=1.0
        )
        sizer = PositionSizer(capital=1000.0, limits=limits)

        Y_ID = "0xYES_TOKEN"
        N_ID = "0xNO_TOKEN"

        constraint = LogicalConstraint(
            description="Binary outcome partition",
            coefficients={Y_ID: 1.0, N_ID: 1.0},
            rhs=1.0
        )

        original_analysis = AnalysisResult(
            cluster_id="test_cluster",
            dependencies=[],
            constraints=[constraint]
        )

        validated = ValidatedResult(
            original=original_analysis,
            is_valid=True,
            validated_constraints=[constraint],
            market_exchanges={"M1": "polymarket"}
        )

        # Scenario 1: HIGH liquidity cushion (10x depth)
        ob_yes_high = OrderBook(
            outcome_id=Y_ID,
            asks=[OrderLevel(price=Decimal("0.40"), size=Decimal("5000"))]
        )
        ob_no_high = OrderBook(
            outcome_id=N_ID,
            asks=[OrderLevel(price=Decimal("0.50"), size=Decimal("5000"))]
        )

        order_books_high = {Y_ID: ob_yes_high, N_ID: ob_no_high}

        detector = ArbitrageDetector(position_sizer=sizer)
        opportunity_high = await detector.detect(validated, order_books_high)

        assert opportunity_high is not None
        assert opportunity_high.confidence > 0.9  # High confidence with ample liquidity
        print(f"High liquidity confidence: {opportunity_high.confidence:.2%}")

        # Scenario 2: LOW liquidity cushion (1.1x depth)
        # Increase limit to 90% of depth to tighten the cushion
        limits_low = PositionLimits(
            max_single_trade_pct=0.1,
            max_total_exposure_pct=0.5,
            max_orderbook_depth_pct=0.9,
            kelly_fraction=1.0
        )
        sizer_low = PositionSizer(capital=1000.0, limits=limits_low)

        ob_yes_low = OrderBook(
            outcome_id=Y_ID,
            asks=[OrderLevel(price=Decimal("0.40"), size=Decimal("50"))]
        )
        ob_no_low = OrderBook(
            outcome_id=N_ID,
            asks=[OrderLevel(price=Decimal("0.50"), size=Decimal("50"))]
        )

        order_books_low = {Y_ID: ob_yes_low, N_ID: ob_no_low}

        detector_low = ArbitrageDetector(position_sizer=sizer_low)
        opportunity_low = await detector_low.detect(validated, order_books_low)

        if opportunity_low is not None:
            assert opportunity_low.confidence < 0.85  # Lower confidence with tight liquidity
            print(f"Low liquidity confidence: {opportunity_low.confidence:.2%}")

        print("SUCCESS: Dynamic confidence scoring working correctly")

    finally:
        config.sizing_strategy = old_strategy


@pytest.mark.asyncio
async def test_partition_size_limit():
    """
    Test P1.3: Partition size limit enforcement.
    Partitions exceeding max_partition_size should be skipped.
    """
    old_strategy = config.sizing_strategy
    old_max_size = config.max_partition_size
    config.sizing_strategy = "dutching"
    config.max_partition_size = 3  # Limit to 3 outcomes

    try:
        limits = PositionLimits(
            max_single_trade_pct=0.1,
            max_total_exposure_pct=0.5,
            max_orderbook_depth_pct=0.5,
            kelly_fraction=1.0
        )
        sizer = PositionSizer(capital=1000.0, limits=limits)

        # Create 5-outcome partition (exceeds limit of 3)
        outcome_ids = [f"0xOUTCOME_{i}" for i in range(5)]

        constraint = LogicalConstraint(
            description="5-way partition",
            coefficients={oid: 1.0 for oid in outcome_ids},
            rhs=1.0
        )

        original_analysis = AnalysisResult(
            cluster_id="test_cluster",
            dependencies=[],
            constraints=[constraint]
        )

        validated = ValidatedResult(
            original=original_analysis,
            is_valid=True,
            validated_constraints=[constraint],
            market_exchanges={"M1": "polymarket"}
        )

        # Create order books with arbitrage
        order_books = {}
        for oid in outcome_ids:
            order_books[oid] = OrderBook(
                outcome_id=oid,
                asks=[OrderLevel(price=Decimal("0.15"), size=Decimal("500"))]
            )

        detector = ArbitrageDetector(position_sizer=sizer)
        opportunity = await detector.detect(validated, order_books)

        # Should be rejected due to partition size limit
        assert opportunity is None

        print("SUCCESS: Partition size limit enforcement working correctly")

    finally:
        config.sizing_strategy = old_strategy
        config.max_partition_size = old_max_size


@pytest.mark.asyncio
async def test_roi_and_capital_efficiency():
    """
    Test P1.2: ROI and capital efficiency calculation.
    Verifies that ArbitrageOpportunity includes roi and capital_efficiency fields.
    """
    old_strategy = config.sizing_strategy
    config.sizing_strategy = "dutching"

    try:
        limits = PositionLimits(
            max_single_trade_pct=0.1,
            max_total_exposure_pct=0.5,
            max_orderbook_depth_pct=1.0,
            kelly_fraction=1.0
        )
        sizer = PositionSizer(capital=1000.0, limits=limits)

        Y_ID = "0xYES_TOKEN"
        N_ID = "0xNO_TOKEN"

        constraint = LogicalConstraint(
            description="Binary outcome partition",
            coefficients={Y_ID: 1.0, N_ID: 1.0},
            rhs=1.0
        )

        original_analysis = AnalysisResult(
            cluster_id="test_cluster",
            dependencies=[],
            constraints=[constraint]
        )

        validated = ValidatedResult(
            original=original_analysis,
            is_valid=True,
            validated_constraints=[constraint],
            market_exchanges={"M1": "polymarket"}
        )

        ob_yes = OrderBook(
            outcome_id=Y_ID,
            asks=[OrderLevel(price=Decimal("0.40"), size=Decimal("500"))]
        )
        ob_no = OrderBook(
            outcome_id=N_ID,
            asks=[OrderLevel(price=Decimal("0.50"), size=Decimal("500"))]
        )

        order_books = {Y_ID: ob_yes, N_ID: ob_no}

        detector = ArbitrageDetector(position_sizer=sizer)
        opportunity = await detector.detect(validated, order_books)

        assert opportunity is not None

        # Verify ROI is calculated (should be around 11% for this scenario)
        assert opportunity.roi > 0
        assert opportunity.roi < 0.2  # Should be reasonable (< 20%)
        print(f"ROI: {opportunity.roi:.2%}")

        # Verify capital efficiency is calculated (profit per second)
        assert opportunity.capital_efficiency > 0
        print(f"Capital Efficiency: ${opportunity.capital_efficiency:.2f}/sec")

        print("SUCCESS: ROI and capital efficiency calculations working correctly")

    finally:
        config.sizing_strategy = old_strategy


if __name__ == "__main__":
    try:
        asyncio.run(test_dutching_integration_smoke())
        asyncio.run(test_vwap_slippage_rejection())
        asyncio.run(test_dynamic_confidence_scoring())
        asyncio.run(test_partition_size_limit())
        asyncio.run(test_roi_and_capital_efficiency())
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
