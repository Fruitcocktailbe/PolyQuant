import math
import pytest
from polyquant.risk.position_sizing import PositionSizer, PositionLimits


def test_calculate_dutching_sizes_valid_arb():
    # Set up conservative limits, turning off capital limits to test liquidity bottleneck
    limits = PositionLimits(
        max_single_trade_pct=1.0, # 100% (No limit)
        max_total_exposure_pct=1.0,
        max_orderbook_depth_pct=0.5, # 50% max depth
        kelly_fraction=1.0 # unused
    )
    sizer = PositionSizer(capital=10000.0, limits=limits)
    
    # Mocking: Limitless YES trading at 0.40 (odds 1.5)
    # Polymarket NO trading at 0.50 (odds 1.0)
    # Implied probs = 0.40 + 0.50 = 0.90. Expected Arb margin = (1/0.9) - 1 = 11.11%
    odds_list = [1.5, 1.0]
    
    # Depth: Limitless=$500, Poly=$1000
    # Limitless max take = 500 * 0.5 = 250
    # Poly max take = 1000 * 0.5 = 500
    depth_list = [500.0, 1000.0]
    
    sizes = sizer.calculate_dutching_sizes(odds_list, depth_list)

    assert len(sizes) == 2

    # Verify Limitless YES: $250 stake at $0.40/share = 625 shares (bottleneck hit)
    assert math.isclose(sizes[0].recommended_size, 625.0, rel_tol=1e-5)
    assert sizes[0].limited_by == "leg_0_liquidity"

    # Verify Polymarket NO: $312.50 stake at $0.50/share = 625 shares
    # (T = 562.5, NO prob = 0.5, Stake = 562.5 * 0.5 / 0.9 = 312.50; shares = 312.50/0.5)
    assert math.isclose(sizes[1].recommended_size, 625.0, rel_tol=1e-5)

    # Verify expected values are 11.11% profit margin
    expected_margin = (1.0 / 0.9) - 1.0
    assert math.isclose(sizes[0].expected_value, expected_margin, rel_tol=1e-5)
    assert math.isclose(sizes[1].expected_value, expected_margin, rel_tol=1e-5)
    
def test_calculate_dutching_sizes_no_arb():
    # If the implied sum is >= 1.0, no arb exists.
    limits = PositionLimits(
        max_single_trade_pct=1.0,
        max_total_exposure_pct=1.0,
        max_orderbook_depth_pct=0.5,
        kelly_fraction=1.0
    )
    sizer = PositionSizer(capital=10000.0, limits=limits)

    # YES price = 0.60 (odds=0.666), NO price = 0.50 (odds=1.0)
    # Implied probs = 0.6 + 0.5 = 1.10 (No arbitrage!)
    odds_list = [0.6666666, 1.0]
    depth_list = [500.0, 1000.0]

    sizes = sizer.calculate_dutching_sizes(odds_list, depth_list)

    assert len(sizes) == 2
    assert sizes[0].recommended_size == 0.0
    assert sizes[1].recommended_size == 0.0
    assert sizes[0].limited_by == "no_arb"
    assert sizes[1].limited_by == "no_arb"


def test_calculate_dutching_roi():
    """Test ROI calculation for dutching opportunities (P1.2)"""
    limits = PositionLimits(
        max_single_trade_pct=1.0,
        max_total_exposure_pct=1.0,
        max_orderbook_depth_pct=0.5,
        kelly_fraction=1.0
    )
    sizer = PositionSizer(capital=10000.0, limits=limits)

    # 11.1% margin opportunity
    odds_list = [1.5, 1.0]
    depth_list = [500.0, 1000.0]

    sizes = sizer.calculate_dutching_sizes(odds_list, depth_list)

    # Verify expected value represents ROI
    # Expected margin = (1/0.9) - 1 = 0.1111...
    expected_roi = (1.0 / 0.9) - 1.0
    assert math.isclose(sizes[0].expected_value, expected_roi, rel_tol=1e-5)

    # Reconstruct dollar stakes from shares: dollar_stake = shares * price = shares * probability
    # Total capital deployed = 250 + 312.50 = 562.50
    # Expected profit = 562.50 * 0.1111 = 62.50
    # ROI = 62.50 / 562.50 = 0.1111
    dollar_stakes = [s.recommended_size * s.probability for s in sizes]
    total_dollar_stake = sum(dollar_stakes)
    expected_profit = total_dollar_stake * expected_roi
    calculated_roi = expected_profit / total_dollar_stake
    assert math.isclose(calculated_roi, expected_roi, rel_tol=1e-5)
    assert math.isclose(total_dollar_stake, 562.50, rel_tol=1e-5)


def test_dutching_liquidity_cushion():
    """Test that liquidity ratios affect confidence scoring (P2.2)"""
    limits = PositionLimits(
        max_single_trade_pct=1.0,
        max_total_exposure_pct=1.0,
        max_orderbook_depth_pct=0.3,  # Low depth cap for tight cushion
        kelly_fraction=1.0
    )
    sizer = PositionSizer(capital=10000.0, limits=limits)

    # Same arbitrage but low depth
    odds_list = [1.5, 1.0]
    depth_list = [300.0, 600.0]  # Lower depth

    sizes = sizer.calculate_dutching_sizes(odds_list, depth_list)

    # Verify sizing is constrained by low depth
    # Max dollar take from leg 0: 300 * 0.3 = 90 → 90 / 0.40 price = 225 shares
    assert math.isclose(sizes[0].recommended_size, 225.0, rel_tol=1e-5)
    assert sizes[0].limited_by == "leg_0_liquidity"


def test_partition_size_limit():
    """Test that large partitions are properly sized (P1.3)"""
    limits = PositionLimits(
        max_single_trade_pct=1.0,
        max_total_exposure_pct=1.0,
        max_orderbook_depth_pct=0.5,
        kelly_fraction=1.0
    )
    sizer = PositionSizer(capital=10000.0, limits=limits)

    # 5-way partition with arbitrage
    # Prices: 0.15, 0.15, 0.20, 0.20, 0.20 (sum = 0.90, 11.1% margin)
    odds_list = [
        (1.0 / 0.15) - 1.0,  # 5.666...
        (1.0 / 0.15) - 1.0,
        (1.0 / 0.20) - 1.0,  # 4.0
        (1.0 / 0.20) - 1.0,
        (1.0 / 0.20) - 1.0,
    ]
    depth_list = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]

    sizes = sizer.calculate_dutching_sizes(odds_list, depth_list)

    # Should process all 5 legs
    assert len(sizes) == 5

    # All sizes should be positive (arbitrage exists)
    for size_res in sizes:
        assert size_res.recommended_size > 0

    # Verify equal-shares property of dutching: shares_i = stake_i / p_i and
    # stake_i = T * (p_i / sum_implied), so shares_i = T / sum_implied — identical for every leg.
    # This is the core dutching invariant: equal shares → equal $1 payout on any winning outcome.
    share_ratio = sizes[0].recommended_size / sizes[2].recommended_size
    assert math.isclose(share_ratio, 1.0, rel_tol=1e-5)

    # Dollar stakes (reconstructed) should still match the probability ratio.
    dollar_stake_0 = sizes[0].recommended_size * sizes[0].probability
    dollar_stake_2 = sizes[2].recommended_size * sizes[2].probability
    prob_ratio = 0.15 / 0.20  # 0.75
    assert math.isclose(dollar_stake_0 / dollar_stake_2, prob_ratio, rel_tol=1e-2)
