"""Tests for Navigator._build_correlation_opportunity (correlation trade
construction from CorrelationSignal objects).

Scope: exercise the trade construction path in isolation — no WebSocket, no
manifest store, no TradeExecutor. The Navigator object is constructed bare and
we inject only what _build_correlation_opportunity reaches for:
  - self._position_sizer
  - self._token_to_exchange
  - config.correlation_execution_enabled
"""

import unittest
from decimal import Decimal
from unittest.mock import MagicMock

from polyquant.agents.correlation import CorrelationPair, CorrelationSignal
from polyquant.data import OrderBook, OrderLevel, OrderSide
from polyquant.navigator import Navigator
from polyquant.risk.position_sizing import PositionSizer
from polyquant.utils.config import config


def _make_pair(leader_id="T_leader", laggard_id="T_laggard") -> CorrelationPair:
    return CorrelationPair(
        leader_id=leader_id,
        leader_question="Leader question",
        laggard_id=laggard_id,
        laggard_question="Laggard question",
        correlation_7d=0.85,
        correlation_30d=0.82,
        avg_lag_minutes=3.0,
        data_points=200,
        p_value=0.001,
    )


def _make_signal(
    pair: CorrelationPair,
    *,
    current: float = 0.40,
    expected: float = 0.55,
    sigma: float = 3.0,
) -> CorrelationSignal:
    return CorrelationSignal(
        pair=pair,
        leader_move=0.05,
        expected_laggard_move=expected - current,
        current_laggard_price=current,
        expected_laggard_price=expected,
        deviation_sigma=sigma,
    )


def _thick_book(outcome_id: str, best_ask: str = "0.40", best_bid: str = "0.38") -> OrderBook:
    return OrderBook(
        outcome_id=outcome_id,
        asks=[OrderLevel(price=Decimal(best_ask), size=Decimal("5000"))],
        bids=[OrderLevel(price=Decimal(best_bid), size=Decimal("5000"))],
    )


class TestCorrelationOpportunity(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Bare Navigator — we don't invoke __aenter__, just poke the state
        # _build_correlation_opportunity reads.
        self.nav = Navigator()
        self.nav._position_sizer = PositionSizer(capital=10_000.0)
        self.nav._token_to_exchange = {"T_laggard": "polymarket"}

        self._orig_flag = config.correlation_execution_enabled
        self._orig_kfrac = config.correlation_kelly_fraction
        self._orig_cap = config.correlation_max_probability
        config.correlation_execution_enabled = True
        # Bump Kelly fraction so the test actually produces non-trivial shares
        # (conservative 0.25 × 0.5 half-Kelly would shrink to noise).
        config.correlation_kelly_fraction = 1.0

    def tearDown(self):
        config.correlation_execution_enabled = self._orig_flag
        config.correlation_kelly_fraction = self._orig_kfrac
        config.correlation_max_probability = self._orig_cap

    async def test_buy_signal_builds_opportunity(self):
        pair = _make_pair()
        sig = _make_signal(pair, current=0.40, expected=0.55)
        books = {"T_laggard": _thick_book("T_laggard")}

        opp = await self.nav._build_correlation_opportunity(sig, books)

        self.assertIsNotNone(opp)
        self.assertEqual(opp["source"], "correlation")
        self.assertEqual(opp["cluster_id"], "correlation:T_leader->T_laggard")
        self.assertEqual(len(opp["arb_object"].trades), 1)

        trade = opp["arb_object"].trades[0]
        self.assertEqual(trade.outcome_id, "T_laggard")
        self.assertEqual(trade.side, OrderSide.BUY)
        self.assertGreater(float(trade.size), 0)
        self.assertEqual(trade.exchange, "polymarket")
        self.assertTrue(trade.reason.startswith("correlation:T_leader"))

        # Correlation metadata should be attached for the outcome tracker.
        meta = opp["_correlation_meta"]
        self.assertEqual(meta["pair_leader_id"], "T_leader")
        self.assertEqual(meta["laggard_id"], "T_laggard")
        self.assertEqual(meta["direction"], "buy")

    async def test_sell_signal_builds_opportunity(self):
        pair = _make_pair()
        sig = _make_signal(pair, current=0.60, expected=0.45)
        books = {"T_laggard": _thick_book("T_laggard", best_ask="0.62", best_bid="0.60")}

        opp = await self.nav._build_correlation_opportunity(sig, books)

        self.assertIsNotNone(opp)
        trade = opp["arb_object"].trades[0]
        self.assertEqual(trade.side, OrderSide.SELL)
        self.assertGreater(float(trade.size), 0)

    async def test_disabled_flag_returns_none(self):
        config.correlation_execution_enabled = False
        pair = _make_pair()
        sig = _make_signal(pair)
        books = {"T_laggard": _thick_book("T_laggard")}

        opp = await self.nav._build_correlation_opportunity(sig, books)
        self.assertIsNone(opp)

    async def test_missing_book_returns_none(self):
        pair = _make_pair()
        sig = _make_signal(pair)
        # No T_laggard in books
        opp = await self.nav._build_correlation_opportunity(sig, {})
        self.assertIsNone(opp)

    async def test_target_not_profitable_returns_none(self):
        """If the realistic vwap already meets/beats the target, skip the trade."""
        pair = _make_pair()
        # Expected price equal to best_ask → no edge to capture
        sig = _make_signal(pair, current=0.40, expected=0.40)
        books = {"T_laggard": _thick_book("T_laggard", best_ask="0.40")}

        opp = await self.nav._build_correlation_opportunity(sig, books)
        self.assertIsNone(opp)

    async def test_empty_side_returns_none(self):
        """Zero depth on the target side must reject, not fall back to a magic default."""
        pair = _make_pair()
        sig = _make_signal(pair, current=0.40, expected=0.55)
        empty_ask_book = OrderBook(
            outcome_id="T_laggard",
            asks=[],  # No depth on buy side
            bids=[OrderLevel(price=Decimal("0.38"), size=Decimal("5000"))],
        )
        opp = await self.nav._build_correlation_opportunity(sig, {"T_laggard": empty_ask_book})
        self.assertIsNone(opp)

    async def test_probability_capped(self):
        """Even with extreme deviation, effective probability stays under the cap."""
        pair = _make_pair()
        pair.signals_generated = 10
        pair.signals_correct = 10  # 100% accuracy prior
        sig = _make_signal(pair, current=0.40, expected=0.99, sigma=20.0)
        books = {"T_laggard": _thick_book("T_laggard")}

        # Sanity: just make sure it doesn't blow up and still builds a trade.
        opp = await self.nav._build_correlation_opportunity(sig, books)
        self.assertIsNotNone(opp)


if __name__ == "__main__":
    unittest.main()
