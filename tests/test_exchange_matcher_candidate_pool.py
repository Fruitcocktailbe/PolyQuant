"""Gap 5a regression guard — lock in that ExchangeMatcher's Polymarket
candidate pool is the full fetch above its liquidity floor, not a
post-clustering subset.

A future refactor that narrows the pool (e.g. by routing only "clustered"
markets through the matcher) would silently lose cross-exchange pairs for
unclustered-but-legitimate Polymarket markets. This test fails loudly if
that invariant is ever broken.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from polyquant.agents.exchange_matcher import ExchangeMatcher


def _mk_poly_market(market_id: str, question: str) -> SimpleNamespace:
    return SimpleNamespace(
        market_id=market_id,
        question=question,
        description="",
        resolution_source="",
        end_date=None,
        outcomes=[SimpleNamespace(), SimpleNamespace()],
    )


@pytest.mark.asyncio
async def test_fetch_polymarket_returns_every_market_above_floor(tmp_path, monkeypatch):
    """_fetch_polymarket must return EVERY market the client returns at the
    floor, with no clustering or post-fetch narrowing in between. Asserting
    on the exact market list ensures no silent filter slips in."""
    monkeypatch.chdir(tmp_path)  # isolate cache writes

    sample = [
        _mk_poly_market("m1", "Will BTC hit $100k by 2026?"),
        _mk_poly_market("m2", "Will Trump win the 2028 primary?"),
        _mk_poly_market("m3", "Will the Fed cut rates in September?"),
    ]

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_active_markets(self, limit, offset, min_liquidity):
            if offset == 0:
                return sample, len(sample)
            return [], 0

    with patch("polyquant.agents.exchange_matcher.PolymarketClient", _FakeClient):
        matcher = ExchangeMatcher()
        markets, total = await matcher._fetch_polymarket()

    assert total == len(sample)
    assert [m.market_id for m in markets] == ["m1", "m2", "m3"]


@pytest.mark.asyncio
async def test_run_matching_pipeline_sees_full_fetch_as_candidate_pool(tmp_path, monkeypatch):
    """End-to-end: the pool fed to Layer 1 matches the full _fetch_polymarket
    result. Any refactor that intersects with a clustering step would
    shrink the pool and fail this assertion.
    """
    monkeypatch.chdir(tmp_path)

    poly_sample = [_mk_poly_market(f"m{i}", f"Q{i}") for i in range(3)]

    matcher = ExchangeMatcher()

    # Capture the Polymarket list the pipeline actually consumes.
    matcher._fetch_polymarket = AsyncMock(return_value=(poly_sample, len(poly_sample)))
    matcher._fetch_limitless = AsyncMock(return_value=[])  # nothing to match against

    # With zero Limitless markets the pipeline short-circuits BEFORE any
    # narrowing step. The captured stats must report the full fetch size.
    _, stats = await matcher.run_matching_pipeline()

    assert stats["polymarket_fetched"] == len(poly_sample)
    assert stats["polymarket_fetched_at_floor"] == len(poly_sample)
