"""Regression tests for the partition_detector tz-naive vs tz-aware bugs.

Two call sites have been implicated:
- `end_date_bucket` (Layer 1/2 date bucketing) — fixed 2026-04-17.
- `layer3_triage` (post-filter freshness check) — fixed shortly after when
  the first fix exposed it: scan_markets crashed with `TypeError: can't
  compare offset-naive and offset-aware datetimes` at the
  `m.end_date <= now` comparison.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from polyquant.agents.partition_detector import end_date_bucket, layer3_triage
from polyquant.data.market_models import Market, Outcome


def _binary_market(
    market_id: str,
    *,
    end_date: datetime | None,
    yes_token: str | None = None,
) -> Market:
    """Minimal binary market with a resolvable YES leg for layer3_triage."""
    return Market(
        market_id=market_id,
        question=f"Will {market_id}?",
        description="",
        outcomes=[
            Outcome(
                outcome_id=f"{market_id}_yes",
                name="Yes",
                price=Decimal("0.5"),
                token_id=yes_token or f"{market_id}_yes_tok",
            ),
            Outcome(
                outcome_id=f"{market_id}_no",
                name="No",
                price=Decimal("0.5"),
                token_id=f"{market_id}_no_tok",
            ),
        ],
        end_date=end_date,
    )


def test_end_date_bucket_accepts_tz_aware_date():
    aware = datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc)
    # Must not raise TypeError: can't subtract offset-naive/aware datetimes.
    bucket = end_date_bucket(aware, window_days=7)
    assert bucket.startswith("bucket_")


def test_end_date_bucket_accepts_tz_naive_date():
    # Legacy callers may still pass naïve datetimes; assume UTC and bucket
    # rather than crash.
    naive = datetime(2026, 12, 31, 23, 59)
    bucket = end_date_bucket(naive, window_days=7)
    assert bucket.startswith("bucket_")


def test_end_date_bucket_groups_nearby_dates_same_bucket():
    # With a 30-day window, dates within a month must share a bucket while
    # dates further apart must not. Picks dates clearly inside/outside the
    # window regardless of where the Jan-2000 anchor lands.
    a = datetime(2026, 5, 1, tzinfo=timezone.utc)
    b = datetime(2026, 5, 10, tzinfo=timezone.utc)
    c = datetime(2026, 9, 15, tzinfo=timezone.utc)
    assert end_date_bucket(a, window_days=30) == end_date_bucket(b, window_days=30)
    assert end_date_bucket(a, window_days=30) != end_date_bucket(c, window_days=30)


def test_end_date_bucket_handles_none():
    assert end_date_bucket(None, window_days=7) == "no_end_date"


# ------------------------------------------------ layer3_triage tz handling


def test_layer3_triage_accepts_tz_aware_end_dates_without_crashing():
    future = datetime.now(timezone.utc) + timedelta(days=30)
    candidate = [
        _binary_market("m1", end_date=future),
        _binary_market("m2", end_date=future),
        _binary_market("m3", end_date=future),
    ]
    # Must not raise "can't compare offset-naive and offset-aware datetimes".
    survivors = layer3_triage([candidate], store=None)
    # All future-dated markets should survive — min_cluster_size defaults to 2
    # in config, and our cluster has 3 usable members.
    assert len(survivors) == 1
    assert len(survivors[0]) == 3


def test_layer3_triage_filters_past_dated_markets():
    past = datetime.now(timezone.utc) - timedelta(days=1)
    future = datetime.now(timezone.utc) + timedelta(days=30)
    candidate = [
        _binary_market("past1", end_date=past),
        _binary_market("past2", end_date=past),
        _binary_market("future1", end_date=future),
        _binary_market("future2", end_date=future),
    ]
    survivors = layer3_triage([candidate], store=None)
    # Only the two future-dated markets should survive.
    assert len(survivors) == 1
    assert {m.market_id for m in survivors[0]} == {"future1", "future2"}


def test_layer3_triage_normalises_stray_naive_end_dates():
    # Legacy / hand-built markets may ship a naïve end_date. The triage step
    # should treat it as UTC rather than crash on the comparison.
    naive_future = (datetime.now(timezone.utc) + timedelta(days=30)).replace(tzinfo=None)
    candidate = [
        _binary_market("m1", end_date=naive_future),
        _binary_market("m2", end_date=naive_future),
    ]
    survivors = layer3_triage([candidate], store=None)
    assert len(survivors) == 1
    assert len(survivors[0]) == 2
