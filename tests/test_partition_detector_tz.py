"""Regression test for the partition_detector tz-naive vs tz-aware bug.

`end_date_bucket` was silently assuming naïve anchors, which crashed the
whole map-maker run once a tz-aware end_date from the Polymarket API
reached it (seen in production 2026-04-17).
"""

from __future__ import annotations

from datetime import datetime, timezone

from polyquant.agents.partition_detector import end_date_bucket


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
