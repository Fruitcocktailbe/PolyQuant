"""Unit tests for the cross-exchange matcher's structural prefilter."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from polyquant.agents.match_prefilter import (
    canonicalize_resolution_source,
    compatible,
    fingerprint_limitless,
    fingerprint_polymarket,
    _classify_domain,
    _expiry_seconds,
    _expiry_week_bucket,
    _extract_numbers,
)


def _poly(question: str, end_date: datetime | None = None, n_outcomes: int = 2):
    return SimpleNamespace(
        question=question,
        end_date=end_date,
        outcomes=[SimpleNamespace() for _ in range(n_outcomes)],
    )


def _limit(title: str, end: datetime | str | int | None = None, n_outcomes: int = 2):
    return {
        "title": title,
        "expirationDate": end.isoformat() if isinstance(end, datetime) else end,
        "outcomes": list(range(n_outcomes)),
    }


# ---------------------------------------------------------------- helpers


def test_extract_numbers_handles_suffixes_and_punctuation():
    assert 100_000.0 in _extract_numbers("Will BTC hit $100k?")
    assert 100_000.0 in _extract_numbers("Will BTC hit $100,000?")
    assert 0.0025 in _extract_numbers("Fed cuts 25bps in September?")
    assert 0.035 in _extract_numbers("Will inflation be above 3.5%?")


def test_classify_domain_picks_strongest_bucket():
    assert _classify_domain("Will BTC hit $100k by year end?") == "crypto"
    assert _classify_domain("Will Trump win the 2024 election?") == "politics"
    assert _classify_domain("Fed cuts rates in September FOMC?") == "macro"
    assert _classify_domain("Random unrelated phrase") == "other"


def test_expiry_week_bucket_groups_nearby_dates():
    a = datetime(2026, 5, 1, tzinfo=timezone.utc)
    b = datetime(2026, 5, 3, tzinfo=timezone.utc)  # same week
    c = datetime(2026, 6, 1, tzinfo=timezone.utc)  # ~4 weeks later
    assert _expiry_week_bucket(a) == _expiry_week_bucket(b)
    assert _expiry_week_bucket(a) != _expiry_week_bucket(c)


def test_expiry_seconds_matches_unix_timestamp():
    dt = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert _expiry_seconds(dt) == dt.timestamp()
    assert _expiry_seconds(None) is None
    assert _expiry_seconds("Not specified") is None


def test_canonicalize_resolution_source_known_aliases():
    assert canonicalize_resolution_source("AP") == "associated press"
    assert canonicalize_resolution_source("ap") == "associated press"
    assert canonicalize_resolution_source("Associated Press") == "associated press"
    assert canonicalize_resolution_source("CG") == "coingecko"
    assert canonicalize_resolution_source("CoinGecko") == "coingecko"
    assert canonicalize_resolution_source("https://www.reuters.com/article/xyz") == "reuters"
    assert canonicalize_resolution_source("NYT") == "new york times"


def test_canonicalize_resolution_source_handles_empty_and_unknown():
    assert canonicalize_resolution_source("") == ""
    assert canonicalize_resolution_source(None) == ""
    assert canonicalize_resolution_source("Not specified") == ""
    assert canonicalize_resolution_source("unknown") == ""
    # Unknown sources pass through (lowercased, punctuation stripped)
    assert canonicalize_resolution_source("Obscure Oracle, Inc.") == "obscure oracle inc"


def test_canonicalize_expands_alias_tokens_in_phrases():
    # Alias table works token-level, not whole-string, so multi-word phrases collapse.
    assert canonicalize_resolution_source("AP official call") == "associated press official call"
    assert canonicalize_resolution_source("CG spot price") == "coingecko spot price"


# ---------------------------------------------------------------- compat: matches


def test_match_clean_btc_pair():
    end = datetime(2026, 5, 30, tzinfo=timezone.utc)
    p = fingerprint_polymarket(_poly("Will BTC hit $100k by May 2026?", end))
    l = fingerprint_limitless(_limit("Will Bitcoin reach $100k before June 2026?", end))
    assert compatible(p, l)


def test_match_election_pair_with_unknown_expiry():
    end = datetime(2024, 11, 5, tzinfo=timezone.utc)
    p = fingerprint_polymarket(_poly("Will Trump win the 2024 presidential election?", end))
    l = fingerprint_limitless(_limit("Donald Trump victor in 2024 election?", end))
    assert compatible(p, l)


def test_match_when_one_side_has_no_extracted_number():
    end = datetime(2026, 9, 30, tzinfo=timezone.utc)
    p = fingerprint_polymarket(_poly("Fed cuts 25bps by September?", end))
    l = fingerprint_limitless(_limit("Will the Fed cut rates by September FOMC?", end))
    # Numeric on poly side only — should defer to LLM, not reject
    assert compatible(p, l)


# ---------------------------------------------------------------- compat: rejects


def test_reject_different_timeframes():
    end_a = datetime(2026, 5, 30, tzinfo=timezone.utc)
    end_b = end_a + timedelta(weeks=8)  # well outside 24h default tolerance
    p = fingerprint_polymarket(_poly("Will BTC hit $100k by May 2026?", end_a))
    l = fingerprint_limitless(_limit("Will BTC hit $100k by July 2026?", end_b))
    assert not compatible(p, l)


def test_reject_timeframe_drift_above_24h_default():
    # New default tolerance is 24h; 36h drift must reject.
    end_a = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    end_b = end_a + timedelta(hours=36)
    p = fingerprint_polymarket(_poly("Will BTC hit $100k by May 30 2026?", end_a))
    l = fingerprint_limitless(_limit("Will BTC hit $100k by June 1 2026?", end_b))
    assert not compatible(p, l)


def test_accept_timeframe_drift_within_24h_default():
    # Same-day drift under default tolerance must accept.
    end_a = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    end_b = end_a + timedelta(hours=6)
    p = fingerprint_polymarket(_poly("Will BTC hit $100k by May 30 2026?", end_a))
    l = fingerprint_limitless(_limit("Will Bitcoin hit $100k on May 30 2026?", end_b))
    assert compatible(p, l)


def test_reject_minute_resolution_under_tight_tolerance():
    # Opt-in 1h tolerance: two BTC snapshots 2 minutes apart must reject.
    end_a = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    end_b = end_a + timedelta(hours=2)
    p = fingerprint_polymarket(_poly("BTC price at 12:00 UTC May 30 2026", end_a))
    l = fingerprint_limitless(_limit("BTC price at 14:00 UTC May 30 2026", end_b))
    assert not compatible(p, l, expiry_tolerance_hours=1.0)


def test_reject_different_numeric_thresholds():
    end = datetime(2026, 5, 30, tzinfo=timezone.utc)
    p = fingerprint_polymarket(_poly("Will BTC hit $100k by May 2026?", end))
    l = fingerprint_limitless(_limit("Will BTC hit $120k by May 2026?", end))
    assert not compatible(p, l)


def test_reject_different_domains():
    end = datetime(2026, 5, 30, tzinfo=timezone.utc)
    p = fingerprint_polymarket(_poly("Will BTC hit $100k by May 2026?", end))
    l = fingerprint_limitless(_limit("Will Trump win the 2024 election?", end))
    assert not compatible(p, l)


def test_reject_binary_vs_multi_outcome():
    end = datetime(2024, 11, 5, tzinfo=timezone.utc)
    p = fingerprint_polymarket(
        _poly("Who will win the 2024 election?", end, n_outcomes=5)
    )
    l = fingerprint_limitless(_limit("Will Trump win the 2024 election?", end))
    assert not compatible(p, l)


def test_reject_same_topic_different_percent():
    end = datetime(2026, 9, 30, tzinfo=timezone.utc)
    p = fingerprint_polymarket(_poly("Will inflation be above 3% in September?", end))
    l = fingerprint_limitless(_limit("Will inflation be above 5% in September?", end))
    assert not compatible(p, l)
