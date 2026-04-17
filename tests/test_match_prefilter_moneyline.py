"""Extended prefilter tests covering the sports moneyline draw-rule path
(§2.1) and the crypto-aware expiry tolerance (§1.6). Complements the
existing test_match_prefilter.py which predates both features.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from polyquant.agents.match_prefilter import (
    Fingerprint,
    _detect_draw_leg,
    compatible,
    fingerprint_limitless,
    fingerprint_polymarket,
)


def _poly_with_outcomes(question: str, names: list[str], end_date: datetime | None = None):
    return SimpleNamespace(
        question=question,
        end_date=end_date,
        outcomes=[SimpleNamespace(name=name) for name in names],
    )


def _limit_with_outcomes(title: str, names: list[str], end: datetime | None = None):
    return {
        "title": title,
        "expirationDate": end.isoformat() if end else None,
        "outcomes": [{"name": n} for n in names],
    }


# ---------------------------------------------------------------- draw leg


def test_detect_draw_leg_matches_common_names():
    assert _detect_draw_leg(["Home", "Draw", "Away"]) is True
    assert _detect_draw_leg(["Manchester", "Tie", "Liverpool"]) is True
    assert _detect_draw_leg(["Home", "X", "Away"]) is True


def test_detect_draw_leg_rejects_substring_confusion():
    # "Drawn test" substring shouldn't fire — exact whole-outcome match only.
    assert _detect_draw_leg(["Home drawn advantage", "Away"]) is False
    # A 3-way market labelled "Noah"/"Other"/"Something" has no draw leg.
    assert _detect_draw_leg(["Noah", "Other", "Something"]) is False


def test_fingerprint_polymarket_carries_has_draw_flag():
    market = _poly_with_outcomes(
        "Premier League: Arsenal vs Man United",
        ["Arsenal", "Draw", "Man United"],
    )
    fp = fingerprint_polymarket(market)
    assert fp.has_draw_leg is True
    assert fp.outcome_count == 3


def test_fingerprint_limitless_missing_outcomes_returns_none_count():
    # Raw Limitless market with no outcomes/tokens keys — group-market case.
    raw = {"title": "Some AMM market", "expirationDate": None}
    fp = fingerprint_limitless(raw)
    assert fp.outcome_count is None
    assert fp.has_draw_leg is False


# ------------------------------------------------ 3-way ↔ 2-way projection


def test_compatible_allows_sports_3way_vs_2way_with_draw_leg():
    fp_3way_sports = Fingerprint(
        expiry_seconds=None,
        outcome_count=3,
        numeric_bounds=(),
        domain="sports",
        has_draw_leg=True,
    )
    fp_2way_sports = Fingerprint(
        expiry_seconds=None,
        outcome_count=2,
        numeric_bounds=(),
        domain="sports",
        has_draw_leg=False,
    )
    assert compatible(fp_3way_sports, fp_2way_sports) is True


def test_compatible_rejects_3way_vs_2way_when_no_draw_leg():
    fp_3way_politics = Fingerprint(
        expiry_seconds=None,
        outcome_count=3,
        numeric_bounds=(),
        domain="politics",
        has_draw_leg=False,
    )
    fp_2way_politics = Fingerprint(
        expiry_seconds=None,
        outcome_count=2,
        numeric_bounds=(),
        domain="politics",
        has_draw_leg=False,
    )
    # No draw leg, and not sports — keep the hard cardinality gate.
    assert compatible(fp_3way_politics, fp_2way_politics) is False


def test_compatible_defers_when_outcome_count_unknown():
    fp_unknown = Fingerprint(
        expiry_seconds=None,
        outcome_count=None,
        numeric_bounds=(),
        domain="other",
    )
    fp_binary = Fingerprint(
        expiry_seconds=None,
        outcome_count=2,
        numeric_bounds=(),
        domain="other",
    )
    # Unknown cardinality should not hard-reject: defer to LLM.
    assert compatible(fp_unknown, fp_binary) is True


# -------------------------------------------------------- crypto tolerance


def test_compatible_applies_crypto_tolerance_when_supplied():
    base = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    a = Fingerprint(
        expiry_seconds=base.timestamp(),
        outcome_count=2,
        numeric_bounds=(100_000.0,),
        domain="crypto",
    )
    b = Fingerprint(
        expiry_seconds=(base + timedelta(hours=12)).timestamp(),
        outcome_count=2,
        numeric_bounds=(100_000.0,),
        domain="crypto",
    )
    # Default tolerance (24h): passes.
    assert compatible(a, b, expiry_tolerance_hours=24.0) is True
    # Crypto tolerance (1h): same pair now fails on expiry drift.
    assert (
        compatible(
            a, b, expiry_tolerance_hours=24.0, crypto_expiry_tolerance_hours=1.0
        )
        is False
    )


def test_compatible_ignores_crypto_tolerance_for_non_crypto_pairs():
    base = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    a = Fingerprint(
        expiry_seconds=base.timestamp(),
        outcome_count=2,
        numeric_bounds=(),
        domain="politics",
    )
    b = Fingerprint(
        expiry_seconds=(base + timedelta(hours=12)).timestamp(),
        outcome_count=2,
        numeric_bounds=(),
        domain="politics",
    )
    # Crypto tolerance override shouldn't touch a politics pair.
    assert (
        compatible(
            a, b, expiry_tolerance_hours=24.0, crypto_expiry_tolerance_hours=1.0
        )
        is True
    )
