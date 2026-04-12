"""
Structural pre-filter for cross-exchange market matching.

Builds a cheap fingerprint per market and decides whether two markets are
structurally compatible BEFORE we spend embedding/LLM cycles on them. This is
the precision layer of the matching funnel — it is meant to kill obvious
non-matches (different timeframe, different numeric bound, different domain,
different outcome cardinality) so the LLM only sees plausible pairs.

Pure Python, no external deps. Works on both Polymarket Market objects and
Limitless raw dicts via duck-typed `extract_*` helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "crypto": (
        "btc", "bitcoin", "eth", "ethereum", "sol", "solana", "xrp", "doge",
        "crypto", "token", "stablecoin", "altcoin", "defi", "nft",
    ),
    "politics": (
        "election", "president", "trump", "biden", "harris", "vance", "putin",
        "congress", "senate", "house", "governor", "primary", "vote", "ballot",
        "republican", "democrat", "gop", "parliament", "prime minister",
    ),
    "sports": (
        "nba", "nfl", "mlb", "nhl", "fifa", "uefa", "premier league", "champions",
        "super bowl", "world cup", "olympics", "tennis", "golf", "f1", "ufc",
        "boxing", "playoff", "finals", "match", "vs ", "v.",
    ),
    "macro": (
        "fed", "fomc", "rate cut", "rate hike", "interest rate", "inflation",
        "cpi", "unemployment", "gdp", "recession", "treasury", "yield",
    ),
    "weather": (
        "hurricane", "storm", "tornado", "snow", "rain", "temperature",
        "weather", "noaa", "celsius", "fahrenheit",
    ),
}

# Numbers like 100, 100k, 1.5m, 25bps, 3.5%, $100,000
_NUMBER_RE = re.compile(
    r"""
    \$?                        # optional currency
    (?P<num>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)
    \s?
    (?P<suffix>bps|k|m|b|%)?   # optional unit (longest first for alternation)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SUFFIX_MULT: dict[str, float] = {
    "": 1.0,
    "k": 1_000.0,
    "m": 1_000_000.0,
    "b": 1_000_000_000.0,
    "bps": 0.0001,  # basis points -> fraction
    "%": 0.01,      # percent -> fraction
}


@dataclass(frozen=True)
class Fingerprint:
    """Structural fingerprint of a market for compatibility checking."""
    expiry_week: int | None       # ISO week-since-epoch bucket, or None if unknown
    outcome_count: int             # 2 for binary, N otherwise
    numeric_bounds: tuple[float, ...]  # normalized numeric bounds extracted from question
    domain: str                    # one of DOMAIN_KEYWORDS keys, or "other"


def _to_datetime(value: Any) -> datetime | None:
    """Best-effort parse of an end-date field from either exchange."""
    if value is None or value == "" or value == "Not specified":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        # Limitless sometimes ships unix seconds, sometimes ms
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Numeric string?
        try:
            return _to_datetime(float(s))
        except ValueError:
            pass
        # ISO 8601
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _expiry_week_bucket(end_date: Any) -> int | None:
    dt = _to_datetime(end_date)
    if dt is None:
        return None
    # Days since unix epoch // 7 = stable weekly bucket independent of ISO calendar quirks
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return int((dt - epoch).days // 7)


def _looks_like_year(raw: str, suffix: str, value: float) -> bool:
    """Filter out 4-digit years — they're date hints, captured by the expiry bucket."""
    if suffix:
        return False
    if "," in raw or "." in raw:
        return False
    return 1900.0 <= value <= 2100.0 and value == int(value) and len(raw) == 4


def _extract_numbers(text: str) -> tuple[float, ...]:
    if not text:
        return ()
    found: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group("num").replace(",", "")
        suffix = (match.group("suffix") or "").lower()
        try:
            base = float(raw)
        except ValueError:
            continue
        if _looks_like_year(raw, suffix, base):
            continue
        mult = _SUFFIX_MULT.get(suffix, 1.0)
        found.append(base * mult)
    # Dedupe (preserve order) — markets often repeat the same threshold in question + description
    seen: set[float] = set()
    out: list[float] = []
    for n in found:
        key = round(n, 6)
        if key not in seen:
            seen.add(key)
            out.append(n)
    return tuple(out)


def _classify_domain(text: str) -> str:
    t = text.lower()
    best: tuple[str, int] = ("other", 0)
    for domain, kws in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in t)
        if hits > best[1]:
            best = (domain, hits)
    return best[0]


def _polymarket_outcome_count(market: Any) -> int:
    outcomes = getattr(market, "outcomes", None)
    if outcomes is None:
        return 2
    return max(2, len(outcomes))


def _limitless_outcome_count(market: dict[str, Any]) -> int:
    # Limitless binary markets typically expose 2 outcomes; multi-outcome markets
    # ship them under "outcomes" or "tokens".
    for key in ("outcomes", "tokens", "outcomeTokens"):
        val = market.get(key)
        if isinstance(val, list) and len(val) > 0:
            return max(2, len(val))
    return 2


def fingerprint_polymarket(market: Any) -> Fingerprint:
    question = getattr(market, "question", "") or ""
    return Fingerprint(
        expiry_week=_expiry_week_bucket(getattr(market, "end_date", None)),
        outcome_count=_polymarket_outcome_count(market),
        numeric_bounds=_extract_numbers(question),
        domain=_classify_domain(question),
    )


def fingerprint_limitless(market: dict[str, Any]) -> Fingerprint:
    title = market.get("title", "") or ""
    end_raw = (
        market.get("expirationDate")
        or market.get("expirationTimestamp")
        or market.get("endDate")
    )
    return Fingerprint(
        expiry_week=_expiry_week_bucket(end_raw),
        outcome_count=_limitless_outcome_count(market),
        numeric_bounds=_extract_numbers(title),
        domain=_classify_domain(title),
    )


def _numbers_compatible(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """If both sides extract numeric bounds, at least one must be ~equal."""
    if not a or not b:
        return True  # one side has no extractable number; defer to LLM
    for na in a:
        for nb in b:
            if na == 0 and nb == 0:
                return True
            denom = max(abs(na), abs(nb), 1e-9)
            if abs(na - nb) / denom < 0.01:  # within 1%
                return True
    return False


def compatible(
    a: Fingerprint,
    b: Fingerprint,
    *,
    expiry_slack_weeks: int = 1,
) -> bool:
    """
    Decide whether two fingerprints are structurally compatible.

    Returns False as soon as a hard mismatch is found. Unknown fields (None
    expiry, "other" domain, empty numeric bounds) defer to downstream layers
    rather than rejecting on missing data.
    """
    # Outcome cardinality: binary must match binary; N-ary must match N-ary
    if (a.outcome_count == 2) != (b.outcome_count == 2):
        return False

    # Domain bucket: if both sides classified into a known domain, they must agree
    if a.domain != "other" and b.domain != "other" and a.domain != b.domain:
        return False

    # Expiry: if both known, must be within slack
    if a.expiry_week is not None and b.expiry_week is not None:
        if abs(a.expiry_week - b.expiry_week) > expiry_slack_weeks:
            return False

    # Numeric bounds
    if not _numbers_compatible(a.numeric_bounds, b.numeric_bounds):
        return False

    return True
