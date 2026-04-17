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


# Canonicalization aliases for resolution-source strings. Used as a HINT only —
# two markets whose sources canonicalize identically still need LLM verification
# that they describe the same real-world event (different AP stories share source
# but not event). Conversely, differing sources don't imply different events
# (AP vs. NYT projections of the same election call are the same event).
_RESOLUTION_SOURCE_ALIASES: dict[str, str] = {
    "ap": "associated press",
    "associated-press": "associated press",
    "cg": "coingecko",
    "coin-gecko": "coingecko",
    "cb": "coinbase",
    "bbg": "bloomberg",
    "reuters.com": "reuters",
    "nyt": "new york times",
    "nytimes": "new york times",
    "wsj": "wall street journal",
    "cnn.com": "cnn",
    "bbc.co.uk": "bbc",
    "bbc.com": "bbc",
}

_PUNCT_RE = re.compile(r"[^\w\s]+")
_WS_RE = re.compile(r"\s+")


def canonicalize_resolution_source(source: Any) -> str:
    """
    Normalize a resolution-source string for weak-positive comparison.

    Returns an empty string for missing / unspecified inputs so callers can
    detect the "unknown" case explicitly.

    This is NOT a source-of-truth equality check — it only collapses obvious
    spelling variants (casing, punctuation, stock abbreviations). The LLM
    still decides whether two markets resolve on the same real-world event.
    """
    if source is None:
        return ""
    s = str(source).strip().lower()
    if not s or s in ("not specified", "unknown", "n/a", "none"):
        return ""
    # Remove URLs' protocol/path noise before alias lookup — sources often
    # ship as "https://ap.org/..." which should collapse to "associated press".
    s = re.sub(r"^https?://(www\.)?", "", s)
    s = s.split("/")[0]  # keep host only
    # Apply whole-host alias lookup BEFORE punctuation stripping so entries
    # like "reuters.com" → "reuters" match. If no host-level alias, fall
    # through to punctuation stripping + token-level alias lookup below.
    if s in _RESOLUTION_SOURCE_ALIASES:
        return _RESOLUTION_SOURCE_ALIASES[s]
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    # Alias expansion — token-level so "ap official call" → "associated press official call".
    tokens = [_RESOLUTION_SOURCE_ALIASES.get(tok, tok) for tok in s.split()]
    return " ".join(tokens)


_DRAW_OUTCOME_TOKENS = frozenset({"draw", "tie", "drawn", "d", "x"})


def _detect_draw_leg(outcome_names: list[str]) -> bool:
    """True when any outcome name is a direct synonym for a draw/tie result.

    Single-character matches ("D", "X") require exact equality — substring
    matching on 1-char tokens would flag virtually everything.
    """
    for raw in outcome_names:
        if not isinstance(raw, str):
            continue
        name = raw.strip().lower()
        if not name:
            continue
        if name in _DRAW_OUTCOME_TOKENS:
            return True
    return False


@dataclass(frozen=True)
class Fingerprint:
    """Structural fingerprint of a market for compatibility checking."""
    expiry_seconds: float | None   # unix seconds (UTC), or None if unknown
    outcome_count: int | None      # 2 for binary, N otherwise; None when the
                                   # source exchange didn't supply an outcome
                                   # list (defer to the LLM rather than silently
                                   # assuming binary and dropping real matches)
    numeric_bounds: tuple[float, ...]  # normalized numeric bounds extracted from question
    domain: str                    # one of DOMAIN_KEYWORDS keys, or "other"
    has_draw_leg: bool = False     # True when outcome names include draw/tie
                                   # (used by 3-way ↔ 2-way sports moneyline
                                   # projection in compatible())


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


def _expiry_seconds(end_date: Any) -> float | None:
    """Parse an end-date into unix seconds (UTC). Returns None if unparseable."""
    dt = _to_datetime(end_date)
    if dt is None:
        return None
    return dt.timestamp()


def _expiry_week_bucket(end_date: Any) -> int | None:
    """Legacy helper retained for backwards compatibility with old callers
    and tests. New code should use `_expiry_seconds` + an hours-based delta."""
    secs = _expiry_seconds(end_date)
    if secs is None:
        return None
    return int(secs // (7 * 86400))


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


def _polymarket_outcome_count(market: Any) -> int | None:
    outcomes = getattr(market, "outcomes", None)
    if outcomes is None:
        return None
    return max(2, len(outcomes))


def _limitless_outcome_count(market: dict[str, Any]) -> int | None:
    # Limitless binary markets typically expose 2 outcomes; multi-outcome markets
    # ship them under "outcomes" or "tokens". When none of those keys exist we
    # can't tell whether the market is binary or an N-way AMM — silently
    # defaulting to 2 caused group markets to be rejected against correctly
    # fingerprinted N-ary Polymarket markets. Return None to signal
    # "unknown" and let the LLM handle the disambiguation.
    for key in ("outcomes", "tokens", "outcomeTokens"):
        val = market.get(key)
        if isinstance(val, list) and len(val) > 0:
            return max(2, len(val))
    return None


def fingerprint_polymarket(market: Any) -> Fingerprint:
    question = getattr(market, "question", "") or ""
    outcomes = getattr(market, "outcomes", None) or []
    outcome_names = [
        getattr(o, "name", "") for o in outcomes
    ]
    return Fingerprint(
        expiry_seconds=_expiry_seconds(getattr(market, "end_date", None)),
        outcome_count=_polymarket_outcome_count(market),
        numeric_bounds=_extract_numbers(question),
        domain=_classify_domain(question),
        has_draw_leg=_detect_draw_leg(outcome_names),
    )


def fingerprint_limitless(market: dict[str, Any]) -> Fingerprint:
    title = market.get("title", "") or ""
    end_raw = (
        market.get("expirationDate")
        or market.get("expirationTimestamp")
        or market.get("endDate")
    )
    outcome_names: list[str] = []
    for key in ("outcomes", "tokens", "outcomeTokens"):
        val = market.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    outcome_names.append(item)
                elif isinstance(item, dict):
                    for field in ("name", "title", "outcome"):
                        if isinstance(item.get(field), str):
                            outcome_names.append(item[field])
                            break
    return Fingerprint(
        expiry_seconds=_expiry_seconds(end_raw),
        outcome_count=_limitless_outcome_count(market),
        numeric_bounds=_extract_numbers(title),
        domain=_classify_domain(title),
        has_draw_leg=_detect_draw_leg(outcome_names),
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
    expiry_tolerance_hours: float = 24.0,
    crypto_expiry_tolerance_hours: float | None = None,
) -> bool:
    """
    Decide whether two fingerprints are structurally compatible.

    Returns False as soon as a hard mismatch is found. Unknown fields (None
    expiry, "other" domain, empty numeric bounds) defer to downstream layers
    rather than rejecting on missing data.

    `expiry_tolerance_hours` is the maximum allowed drift between resolution
    deadlines for general-domain pairs. `crypto_expiry_tolerance_hours` is a
    tighter fallback applied when either side is classified as `crypto` —
    BTC/ETH snapshot markets resolve on point-in-time prices that diverge
    meaningfully inside a 24h window, so the default tolerance is too loose
    there. Pass None to reuse `expiry_tolerance_hours` for crypto too (legacy
    behaviour).
    """
    # Outcome cardinality: binary must match binary; N-ary must match N-ary,
    # except for the sports moneyline projection case — a 3-way market with a
    # draw leg can legitimately match a 2-way market on the same sports event
    # once the LLM confirms the draw resolution rule (draw_is_no vs
    # double_chance). Gate that narrow exception on BOTH sides being sports
    # and the N-ary side actually having a draw leg; otherwise keep the
    # original hard reject.
    #
    # Unknown outcome_count (None) means the source exchange didn't expose an
    # outcome list — defer to the LLM rather than hard-reject, which used to
    # silently drop group markets that shipped without the "outcomes" key.
    if a.outcome_count is not None and b.outcome_count is not None:
        a_binary = a.outcome_count == 2
        b_binary = b.outcome_count == 2
        if a_binary != b_binary:
            nary_side = b if a_binary else a
            both_sports = a.domain == "sports" and b.domain == "sports"
            moneyline_projection_ok = (
                nary_side.outcome_count == 3
                and nary_side.has_draw_leg
                and both_sports
            )
            if not moneyline_projection_ok:
                return False

    # Domain bucket: if both sides classified into a known domain, they must agree
    if a.domain != "other" and b.domain != "other" and a.domain != b.domain:
        return False

    # Expiry: if both known, must be within tolerance. Crypto pairs get a
    # tighter window because a 12:00 UTC vs 12:25 UTC snapshot are different
    # events even though they share a 24h window.
    effective_tolerance = expiry_tolerance_hours
    if crypto_expiry_tolerance_hours is not None and "crypto" in (a.domain, b.domain):
        effective_tolerance = crypto_expiry_tolerance_hours
    if a.expiry_seconds is not None and b.expiry_seconds is not None:
        if abs(a.expiry_seconds - b.expiry_seconds) > effective_tolerance * 3600.0:
            return False

    # Numeric bounds
    if not _numbers_compatible(a.numeric_bounds, b.numeric_bounds):
        return False

    return True
