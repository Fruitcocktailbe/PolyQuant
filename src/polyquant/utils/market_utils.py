"""
Market-related utility functions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polyquant.data.market_models import Market, Outcome


def extract_market_id(outcome_id: str) -> str:
    """
    Extract the market ID from an outcome ID.

    Format expected: "marketID_outcomeID" or just "marketID" if no underscore.

    Args:
        outcome_id: The outcome ID string

    Returns:
        The market ID part, or empty string if invalid
    """
    if not outcome_id:
        return ""

    if "_" in outcome_id:
        return outcome_id.split("_")[0]

    return outcome_id


def get_yes_outcome(market: "Market") -> "Outcome | None":
    """
    Resolve a market's YES outcome by name.

    Returns the first outcome whose name contains "yes" (case-insensitive).
    Returns None if no such outcome exists — callers MUST handle the None
    case explicitly (log + skip) rather than falling back to index 0.

    Indexing outcomes positionally is unsafe: Polymarket and Limitless both
    return outcomes in non-deterministic order, so `market.outcomes[0]` is
    not guaranteed to be YES.
    """
    for outcome in market.outcomes:
        if "yes" in outcome.name.lower():
            return outcome
    return None


def get_no_outcome(market: "Market") -> "Outcome | None":
    """
    Resolve a market's NO outcome by name.

    Returns the first outcome whose name contains "no" (case-insensitive).
    Returns None if no such outcome exists — callers MUST handle the None
    case explicitly rather than falling back to index 1.
    """
    for outcome in market.outcomes:
        if "no" in outcome.name.lower():
            return outcome
    return None


def has_binary_polarity(market: "Market") -> bool:
    """
    Check if a market has resolvable YES/NO polarity.

    Used as a discovery-time filter: markets without clear "yes"/"no" named
    outcomes (e.g., "Trump wins"/"Trump loses") cannot be safely fed to the
    solver, because every downstream module assumes the YES token can be
    identified by name.
    """
    return get_yes_outcome(market) is not None and get_no_outcome(market) is not None
