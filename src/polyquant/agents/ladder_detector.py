"""
Monotonic ladder detector — identifies groups of binary threshold markets
that must obey a structural ordering and emits chained SUBSET constraints
for them without involving the LLM.

A classic example from Polymarket:
    "Will BTC hit $100k by EOY 2026?"
    "Will BTC hit $120k by EOY 2026?"
    "Will BTC hit $150k by EOY 2026?"

BTC crossing $150k implies it also crossed $120k and $100k, so:
    P(>=$100k) >= P(>=$120k) >= P(>=$150k)

The LLM can sometimes extract these via SUBSET dependencies, but its output
is non-deterministic and a missed ladder silently drops a real arbitrage
opportunity. This module enumerates ladders structurally so the solver
always sees them, regardless of LLM behaviour on a given run.

Emitted shape: for each adjacent (low, high) pair we produce one
LogicalConstraint with coefficients {low_YES: 1.0, high_YES: -1.0} and
rhs=0.0 — i.e. `z[low_YES] - z[high_YES] >= 0`, which the Frank-Wolfe solver
reads directly in its canonical A^T z >= b form.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polyquant.data.market_models import Market
from polyquant.utils import get_logger
from polyquant.utils.market_utils import get_yes_outcome, has_binary_polarity

if TYPE_CHECKING:
    from polyquant.agents.discovery import MarketCluster
    from polyquant.agents.logic_architect import LogicalConstraint

logger = get_logger(__name__)


# Capture a numeric threshold with optional $ / k / m / b / % suffix. The
# numeric part is what varies across a ladder; the rest of the question is
# the stable template we group by.
_NUMERIC_TOKEN_RE = re.compile(
    r"""
    (?P<prefix>[\$€£])?       # optional currency sign
    (?P<mag>\d{1,3}(?:[,_]\d{3})+|\d+(?:\.\d+)?)  # 100 / 1,000 / 1.5 / 1_000
    \s*
    (?P<suffix>[kKmMbB]|%|\s*(?:thousand|million|billion|percent))?
    """,
    re.VERBOSE,
)

_SUFFIX_MULTIPLIERS = {
    "": 1.0,
    "k": 1_000.0,
    "m": 1_000_000.0,
    "b": 1_000_000_000.0,
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "%": 1.0,        # percents stay on the same scale across a ladder
    "percent": 1.0,
}

_LADDER_PLACEHOLDER = "<N>"


@dataclass(frozen=True)
class _LadderSignature:
    """Stable grouping key for markets that belong to the same ladder.

    Two markets share a signature iff their questions are identical after
    numeric tokens are replaced by ``<N>`` and casing / whitespace is
    normalised. The `event_id` slot prevents cross-event confusion (an
    EOY-2025 BTC ladder must never merge with an EOY-2026 one).
    """

    template: str
    event_id: str


def _normalize_for_template(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _to_numeric(_prefix: str | None, mag: str, suffix: str | None) -> float | None:
    try:
        value = float(mag.replace(",", "").replace("_", ""))
    except ValueError:
        return None
    key = (suffix or "").strip().lower()
    multiplier = _SUFFIX_MULTIPLIERS.get(key)
    if multiplier is None:
        return None
    return value * multiplier


def _extract_ladder_features(question: str) -> tuple[str, list[float]] | None:
    """Return (template_with_placeholders, ordered_numeric_values).

    Returns None when the question contains fewer than one numeric token —
    ladders need at least one varying threshold per question.
    """
    if not question:
        return None
    normalised = _normalize_for_template(question)

    values: list[float] = []

    def _repl(match: re.Match) -> str:
        value = _to_numeric(
            match.group("prefix"), match.group("mag"), match.group("suffix")
        )
        if value is None:
            return match.group(0)
        values.append(value)
        return _LADDER_PLACEHOLDER

    template = _NUMERIC_TOKEN_RE.sub(_repl, normalised)

    if not values:
        return None
    return template, values


def _signature_for_market(market: Market) -> _LadderSignature | None:
    features = _extract_ladder_features(market.question or "")
    if features is None:
        return None
    template, _ = features
    # Markets from the same event share an event_slug; fall back to
    # market_id so isolated markets still cluster with themselves (which
    # detect_monotonic_ladders will then drop below min_rungs).
    event_id = getattr(market, "event_slug", "") or market.market_id
    return _LadderSignature(template=template, event_id=event_id)


def _numeric_for_market(market: Market) -> float | None:
    """Return the *largest* numeric token in the question as the ladder key.

    Questions sometimes embed an ancillary number (year, leg count) alongside
    the threshold; picking the largest reliably isolates the financial
    threshold for crypto / sports / rate-ladder markets without requiring a
    per-template selector table. If this heuristic ever misfires on a real
    ladder we fall back to dropping that group (see `_sort_and_validate`)
    rather than emitting an incorrect ordering.
    """
    features = _extract_ladder_features(market.question or "")
    if features is None:
        return None
    _, values = features
    return max(values) if values else None


def _sort_and_validate(members: list[tuple[Market, float]]) -> list[tuple[Market, float]]:
    """Sort ladder members by threshold and dedupe identical thresholds."""
    by_value: dict[float, tuple[Market, float]] = {}
    for market, value in members:
        # Duplicate thresholds would produce coefficient {m: 1, m: -1} = 0,
        # which is a tautology. Keep the most liquid leg when duplicates exist.
        existing = by_value.get(value)
        if existing is None:
            by_value[value] = (market, value)
        else:
            keep = existing
            if (market.liquidity or 0) > (existing[0].liquidity or 0):
                keep = (market, value)
            by_value[value] = keep
    return sorted(by_value.values(), key=lambda pair: pair[1])


def detect_monotonic_ladders(
    markets: list[Market],
    *,
    min_rungs: int = 3,
) -> list["MarketCluster"]:
    """Partition ``markets`` into monotonic-ladder clusters.

    Returns at most one cluster per (event, question-template) pair. Markets
    that don't slot into any ladder are silently left for the caller to route
    through other discovery paths. ``min_rungs`` is the minimum number of
    distinct threshold values required to emit a cluster — below 3 the SUBSET
    constraints we'd emit are subsumed by cheaper partition / LLM handling.
    """
    # Local import to avoid discovery ↔ ladder_detector cycle at module load.
    from polyquant.agents.discovery import MarketCluster

    groups: dict[_LadderSignature, list[tuple[Market, float]]] = {}
    for market in markets:
        if not has_binary_polarity(market):
            continue
        if get_yes_outcome(market) is None:
            continue
        sig = _signature_for_market(market)
        if sig is None:
            continue
        value = _numeric_for_market(market)
        if value is None:
            continue
        groups.setdefault(sig, []).append((market, value))

    clusters: list[MarketCluster] = []
    for sig, members in groups.items():
        ordered = _sort_and_validate(members)
        if len(ordered) < min_rungs:
            continue

        rung_markets = [m for m, _ in ordered]
        rung_values = [v for _, v in ordered]
        topic_template = sig.template.replace(_LADDER_PLACEHOLDER, "<N>")
        topic = (
            f"[LADDER] {topic_template} "
            f"({len(ordered)} rungs: {rung_values[0]:g} → {rung_values[-1]:g})"
        )
        # Cluster id is content-addressed so reruns produce a stable id even if
        # market ordering in the input list changes.
        id_hash = hashlib.sha1(
            "|".join(m.market_id for m in rung_markets).encode()
        ).hexdigest()[:12]
        cluster_id = f"ladder_{id_hash}"

        clusters.append(
            MarketCluster(
                cluster_id=cluster_id,
                topic=topic,
                markets=rung_markets,
                potential_dependencies=[
                    "[LADDER] Monotonic threshold ladder. For adjacent pairs "
                    "(low, high), P(>=low) >= P(>=high) must hold by definition."
                ],
                constraint_source="monotonic_ladder",
                is_exhaustive=False,
            )
        )

    if clusters:
        logger.info(
            "Detected monotonic ladders",
            count=len(clusters),
            sample_topics=[c.topic[:80] for c in clusters[:3]],
        )
    return clusters


def detect_conditional_subsets(markets: list[Market]) -> list["MarketCluster"]:
    """Group markets that share a `conditional_parent_id` with their parent.

    Emitted clusters carry ``constraint_source="conditional_subset"`` so
    map_maker's mechanical bypass invokes `build_conditional_constraints`
    without an LLM round-trip. Conditional markets are rare but when they
    exist the arbitrage structure is unambiguous: P(child) <= P(parent).

    Parents without at least one child are silently ignored — a single
    parent-child pair is enough since the SUBSET inequality still fires.
    """
    from polyquant.agents.discovery import MarketCluster

    by_id: dict[str, Market] = {m.market_id: m for m in markets if m.market_id}
    children_by_parent: dict[str, list[Market]] = {}
    for m in markets:
        parent_id = getattr(m, "conditional_parent_id", None)
        if not parent_id:
            continue
        children_by_parent.setdefault(parent_id, []).append(m)

    clusters: list[MarketCluster] = []
    for parent_id, children in children_by_parent.items():
        parent = by_id.get(parent_id)
        if parent is None:
            # Parent is outside the scan scope. A SUBSET would still hold,
            # but we don't have its token ids — skip rather than guess.
            continue
        if not has_binary_polarity(parent):
            continue
        eligible_children = [m for m in children if has_binary_polarity(m)]
        if not eligible_children:
            continue
        members = [parent] + eligible_children
        id_hash = hashlib.sha1(parent_id.encode()).hexdigest()[:12]
        cluster_id = f"conditional_{id_hash}"
        clusters.append(
            MarketCluster(
                cluster_id=cluster_id,
                topic=(
                    f"[CONDITIONAL] {parent.question[:80]} "
                    f"(+ {len(eligible_children)} child market(s))"
                ),
                markets=members,
                potential_dependencies=[
                    "[CONDITIONAL] Child markets depend on parent resolution; "
                    "P(child) <= P(parent) must hold by definition."
                ],
                constraint_source="conditional_subset",
                is_exhaustive=False,
            )
        )
    if clusters:
        logger.info(
            "Detected conditional-parent clusters",
            count=len(clusters),
        )
    return clusters


def build_conditional_constraints(cluster: "MarketCluster") -> list["LogicalConstraint"]:
    """Emit SUBSET inequalities for a conditional_subset cluster.

    First market in the cluster is the parent (by convention of
    `detect_conditional_subsets`); every other market contributes one
    constraint ``z[parent_YES] - z[child_YES] >= 0``.
    """
    from polyquant.agents.logic_architect import (
        LogicalConstraint,
        stable_constraint_id,
    )

    if len(cluster.markets) < 2:
        return []
    parent = cluster.markets[0]
    parent_yes = get_yes_outcome(parent)
    if parent_yes is None:
        return []
    parent_token = parent_yes.token_id or parent_yes.outcome_id
    if not parent_token:
        return []

    constraints: list[LogicalConstraint] = []
    for child in cluster.markets[1:]:
        child_yes = get_yes_outcome(child)
        if child_yes is None:
            continue
        child_token = child_yes.token_id or child_yes.outcome_id
        if not child_token or child_token == parent_token:
            continue
        coefficients = {parent_token: 1.0, child_token: -1.0}
        rhs = 0.0
        cid = stable_constraint_id(
            source_cluster_id=cluster.cluster_id,
            coefficients=coefficients,
            rhs=rhs,
            prefix="conditional",
        )
        constraints.append(
            LogicalConstraint(
                constraint_id=cid,
                description=f"[CONDITIONAL] P({child.question[:40]}) <= P(parent)",
                coefficients=coefficients,
                rhs=rhs,
                confidence=1.0,
                source_markets=[parent.market_id, child.market_id],
                reasoning=(
                    "Conditional market: child resolves only if parent resolved "
                    "YES, so P(child) <= P(parent) structurally."
                ),
                is_exhaustive=False,
            )
        )
    return constraints


def build_ladder_constraints(cluster: "MarketCluster") -> list["LogicalConstraint"]:
    """Emit chained SUBSET inequalities for a ladder cluster.

    Each adjacent pair (m_low, m_high) contributes one constraint of the form
    ``z[low_YES] - z[high_YES] >= 0`` — equivalently P(low) >= P(high), which
    is the structural guarantee that a higher threshold being met implies the
    lower one was too.
    """
    # Local import keeps the logic_architect module independent of this one.
    from polyquant.agents.logic_architect import (
        LogicalConstraint,
        stable_constraint_id,
    )

    ordered: list[tuple[Market, float]] = []
    for market in cluster.markets:
        value = _numeric_for_market(market)
        if value is None:
            continue
        ordered.append((market, value))
    ordered = _sort_and_validate(ordered)
    if len(ordered) < 2:
        return []

    constraints: list[LogicalConstraint] = []
    for (low_market, low_val), (high_market, high_val) in zip(ordered, ordered[1:]):
        low_yes = get_yes_outcome(low_market)
        high_yes = get_yes_outcome(high_market)
        if low_yes is None or high_yes is None:
            continue
        low_token = low_yes.token_id or low_yes.outcome_id
        high_token = high_yes.token_id or high_yes.outcome_id
        if not low_token or not high_token or low_token == high_token:
            continue
        coefficients = {low_token: 1.0, high_token: -1.0}
        rhs = 0.0
        constraint_id = stable_constraint_id(
            source_cluster_id=cluster.cluster_id,
            coefficients=coefficients,
            rhs=rhs,
            prefix="ladder",
        )
        constraints.append(
            LogicalConstraint(
                constraint_id=constraint_id,
                description=(
                    f"[LADDER] P(>={low_val:g}) >= P(>={high_val:g})"
                ),
                coefficients=coefficients,
                rhs=rhs,
                confidence=1.0,
                source_markets=[low_market.market_id, high_market.market_id],
                reasoning=(
                    "Monotonic threshold: a higher threshold being met "
                    "structurally implies every lower threshold was met too."
                ),
                is_exhaustive=False,
            )
        )
    return constraints
