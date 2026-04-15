"""
Cross-Market Partition Detector.

Finds groups of separate binary YES/NO markets that implicitly partition a
single real-world event (e.g. N standalone "Will <candidate> win ...?" markets
that together cover one election). Such groups are valuable because their YES
prices should sum to ≈ 1.0 — any deviation is a cross-market arbitrage.

Pipeline (all except Layer 4 runs without LLM calls):

    Layer 1: Mechanical pre-clustering (tags + question template + end_date)
    Layer 2: Semantic pre-clustering (sentence-transformer embeddings)
    Layer 3: Structural triage (no price fetching — map_maker ≠ navigator)
    Layer 4: LLM verification of candidate clusters (one Flash-Lite call each)

Verified clusters become `MarketCluster(constraint_source="cross_market_partition")`
objects that map_maker bypasses LogicArchitect for, emitting a mechanical
partition `LogicalConstraint` directly.

This module's correctness hinges on the verification prompt — see
`VERIFICATION_PROMPT` below for its design rationale.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from polyquant.data import Market
from polyquant.utils import config, get_logger
from polyquant.utils.llm_client import call_llm_json
from polyquant.utils.market_utils import get_yes_outcome, has_binary_polarity

if TYPE_CHECKING:
    from polyquant.agents.discovery import MarketCluster
    from polyquant.data.constraint_store import ConstraintStore

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Layer 1 — template extraction
# -----------------------------------------------------------------------------

# Hand-written, high-precision / low-recall patterns. Each captures:
#   (template_id, slot_group) where slot_group is the varying piece.
# Add more over time as unmatched questions are logged.
TEMPLATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "will_x_win_year_event",
        re.compile(
            r"^will\s+(?P<slot>.+?)\s+win\s+the\s+(?P<year>\d{4})\s+(?P<event>.+?)\??$",
            re.IGNORECASE,
        ),
    ),
    (
        "who_win_year_event",
        re.compile(
            r"^who\s+will\s+win\s+the\s+(?P<year>\d{4})\s+(?P<event>.+?)\??$",
            re.IGNORECASE,
        ),
    ),
    (
        "will_x_be_year_role",
        re.compile(
            r"^will\s+(?P<slot>.+?)\s+be\s+the\s+(?P<year>\d{4})\s+(?P<role>.+?)\??$",
            re.IGNORECASE,
        ),
    ),
    (
        "will_fed_action_month",
        re.compile(
            r"^will\s+the\s+fed\s+(?P<action>raise|hold|cut|lower)\s+rates?\s+in\s+(?P<month>.+?)\??$",
            re.IGNORECASE,
        ),
    ),
    (
        "will_x_win_event",
        re.compile(
            r"^will\s+(?P<slot>.+?)\s+win\s+(?P<event>.+?)\??$",
            re.IGNORECASE,
        ),
    ),
]


def extract_template(question: str) -> tuple[str, str] | None:
    """
    Extract a template signature from a market question.

    Returns (template_key, slot_value) or None if no pattern matches.
    The template_key is a stable string that groups markets sharing the same
    underlying event structure (e.g. same election, same Fed meeting). The
    slot_value is the varying piece (candidate name, action) used for
    debugging only — it is not part of the grouping key.
    """
    if not question:
        return None
    q = question.strip()
    for template_id, pattern in TEMPLATE_PATTERNS:
        match = pattern.match(q)
        if not match:
            continue
        groups = match.groupdict()
        # Build a key from every named group EXCEPT the varying `slot`.
        fixed_parts = [
            f"{k}={v.strip().lower()}"
            for k, v in sorted(groups.items())
            if k != "slot" and v
        ]
        template_key = f"{template_id}|{'|'.join(fixed_parts)}"
        slot_value = groups.get("slot", "").strip()
        return (template_key, slot_value)
    return None


# -----------------------------------------------------------------------------
# Layer 1 / 2 — end_date bucketing
# -----------------------------------------------------------------------------

def end_date_bucket(end_date: datetime | None, window_days: int) -> str:
    """
    Coarse time bucket for hard prefiltering in Layer 2.

    Markets whose end_dates differ by more than `window_days` end up in
    different buckets and never enter the same similarity computation — this
    is the mitigation for Hole #1 (embeddings silently cluster "2024 primary"
    with "2028 primary" because their text is near-identical).
    """
    if end_date is None:
        return "no_end_date"
    anchor = datetime(2000, 1, 1)
    days_since = (end_date - anchor).days
    bucket_index = days_since // max(1, window_days)
    return f"bucket_{bucket_index}"


def _tag_signature(tags: list[str] | None) -> str:
    """Stable tag signature for Layer 1 grouping. Ignores casing / order."""
    if not tags:
        return "no_tags"
    cleaned = sorted({t.strip().lower() for t in tags if t})
    if not cleaned:
        return "no_tags"
    return "|".join(cleaned)


def _market_tags(market: Market) -> list[str]:
    """Best-effort extraction of tags from a Market. Tags live on the event,
    which isn't attached to Market here, so we fall back to no_tags."""
    del market  # tags are event-level, not market-level — fallback per plan
    return []


# -----------------------------------------------------------------------------
# Layer 1 — template-based clustering (strict, high-precision)
# -----------------------------------------------------------------------------

def layer1_template_cluster(
    markets: list[Market],
) -> tuple[list[list[Market]], set[str]]:
    """
    Group markets by (template_key, tag_signature, end_date_bucket).

    Returns (clusters, matched_market_ids). Markets that didn't match any
    template are excluded from `matched_market_ids` so Layer 2 can process
    them.
    """
    buckets: dict[tuple[str, str, str], list[Market]] = {}
    matched: set[str] = set()
    unmatched_samples: list[str] = []

    window_days = config.partition_end_date_window_days

    for market in markets:
        extracted = extract_template(market.question)
        if not extracted:
            if len(unmatched_samples) < 5:
                unmatched_samples.append(market.question[:80])
            continue
        template_key, _slot = extracted
        key = (
            template_key,
            _tag_signature(_market_tags(market)),
            end_date_bucket(market.end_date, window_days),
        )
        buckets.setdefault(key, []).append(market)
        matched.add(market.market_id)

    clusters: list[list[Market]] = [
        ms for ms in buckets.values() if len(ms) >= config.partition_min_cluster_size
    ]

    # Drop singletons from `matched` so Layer 2 retries them.
    singletons = {
        m.market_id
        for ms in buckets.values()
        if len(ms) < config.partition_min_cluster_size
        for m in ms
    }
    matched -= singletons

    logger.info(
        "Partition Layer 1 (template)",
        input_markets=len(markets),
        clusters=len(clusters),
        matched_markets=len(matched),
        unmatched_sample=unmatched_samples,
    )
    return clusters, matched


# -----------------------------------------------------------------------------
# Layer 2 — embedding-based clustering (soft, catches Layer 1 misses)
# -----------------------------------------------------------------------------

def layer2_embedding_cluster(markets: list[Market]) -> list[list[Market]]:
    """
    Cluster markets by cosine similarity of their question embeddings.

    Applies a hard end_date_bucket prefilter first — markets more than
    `partition_end_date_window_days` apart never enter the same similarity
    matrix. Returns clusters of size >= `partition_min_cluster_size`.
    """
    if len(markets) < config.partition_min_cluster_size:
        return []

    try:
        import gc
        from sentence_transformers import SentenceTransformer, util
        import torch  # type: ignore
    except Exception as e:
        logger.warning(
            "Layer 2 embeddings unavailable (sentence-transformers missing): skipping",
            error=str(e),
        )
        return []

    window_days = config.partition_end_date_window_days

    # Hard prefilter: group by end_date_bucket so embeddings never merge
    # distant-in-time markets.
    date_buckets: dict[str, list[Market]] = {}
    for m in markets:
        date_buckets.setdefault(end_date_bucket(m.end_date, window_days), []).append(m)

    all_clusters: list[list[Market]] = []

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.warning("Failed to load embedding model for Layer 2", error=str(e))
        return []

    threshold = config.partition_embedding_threshold
    min_size = config.partition_min_cluster_size
    max_size = config.partition_max_cluster_size

    for bucket_key, bucket_markets in date_buckets.items():
        if len(bucket_markets) < min_size:
            continue

        questions = [m.question for m in bucket_markets]
        try:
            embeddings = model.encode(questions, convert_to_tensor=True)
            sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        except Exception as e:
            logger.warning(
                "Embedding computation failed in Layer 2",
                bucket=bucket_key,
                error=str(e),
            )
            continue

        # Greedy agglomerative clustering
        visited: set[int] = set()
        for i in range(len(bucket_markets)):
            if i in visited:
                continue
            cluster = [bucket_markets[i]]
            visited.add(i)
            for j in range(i + 1, len(bucket_markets)):
                if j in visited:
                    continue
                if sim_matrix[i][j] >= threshold:
                    cluster.append(bucket_markets[j])
                    visited.add(j)
                    if len(cluster) >= max_size:
                        break
            if len(cluster) >= min_size:
                all_clusters.append(cluster)

    # Free model to save RAM (mirrors existing discovery.py pattern).
    del model
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    logger.info(
        "Partition Layer 2 (embeddings)",
        input_markets=len(markets),
        clusters=len(all_clusters),
        threshold=threshold,
    )
    return all_clusters


# -----------------------------------------------------------------------------
# Layer 3 — structural triage (NO price fetching)
# -----------------------------------------------------------------------------

def layer3_triage(
    candidates: list[list[Market]],
    store: "ConstraintStore | None",
) -> list[list[Market]]:
    """
    Filter candidate clusters by structural properties only. No network I/O,
    no price checks — that's the Navigator's job, not the Map Maker's.
    """
    survivors: list[list[Market]] = []
    rejection_reasons: dict[str, int] = {
        "size_bounds": 0,
        "duplicate_token": 0,
        "post_filter_below_min": 0,
        "fresh_manifest": 0,
    }
    now = datetime.utcnow()
    ttl_hours = config.partition_manifest_ttl_hours
    min_size = config.partition_min_cluster_size
    max_size = config.partition_max_cluster_size

    for candidate in candidates:
        # Size bounds
        if not (min_size <= len(candidate) <= max_size):
            rejection_reasons["size_bounds"] += 1
            continue

        # Freshness + polarity checks
        usable: list[Market] = []
        seen_tokens: set[str] = set()
        duplicate_token = False
        for m in candidate:
            if m.resolved:
                continue
            if m.end_date is not None and m.end_date <= now:
                continue
            yes = get_yes_outcome(m)
            if yes is None:
                continue
            token_id = yes.token_id or yes.outcome_id
            if not token_id:
                continue
            if token_id in seen_tokens:
                duplicate_token = True
                break
            seen_tokens.add(token_id)
            usable.append(m)

        if duplicate_token:
            rejection_reasons["duplicate_token"] += 1
            continue
        if len(usable) < min_size:
            rejection_reasons["post_filter_below_min"] += 1
            continue

        # Dedup against recently-persisted constraint manifests (if store is wired)
        if store is not None:
            token_hash = hashlib.sha256(
                ",".join(sorted(seen_tokens)).encode("utf-8")
            ).hexdigest()[:16]
            candidate_constraint_id = f"cross_{token_hash}"
            if store.is_constraint_fresh(candidate_constraint_id, ttl_hours):
                rejection_reasons["fresh_manifest"] += 1
                continue

        survivors.append(usable)

    logger.info(
        "Partition Layer 3 (triage)",
        input_candidates=len(candidates),
        survivors=len(survivors),
        rejection_reasons=rejection_reasons,
    )
    return survivors


# -----------------------------------------------------------------------------
# Layer 4 — LLM verification
# -----------------------------------------------------------------------------

VERIFICATION_PROMPT = """You are verifying whether a set of prediction markets form a valid PARTITION for
arbitrage coupling. Your verification directly controls which market groups the
trading system treats as correlated. False positives CORRUPT the solver by
coupling unrelated markets. False negatives MISS arbitrage. Both are costly.

=====================================================================
DEFINITIONS
=====================================================================

A group of markets is a valid PARTITION if and only if:

  (A) EVENT IDENTITY: Every market in the group resolves based on the SAME
      real-world event — same event, same date, same resolution criterion,
      same resolution source, same geographic/categorical scope.

  (B) MUTUAL EXCLUSIVITY: At most ONE market in the group can resolve YES.
      For every pair (i, j), if market i resolves YES, market j must
      necessarily resolve NO under any reading of the resolution rules.

  (C) EXHAUSTIVENESS (OPTIONAL): Under ALL possible outcomes of the underlying
      event, AT LEAST one market in the group resolves YES. If this fails, the
      partition is PARTIAL — still useful for sell-side arbitrage, but not for
      buy-side.

MUTUAL EXCLUSIVITY is mandatory. EXHAUSTIVENESS is reported separately.

=====================================================================
REASONING PROTOCOL (follow in order)
=====================================================================

STEP 1 — Articulate the underlying event in one sentence.
   Format: "[event type] on [specific date] resolved by [source] with
   outcome space [enumerated possible outcomes]"
   If you cannot write this sentence cleanly, set is_partition=false and stop.

STEP 2 — Event-identity check, per market.
   For each market, answer: "Does this market's YES condition correspond to
   exactly one of the possible outcomes of the event in Step 1?"
   RED FLAGS (any one of these = exclude this market from the partition):
     * Different year, cycle, or election date
     * Different resolution source that could diverge (UMA vs Polymarket vs custom)
     * Conditional language ("if X then Y") when others are unconditional
     * Different geographic scope (federal vs state, national vs league)
     * Different resolution window or end date that doesn't align
     * Market description references a different underlying event than the
       question suggests

STEP 3 — Pairwise mutual-exclusivity check.
   For every ordered pair (i, j) of markets that survived Step 2:
     "If market i resolves YES, is it LOGICALLY NECESSARY that market j
      resolves NO under the written resolution rules?"
   If ANY pair fails this check, the surviving set is NOT mutually exclusive.
   Prune the weakest member and re-check, OR set is_partition=false.

STEP 4 — Exhaustiveness check.
   List every possible outcome of the underlying event from Step 1.
   For each outcome, identify which market (if any) resolves YES on it.
   If one or more outcomes are not covered by ANY market in the surviving
   set, set is_exhaustive=false and list the uncovered outcomes.

STEP 5 — Self-check and calibration.
   Re-read your Step 1 event sentence. For each surviving market, verbally
   confirm the template: "If this market resolves YES, it means [specific
   outcome] of [the event in Step 1] on [the specific date]."
   Any market that doesn't fit cleanly → exclude it.
   If fewer than 2 markets survive, set is_partition=false.

   Confidence calibration:
     1.0  — resolution rules are unambiguous and every surviving market fits
            the template with identical dates and resolution sources
     0.8  — rules are clear but resolution sources differ slightly or dates
            are close but not identical
     0.6  — template fit is good but some ambiguity in resolution language
     0.4  — significant uncertainty; should probably not trade this cluster
     < 0.4 — set is_partition=false instead

=====================================================================
OUTPUT FORMAT — return exactly this JSON object
=====================================================================

{
  "underlying_event": "<one-sentence event description from Step 1>",
  "is_partition": <bool>,
  "verified_market_ids": [<ids of markets that passed all checks>],
  "excluded_market_ids": [
    {"market_id": "<id>", "reason": "<which step excluded this market and why>"}
  ],
  "is_mutually_exclusive": <bool>,
  "is_exhaustive": <bool>,
  "uncovered_outcomes": [<list of outcome descriptions not in the verified set>],
  "confidence": <float in [0, 1]>,
  "reasoning": "<2-3 sentences explaining the key decision factor>"
}

=====================================================================
GUARDRAILS — read before answering
=====================================================================

1. Err on the side of EXCLUDING markets from the verified set, not rejecting
   whole clusters. A partition of 3 that drops to 2 is better than a rejected
   cluster of 3 with one bad member.

2. MUTUAL EXCLUSIVITY is sacred. Only mark is_mutually_exclusive=true if you
   can articulate WHY market i YES forces market j NO under the written rules.
   "They seem related" is not sufficient.

3. EXHAUSTIVENESS is optional. When in doubt, set is_exhaustive=false. A
   partial partition is still useful for overpriced arbitrage, and a false
   exhaustive claim would corrupt underpriced arbitrage detection.

4. Different ELECTION YEARS, RESOLUTION DATES, or RESOLUTION SOURCES are
   automatic disqualifiers. Do not group "2024 primary" with "2028 primary"
   even if the candidate names overlap.

5. If a market's description mentions CONDITIONAL resolution ("conditional on
   X being nominated", "if Y qualifies"), exclude it. Partitions require
   unconditional resolution.

6. Output JSON ONLY. No prose before or after. No code fences.
"""


class ExcludedMarket(BaseModel):
    market_id: str
    reason: str = ""


class PartitionVerification(BaseModel):
    underlying_event: str = ""
    is_partition: bool = False
    verified_market_ids: list[str] = Field(default_factory=list)
    excluded_market_ids: list[ExcludedMarket] = Field(default_factory=list)
    is_mutually_exclusive: bool = False
    is_exhaustive: bool = False
    uncovered_outcomes: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""

    model_config = {"extra": "ignore"}


def _format_market_for_prompt(market: Market) -> str:
    description = (market.description or "")[:300]
    end_date_str = market.end_date.isoformat() if market.end_date else "unknown"
    resolution = getattr(market, "resolution_source", None) or "unspecified"
    outcome_list = ", ".join(
        f"{o.name} ({float(o.price):.2f})"
        for o in market.outcomes[:8]
    )
    return (
        f"- market_id: {market.market_id}\n"
        f"  question: {market.question}\n"
        f"  description: {description}\n"
        f"  end_date: {end_date_str}\n"
        f"  resolution_source: {resolution}\n"
        f"  outcomes: {outcome_list}"
    )


async def layer4_verify(cluster: list[Market]) -> PartitionVerification | None:
    """
    Send one candidate cluster to the LLM verifier.

    Uses Flash-Lite via config.llm_model_matcher. Returns None on hard failure
    (LLM unavailable, unparseable output, malformed schema).
    """
    if not cluster:
        return None

    markets_text = "\n\n".join(_format_market_for_prompt(m) for m in cluster)
    user_prompt = (
        "=====================================================================\n"
        "MARKETS TO VERIFY\n"
        "=====================================================================\n\n"
        f"{markets_text}"
    )

    raw = await asyncio.to_thread(
        call_llm_json,
        prompt=user_prompt,
        system_prompt=VERIFICATION_PROMPT,
        temperature=0.0,
        model=config.llm_model_matcher,
    )
    if raw is None:
        logger.debug("Layer 4: LLM returned None", cluster_size=len(cluster))
        return None

    try:
        verification = PartitionVerification.model_validate(raw)
    except Exception as e:
        logger.warning(
            "Layer 4: LLM output failed PartitionVerification schema",
            error=str(e),
            raw_keys=list(raw.keys()) if isinstance(raw, dict) else None,
        )
        return None

    return verification


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

async def detect_cross_market_partitions(
    markets: list[Market],
    store: "ConstraintStore | None",
) -> list["MarketCluster"]:
    """
    End-to-end cross-market partition detection.

    Takes a list of residual binary YES/NO markets that weren't already
    auto-clustered (NegRisk / native partition), runs Layers 1-4, and returns
    `MarketCluster` objects tagged with `constraint_source="cross_market_partition"`.
    Map_maker bypasses LogicArchitect for these clusters.

    Non-exhaustive verifications are logged and dropped in v1 — see plan scope.
    """
    # Lazy import to avoid circular dependency (discovery → partition_detector → discovery).
    from polyquant.agents.discovery import MarketCluster

    if not markets:
        logger.info(
            "Cross-market detection skipped: no residual polar markets "
            "(all consumed by NegRisk/native_partition)"
        )
        return []

    # Binary path only — cross-market partitions assume per-market YES/NO legs.
    polar_markets = [m for m in markets if has_binary_polarity(m)]
    if len(polar_markets) < config.partition_min_cluster_size:
        logger.info(
            "Cross-market detection skipped: fewer than min_cluster_size polar markets",
            polar=len(polar_markets),
            min_size=config.partition_min_cluster_size,
        )
        return []

    # Layer 1
    l1_clusters, matched = layer1_template_cluster(polar_markets)

    # Layer 2 on the residual
    residual = [m for m in polar_markets if m.market_id not in matched]
    l2_clusters = layer2_embedding_cluster(residual) if residual else []

    # Layer 3 triage (combined)
    candidates = l1_clusters + l2_clusters
    survivors = layer3_triage(candidates, store)

    # Layer 4 verify each survivor
    result_clusters: list[MarketCluster] = []
    conf_threshold = config.partition_confidence_threshold
    layer4_sent = 0
    layer4_accepted = 0
    layer4_rejected = 0

    for idx, candidate in enumerate(survivors):
        layer4_sent += 1
        verification = await layer4_verify(candidate)
        if verification is None:
            layer4_rejected += 1
            logger.debug("Layer 4: no verification", idx=idx, size=len(candidate))
            continue

        if not verification.is_partition or not verification.is_mutually_exclusive:
            layer4_rejected += 1
            logger.info(
                "Layer 4: rejected — not a mutually exclusive partition",
                event=verification.underlying_event[:80],
                reasoning=verification.reasoning[:120],
            )
            continue

        if verification.confidence < conf_threshold:
            layer4_rejected += 1
            logger.info(
                "Layer 4: rejected — confidence below threshold",
                confidence=verification.confidence,
                threshold=conf_threshold,
                event=verification.underlying_event[:80],
            )
            continue

        if not verification.is_exhaustive:
            layer4_rejected += 1
            logger.info(
                "Layer 4: dropped non-exhaustive partition (v1 scope)",
                event=verification.underlying_event[:80],
                uncovered=verification.uncovered_outcomes,
            )
            continue

        verified_ids = set(verification.verified_market_ids)
        verified_markets = [m for m in candidate if m.market_id in verified_ids]
        if len(verified_markets) < config.partition_min_cluster_size:
            layer4_rejected += 1
            logger.info(
                "Layer 4: rejected — verified set below min size",
                verified=len(verified_markets),
                min_size=config.partition_min_cluster_size,
            )
            continue

        layer4_accepted += 1

        # Cluster ID matches the mechanical constraint_id prefix used later by
        # build_partition_constraint, so Layer 3 dedup lines up on re-runs.
        token_ids = sorted(
            (get_yes_outcome(m).token_id or get_yes_outcome(m).outcome_id)  # type: ignore[union-attr]
            for m in verified_markets
        )
        token_hash = hashlib.sha256(",".join(token_ids).encode("utf-8")).hexdigest()[:16]
        cluster_id = f"cross_{token_hash}"
        topic = (
            f"[CROSS] {verification.underlying_event[:100]}"
            if verification.underlying_event
            else f"[CROSS] Cross-market partition ({len(verified_markets)} markets)"
        )

        cluster = MarketCluster(
            cluster_id=cluster_id,
            topic=topic,
            markets=verified_markets,
            potential_dependencies=[
                f"[CROSS_PARTITION] {verification.reasoning[:200]}"
            ],
            constraint_source="cross_market_partition",
            is_exhaustive=True,
        )
        result_clusters.append(cluster)

    logger.info(
        "Partition Layer 4 (LLM verify)",
        candidates_sent=layer4_sent,
        accepted=layer4_accepted,
        rejected=layer4_rejected,
    )
    logger.info(
        "Cross-market partition detection complete",
        input_markets=len(markets),
        verified_clusters=len(result_clusters),
    )
    return result_clusters
