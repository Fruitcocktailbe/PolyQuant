"""
Discovery Agent for PolyQuant 2.0 - Phase 1

The Discovery Agent is the first phase of the pipeline. Its job is to scan
Polymarket for all active markets and cluster them by topic to identify
potential logical dependencies.

RESPONSIBILITIES:
-----------------
1. Fetch all active markets from Polymarket
2. Cluster markets by topic (e.g., "2024 Election", "Fed Rate Decision")
3. Identify potential logical dependencies between markets in each cluster
4. Pass clusters to the Logic Architect for deeper analysis

WHY GEMINI 2.0 FLASH?
---------------------
Gemini 2.0 Flash was chosen for this phase because:
1. Extremely fast for clustering and categorization tasks
2. Cost-effective with a generous free tier
3. Excellent at understanding market questions and grouping by topic
4. Strong JSON mode for structured output

USAGE:
------
    discovery = DiscoveryAgent()
    
    async with discovery:
        clusters = await discovery.scan_markets()
        
        for cluster in clusters:
            print(f"Cluster: {cluster.topic}")
            for market in cluster.markets:
                print(f"  - {market.question}")
"""

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Any, TYPE_CHECKING

from polyquant.utils.llm_client import call_llm_json
from pydantic import BaseModel, Field, model_validator

from polyquant.data import Market, PolymarketClient
from polyquant.utils import config, get_logger
from polyquant.utils.cache import cache
from polyquant.utils.market_utils import get_yes_outcome, has_binary_polarity

if TYPE_CHECKING:
    from polyquant.data.constraint_store import ConstraintStore

logger = get_logger(__name__)

# Shared volume threshold used by NegRisk auto-clustering and native-partition
# detection. Outcomes below this are phantom/dead and would produce false sums.
_MIN_OUTCOME_VOLUME = 100


def _native_partition_cluster(market: Market) -> "MarketCluster | None":
    """
    Build a native-partition cluster for a single market whose outcome set is
    intrinsically exhaustive (the outcomes are the whole event space by
    definition). Works for both polar YES/NO binaries (YES + NO = 1 by CTF
    split/merge mechanics) and non-polar multi-outcome markets ("Trump wins /
    Trump loses", N-way sports, etc.).

    These clusters are tagged `constraint_source="native_partition"` so
    map_maker's mechanical bypass persists them without an LLM call. For
    binary markets this gives the solver an explicit `YES + NO = 1` identity
    that lets it exploit intra-market YES + NO < $1 arb (rare but real).

    Returns None if the market has fewer than 2 live outcomes, or if it looks
    like a degenerate single-outcome event, or if volume is below the phantom
    floor.
    """
    if not market.outcomes or len(market.outcomes) < 2:
        return None

    # Require at least 2 distinct tokens so build_partition_constraint can
    # emit a valid LogicalConstraint.
    live_tokens = [
        (o.token_id or o.outcome_id)
        for o in market.outcomes
        if (o.token_id or o.outcome_id)
    ]
    if len(set(live_tokens)) < 2:
        return None

    # Volume floor: intra-market arbitrage is rare, but phantom-priced outcomes
    # would produce false deviation signals. Skip markets with essentially no
    # trading activity.
    if not market.volume or market.volume < _MIN_OUTCOME_VOLUME:
        return None

    cluster_id = f"native_{market.market_id}"
    topic = f"[NATIVE] {market.question}"
    dependency_desc = (
        "[PARTITION] Native multi-outcome market. Outcomes are mutually "
        "exclusive and exhaustive; sum of prices must equal 1.0."
    )
    return MarketCluster(
        cluster_id=cluster_id,
        topic=topic,
        markets=[market],
        potential_dependencies=[dependency_desc],
        constraint_source="native_partition",
        is_exhaustive=True,
    )


def _hash_cluster_id(market_ids: list[str]) -> str:
    """Stable content-addressed cluster id from sorted member market_ids."""
    joined = ",".join(sorted(market_ids))
    return hashlib.sha1(joined.encode()).hexdigest()[:16]


class MarketCluster(BaseModel):
    """
    A cluster of related markets that may have logical dependencies.

    Attributes:
        cluster_id: Content-addressed hash of sorted member market_ids.
            Rerunning map over the same cluster overwrites the same manifest
            instead of leaking a new file per run.
        topic: Human-readable topic description
        markets: List of Market objects in this cluster
        potential_dependencies: Initial guesses at dependencies (for Logic Architect)
        created_at: When this cluster was created
        constraint_source: Where this cluster's constraints come from. Mechanical
            sources (negrisk, native_partition, cross_market_partition) bypass
            LogicArchitect + Validator LLM calls in map_maker. "llm_analysis" is
            the default and uses the full LogicArchitect path.
        is_exhaustive: Whether the outcome set covers the full event space.
            Non-exhaustive partitions are dropped in v1 (solver is buy-side-only).
    """
    cluster_id: str = ""
    topic: str
    markets: list[Market] = Field(default_factory=list)
    potential_dependencies: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    constraint_source: str = "llm_analysis"
    is_exhaustive: bool = True

    @model_validator(mode="after")
    def _assign_cluster_id(self) -> "MarketCluster":
        # Preserve explicitly set ids (e.g. "negrisk_{event_id}"). Only fill in
        # a hash when the caller left cluster_id blank AND we have markets to
        # hash. Empty-market fallback clusters keep an empty id, which signals
        # downstream code to skip persistence.
        if not self.cluster_id and self.markets:
            self.cluster_id = _hash_cluster_id([m.market_id for m in self.markets])
        return self


class DiscoveryAgent:
    """
    Phase 1: Market Discovery and Clustering
    
    The Discovery Agent uses Gemini 2.0 Flash to scan Polymarket
    and identify clusters of related markets for dependency analysis.
    
    Architecture:
    - Fetches markets via PolymarketClient
    - Uses Gemini for intelligent clustering
    - Maintains a cache of processed markets to avoid duplicate work
    
    Example:
        discovery = DiscoveryAgent()
        
        async with discovery:
            # Scan for new market clusters
            clusters = await discovery.scan_markets(limit=100)
            
            for cluster in clusters:
                print(f"Found cluster: {cluster.topic}")
                print(f"  Markets: {len(cluster.markets)}")
    """
    
    # System prompt for LLM clustering - optimized for arbitrage detection
    CLUSTERING_PROMPT = """You are a prediction market arbitrage detector. Your ONLY job is to find markets whose outcomes are LOGICALLY LINKED, creating potential arbitrage opportunities.

## CONSTRAINT TYPES TO FIND:
1. **MUTUALLY_EXCLUSIVE**: Only one can be YES. Sum of YES prices must be <= 1.
   Example: "Will Trump win 2024?" vs "Will Biden win 2024?" (same election)
2. **EXHAUSTIVE**: The outcomes cover ALL possibilities. Sum of YES prices must = 1.
   Example: All candidates in a single race listed as separate markets.
3. **IMPLICATION**: If A is YES, B MUST be YES. So P(B) >= P(A).
   Example: "Trump wins" -> "A Republican wins" (Trump IS a Republican)
4. **CONDITIONAL**: A's outcome significantly changes B's probability.
   Example: "Fed cuts rates" affects "S&P hits 6000"

## INPUT FORMAT:
Each market has: ID, Question, YES Price, Liquidity ($)
The YES Price is the current market probability (0.00 to 1.00).

## YOUR TASK:
1. Scan ALL markets for pairs or groups with logical dependencies.
2. Group them into clusters where arbitrage may exist.
3. For each cluster, specify the EXACT constraint type and which markets are involved.
4. Flag any obvious price violations (e.g., P(Trump) = 0.60 but P(Republican) = 0.55 violates IMPLICATION).

## OUTPUT (strict JSON, no comments):
{
  "clusters": [
    {
      "topic": "Short description (e.g. '2024 US Presidential Election')",
      "market_ids": ["id1", "id2"],
      "constraints": [
        {
          "type": "IMPLICATION",
          "market_ids": ["id1", "id2"],
          "description": "If id1 (Trump wins) is YES, id2 (Republican wins) MUST be YES"
        }
      ],
      "arbitrage_signal": "Describe any spotted price violation, or 'none' if prices look consistent",
      "confidence": 0.95
    }
  ]
}

## CRITICAL RULES:
- ONLY group markets with LOGICAL dependencies, NOT just topic similarity.
- Markets with NO logical link to ANY other market must be EXCLUDED entirely.
- Prefer clusters where you can spot an actual price violation.
- If a market could belong to multiple clusters, place it in the one with the strongest logical link.
- Return an empty clusters array if no logical links exist.
"""

    def __init__(self):
        """Initialize the Discovery Agent."""
        self._polymarket: PolymarketClient | None = None
        self._llm_available = False
        self._processed_markets: set[str] = set()
        # Per-run telemetry — map_maker reads these when assembling its
        # end-of-run report. Updated by scan_markets() on every call.
        self.last_tag_skip_counts: dict[str, int] = {}
        self.last_standalone_binary_count: int = 0
        self.last_limitless_discovery_stats: dict[str, int] = {}
        self.last_cross_event_cluster_count: int = 0

        logger.info("DiscoveryAgent initialized")
    
    async def __aenter__(self) -> "DiscoveryAgent":
        """Async context manager - initialize clients."""
        # Initialize Polymarket client
        self._polymarket = PolymarketClient()
        await self._polymarket.__aenter__()
        
        # Check if LLM is available (via OpenRouter)
        from polyquant.utils.llm_client import get_llm_client
        self._llm_available = get_llm_client() is not None
        if not self._llm_available:
            logger.warning("LLM not available - running in No-LLM mode")
        
        # Connect to Redis
        await cache.connect()
        
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager - cleanup."""
        if self._polymarket:
            await self._polymarket.__aexit__(*args)
    
    def _is_zombie_market(self, market: Market) -> bool:
        """
        Detect "zombie" markets that shouldn't be traded.

        A zombie market has ALL outcomes at extreme prices (< 0.02 or > 0.98),
        suggesting the market is already resolved or broken.

        ENHANCED (Week 5): Changed from ANY to ALL logic to avoid filtering
        markets with mixed prices that could be arbitrage opportunities.

        Args:
            market: Market to check

        Returns:
            True if zombie (should be filtered), False if valid
        """
        if not market.outcomes:
            return True  # No outcomes = invalid market

        # Count extreme outcomes using config thresholds
        extreme_count = sum(
            1 for o in market.outcomes
            if o.price < config.zombie_low_threshold or o.price > config.zombie_high_threshold
        )

        # Only filter if ALL outcomes are extreme
        # This indicates the market is resolved or broken
        if extreme_count == len(market.outcomes):
            logger.debug(
                "Zombie market detected (all outcomes extreme)",
                market_id=market.market_id,
                extreme_count=extreme_count,
                total_outcomes=len(market.outcomes)
            )
            return True

        # If some (but not all) outcomes are extreme, this might be
        # a partially resolved market or mispricing opportunity
        if extreme_count > 0:
            logger.info(
                "Market has extreme outcomes but keeping for arbitrage analysis",
                market_id=market.market_id,
                extreme_count=extreme_count,
                total_outcomes=len(market.outcomes),
                reason="Mixed prices may indicate arbitrage opportunity"
            )

        return False

    
    async def scan_markets(
        self,
        limit: int = 100,
        min_liquidity: float | None = None,  # Defaults to config
        skip_processed: bool = True,
        start_offset: int = 0,
        constraint_store: "ConstraintStore | None" = None,
    ) -> list[MarketCluster]:
        """
        Scan Polymarket for markets and cluster them to find arbitrage.
        
        Uses a 3-phase pipeline:
          1. Fetch events from /events API (sorted by liquidity, fast)
          2. Auto-cluster NegRisk groups (no LLM needed)
          3. Send multi-market events to LLM for constraint analysis
        
        Args:
            limit: Max events to fetch (0 for all above threshold)
            min_liquidity: Stop when event liquidity drops below this
            skip_processed: Skip events we've already analyzed
            start_offset: Offset for pagination (unused in event mode)
            
        Returns:
            List of MarketCluster objects
        """
        if not self._polymarket:
            raise RuntimeError("DiscoveryAgent not initialized. Use 'async with discovery:'")
        
        min_liquidity_val = min_liquidity if min_liquidity is not None else config.min_liquidity

        logger.info(
            "Scanning markets (event-based pipeline)",
            min_liquidity=min_liquidity_val,
            skip_processed=skip_processed,
        )
        
        # ── Phase 1: Fetch events from API (sorted by liquidity desc) ──
        events = await self._polymarket.get_active_events(
            min_liquidity=min_liquidity_val,
            max_events=limit,
        )
        
        if not events:
            logger.info("No events found matching criteria")
            return []
        
        logger.info(
            "Phase 1 complete: events fetched",
            events=len(events),
            total_markets=sum(len(e["markets"]) for e in events),
        )
        
        # ── Phase 2: Pre-filter and auto-cluster ──
        total_markets_fetched = sum(len(e["markets"]) for e in events)
        logger.info(
            "Entering Phase 2 pre-filtering",
            total_markets_fetched=total_markets_fetched,
            total_events=len(events),
        )
        clusters: list[MarketCluster] = []
        events_for_llm: list[dict] = []
        auto_clustered_market_ids: set[str] = set()
        skipped_by_memory = 0
        skipped_by_cache = 0

        # Gap 4: case-insensitive excluded-tag filter. Build once per run so
        # the per-event check is an O(1) set intersection. `skipped_by_tag`
        # tallies per-tag drops for the report.
        excluded_tags_lc: set[str] = {t.strip().lower() for t in (config.excluded_tags or []) if t}
        skipped_by_tag: dict[str, int] = {}
        # Market-id → event tag labels (original casing). Threaded into
        # partition_detector's Layer 1 so tag-sharing markets group tighter.
        market_tag_map: dict[str, list[str]] = {}

        for event in events:
            event_id = event["event_id"]
            event_title = event["title"]
            markets: list[Market] = event["markets"]
            neg_risk_id = event.get("neg_risk_market_id")
            event_tags: list[str] = event.get("tags", []) or []

            # Skip already-processed events
            if skip_processed and event_id in self._processed_markets:
                skipped_by_memory += 1
                continue

            # Skip if Redis says processed
            if skip_processed and await cache.is_market_processed(f"event_{event_id}"):
                self._processed_markets.add(event_id)
                skipped_by_cache += 1
                continue

            # Gap 4: skip events whose tag set intersects excluded_tags.
            if excluded_tags_lc:
                event_tags_lc = {t.strip().lower() for t in event_tags if t}
                hit = event_tags_lc & excluded_tags_lc
                if hit:
                    for tag in hit:
                        skipped_by_tag[tag] = skipped_by_tag.get(tag, 0) + 1
                    continue

            # Populate the tag map for every market we're about to process so
            # downstream partition_detector can use tag priors.
            for _m in markets:
                market_tag_map[_m.market_id] = list(event_tags)

            # Drop already-resolved markets before any other filtering. The
            # /events endpoint occasionally returns markets with closed=True
            # that haven't been purged from the active set yet. Keeping them
            # wastes LLM budget on constraints the Navigator would refuse to
            # trade anyway, and the settled prices can distort deviation
            # signals (price "sums" far away from 1.0 that aren't real arbs).
            markets = [m for m in markets if not getattr(m, "resolved", False)]

            # Filter zombie markets (extreme prices)
            valid_markets = [m for m in markets if not self._is_zombie_market(m)]

            # Split by polarity instead of dropping. Markets with YES/NO
            # outcomes flow through the standard path; markets without YES/NO
            # polarity (e.g. "Trump wins / Trump loses" or 3-way sports) go
            # through _native_partition_cluster, which emits a mechanical
            # partition constraint without touching LogicArchitect.
            polar_markets: list[Market] = []
            ambiguous_markets: list[Market] = []
            for m in valid_markets:
                if has_binary_polarity(m):
                    polar_markets.append(m)
                else:
                    ambiguous_markets.append(m)

            # Phase 2b: native partition for ambiguous multi-outcome markets
            for m in ambiguous_markets:
                native_cluster = _native_partition_cluster(m)
                if native_cluster:
                    clusters.append(native_cluster)
                    auto_clustered_market_ids.add(m.market_id)

            if not polar_markets:
                continue
            valid_markets = polar_markets
            
            # Auto-cluster: NegRisk events with price deviation detection
            # ENHANCED (Week 5): Explicit deviation detection for arbitrage opportunities
            # ENHANCED (Week 6): Filter zero-volume outcomes to prevent phantom signals
            if neg_risk_id and len(valid_markets) > 1:
                # Filter out dead outcomes before emitting the partition constraint.
                # Low-volume outcomes often have phantom quotes that would pollute
                # the coefficient set; require each sub-market to have real volume
                # AND a resolvable YES leg (token the solver can reference).
                # NOTE: this is structural cleanup, not price-based arbitrage
                # detection. The Navigator handles all price/arb logic at trade time.
                MIN_OUTCOME_VOLUME = 100  # $100 minimum volume to be considered "real"
                priced_markets = [
                    m for m in valid_markets
                    if (
                        m.outcomes
                        and m.volume
                        and m.volume > MIN_OUTCOME_VOLUME
                        and get_yes_outcome(m) is not None
                    )
                ]
                filtered_count = len(valid_markets) - len(priced_markets)

                if not priced_markets:
                    # All outcomes are dead — skip this event entirely
                    logger.debug(
                        "NegRisk event skipped: all outcomes below volume threshold",
                        event_title=event_title,
                        total_outcomes=len(valid_markets),
                        threshold=MIN_OUTCOME_VOLUME,
                    )
                    continue

                if filtered_count > 0:
                    logger.debug(
                        "NegRisk: filtered low-volume outcomes",
                        event_title=event_title,
                        kept=len(priced_markets),
                        filtered=filtered_count,
                        threshold=MIN_OUTCOME_VOLUME,
                    )

                clusters.append(
                    MarketCluster(
                        cluster_id=f"negrisk_{event_id}",
                        topic=f"[NEGRISK] {event_title} ({len(priced_markets)} outcomes)",
                        markets=priced_markets,
                        potential_dependencies=[
                            "[PARTITION] NegRisk event: outcomes are mutually "
                            "exclusive and exhaustive; YES prices must sum to 1.0."
                        ],
                        constraint_source="negrisk",
                        is_exhaustive=True,
                    )
                )
                for _m in priced_markets:
                    auto_clustered_market_ids.add(_m.market_id)

                logger.info(
                    "Auto-clustered NegRisk event",
                    event_title=event_title,
                    outcomes=len(priced_markets),
                )
            elif len(valid_markets) > 1:
                # Multi-market event → needs LLM to determine constraint types
                events_for_llm.append({
                    "event_id": event_id,
                    "title": event_title,
                    "markets": valid_markets,
                    "liquidity": event["liquidity"],
                    "tags": event.get("tags", []),
                })
            # Solo-market events → no constraints possible within event,
            # but could have cross-event links, so include in LLM batch
            else:
                events_for_llm.append({
                    "event_id": event_id,
                    "title": event_title,
                    "markets": valid_markets,
                    "liquidity": event["liquidity"],
                    "tags": event.get("tags", []),
                })
        
        auto_cluster_count = len(clusters)
        skipped_by_tag_total = sum(skipped_by_tag.values())
        events_processed = (
            len(events) - skipped_by_memory - skipped_by_cache - skipped_by_tag_total
        )
        logger.info(
            "Phase 2 cache-skip summary",
            events_fetched=len(events),
            skipped_by_memory=skipped_by_memory,
            skipped_by_cache=skipped_by_cache,
            skipped_by_tag=skipped_by_tag_total,
            skipped_by_tag_breakdown=dict(skipped_by_tag) if skipped_by_tag else None,
            events_processed=events_processed,
            hint=(
                "Use --force to bypass the 24h event-processed cache"
                if skipped_by_cache > 0
                else None
            ),
        )

        # Loud alert: if we silently dropped most of the event universe to the
        # cache, surface it so the user knows why a run looks sparse.
        if len(events) > 0 and skipped_by_cache / len(events) > 0.5:
            logger.warning(
                "More than half of fetched events were silently skipped by the "
                "Redis event-processed cache (24h TTL). Re-run with --force to "
                "bypass the cache and re-process events.",
                skipped_by_cache=skipped_by_cache,
                events_fetched=len(events),
                skip_ratio=f"{skipped_by_cache / len(events):.1%}",
            )

        logger.info(
            "Phase 2 complete: pre-filtering done",
            auto_clusters=auto_cluster_count,
            events_for_llm=len(events_for_llm),
        )

        # ── Phase 2c: cross-market partition detection on residual polar markets ──
        # Residual = polar markets from events_for_llm (not already auto-clustered
        # via NegRisk / native partition paths). Layers 1-4 find groups of separate
        # binary YES/NO markets that implicitly partition one real-world event.
        residual_polar: list[Market] = []
        for ev in events_for_llm:
            for m in ev["markets"]:
                if m.market_id not in auto_clustered_market_ids:
                    residual_polar.append(m)

        cross_clustered_ids: set[str] = set()
        if residual_polar:
            from polyquant.agents.partition_detector import detect_cross_market_partitions
            cross_clusters = await detect_cross_market_partitions(
                residual_polar, constraint_store, tag_map=market_tag_map
            )
            for c in cross_clusters:
                clusters.append(c)
                for m in c.markets:
                    cross_clustered_ids.add(m.market_id)
            if cross_clusters:
                logger.info(
                    "Cross-market partition detection added clusters",
                    new_clusters=len(cross_clusters),
                    markets_consumed=len(cross_clustered_ids),
                )

        # ── Phase 2d: standalone binary YES/NO partition emission (Gap 3) ──
        # Every binary YES/NO market not absorbed by NegRisk (2a) or cross-market
        # partition detection (2c) gets its own trivial `native_partition` cluster
        # encoding the CTF-mechanical identity YES + NO = 1. Gives the solver an
        # explicit structural constraint for intra-market YES + NO < $1 arb
        # (rare but legitimate) and costs zero LLM quota since it routes through
        # map_maker's mechanical bypass.
        standalone_binary_count = 0
        for ev in events_for_llm:
            for m in ev["markets"]:
                if m.market_id in auto_clustered_market_ids:
                    continue
                if m.market_id in cross_clustered_ids:
                    continue
                if not has_binary_polarity(m):
                    continue
                binary_cluster = _native_partition_cluster(m)
                if binary_cluster is None:
                    continue
                clusters.append(binary_cluster)
                auto_clustered_market_ids.add(m.market_id)
                standalone_binary_count += 1
        if standalone_binary_count:
            logger.info(
                "Phase 2d: emitted standalone binary native_partition clusters",
                clusters=standalone_binary_count,
            )
        self.last_standalone_binary_count = standalone_binary_count
        self.last_tag_skip_counts = dict(skipped_by_tag)

        # ── Phase 2d.25: conditional-parent clustering ──
        # Polymarket's conditional markets have a parent_market_id whose
        # outcome must resolve YES for the child to pay out. That imposes
        # P(child) <= P(parent) as a hard structural constraint. Cluster
        # by parent_id and emit SUBSET constraints deterministically — the
        # LLM path was never primed for this shape and silently missed it.
        from polyquant.agents.ladder_detector import detect_conditional_subsets

        conditional_candidates = [
            m
            for ev in events_for_llm
            for m in ev["markets"]
            if m.market_id not in auto_clustered_market_ids
            and m.market_id not in cross_clustered_ids
        ]
        conditional_clusters = detect_conditional_subsets(conditional_candidates)
        if conditional_clusters:
            clusters.extend(conditional_clusters)
            for cc in conditional_clusters:
                for m in cc.markets:
                    auto_clustered_market_ids.add(m.market_id)
            logger.info(
                "Phase 2d.25: emitted conditional-parent clusters",
                clusters=len(conditional_clusters),
            )

        # ── Phase 2d.5: monotonic ladder detection ──
        # Classic Polymarket structure: several binary YES/NO markets sharing a
        # common question template but differing in a numeric threshold (e.g.
        # "BTC >= $100k?" / ">= $120k?" / ">= $150k?"). Higher thresholds must
        # imply lower ones, producing chained SUBSET inequalities the solver
        # can exploit. Deterministic detection here replaces reliance on the
        # LLM to spot ladders — otherwise a noisy LLM run silently drops the
        # opportunity. Ladder clusters consume markets the same way NegRisk
        # clusters do so they don't double-count in later phases.
        from polyquant.agents.ladder_detector import detect_monotonic_ladders

        ladder_candidates = [
            m
            for ev in events_for_llm
            for m in ev["markets"]
            if m.market_id not in auto_clustered_market_ids
            and m.market_id not in cross_clustered_ids
        ]
        ladder_clusters = detect_monotonic_ladders(ladder_candidates)
        if ladder_clusters:
            clusters.extend(ladder_clusters)
            for lc in ladder_clusters:
                for m in lc.markets:
                    auto_clustered_market_ids.add(m.market_id)
            logger.info(
                "Phase 2d.5: emitted monotonic ladder clusters",
                clusters=len(ladder_clusters),
                rung_total=sum(len(c.markets) for c in ladder_clusters),
            )

        # ── Phase 2e: cross-event logical clustering (Gap 1) ──
        # Two strategies:
        # * representative_top_k (legacy): pull top-K markets per mechanical
        #   cluster and semantic-cluster only those — cheap but drops ~70% of
        #   candidate pairs on the floor when top-3² is the visible universe.
        # * all_pairs_ann (default): embed EVERY polar market and find
        #   neighbours across cluster boundaries. Much better coverage for the
        #   cost of one extra embedding pass per run.
        cross_strategy = getattr(
            config, "cross_event_pool_strategy", "representative_top_k"
        )
        if cross_strategy == "all_pairs_ann":
            all_polar_for_ann: list[Market] = []
            polar_seen: set[str] = set()
            for c in clusters:
                for m in c.markets:
                    if not m.market_id or m.market_id in polar_seen:
                        continue
                    if not has_binary_polarity(m):
                        continue
                    polar_seen.add(m.market_id)
                    all_polar_for_ann.append(m)
            cross_event_clusters = self._build_cross_event_clusters_all_pairs(
                all_polar_markets=all_polar_for_ann
            )
        else:
            cross_event_clusters = self._build_cross_event_clusters(clusters)
        if cross_event_clusters:
            clusters.extend(cross_event_clusters)
            logger.info(
                "Phase 2e: emitted cross_event_logical clusters",
                new_clusters=len(cross_event_clusters),
            )
        self.last_cross_event_cluster_count = len(cross_event_clusters)

        # ── Phase 3: LLM analysis for what's left after all mechanical paths ──
        if events_for_llm:
            # Collect markets from events that weren't consumed by Phase 2a/2b/2c
            all_llm_markets: list[Market] = []
            for ev in events_for_llm:
                for m in ev["markets"]:
                    if (
                        m.market_id not in auto_clustered_market_ids
                        and m.market_id not in cross_clustered_ids
                    ):
                        all_llm_markets.append(m)

            if all_llm_markets:
                logger.info(
                    f"Sending {len(all_llm_markets)} markets from "
                    f"{len(events_for_llm)} events to LLM for clustering..."
                )
                llm_clusters = await self._cluster_markets(all_llm_markets)
                clusters.extend(llm_clusters)
        
        # Mark all events as processed
        for event in events:
            event_id = event["event_id"]
            self._processed_markets.add(event_id)
            await cache.mark_market_processed(f"event_{event_id}")
        
        logger.info(
            f"Market scan complete: {len(clusters)} clusters (auto={auto_cluster_count}, llm={len(clusters) - auto_cluster_count})",
            auto_clusters=auto_cluster_count,
            llm_clusters=len(clusters) - auto_cluster_count,
            total_clusters=len(clusters),
            total_markets=sum(len(c.markets) for c in clusters),
        )

        return clusters

    def _select_mechanical_cluster_representatives(
        self,
        clusters: list[MarketCluster],
        k: int = 3,
    ) -> list[tuple[MarketCluster, Market]]:
        """Gap 1: top-K representative markets per mechanical cluster.

        "Top-K" = highest-liquidity Polymarket markets in each mechanical
        cluster (negrisk / native_partition / cross_market_partition). Returns
        a list of (source_cluster, market) tuples so the caller can reason
        about which event a representative came from.

        Markets appearing in multiple mechanical clusters are de-duplicated by
        market_id — the first cluster to pick them wins. Clusters with <2
        markets (degenerate) are skipped entirely.
        """
        MECHANICAL_SOURCES = {
            "negrisk", "native_partition", "cross_market_partition"
        }
        seen: set[str] = set()
        reps: list[tuple[MarketCluster, Market]] = []
        for cluster in clusters:
            if cluster.constraint_source not in MECHANICAL_SOURCES:
                continue
            if len(cluster.markets) < 2:
                continue
            sorted_markets = sorted(
                cluster.markets,
                key=lambda m: (m.liquidity or 0.0),
                reverse=True,
            )
            picked = 0
            for m in sorted_markets:
                if m.market_id in seen:
                    continue
                # Representative must have a resolvable YES leg so the
                # cross-event constraint can coefficient it directly.
                if not has_binary_polarity(m):
                    continue
                seen.add(m.market_id)
                reps.append((cluster, m))
                picked += 1
                if picked >= k:
                    break
        return reps

    def _build_cross_event_clusters_all_pairs(
        self,
        *,
        all_polar_markets: list[Market],
    ) -> list[MarketCluster]:
        """Gap 1 (§2.4 upgrade): embed every polar market in scope and cluster
        by semantic similarity across cluster boundaries.

        Old path (representative_top_k) evaluated only K markets per mechanical
        cluster — with K=3 on 50 clusters that's 150/500 markets considered,
        dropping ~70% of candidate cross-event pairs on the floor. The all-
        pairs approach embeds the full polar universe once (MiniLM, cheap),
        builds a cosine-sim graph, and forms connected components above the
        similarity threshold. Components whose members span ≥2 source events
        become cross_event_logical clusters.

        Returns [] when sentence-transformers is unavailable or fewer than 2
        markets are eligible, matching the fallback behaviour of the legacy
        top-K path.
        """
        if len(all_polar_markets) < 2:
            return []

        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
        except Exception as e:
            logger.info(
                "All-pairs cross-event clustering skipped: sentence-transformers missing",
                error=str(e),
            )
            return []

        questions = [m.question for m in all_polar_markets]
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(questions, convert_to_tensor=True)
            sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        except Exception as e:
            logger.warning("All-pairs cross-event embedding failed", error=str(e))
            return []

        threshold = config.cross_event_similarity_threshold
        top_k = config.cross_event_all_pairs_top_k
        n = len(all_polar_markets)

        # Per-node top-K neighbours above threshold. Symmetric graph — we
        # take the union of (i→j) and (j→i) edges.
        adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
        import numpy as _np
        for i in range(n):
            row = sim_matrix[i].copy()
            row[i] = -1.0  # drop self-edge
            order = _np.argsort(row)[::-1][:top_k]
            for j in order:
                j_int = int(j)
                if row[j_int] < threshold:
                    break
                adjacency[i].add(j_int)
                adjacency[j_int].add(i)

        # Union-find to extract connected components.
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        for i, neighbours in adjacency.items():
            for j in neighbours:
                _union(i, j)

        components: dict[int, list[int]] = {}
        for i in range(n):
            components.setdefault(_find(i), []).append(i)

        max_size = config.cross_event_max_cluster_size
        cross_clusters: list[MarketCluster] = []
        for root, idxs in components.items():
            if len(idxs) < 2:
                continue
            # Keep the top-`max_size` markets in the component by liquidity
            # so the LLM prompt stays bounded. Use market's liquidity field.
            component_markets = [all_polar_markets[i] for i in idxs]
            component_markets.sort(
                key=lambda m: m.liquidity or 0.0, reverse=True
            )
            component_markets = component_markets[:max_size]
            source_event_ids = {
                (getattr(m, "event_slug", "") or m.market_id)
                for m in component_markets
            }
            if len(source_event_ids) < 2:
                # All in one event — not cross-event, skip.
                continue
            content = ",".join(sorted(m.market_id for m in component_markets))
            cluster_hash = hashlib.sha1(content.encode()).hexdigest()[:16]
            topic = (
                f"[CROSS-EVENT-ANN] {component_markets[0].question[:60]} "
                f"… (+{len(component_markets) - 1} more)"
            )
            cross_clusters.append(
                MarketCluster(
                    cluster_id=f"cross_event_ann_{cluster_hash}",
                    topic=topic,
                    markets=component_markets,
                    potential_dependencies=[
                        "[CROSS-EVENT-ANN] Markets semantically related across "
                        "mechanical-cluster boundaries. LLM should look for "
                        "SUBSET / MUTUALLY_EXCLUSIVE / COALITION relationships."
                    ],
                    constraint_source="cross_event_logical",
                    is_exhaustive=False,
                )
            )
        return cross_clusters

    def _build_cross_event_clusters(
        self,
        clusters: list[MarketCluster],
    ) -> list[MarketCluster]:
        """Gap 1: semantic pre-cluster representatives drawn from *different*
        mechanical clusters, emit one `cross_event_logical` cluster per
        semantic group of size ≥2 whose members span ≥2 source events.

        Returns [] if:
        - sentence-transformers is unavailable (keeps map_maker resilient on
          low-resource hosts)
        - fewer than 2 representatives survive selection
        - no semantic group spans >1 source event

        The emitted cluster routes through LogicArchitect/Validator in map_maker
        (NOT the mechanical bypass), so the LLM does the reasoning; the
        `cluster_type` stamp makes it identifiable downstream.
        """
        k = config.cross_event_representatives_per_cluster
        reps = self._select_mechanical_cluster_representatives(clusters, k=k)
        if len(reps) < 2:
            return []

        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
        except Exception as e:
            logger.info(
                "Cross-event pre-clustering skipped: sentence-transformers missing",
                error=str(e),
            )
            return []

        questions = [m.question for _, m in reps]
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(questions, convert_to_tensor=True)
            sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        except Exception as e:
            logger.warning("Cross-event embedding failed", error=str(e))
            return []

        threshold = config.cross_event_similarity_threshold
        visited: set[int] = set()
        groups: list[list[int]] = []
        for i in range(len(reps)):
            if i in visited:
                continue
            group = [i]
            visited.add(i)
            for j in range(i + 1, len(reps)):
                if j in visited:
                    continue
                if sim_matrix[i][j] >= threshold:
                    group.append(j)
                    visited.add(j)
            if len(group) >= 2:
                groups.append(group)

        cross_clusters: list[MarketCluster] = []
        for group in groups:
            source_cluster_ids = {reps[i][0].cluster_id for i in group}
            if len(source_cluster_ids) < 2:
                # All reps from the same mechanical cluster — not cross-event.
                continue
            member_markets = [reps[i][1] for i in group]
            topic = f"[CROSS-EVENT] {member_markets[0].question[:60]} … (+{len(member_markets) - 1} more)"
            content = ",".join(sorted(m.market_id for m in member_markets))
            cluster_hash = hashlib.sha1(content.encode()).hexdigest()[:16]
            cross_clusters.append(
                MarketCluster(
                    cluster_id=f"cross_event_{cluster_hash}",
                    topic=topic,
                    markets=member_markets,
                    potential_dependencies=[
                        "[CROSS-EVENT] Representatives drawn from multiple mechanical "
                        "clusters. LLM should look for SUBSET / MUTUALLY_EXCLUSIVE "
                        "relationships BETWEEN events; partition constraints are "
                        "already emitted upstream and must NOT be re-emitted."
                    ],
                    constraint_source="cross_event_logical",
                    is_exhaustive=False,
                )
            )
        return cross_clusters

    async def discover_standalone_limitless_clusters(
        self,
        mapped_limitless_slugs: set[str],
    ) -> list[MarketCluster]:
        """
        Gap 5b: emit `native_partition` clusters for standalone Limitless binary
        markets not covered by cross-exchange matching.

        `mapped_limitless_slugs` is the set of Limitless slugs that already
        appear as aliases on Polymarket-anchored clusters (per the
        ExchangeMatcher). Everything else in the active Limitless universe
        above the liquidity floor gets its own YES+NO=1 cluster so the
        solver can exploit pure-Limitless intra-market arb.

        Returns [] if Limitless fetch fails — this path must never take
        down the whole map build.
        """
        from polyquant.data.limitless_client import (
            LimitlessClient,
            limitless_market_to_market,
        )

        floor = config.min_liquidity_limitless_discovery
        markets_raw: list[dict] = []
        try:
            async with LimitlessClient() as l_client:
                markets_raw = await l_client.get_markets(limit=20)
        except Exception as e:
            logger.warning(
                "Limitless-first discovery: fetch failed, skipping",
                error=str(e),
            )
            return []

        # Apply the floor here (Limitless API doesn't support server-side
        # liquidity filtering) and drop anything already aliased to a
        # Polymarket cluster via the ExchangeMatcher.
        standalone_clusters: list[MarketCluster] = []
        emitted_slugs: set[str] = set()
        skipped_already_mapped = 0
        skipped_below_floor = 0
        skipped_parse_fail = 0

        for raw in markets_raw:
            slug = raw.get("slug")
            if not isinstance(slug, str) or not slug:
                continue
            if slug in mapped_limitless_slugs:
                skipped_already_mapped += 1
                continue
            try:
                vol = float(raw.get("volumeFormatted") or raw.get("volume") or 0)
                liq = float(raw.get("liquidityFormatted") or raw.get("liquidity") or 0)
            except (TypeError, ValueError):
                vol, liq = 0.0, 0.0
            if vol > 1_000_000:
                vol /= 1_000_000
            if liq > 1_000_000:
                liq /= 1_000_000
            if max(vol, liq) < floor:
                skipped_below_floor += 1
                continue

            market = limitless_market_to_market(raw)
            if market is None:
                skipped_parse_fail += 1
                continue

            cluster = _native_partition_cluster(market)
            if cluster is None:
                continue
            # Distinguish from Polymarket native_partition clusters so the
            # report (and future Navigator consumers) can filter by exchange.
            cluster = cluster.model_copy(update={
                "cluster_id": f"limitless_native_{market.market_id}",
                "topic": f"[LIMITLESS-NATIVE] {market.question}",
            })
            standalone_clusters.append(cluster)
            emitted_slugs.add(slug)

        logger.info(
            "Limitless-first discovery complete",
            markets_fetched=len(markets_raw),
            standalone_clusters=len(standalone_clusters),
            skipped_already_mapped=skipped_already_mapped,
            skipped_below_floor=skipped_below_floor,
            skipped_parse_fail=skipped_parse_fail,
            floor=floor,
        )
        # Expose on the agent so map_maker can pull counts for the report.
        self.last_limitless_discovery_stats = {
            "markets_fetched": len(markets_raw),
            "standalone_clusters": len(standalone_clusters),
            "skipped_already_mapped": skipped_already_mapped,
            "skipped_below_floor": skipped_below_floor,
            "skipped_parse_fail": skipped_parse_fail,
            "floor": floor,
        }
        return standalone_clusters

    async def _cluster_markets(self, markets: list[Market]) -> list[MarketCluster]:
        """
        Use LLM (via OpenRouter) to cluster markets by topic.
        """
        # Re-check availability using the hardened client (handles missing/short keys)
        from polyquant.utils.llm_client import get_llm_client
        if get_llm_client() is None:
            logger.info("LLM keys not found - using basic clustering only")
            return [
                MarketCluster(
                    topic="Uncategorized (No AI)",
                    markets=markets,
                    potential_dependencies=[],
                )
            ]

        # Prevent massive prompts that break the LLM API / Context window
        # Reduced from 150 to 50 to prevent free-tier models from returning unterminated JSON strings
        MAX_BATCH_SIZE = 50
        if len(markets) > MAX_BATCH_SIZE:
            logger.info(f"Chunking {len(markets)} markets into batches (max {MAX_BATCH_SIZE})...")
            
            batches = []
            if config.enable_semantic_matching:
                try:
                    from sentence_transformers import SentenceTransformer, util
                    import gc
                    import torch
                    
                    logger.info("Loading semantic model for pre-clustering targets...")
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Get embeddings
                    questions = [m.question for m in markets]
                    embeddings = model.encode(questions, convert_to_tensor=True)
                    
                    # Compute sim matrix
                    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
                    
                    # Free model immediately to save RAM on 2GB instances
                    del model
                    del embeddings
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Greedy clustering
                    visited = set()
                    clusters = []
                    for i in range(len(markets)):
                        if i in visited:
                            continue
                        
                        # Start a new cluster with market i
                        cluster = [markets[i]]
                        visited.add(i)
                        
                        # Find all related markets
                        for j in range(i + 1, len(markets)):
                            if j not in visited and sim_matrix[i][j] >= 0.40:
                                cluster.append(markets[j])
                                visited.add(j)
                        
                        clusters.append(cluster)
                    
                    # Bin packing clusters into batches of MAX_BATCH_SIZE
                    current_batch = []
                    for cluster in clusters:
                        if len(cluster) > MAX_BATCH_SIZE:
                            # Flush current batch
                            if current_batch:
                                batches.append(current_batch)
                                current_batch = []
                            # Slice huge cluster
                            for i in range(0, len(cluster), MAX_BATCH_SIZE):
                                batches.append(cluster[i:i + MAX_BATCH_SIZE])
                        else:
                            if len(current_batch) + len(cluster) > MAX_BATCH_SIZE:
                                batches.append(current_batch)
                                current_batch = []
                            current_batch.extend(cluster)
                    
                    if current_batch:
                        batches.append(current_batch)
                        
                    logger.info(f"Semantically clustered into {len(batches)} batches.")

                except Exception as e:
                    logger.warning(f"Semantic pre-clustering failed ({e}), falling back to arbitrary slicing.")
                    batches = [markets[i:i + MAX_BATCH_SIZE] for i in range(0, len(markets), MAX_BATCH_SIZE)]
            else:
                batches = [markets[i:i + MAX_BATCH_SIZE] for i in range(0, len(markets), MAX_BATCH_SIZE)]
            
            all_clusters = []
            
            # Limit concurrency to exactly 2 with a delay to respect OpenRouter's 20 RPM free tier
            sem = asyncio.Semaphore(2)
            
            async def process_batch(b: list[Market], batch_idx: int) -> list[MarketCluster]:
                async with sem:
                    try:
                        res = await self._cluster_markets(b)
                        await asyncio.sleep(6.0)  # Stay under 20 RPM
                        return res
                    except Exception as e:
                        logger.warning(f"Batch {batch_idx} LLM failed: {e}")
                        # Hold the semaphore slot for the full window even on failure
                        # so a 429 or timeout doesn't immediately release capacity.
                        await asyncio.sleep(6.0)
                        return [MarketCluster(topic="Uncategorized (Batch Failed)", markets=b)]

            tasks = [process_batch(b, i) for i, b in enumerate(batches)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            retry_batches: list[list[Market]] = []
            for i, r in enumerate(results):
                if isinstance(r, list):
                    all_clusters.extend(r)
                else:
                    # BaseException (e.g. CancelledError) — not caught by process_batch.
                    # Queue for sequential retry at the end of this run.
                    logger.warning(
                        f"Batch {i} did not complete ({type(r).__name__}), queued for retry"
                    )
                    retry_batches.append(batches[i])

            if retry_batches:
                logger.info(f"Retrying {len(retry_batches)} interrupted batch(es) sequentially...")
                for idx, b in enumerate(retry_batches):
                    if idx > 0:
                        await asyncio.sleep(6.0)  # Rate-limit between retries
                    try:
                        retry_result = await self._cluster_markets(b)
                        all_clusters.extend(retry_result)
                        logger.info(f"Retry {idx + 1}/{len(retry_batches)} succeeded.")
                    except Exception as e:
                        logger.warning(f"Retry {idx + 1}/{len(retry_batches)} failed: {e}. Using fallback cluster.")
                        all_clusters.append(MarketCluster(topic="Uncategorized (Retry Failed)", markets=b))

            return all_clusters

        # Format markets for the prompt - include prices and liquidity
        market_lines = []
        for m in markets:
            # Resolve YES by name, not by index — polarity is not guaranteed.
            yes_outcome = get_yes_outcome(m)
            yes_price = yes_outcome.price if yes_outcome else 0.0
            market_lines.append(
                f"ID: {m.market_id} | Question: {m.question} | YES Price: {yes_price:.2f} | Liquidity: ${m.liquidity:,.0f}"
            )
        market_descriptions = "\n".join(market_lines)
        
        # Create market lookup for quick access
        market_lookup = {m.market_id: m for m in markets}

        logger.info(f"Calling LLM for market clustering ({len(markets)} targets)", market_count=len(markets))

        try:
            # call_llm_json owns retries (3x per model across the fallback chain)
            # and the OpenAI client owns the per-request timeout (45s). The outer
            # batching layer in cluster_markets() handles batch-level retries.
            result = await asyncio.to_thread(
                call_llm_json,
                prompt=f"Analyze these {len(markets)} markets for arbitrage opportunities:\n\n{market_descriptions}",
                system_prompt=self.CLUSTERING_PROMPT,
                temperature=0.2,
                model=config.llm_model_discovery,
            )

            if not result:
                raise ValueError("LLM clustering returned no result")

        except Exception as e:
            logger.warning(f"LLM clustering failed, falling back to manual groups: {e}")
            return [
                MarketCluster(
                    topic="Uncategorized (LLM Failed)",
                    markets=markets,
                    potential_dependencies=[],
                )
            ]
        
        # Convert response to MarketCluster objects
        clusters = []
        for cluster_data in result.get("clusters", []):
            cluster_markets = [
                market_lookup[mid]
                for mid in cluster_data.get("market_ids", [])
                if mid in market_lookup
            ]
            
            if not cluster_markets:
                continue
            
            # Parse structured constraints into dependency strings
            dependencies = []
            for constraint in cluster_data.get("constraints", []):
                c_type = constraint.get("type", "UNKNOWN")
                c_desc = constraint.get("description", "")
                dependencies.append(f"[{c_type}] {c_desc}")
            
            # Also capture legacy format if present
            for dep in cluster_data.get("potential_dependencies", []):
                if dep not in dependencies:
                    dependencies.append(dep)
            
            # Log arbitrage signals
            arb_signal = cluster_data.get("arbitrage_signal", "none")
            if arb_signal and arb_signal.lower() != "none":
                logger.info(
                    "Arbitrage signal detected",
                    topic=cluster_data.get("topic"),
                    signal=arb_signal,
                )
            
            confidence = cluster_data.get("confidence", 0.0)
            
            clusters.append(
                MarketCluster(
                    topic=cluster_data.get("topic", "Unknown"),
                    markets=cluster_markets,
                    potential_dependencies=dependencies,
                )
            )
            
            logger.info(
                "Cluster found",
                topic=cluster_data.get("topic"),
                markets=len(cluster_markets),
                constraints=len(dependencies),
                confidence=confidence,
            )
        
        return clusters
    
    async def continuous_scan(
        self,
        interval_seconds: int = 300,
        callback: Any | None = None,
    ) -> None:
        """
        Continuously scan for new markets.
        
        This runs indefinitely, scanning at the specified interval.
        New clusters are passed to the callback function.
        
        Args:
            interval_seconds: Seconds between scans
            callback: Async function to call with new clusters
        """
        logger.info(
            "Starting continuous market scan",
            interval=interval_seconds,
        )
        
        while True:
            try:
                clusters = await self.scan_markets()
                
                if clusters and callback:
                    await callback(clusters)
                    
            except Exception as e:
                logger.error(f"Scan failed: {e}")
            
            await asyncio.sleep(interval_seconds)


# Convenience function for simple usage
async def discover_markets(
    limit: int = 100,
    min_liquidity: float | None = None,
) -> list[MarketCluster]:
    """
    Convenience function to discover and cluster markets.
    
    Example:
        clusters = await discover_markets()
        for cluster in clusters:
            print(cluster.topic)
    """
    async with DiscoveryAgent() as discovery:
        return await discovery.scan_markets(limit=limit, min_liquidity=min_liquidity)
