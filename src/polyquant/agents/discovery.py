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
import json
from datetime import datetime
from typing import Any

from polyquant.utils.llm_client import call_llm_json
from pydantic import BaseModel, Field

from polyquant.data import Market, PolymarketClient
from polyquant.utils import config, get_logger
from polyquant.utils.cache import cache

logger = get_logger(__name__)


class MarketCluster(BaseModel):
    """
    A cluster of related markets that may have logical dependencies.
    
    Attributes:
        cluster_id: Unique identifier for this cluster
        topic: Human-readable topic description
        markets: List of Market objects in this cluster
        potential_dependencies: Initial guesses at dependencies (for Logic Architect)
        created_at: When this cluster was created
    """
    cluster_id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()))
    topic: str
    markets: list[Market] = Field(default_factory=list)
    potential_dependencies: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


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
        
        for event in events:
            event_id = event["event_id"]
            event_title = event["title"]
            markets: list[Market] = event["markets"]
            neg_risk_id = event.get("neg_risk_market_id")
            
            # Skip already-processed events
            if skip_processed and event_id in self._processed_markets:
                continue
            
            # Skip if Redis says processed
            if skip_processed and await cache.is_market_processed(f"event_{event_id}"):
                self._processed_markets.add(event_id)
                continue
            
            # Filter zombie markets (extreme prices)
            valid_markets = [m for m in markets if not self._is_zombie_market(m)]
            if not valid_markets:
                continue
            
            # Auto-cluster: NegRisk events with price deviation detection
            # ENHANCED (Week 5): Explicit deviation detection for arbitrage opportunities
            # ENHANCED (Week 6): Filter zero-volume outcomes to prevent phantom signals
            if neg_risk_id and len(valid_markets) > 1:
                # Filter out outcomes with negligible volume before calculating price sum.
                # Dead/zero-volume outcomes often have phantom mid-prices that inflate
                # the sum far beyond 1.0 (e.g., 14.43 or 4.50), producing false positives.
                MIN_OUTCOME_VOLUME = 100  # $100 minimum volume to be considered "real"
                priced_markets = [
                    m for m in valid_markets
                    if m.outcomes and m.volume and m.volume > MIN_OUTCOME_VOLUME
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
                        "NegRisk: filtered low-volume outcomes from price sum",
                        event_title=event_title,
                        kept=len(priced_markets),
                        filtered=filtered_count,
                        threshold=MIN_OUTCOME_VOLUME,
                    )

                # Calculate actual price sum using only outcomes with real volume
                total_price = sum(
                    m.outcomes[0].price for m in priced_markets
                )

                # Detect deviation from theoretical sum of 1.0 (ensure types match)
                deviation = abs(float(total_price) - 1.0)
                deviation_pct = deviation * 100

                # Classify market state based on deviation
                if float(total_price) < 0.98:
                    market_state = "UNDERPRICED"
                    arbitrage_type = "Buy Arbitrage (prices sum < 1.0)"
                elif float(total_price) > 1.02:
                    market_state = "OVERPRICED"
                    arbitrage_type = "Sell Arbitrage (prices sum > 1.0)"
                else:
                    market_state = "FAIR"
                    arbitrage_type = "No deviation"

                # Create dependency description with deviation info
                dependency_desc = (
                    f"[PARTITION] NegRisk group must sum to 1.0. "
                    f"Actual: {total_price:.4f} ({market_state}). "
                    f"Deviation: {deviation_pct:.2f}%. "
                    f"Opportunity: {arbitrage_type}"
                )

                clusters.append(
                    MarketCluster(
                        cluster_id=f"negrisk_{event_id}",
                        topic=f"[AUTO] {event_title} (NegRisk, Sum={total_price:.4f}, {market_state})",
                        markets=valid_markets,
                        potential_dependencies=[dependency_desc],
                    )
                )

                # Log arbitrage signals for monitoring
                if deviation > 0.02:  # >2% deviation
                    logger.warning(
                        "ARBITRAGE SIGNAL: Price deviation detected in NegRisk event",
                        event_title=event_title,
                        markets=len(valid_markets),
                        price_sum=f"{total_price:.4f}",
                        deviation_pct=f"{deviation_pct:.2f}%",
                        state=market_state,
                        opportunity=arbitrage_type,
                    )
                else:
                    logger.info(
                        "Auto-clustered NegRisk event (fair price)",
                        event_title=event_title,
                        markets=len(valid_markets),
                        price_sum=f"{total_price:.4f}",
                        state=market_state,
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
        logger.info(
            "Phase 2 complete: pre-filtering done",
            auto_clusters=auto_cluster_count,
            events_for_llm=len(events_for_llm),
        )
        
        # ── Phase 3: LLM analysis for within-event + cross-event constraints ──
        if events_for_llm:
            # Collect all markets from events needing LLM analysis
            all_llm_markets: list[Market] = []
            for event in events_for_llm:
                all_llm_markets.extend(event["markets"])
            
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
                        await asyncio.sleep(6.0) # Stay under 20 RPM
                        return res
                    except Exception as e:
                        logger.warning(f"Batch {batch_idx} LLM failed: {e}")
                        return [MarketCluster(topic="Uncategorized (Batch Failed)", markets=b)]
            
            tasks = [process_batch(b, i) for i, b in enumerate(batches)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, list):
                    all_clusters.extend(r)
                elif isinstance(r, Exception):
                    logger.warning(f"LLM batch failed, skipping that chunk: {r}")
                    # Fallback for this chunk so we don't lose the markets
                    all_clusters.append(MarketCluster(
                        topic="Uncategorized (Batch Failed)",
                        markets=batches[results.index(r)],
                        potential_dependencies=[],
                    ))
            return all_clusters

        # Format markets for the prompt - include prices and liquidity
        market_lines = []
        for m in markets:
            # Get YES price from first outcome, or 0 if unavailable
            yes_price = m.outcomes[0].price if m.outcomes else 0.0
            market_lines.append(
                f"ID: {m.market_id} | Question: {m.question} | YES Price: {yes_price:.2f} | Liquidity: ${m.liquidity:,.0f}"
            )
        market_descriptions = "\n".join(market_lines)
        
        # Create market lookup for quick access
        market_lookup = {m.market_id: m for m in markets}

        logger.info(f"Calling LLM for market clustering ({len(markets)} targets)", market_count=len(markets))

        try:
            # Call LLM via OpenRouter with an outer timeout guard.
            # The OpenAI client has a 45s timeout, but some free models
            # (e.g., glm-4.5-air:free) ignore it and hang for 10+ minutes.
            # asyncio.wait_for provides a hard upper bound.
            # Retry once on timeout/failure — OpenRouter rotates to a different
            # free model on each call, so the second attempt often succeeds.
            LLM_CLUSTERING_TIMEOUT = 60  # seconds — generous but finite
            MAX_ATTEMPTS = 2
            result = None

            for attempt in range(MAX_ATTEMPTS):
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            call_llm_json,
                            prompt=f"Analyze these {len(markets)} markets for arbitrage opportunities:\n\n{market_descriptions}",
                            system_prompt=self.CLUSTERING_PROMPT,
                            temperature=0.2,
                        ),
                        timeout=LLM_CLUSTERING_TIMEOUT,
                    )
                    if result:
                        break  # Success — exit retry loop
                    else:
                        logger.warning(
                            f"LLM returned empty response (attempt {attempt + 1}/{MAX_ATTEMPTS})",
                            market_count=len(markets),
                        )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"LLM clustering timed out after {LLM_CLUSTERING_TIMEOUT}s "
                        f"(attempt {attempt + 1}/{MAX_ATTEMPTS})",
                        market_count=len(markets),
                    )
                except Exception as e:
                    logger.warning(
                        f"LLM clustering error (attempt {attempt + 1}/{MAX_ATTEMPTS}): {e}",
                        market_count=len(markets),
                    )

                if attempt < MAX_ATTEMPTS - 1:
                    logger.info("Retrying LLM clustering with fresh model rotation...")
                    await asyncio.sleep(2)  # Brief pause before retry

            if not result:
                raise ValueError("LLM clustering failed after all retry attempts")

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
        import asyncio
        
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
