"""
Map Maker - Offline analysis and constraint generation.

The Map Maker is the "slow brain" of PolyQuant. It runs periodically to:
1. Discover markets from Polymarket.
2. Analyze logical dependencies using DeepSeek.
3. Validate constraints using Gemini.
4. Persist the validated "Constraint Map" to disk.

The Navigator then loads this map for real-time trading.

USAGE:
------
    # Run as a script
    python -m polyquant.map_maker
    
    # Or programmatically
    from polyquant.map_maker import MapMaker
    
    async def run():
        map_maker = MapMaker()
        await map_maker.build_map()
"""

import asyncio
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polyquant.agents import (
    DiscoveryAgent,
    LogicArchitect,
    ValidatorAgent,
    CorrelationEngine,
    MarketCluster,
)
from polyquant.data import PolymarketClient
from polyquant.data.constraint_store import (
    ConstraintStore,
    ConstraintManifest,
    StoredConstraint,
    StoredDependency,
)
from polyquant.data.market_models import Market # Added based on instruction's implied source for Market
from polyquant.data.limitless_client import limitless_yes_token, limitless_no_token
from polyquant.agents.exchange_matcher import ExchangeMatcher
from polyquant.utils import config, get_logger
from polyquant.api.server import monitor
from polyquant.utils.cache import cache
from polyquant.utils.market_utils import get_yes_outcome, get_no_outcome

logger = get_logger(__name__)


class MapMaker:
    """
    Offline analysis engine for building the Constraint Map.
    
    The Map Maker performs deep analysis of market structures:
    - Discovery: Find related markets (clusters).
    - Logic: Analyze dependencies using LLM (DeepSeek).
    - Validation: Verify constraints using LLM (Gemini).
    - Persistence: Save validated constraints to disk.
    
    This is designed to run infrequently (e.g., hourly) since market
    structure is immutable on Polymarket.
    
    Example:
        async with MapMaker() as map_maker:
            await map_maker.build_map()
    """
    
    def __init__(self):
        """Initialize the Map Maker."""
        self._discovery: DiscoveryAgent | None = None
        self._logic_architect: LogicArchitect | None = None
        self._validator: ValidatorAgent | None = None
        self._correlation_agent: CorrelationEngine | None = None
        self._polymarket: PolymarketClient | None = None
        self._store: ConstraintStore | None = None
        self._exchange_matcher: ExchangeMatcher | None = None
        
        logger.info("MapMaker initialized")
    
    async def __aenter__(self) -> "MapMaker":
        """Initialize all components."""
        logger.info("Starting Map Maker...")

        # Connect to cache
        await cache.connect()

        # Initialize agents
        self._discovery = DiscoveryAgent()
        await self._discovery.__aenter__()

        self._logic_architect = LogicArchitect()
        await self._logic_architect.__aenter__()

        self._validator = ValidatorAgent()
        await self._validator.__aenter__()

        self._correlation_agent = CorrelationEngine()

        # Initialize Polymarket client (for market data)
        self._polymarket = PolymarketClient()
        await self._polymarket.__aenter__()

        # Initialize Cross-Exchange Matcher
        self._exchange_matcher = ExchangeMatcher()

        # Initialize constraint store
        self._store = ConstraintStore()

        logger.info("Map Maker started")
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Cleanup all components."""
        logger.info("Shutting down Map Maker...")
        
        if self._discovery:
            await self._discovery.__aexit__(*args)
        if self._logic_architect:
            await self._logic_architect.__aexit__(*args)
        if self._validator:
            await self._validator.__aexit__(*args)
        if self._correlation_agent:
            # Cleanup if needed
            pass
        if self._polymarket:
            await self._polymarket.__aexit__(*args)
        
        logger.info("Map Maker shutdown complete")
    
    async def build_map(
        self,
        limit: int = 0,
        min_liquidity: float = 1000,
        skip_processed: bool = True,
    ) -> dict[str, Any]:
        """
        Build the complete constraint map.
        
        This is the main entry point. It runs the full analysis pipeline
        and persists the results.
        
        Args:
            limit: Maximum number of markets to analyze.
            min_liquidity: Minimum liquidity threshold for markets.
            skip_processed: Whether to skip markets already in the constraint store.
            
        Returns:
            Summary of the map building process.
        """
        start_time = datetime.now(timezone.utc)

        # Get and increment version for this build
        current_version = await cache.get_manifest_version()
        new_version = await cache.increment_manifest_version()

        logger.info(
            "Building constraint map",
            limit=limit,
            min_liquidity=min_liquidity,
            version=new_version,
            previous_version=current_version,
        )

        results: dict[str, Any] = {
            "start_time": start_time.isoformat(),
            "status": "running",
            "version": new_version,
        }
        
        try:
            # ================================================================
            # PHASE 1: DISCOVERY
            # ================================================================
            # Clear previous run's events and emit start banner
            monitor.state.pipeline_events = []
            # Reset both progress buckets so a fresh run starts from 0/0
            await monitor.update_status(llm_progress={
                "LOGIC":    {"done": 0, "total": 0, "current": ""},
                "MATCHING": {"done": 0, "total": 0, "current": ""},
            })
            print(f"\n{'='*60}")
            print(f"  \U0001f680 MAP MAKER STARTED — {start_time.strftime('%H:%M:%S UTC')}")
            print(f"{'='*60}")
            await monitor.emit_pipeline_event(
                "DISCOVERY", "info",
                f"MapMaker started (v{new_version})",
            )

            logger.info("Phase 1: Discovery - Scanning markets...")
            await monitor.update_status(pipeline_stage="DISCOVERY", mapped_pairs=[])
            
            if not self._discovery:
                raise RuntimeError("Discovery agent not initialized")
            
            clusters = await self._discovery.scan_markets(
                limit=limit,
                min_liquidity=min_liquidity,
                skip_processed=skip_processed,
            )
            
            # Record in monitor for UI
            await monitor.update_status(
                clusters=[{
                    "id": c.cluster_id,
                    "topic": c.topic,
                    "count": len(c.markets),
                    "status": "pending_analysis"
                } for c in clusters]
            )

            total_markets = sum(len(c.markets) for c in clusters)
            print(f"--- \U0001f4e1 DISCOVERY: Found {len(clusters)} clusters ({total_markets} markets) ---")
            logger.info(f"Discovered {len(clusters)} clusters")

            await monitor.emit_pipeline_event(
                "DISCOVERY", "info",
                f"Discovered {len(clusters)} clusters ({total_markets} markets)",
            )

            results["discovery"] = {
                "clusters_found": len(clusters),
                "total_markets": total_markets,
            }
            
            if not clusters:
                results["status"] = "complete"
                results["outcome"] = "no_clusters"
                await monitor.update_status(pipeline_stage="IDLE")
                return results
            
            # ================================================================
            # PHASE 2, 3 & 4: CONCURRENT REASONING AND CROSS-EXCHANGE MAPPING
            # ================================================================
            logger.info("Starting concurrent Reasoning Phase (Intra-market) and Matching Phase (Cross-exchange)...")
            
            async def run_reasoning_pipeline() -> dict[str, Any]:
                await monitor.update_status(pipeline_stage="LOGIC")
                manifests_saved = 0
                total_constraints = 0
                total_dependencies = 0
                cache_hits = 0

                total_clusters = len(clusters)
                await monitor.update_llm_progress(
                    "LOGIC", done=0, total=total_clusters, current=""
                )
                
                # Helper to update a single cluster's status in the UI
                async def update_ui_cluster_status(cid: str, new_status: str):
                    current_clusters = monitor.state.clusters
                    for c in current_clusters:
                        if c["id"] == cid:
                            c["status"] = new_status
                            break
                    await monitor.update_status(clusters=current_clusters)

                for idx, cluster in enumerate(clusters, 1):
                    # Progress logging
                    logger.info(
                        f"Processing cluster {idx}/{total_clusters}",
                        cluster_id=cluster.cluster_id,
                        topic=cluster.topic,
                    )

                    await update_ui_cluster_status(cluster.cluster_id, "Analyzing (LLM)...")
                    # Surface which cluster is currently being analyzed (before LLM call starts)
                    await monitor.update_llm_progress(
                        "LOGIC", current=cluster.topic[:80]
                    )

                    print(f"\n--- \U0001f9e9 LOGIC [{idx}/{total_clusters}]: '{cluster.topic}' ({len(cluster.markets)} markets) ---")
                    await monitor.emit_pipeline_event(
                        "LOGIC", "llm_start",
                        f"Analyzing cluster {idx}/{total_clusters}: {cluster.topic}",
                        detail=f"{len(cluster.markets)} markets",
                    )

                    t_cluster = time.time()
                    manifest = await self._analyze_cluster(cluster)
                    cluster_elapsed = time.time() - t_cluster

                    if manifest:
                        # Check if this was a cache hit
                        is_cached = hasattr(manifest, '_from_cache') and manifest._from_cache
                        if is_cached:
                            cache_hits += 1
                            print(f"--- \u26a1 CACHE HIT: '{cluster.topic}' ({manifest.constraint_count} constraints) ---")
                            await update_ui_cluster_status(cluster.cluster_id, f"Cached ({manifest.constraint_count} found)")
                            await monitor.emit_pipeline_event(
                                "LOGIC", "cache_hit",
                                f"Cache hit for '{cluster.topic}'",
                                detail=f"{manifest.constraint_count} constraints reused",
                                duration=cluster_elapsed,
                            )
                        else:
                            print(f"--- \u2705 ANALYZED: '{cluster.topic}' [{cluster_elapsed:.1f}s] -> {manifest.constraint_count} constraints, {manifest.dependency_count} deps ---")
                            await update_ui_cluster_status(cluster.cluster_id, f"Parsed ({manifest.constraint_count} found)")
                            await monitor.emit_pipeline_event(
                                "LOGIC", "llm_success",
                                f"Analyzed '{cluster.topic}'",
                                detail=f"{manifest.constraint_count} constraints, {manifest.dependency_count} dependencies",
                                duration=cluster_elapsed,
                            )

                        if manifest.constraint_count > 0:
                            if self._store:
                                await self._store.save_manifest(manifest)
                            manifests_saved += 1
                            total_constraints += manifest.constraint_count
                            total_dependencies += manifest.dependency_count
                    else:
                        print(f"--- \u2796 NO CONSTRAINTS: '{cluster.topic}' [{cluster_elapsed:.1f}s] ---")
                        await update_ui_cluster_status(cluster.cluster_id, "No constraints found")
                        await monitor.emit_pipeline_event(
                            "LOGIC", "info",
                            f"No constraints found for '{cluster.topic}'",
                            duration=cluster_elapsed,
                        )

                    # Progress percentage
                    progress_pct = (idx / total_clusters) * 100
                    logger.info(f"Progress: {progress_pct:.1f}% complete")
                    await monitor.update_llm_progress("LOGIC", done=idx)
                
                return {
                    "manifests_saved": manifests_saved,
                    "total_constraints": total_constraints,
                    "total_dependencies": total_dependencies,
                    "cache_hits": cache_hits,
                    "cache_hit_rate": f"{(cache_hits / total_clusters * 100):.1f}%" if total_clusters > 0 else "0%",
                }

            async def run_cross_exchange_pipeline() -> dict[str, Any]:
                await monitor.update_status(pipeline_stage="MATCHING")
                print(f"\n{'='*60}")
                print(f"--- \U0001f310 MATCHING: Starting cross-exchange pipeline ---")
                print(f"{'='*60}")
                await monitor.emit_pipeline_event(
                    "MATCHING", "info",
                    "Starting cross-exchange matching pipeline",
                )
                t_match = time.time()
                if self._exchange_matcher:
                    new_mappings, matching_stats = await self._exchange_matcher.run_matching_pipeline()
                    matching_stats["total_pairs"] = len(new_mappings)
                    match_elapsed = time.time() - t_match
                    new_found = matching_stats.get("new_pairs_found", 0)
                    total = matching_stats.get("total_pairs_after", len(new_mappings))
                    print(f"--- \u2705 MATCHING COMPLETE [{match_elapsed:.1f}s]: {new_found} new pairs, {total} total ---\n")
                    await monitor.emit_pipeline_event(
                        "MATCHING", "info",
                        f"Matching complete: {new_found} new pairs found ({total} total cached)",
                        duration=match_elapsed,
                    )
                    # Emit individual match events for any newly found pairs
                    for pair in matching_stats.get("matched_pairs_detail", []):
                        await monitor.emit_pipeline_event(
                            "MATCHING", "match",
                            f"Matched: {pair.get('polymarket', '')[:50]}",
                            detail=f"↔ {pair.get('limitless', '')[:50]} (sim: {pair.get('similarity', 0):.2f})",
                        )
                    return matching_stats
                return {"skipped": True}

            # Run cross exchange matching first so reasoning pipeline can use the newly discovered Limitless pairs
            matching_stats = await run_cross_exchange_pipeline()
            analysis_data = await run_reasoning_pipeline()
            
            results["analysis"] = analysis_data
            results["matching"] = matching_stats
            
            results["status"] = "complete"
            results["outcome"] = "success"
            
            await monitor.update_status(pipeline_stage="COMPLETE")
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Map building failed", error=str(e), exc_type=type(e).__name__, exc_info=True)
            results["status"] = "error"
            results["error"] = f"{type(e).__name__}: {str(e)}"
            results["traceback"] = tb
        
        # Record timing
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(results["start_time"])).total_seconds()
        results["elapsed_seconds"] = elapsed

        # Final summary
        print(f"\n{'='*60}")
        print(f"--- \U0001f3c1 MAP MAKER COMPLETE [{elapsed:.0f}s] | Status: {results['status'].upper()} ---")
        print(f"{'='*60}\n")
        await monitor.emit_pipeline_event(
            "COMPLETE", "info",
            f"MapMaker finished in {elapsed:.0f}s — {results['status'].upper()}",
        )
        
        logger.info(
            "Map building complete",
            status=results["status"],
            elapsed=elapsed,
        )

        # Generate and save report
        report_path = self._generate_report(results)
        results["report_path"] = str(report_path)
        
        return results

    def _generate_report(self, results: dict[str, Any]) -> Path:
        """Generate a comprehensive run report and save it to .polyquant/reports/."""
        reports_dir = Path.cwd() / ".polyquant" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"mapmaker_report_{timestamp}.txt"
        
        lines = []
        lines.append("=" * 70)
        lines.append("  POLYQUANT MAP MAKER — RUN REPORT")
        lines.append("=" * 70)
        lines.append(f"  Timestamp    : {results.get('start_time', 'N/A')}")
        lines.append(f"  Status       : {results.get('status', 'unknown').upper()}")
        lines.append(f"  Version      : {results.get('version', 'N/A')}")
        lines.append(f"  Duration     : {results.get('elapsed_seconds', 0):.1f} seconds")
        lines.append("")
        
        # Discovery
        disc = results.get("discovery", {})
        lines.append("-" * 70)
        lines.append("  PHASE 1: DISCOVERY")
        lines.append("-" * 70)
        lines.append(f"  Clusters found        : {disc.get('clusters_found', 0)}")
        lines.append(f"  Total markets in scope: {disc.get('total_markets', 0)}")
        lines.append("")
        
        # Cross-Exchange Matching
        match = results.get("matching", {})
        if not match.get("skipped"):
            lines.append("-" * 70)
            lines.append("  PHASE 2: CROSS-EXCHANGE MATCHING (Polymarket ↔ Limitless)")
            lines.append("-" * 70)
            lines.append(f"  Polymarket markets fetched  : {match.get('polymarket_fetched', 'N/A')}")
            lines.append(f"  Polymarket after $2500 filt : {match.get('polymarket_after_filter', 'N/A')}")
            lines.append(f"  Limitless markets fetched   : {match.get('limitless_fetched', 'N/A')}")
            lines.append(f"  Limitless after $2500 filt  : {match.get('limitless_after_filter', 'N/A')}")
            lines.append(f"  Limitless discarded (low $) : {match.get('limitless_discarded', 'N/A')}")
            lines.append(f"  Already accepted (cache)    : {match.get('already_accepted', 0)}")
            lines.append(f"  Already rejected (cache)    : {match.get('already_rejected', 0)}")
            lines.append(f"  Prefilter dropped pairs     : {match.get('prefilter_dropped', 0)}")
            lines.append(f"  Rejection cache skips       : {match.get('rejection_cache_hits', 0)}")
            lines.append(f"  Candidate pairs after pref. : {match.get('candidate_pairs_after_prefilter', 0)}")
            lines.append(f"  LLM verifications sent      : {match.get('llm_verifications_sent', 0)}")
            lines.append(f"  LLM matches confirmed       : {match.get('llm_matches_confirmed', 0)}")
            lines.append(f"  LLM matches rejected        : {match.get('llm_matches_rejected', 0)}")
            lines.append(f"  LLM errors                  : {match.get('llm_errors', 0)}")
            lines.append(f"  NEW pairs found this run    : {match.get('new_pairs_found', 0)}")
            lines.append(f"  NEW rejections cached       : {match.get('new_rejections_cached', 0)}")
            lines.append(f"  Total pairs (cumulative)    : {match.get('total_pairs_after', 0)}")
            
            pairs_detail = match.get("matched_pairs_detail", [])
            if pairs_detail:
                lines.append("")
                lines.append("  Matched Pairs Detail:")
                for i, p in enumerate(pairs_detail, 1):
                    lines.append(f"    {i}. [{p['similarity']:.3f}] {p['method']}")
                    lines.append(f"       PM: {p['polymarket']}")
                    lines.append(f"       LM: {p['limitless']}")
            lines.append("")
        else:
            lines.append("-" * 70)
            lines.append("  PHASE 2: CROSS-EXCHANGE MATCHING — SKIPPED")
            lines.append("-" * 70)
            lines.append("")
        
        # Constraint Analysis
        analysis = results.get("analysis", {})
        lines.append("-" * 70)
        lines.append("  PHASE 3: CONSTRAINT ANALYSIS (LLM Reasoning)")
        lines.append("-" * 70)
        lines.append(f"  Manifests saved        : {analysis.get('manifests_saved', 0)}")
        lines.append(f"  Total constraints      : {analysis.get('total_constraints', 0)}")
        lines.append(f"  Total dependencies     : {analysis.get('total_dependencies', 0)}")
        lines.append(f"  Cache hits             : {analysis.get('cache_hits', 0)}")
        lines.append(f"  Cache hit rate         : {analysis.get('cache_hit_rate', 'N/A')}")
        lines.append("")
        
        # Error
        if results.get("error"):
            lines.append("-" * 70)
            lines.append("  ERROR")
            lines.append("-" * 70)
            lines.append(f"  {results['error']}")
            tb = results.get("traceback", "")
            if tb:
                for tb_line in tb.strip().split("\n"):
                    lines.append(f"  {tb_line}")
            lines.append("")
        
        lines.append("=" * 70)
        lines.append("  END OF REPORT")
        lines.append("=" * 70)
        
        report_text = "\n".join(lines)
        
        # Save to file
        report_file.write_text(report_text, encoding="utf-8")
        
        # Also print to console
        print("\n" + report_text)
        
        logger.info(f"Report saved to {report_file}")
        return report_file
    
    @staticmethod
    def _compute_cluster_hash(cluster: MarketCluster) -> str:
        """
        Compute a stable hash of a cluster for caching.

        The hash is based on:
        - Market IDs (stable)
        - Market questions (stable)
        - Topic (stable)

        This allows us to cache LLM results and avoid re-analyzing
        identical clusters.

        Args:
            cluster: The cluster to hash

        Returns:
            SHA256 hex digest of the cluster
        """
        # Create a deterministic representation
        market_data = []
        for market in sorted(cluster.markets, key=lambda m: m.market_id):
            market_data.append(f"{market.market_id}:{market.question}")

        cluster_repr = f"{cluster.topic}|{'|'.join(market_data)}"
        return hashlib.sha256(cluster_repr.encode()).hexdigest()

    async def _analyze_cluster(
        self,
        cluster: MarketCluster,
    ) -> ConstraintManifest | None:
        """
        Analyze a single cluster and create a ConstraintManifest.

        This runs the Logic Architect and Validator on the cluster.
        Uses caching to avoid redundant LLM calls for identical clusters.

        Args:
            cluster: The market cluster to analyze.

        Returns:
            ConstraintManifest if analysis succeeds, None otherwise.
        """
        logger.info(
            "Analyzing cluster",
            cluster_id=cluster.cluster_id,
            topic=cluster.topic,
            markets=len(cluster.markets),
        )

        # Compute cluster hash for caching
        cluster_hash = self._compute_cluster_hash(cluster)

        # Check cache first
        cached_result = await cache.get_llm_result(cluster_hash)
        if cached_result:
            logger.info(
                "Cache hit - reusing previous analysis",
                cluster_id=cluster.cluster_id,
                cluster_hash=cluster_hash[:8],
            )

            # Reconstruct manifest from cached data
            try:
                manifest = ConstraintManifest(**cached_result)
                # Mark as from cache for statistics
                manifest._from_cache = True  # type: ignore
                return manifest
            except Exception as e:
                logger.warning(
                    "Failed to reconstruct manifest from cache",
                    error=str(e),
                )
                # Fall through to re-analyze

        try:
            # Run Logic Architect
            if not self._logic_architect:
                raise RuntimeError("Logic Architect not initialized")
            
            analysis = await self._logic_architect.analyze_cluster(cluster)
            
            if not analysis.constraints and not analysis.dependencies:
                logger.info("No dependencies found", cluster_id=cluster.cluster_id)
                return None
            
            # Run Validator
            if not self._validator:
                raise RuntimeError("Validator not initialized")
            
            validated = await self._validator.validate(analysis)

            if not validated.is_valid:
                issue_reasons = [i.description for i in validated.issues[:3]]
                reason_str = "; ".join(issue_reasons) if issue_reasons else "Unknown"
                print(f"--- \u26a0\ufe0f  VALIDATOR REJECTED: '{cluster.topic}' | Reason: {reason_str} ---")
                logger.warning(
                    "Validation failed - discarding cluster",
                    cluster_id=cluster.cluster_id,
                    issues=len(validated.issues),
                )
                await monitor.emit_pipeline_event(
                    "LOGIC", "validation_fail",
                    f"Validation issues for '{cluster.topic}'",
                    detail=reason_str,
                )
                # Rejected clusters must NOT be persisted. Navigator's
                # _manifest_to_validated() unconditionally marks loaded manifests
                # as is_valid=True, so any file on disk is treated as fully
                # validated — a rejected-but-saved cluster would drive live trades.
                return None
            
            # Convert to StoredConstraint format
            stored_constraints = [
                StoredConstraint(
                    constraint_id=c.constraint_id,
                    description=c.description,
                    coefficients=c.coefficients,
                    rhs=c.rhs,
                    confidence=c.confidence,
                    reasoning=c.reasoning,
                    source_markets=c.source_markets,
                )
                for c in validated.validated_constraints
            ]
            
            # Convert to StoredDependency format
            stored_dependencies = [
                StoredDependency(
                    source_market_id=d.source_market_id,
                    source_outcome=d.source_outcome,
                    target_market_id=d.target_market_id,
                    target_outcome=d.target_outcome,
                    relationship=d.relationship,
                    confidence=d.confidence,
                )
                for d in validated.validated_dependencies
            ]
            
            # Run Correlation Agent
            correlations = []
            if self._correlation_agent and self._polymarket:
                async def history_provider(mid: str):
                    # get_history returns a list of {"t": timestamp, "p": price} dicts
                    # Correlation engine expects exactly that shape.
                    return await self._polymarket.get_history(mid)
                
                # Correlate on the YES token specifically. Resolve YES by name
                # so we never accidentally correlate NO prices — outcomes[0] is
                # not guaranteed to be the YES side.
                token_to_market = {}
                token_markets = []
                for m in cluster.markets:
                    yes_out = get_yes_outcome(m)
                    if not yes_out or not yes_out.token_id:
                        continue
                    token_id = yes_out.token_id
                    token_to_market[token_id] = m
                    # Create a dummy market with the YES token_id as its primary
                    # ID so the correlation engine tests the correct string
                    token_market = Market(
                        market_id=token_id,
                        question=m.question,
                        outcomes=m.outcomes,
                        liquidity=m.liquidity,
                        volume=m.volume,
                    )
                    token_markets.append(token_market)
                
                # CorrelationAgent might need to be async or we fetch here
                # Let's assume we can pass the provider and it handles it
                # Logic: scan_for_pairs(token_markets, history_provider)
                signals = await self._correlation_agent.scan_for_pairs(
                    token_markets, 
                    history_provider
                )
                correlations = [s.model_dump(mode="json") for s in signals]

            # Create limitless equivalencies
            market_ids = [m.market_id for m in cluster.markets]
            market_exchanges = {m.market_id: "polymarket" for m in cluster.markets}
            market_titles = {m.market_id: m.question for m in cluster.markets}
            
            if self._exchange_matcher:
                mapped = self._exchange_matcher.mapped_pairs
                for m in cluster.markets:
                    l_val = mapped.get(m.market_id)
                    if l_val:
                        # Value might be "l_id|slug" or just "l_id"
                        if "|" in l_val:
                            l_id, l_slug = l_val.split("|", 1)
                        else:
                            l_id = l_val
                            l_slug = ""
                            
                        if l_id not in market_ids:
                            market_ids.append(l_id)
                        
                        # Add slug to market exchanges for downstream
                        if l_slug:
                            market_exchanges[l_id] = f"limitless:{l_slug}"
                        else:
                            market_exchanges[l_id] = "limitless"
                            
                        market_titles[l_id] = f"{m.question} (Limitless)"
                            
                        # Resolve Polymarket YES/NO by outcome name — never by
                        # index. Falls back to skipping the cross-exchange
                        # mapping if either side is unresolvable, rather than
                        # silently writing the wrong token_id into coefficients.
                        pm_yes_out = get_yes_outcome(m)
                        pm_no_out = get_no_outcome(m)
                        if pm_yes_out is None or pm_no_out is None:
                            logger.warning(
                                "Skipping Limitless equivalence: polarity unresolved",
                                market_id=m.market_id,
                                question=m.question,
                            )
                            continue
                        pm_yes = pm_yes_out.token_id or pm_yes_out.outcome_id
                        pm_no = pm_no_out.token_id or pm_no_out.outcome_id

                        # Build Limitless token_ids via the canonical helper so
                        # the suffix convention stays in one place.
                        l_yes = limitless_yes_token(l_id)
                        l_no = limitless_no_token(l_id)

                        # Append to coefficients for all constraints so solver treats them as perfect substitutes
                        for c in stored_constraints:
                            if pm_yes in c.coefficients:
                                c.coefficients[l_yes] = c.coefficients[pm_yes]
                            if pm_no in c.coefficients:
                                c.coefficients[l_no] = c.coefficients[pm_no]
            # Validate coefficient keys to ensure they are properly mapped
            for c in stored_constraints:
                invalid_keys = []
                for k in c.coefficients.keys():
                    if not (k.startswith("0x") or "_" in k or len(k) > 20):
                        invalid_keys.append(k)
                if invalid_keys:
                    logger.warning("Constraint has potentially unmapped/invalid token IDs", constraint_id=c.constraint_id, invalid_keys=invalid_keys)

            # Create manifest

            manifest = ConstraintManifest(
                cluster_id=cluster.cluster_id,
                topic=cluster.topic,
                market_ids=market_ids,
                market_exchanges=market_exchanges,
                market_titles=market_titles,
                constraints=stored_constraints,
                dependencies=stored_dependencies,
                correlations=correlations,
            )

            # Cache the result for future runs (5 minute TTL)
            cache_data = manifest.model_dump(mode="json")
            await cache.set_llm_result(cluster_hash, cache_data, ttl_seconds=300)

            logger.info(
                "Cluster analysis complete",
                cluster_id=cluster.cluster_id,
                constraints=len(stored_constraints),
                dependencies=len(stored_dependencies),
                cached=True,
            )

            return manifest
            
        except Exception as e:
            logger.error(
                "Cluster analysis failed",
                cluster_id=cluster.cluster_id,
                error=str(e),
            )
            return None


async def main() -> None:
    """Main entry point for the Map Maker."""
    from polyquant.api.server import monitor, start_api_server

    print("""
    ===============================================================
                      PolyQuant Map Maker
              Offline Constraint Analysis Engine
    ===============================================================
    """)

    server, server_task = await start_api_server()
    await monitor.update_status(status="MAPPING")

    try:
        async with MapMaker() as map_maker:
            result = await map_maker.build_map(
                limit=500,
                min_liquidity=1000,
            )

            report_path = result.get("report_path", "N/A")
            print(f"\n  Report saved to: {report_path}")
            await monitor.update_status(status="MAPPING_COMPLETE")
            await asyncio.sleep(5)  # let final WS frames flush to dashboard
    finally:
        server.should_exit = True
        await server_task


if __name__ == "__main__":
    asyncio.run(main())
