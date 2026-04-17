"""
Map Maker - Offline analysis and constraint generation.

The Map Maker is the "slow brain" of PolyQuant. It runs periodically to:
1. Discover markets from Polymarket.
2. Analyze logical dependencies using Gemini (LogicArchitect).
3. Validate constraints using Gemini (ValidatorAgent).
4. Persist the validated "Constraint Map" to disk.

The Navigator then loads this map for real-time trading.

USAGE:
------
    python -m polyquant.main map [--limit N] [--min-liquidity M] [--force]
"""

import hashlib
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polyquant.agents import (
    DiscoveryAgent,
    LogicArchitect,
    ValidatorAgent,
    MarketCluster,
)
from polyquant.data import PolymarketClient
from polyquant.data.constraint_store import (
    ConstraintStore,
    ConstraintManifest,
    StoredConstraint,
    StoredDependency,
    TokenLabel,
)
from polyquant.data.market_models import Market
from polyquant.data.limitless_client import limitless_yes_token, limitless_no_token
from polyquant.agents.exchange_matcher import ExchangeMatcher
from polyquant.utils import config, get_logger
from polyquant.api.server import monitor
from polyquant.utils.cache import cache
from polyquant.utils.market_utils import get_yes_outcome, get_no_outcome

logger = get_logger(__name__)


# Recognized token_id shapes. Polymarket token_ids are long decimal strings;
# Limitless token_ids end in `_yes`/`_no` (or legacy `_0`/`_1` from manifests
# written before the suffix migration).
_POLY_TOKEN_RE = re.compile(r"^\d{50,80}$")
_LIMITLESS_TOKEN_RE = re.compile(r".+_(yes|no|0|1)$")


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
        self._polymarket: PolymarketClient | None = None
        self._store: ConstraintStore | None = None
        self._exchange_matcher: ExchangeMatcher | None = None

        # Per-run counters for Limitless equivalence injection diagnostics.
        # Reset at the start of each build_map() call so the run report
        # reflects this run only.
        self._limitless_skip_native_partition: int = 0
        self._limitless_skip_polarity_mismatch: int = 0

        logger.info("MapMaker initialized")

    async def __aenter__(self) -> "MapMaker":
        """Initialize all components."""
        logger.info("Starting Map Maker...")

        await cache.connect()

        self._discovery = DiscoveryAgent()
        await self._discovery.__aenter__()

        self._logic_architect = LogicArchitect()
        await self._logic_architect.__aenter__()

        self._validator = ValidatorAgent()
        await self._validator.__aenter__()

        self._polymarket = PolymarketClient()
        await self._polymarket.__aenter__()

        self._exchange_matcher = ExchangeMatcher()
        self._store = ConstraintStore()

        logger.info("Map Maker started")
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Cleanup context-managed components. ExchangeMatcher and
        ConstraintStore are not context managers and need no teardown."""
        logger.info("Shutting down Map Maker...")

        if self._discovery:
            await self._discovery.__aexit__(*args)
        if self._logic_architect:
            await self._logic_architect.__aexit__(*args)
        if self._validator:
            await self._validator.__aexit__(*args)
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

        # Reset per-run Limitless equivalence counters
        self._limitless_skip_native_partition = 0
        self._limitless_skip_polarity_mismatch = 0

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
                constraint_store=self._store,
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

            cluster_type_breakdown: dict[str, int] = {}
            for c in clusters:
                ct = c.constraint_source or "llm_analysis"
                cluster_type_breakdown[ct] = cluster_type_breakdown.get(ct, 0) + 1

            results["discovery"] = {
                "clusters_found": len(clusters),
                "total_markets": total_markets,
                "tag_skip_counts": dict(getattr(self._discovery, "last_tag_skip_counts", {})),
                "standalone_binary_partitions": getattr(
                    self._discovery, "last_standalone_binary_count", 0
                ),
                "cross_event_logical_clusters": getattr(
                    self._discovery, "last_cross_event_cluster_count", 0
                ),
                "cluster_type_breakdown": cluster_type_breakdown,
            }
            
            if not clusters:
                results["status"] = "complete"
                results["outcome"] = "no_clusters"
                await monitor.update_status(pipeline_stage="IDLE")
                return results
            
            # ================================================================
            # PHASE 2: CROSS-EXCHANGE MATCHING (must run first so the reasoning
            #          loop sees the up-to-date Limitless equivalencies)
            # PHASE 3: REASONING (LLM constraint analysis)
            # ================================================================
            logger.info("Phase 2: Cross-exchange matching, then Phase 3: Reasoning")
            
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
                    result = await self._analyze_cluster(cluster)
                    cluster_elapsed = time.time() - t_cluster

                    if result is not None:
                        manifest, is_cached = result
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
                    "limitless_skip_native_partition": self._limitless_skip_native_partition,
                    "limitless_skip_polarity_mismatch": self._limitless_skip_polarity_mismatch,
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

            # Run cross-exchange matching first so the reasoning loop can see
            # newly-discovered Limitless pairs. A matching failure must NOT
            # take down the whole build — reasoning can still produce
            # Polymarket-only manifests with an empty mapped_pairs.
            try:
                matching_stats = await run_cross_exchange_pipeline()
            except Exception as match_err:
                logger.error(
                    "Cross-exchange matching failed; continuing without Limitless equivalencies",
                    error=str(match_err),
                    exc_info=True,
                )
                await monitor.emit_pipeline_event(
                    "MATCHING", "error",
                    f"Matching pipeline failed: {type(match_err).__name__}",
                    detail=str(match_err),
                )
                matching_stats = {"error": str(match_err), "skipped": True}

            # ================================================================
            # PHASE 2.5: LIMITLESS-FIRST DISCOVERY (Gap 5b)
            # Emit native_partition clusters for standalone Limitless binary
            # markets not already aliased to a Polymarket cluster. Closes the
            # pure-Limitless intra-market arb hole.
            # ================================================================
            if config.enable_limitless_discovery and self._discovery and self._exchange_matcher:
                try:
                    mapped_limitless_slugs: set[str] = set(
                        self._exchange_matcher.mapped_pairs.values()
                    )
                    limitless_clusters = await self._discovery.discover_standalone_limitless_clusters(
                        mapped_limitless_slugs
                    )
                    if limitless_clusters:
                        logger.info(
                            "Adding standalone Limitless clusters to reasoning queue",
                            new_clusters=len(limitless_clusters),
                        )
                        clusters.extend(limitless_clusters)
                        # Keep the discovery summary in results accurate.
                        if isinstance(results.get("discovery"), dict):
                            results["discovery"]["clusters_found"] = len(clusters)
                            results["discovery"]["limitless_standalone_clusters"] = len(
                                limitless_clusters
                            )
                            results["discovery"]["limitless_discovery_stats"] = dict(
                                getattr(
                                    self._discovery,
                                    "last_limitless_discovery_stats",
                                    {},
                                )
                            )
                except Exception as limit_err:
                    logger.error(
                        "Limitless-first discovery failed; continuing without it",
                        error=str(limit_err),
                        exc_info=True,
                    )

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
        breakdown = disc.get("cluster_type_breakdown") or {}
        if breakdown:
            lines.append("  Clusters by type:")
            for ct in sorted(breakdown):
                lines.append(f"    - {ct}: {breakdown[ct]}")
        standalone_binaries = disc.get("standalone_binary_partitions", 0)
        if standalone_binaries:
            lines.append(f"  Standalone binary YES+NO=1 clusters (Gap 3): {standalone_binaries}")
        tag_skip_counts = disc.get("tag_skip_counts") or {}
        if tag_skip_counts:
            skipped_total = sum(tag_skip_counts.values())
            breakdown = ", ".join(f"{tag}={n}" for tag, n in sorted(tag_skip_counts.items()))
            lines.append(f"  Events skipped by excluded_tags: {skipped_total} ({breakdown})")
        limitless_stats = disc.get("limitless_discovery_stats") or {}
        if limitless_stats:
            lines.append(
                f"  Limitless-first discovery (Gap 5b): "
                f"{limitless_stats.get('standalone_clusters', 0)} standalone clusters from "
                f"{limitless_stats.get('markets_fetched', 0)} fetched "
                f"(already_mapped={limitless_stats.get('skipped_already_mapped', 0)}, "
                f"below_floor={limitless_stats.get('skipped_below_floor', 0)})"
            )
        lines.append("")
        
        # Cross-Exchange Matching
        match = results.get("matching", {})
        if not match.get("skipped"):
            lines.append("-" * 70)
            lines.append("  PHASE 2: CROSS-EXCHANGE MATCHING (Polymarket ↔ Limitless)")
            lines.append("-" * 70)
            poly_floor = config.min_liquidity_matcher_polymarket
            limitless_floor = config.min_liquidity_matcher_limitless
            lines.append(f"  Polymarket fetched (≥${poly_floor:,.0f}): {match.get('polymarket_fetched_at_floor', match.get('polymarket_fetched', 'N/A'))}")
            lines.append(f"  Limitless raw fetched      : {match.get('limitless_fetched_raw', match.get('limitless_fetched', 'N/A'))}")
            lines.append(f"  Limitless kept (≥${limitless_floor:,.0f}) : {match.get('limitless_kept_at_floor', match.get('limitless_after_filter', 'N/A'))}")
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

        # Phase 4 section is rendered inside Phase 3's block so the overall
        # report structure stays readable. Counts surface from discovery.
        cross_event_ce = disc.get("cross_event_logical_clusters", 0)
        if cross_event_ce:
            lines.append(f"  Cross-event logical clusters (Gap 1): {cross_event_ce} (routed through LLM)")
        lines.append(f"  Manifests saved        : {analysis.get('manifests_saved', 0)}")
        lines.append(f"  Total constraints      : {analysis.get('total_constraints', 0)}")
        lines.append(f"  Total dependencies     : {analysis.get('total_dependencies', 0)}")
        lines.append(f"  Cache hits             : {analysis.get('cache_hits', 0)}")
        lines.append(f"  Cache hit rate         : {analysis.get('cache_hit_rate', 'N/A')}")
        lines.append(f"  Limitless skip (native)   : {analysis.get('limitless_skip_native_partition', 0)} (expected)")
        lines.append(f"  Limitless skip (polarity) : {analysis.get('limitless_skip_polarity_mismatch', 0)} (diagnostic)")
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
    
    # Cache hash format version. Bump when the preimage structure changes so
    # older cache entries are naturally invalidated on the next run.
    _CACHE_HASH_VERSION = "v2"

    @staticmethod
    def _compute_cluster_hash(cluster: MarketCluster) -> str:
        """
        Compute a stable hash of a cluster for caching.

        The hash is based on:
        - constraint_source (so clusters with the same markets but different
          semantics — e.g. a mechanical `negrisk` vs. an LLM `llm_analysis`
          vs. the upcoming `cross_event_logical` — never collide)
        - Market IDs (stable)
        - Market questions (stable)
        - Topic (stable)

        Args:
            cluster: The cluster to hash

        Returns:
            SHA256 hex digest of the cluster
        """
        market_data = []
        for market in sorted(cluster.markets, key=lambda m: m.market_id):
            market_data.append(f"{market.market_id}:{market.question}")

        cluster_repr = (
            f"{MapMaker._CACHE_HASH_VERSION}"
            f"|{cluster.constraint_source}"
            f"|{cluster.topic}"
            f"|{'|'.join(market_data)}"
        )
        return hashlib.sha256(cluster_repr.encode()).hexdigest()

    @staticmethod
    def _cache_ttl_seconds() -> int:
        """LLM cache lives until ~10% before the next scheduled MapMaker run,
        so a fresh build never re-hits stale entries from the previous run."""
        return max(60, int(config.map_interval_seconds * 0.9))

    async def _analyze_mechanical_cluster(
        self,
        cluster: MarketCluster,
    ) -> tuple[ConstraintManifest, bool] | None:
        """
        Build a ConstraintManifest for a mechanical cluster without any LLM call.

        Handles NegRisk, native_partition, cross_market_partition, and
        monotonic_ladder clusters. The constraint shape is fully determined by
        the cluster's market set — no semantic reasoning is needed.

        Cross-exchange equivalencies are still injected on top, so Limitless
        matches land on these clusters too.
        """
        from polyquant.agents.logic_architect import build_partition_constraint

        logic_constraints = []
        if cluster.constraint_source == "monotonic_ladder":
            from polyquant.agents.ladder_detector import build_ladder_constraints

            logic_constraints = build_ladder_constraints(cluster)
        elif cluster.constraint_source == "conditional_subset":
            from polyquant.agents.ladder_detector import build_conditional_constraints

            logic_constraints = build_conditional_constraints(cluster)
        else:
            single = build_partition_constraint(cluster)
            if single is not None:
                logic_constraints = [single]

        if not logic_constraints:
            logger.warning(
                "Mechanical cluster produced no constraint — skipping",
                cluster_id=cluster.cluster_id,
                constraint_source=cluster.constraint_source,
                market_count=len(cluster.markets),
            )
            return None

        stored_constraints: list[StoredConstraint] = [
            StoredConstraint(
                constraint_id=lc.constraint_id,
                description=lc.description,
                coefficients=lc.coefficients,
                rhs=lc.rhs,
                confidence=lc.confidence,
                reasoning=lc.reasoning,
                source_markets=lc.source_markets,
            )
            for lc in logic_constraints
        ]
        stored_dependencies: list[StoredDependency] = []

        (
            market_ids,
            market_exchanges,
            market_titles,
            market_urls,
            token_labels,
        ) = self._build_market_metadata(cluster)
        self._inject_limitless_equivalencies(
            cluster,
            stored_constraints,
            market_ids,
            market_exchanges,
            market_titles,
            market_urls,
            token_labels,
        )
        self._validate_token_id_shapes(stored_constraints)

        manifest = ConstraintManifest(
            cluster_id=cluster.cluster_id,
            cluster_type=cluster.constraint_source,
            topic=cluster.topic,
            market_ids=market_ids,
            market_exchanges=market_exchanges,
            market_titles=market_titles,
            market_urls=market_urls,
            token_labels=token_labels,
            constraints=stored_constraints,
            dependencies=stored_dependencies,
        )

        logger.info(
            "Mechanical cluster analysis complete",
            cluster_id=cluster.cluster_id,
            constraint_source=cluster.constraint_source,
            constraints=len(stored_constraints),
        )
        return manifest, False

    async def _analyze_cluster(
        self,
        cluster: MarketCluster,
    ) -> tuple[ConstraintManifest, bool] | None:
        """
        Analyze a single cluster and create a ConstraintManifest.

        Cache layout: only the LLM-derived portion (constraints + dependencies)
        is cached. Cross-exchange equivalencies are re-injected on every run,
        whether the LLM output came from cache or fresh analysis — otherwise
        newly-discovered Limitless pairs would not land on cached clusters
        until the LLM cache expired.

        Returns:
            (manifest, from_cache) on success, None on failure / empty.
        """
        logger.info(
            "Analyzing cluster",
            cluster_id=cluster.cluster_id,
            topic=cluster.topic,
            markets=len(cluster.markets),
            constraint_source=cluster.constraint_source,
        )

        # Mechanical cluster bypass: NegRisk / native / cross-market partitions
        # and monotonic ladders all emit LogicalConstraint(s) with known
        # structure. Skip LogicArchitect and Validator entirely — they'd just
        # re-derive what we already know, at real Pro-tier quota cost.
        if cluster.constraint_source in (
            "negrisk",
            "native_partition",
            "cross_market_partition",
            "monotonic_ladder",
            "conditional_subset",
        ):
            return await self._analyze_mechanical_cluster(cluster)

        cluster_hash = self._compute_cluster_hash(cluster)

        stored_constraints: list[StoredConstraint] | None = None
        stored_dependencies: list[StoredDependency] | None = None
        from_cache = False

        cached_payload = await cache.get_llm_result(cluster_hash)
        if cached_payload:
            try:
                stored_constraints = [
                    StoredConstraint(**c) for c in cached_payload.get("constraints", [])
                ]
                stored_dependencies = [
                    StoredDependency(**d) for d in cached_payload.get("dependencies", [])
                ]
                from_cache = True
                logger.info(
                    "Cache hit - reusing previous LLM analysis",
                    cluster_id=cluster.cluster_id,
                    cluster_hash=cluster_hash[:8],
                )
            except Exception as e:
                logger.warning(
                    "Failed to reconstruct cached constraints; re-analyzing",
                    cluster_id=cluster.cluster_id,
                    error=str(e),
                )
                stored_constraints = None
                stored_dependencies = None
                from_cache = False

        if stored_constraints is None or stored_dependencies is None:
            try:
                if not self._logic_architect:
                    raise RuntimeError("Logic Architect not initialized")
                if not self._validator:
                    raise RuntimeError("Validator not initialized")

                analysis = await self._logic_architect.analyze_cluster(cluster)

                if not analysis.constraints and not analysis.dependencies:
                    logger.info("No dependencies found", cluster_id=cluster.cluster_id)
                    return None

                validated = await self._validator.validate(analysis, cluster=cluster)

                # Validator-bypass guard: if the LLM was unavailable, the
                # validator returns is_valid=True with no real review. The
                # Navigator stamps loaded manifests as is_valid=True, so
                # persisting unreviewed output would drive live trades from
                # raw LLM constraints. Refuse to persist instead.
                if validated.validation_notes.startswith("No-LLM mode"):
                    logger.error(
                        "Refusing to persist cluster: validator ran in No-LLM mode "
                        "(LLM unavailable). Check GEMINI_API_KEY / OpenRouter config.",
                        cluster_id=cluster.cluster_id,
                    )
                    await monitor.emit_pipeline_event(
                        "LOGIC", "validation_skipped",
                        f"Validator was unavailable for '{cluster.topic}' — refusing to persist",
                    )
                    return None

                rejected_count = (
                    len(analysis.constraints) - len(validated.validated_constraints)
                )
                if rejected_count > 0:
                    rejected_ids: list[str] = []
                    for issue in validated.issues:
                        if issue.severity == "error":
                            rejected_ids.extend(issue.affected_constraints)
                    logger.warning(
                        "Validator filtered constraints",
                        cluster_id=cluster.cluster_id,
                        rejected=rejected_count,
                        kept=len(validated.validated_constraints),
                        rejected_ids=rejected_ids[:10],
                    )

                if not validated.is_valid:
                    logger.info(
                        "All constraints rejected by validator — nothing to persist",
                        cluster_id=cluster.cluster_id,
                    )
                    return None

                # Gap 1: cross-event clusters get a higher confidence floor —
                # without upstream mechanical ground truth, low-confidence
                # LLM output would pollute the solver. Polymarket-anchored
                # mechanical clusters aren't filtered this way since they
                # don't go through this code path (mechanical bypass).
                if cluster.constraint_source == "cross_event_logical":
                    conf_floor = config.cross_event_confidence_floor
                    kept_constraints = [
                        c for c in validated.validated_constraints
                        if (c.confidence or 0.0) >= conf_floor
                    ]
                    if len(kept_constraints) < len(validated.validated_constraints):
                        logger.info(
                            "Cross-event confidence floor filtered constraints",
                            cluster_id=cluster.cluster_id,
                            kept=len(kept_constraints),
                            dropped=len(validated.validated_constraints) - len(kept_constraints),
                            floor=conf_floor,
                        )
                    if not kept_constraints:
                        return None
                else:
                    kept_constraints = validated.validated_constraints

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
                    for c in kept_constraints
                ]
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

                cache_payload = {
                    "constraints": [c.model_dump(mode="json") for c in stored_constraints],
                    "dependencies": [d.model_dump(mode="json") for d in stored_dependencies],
                }
                await cache.set_llm_result(
                    cluster_hash, cache_payload, ttl_seconds=self._cache_ttl_seconds()
                )

            except Exception as e:
                logger.error(
                    "Cluster analysis failed",
                    cluster_id=cluster.cluster_id,
                    error=str(e),
                    exc_info=True,
                )
                return None

        # Cross-exchange injection runs on BOTH the cache-hit and fresh paths
        # so newly-discovered Limitless matches always land on the manifest.
        (
            market_ids,
            market_exchanges,
            market_titles,
            market_urls,
            token_labels,
        ) = self._build_market_metadata(cluster)
        self._inject_limitless_equivalencies(
            cluster,
            stored_constraints,
            market_ids,
            market_exchanges,
            market_titles,
            market_urls,
            token_labels,
        )
        self._validate_token_id_shapes(stored_constraints)

        manifest = ConstraintManifest(
            cluster_id=cluster.cluster_id,
            cluster_type=cluster.constraint_source,
            topic=cluster.topic,
            market_ids=market_ids,
            market_exchanges=market_exchanges,
            market_titles=market_titles,
            market_urls=market_urls,
            token_labels=token_labels,
            constraints=stored_constraints,
            dependencies=stored_dependencies,
        )

        logger.info(
            "Cluster analysis complete",
            cluster_id=cluster.cluster_id,
            constraints=len(stored_constraints),
            dependencies=len(stored_dependencies),
            from_cache=from_cache,
        )

        return manifest, from_cache

    @staticmethod
    def _polymarket_url(market: Market) -> str:
        """Best-effort deep link to the Polymarket page for a market.

        Prefers the event slug (ideal for NegRisk clusters where the whole
        event renders on one page), then the market slug, then a search URL
        as a never-dead fallback."""
        from urllib.parse import quote
        if market.event_slug:
            return f"https://polymarket.com/event/{market.event_slug}"
        if market.slug:
            return f"https://polymarket.com/market/{market.slug}"
        return f"https://polymarket.com/markets?_s={quote(market.question or '')}"

    @staticmethod
    def _is_limitless_market(market: Market) -> bool:
        """Heuristic: Limitless markets we synthesize in Gap 5b use the slug
        as market_id and carry YES/NO outcomes whose token_ids end in
        `_yes`/`_no`. Any one of those matchers is enough to be confident."""
        for o in market.outcomes:
            tid = o.token_id or o.outcome_id
            if tid and _LIMITLESS_TOKEN_RE.match(tid):
                return True
        return False

    @staticmethod
    def _build_market_metadata(
        cluster: MarketCluster,
    ) -> tuple[
        list[str],
        dict[str, str],
        dict[str, str],
        dict[str, str],
        dict[str, TokenLabel],
    ]:
        market_ids = [m.market_id for m in cluster.markets]
        # Exchange is per-market, not per-cluster — standalone Limitless
        # clusters emitted by Gap 5b have `_yes`/`_no` token shapes.
        market_exchanges: dict[str, str] = {}
        market_urls: dict[str, str] = {}
        for m in cluster.markets:
            if MapMaker._is_limitless_market(m):
                market_exchanges[m.market_id] = "limitless"
                market_urls[m.market_id] = (
                    f"https://limitless.exchange/markets/{m.slug or m.market_id}"
                )
            else:
                market_exchanges[m.market_id] = "polymarket"
                market_urls[m.market_id] = MapMaker._polymarket_url(m)
        market_titles = {m.market_id: m.question for m in cluster.markets}
        token_labels: dict[str, TokenLabel] = {}
        for m in cluster.markets:
            exchange_name = market_exchanges.get(m.market_id, "polymarket")
            for o in m.outcomes:
                tid = o.token_id or o.outcome_id
                if not tid:
                    continue
                token_labels[tid] = TokenLabel(
                    market_id=m.market_id,
                    market_title=m.question,
                    outcome_name=o.name,
                    exchange=exchange_name,
                )
        return market_ids, market_exchanges, market_titles, market_urls, token_labels

    def _inject_limitless_equivalencies(
        self,
        cluster: MarketCluster,
        stored_constraints: list[StoredConstraint],
        market_ids: list[str],
        market_exchanges: dict[str, str],
        market_titles: dict[str, str],
        market_urls: dict[str, str],
        token_labels: dict[str, TokenLabel],
    ) -> None:
        """Add Limitless YES/NO token coefficients alongside their Polymarket
        counterparts so the solver treats matched pairs as the same security.

        Polarity is resolved by exact outcome name (never by index). When the
        ExchangeMatcher flagged a pair as polarity-inverted (PM-YES ≡ LM-NO),
        the YES↔NO bindings are swapped during injection — the pair is still
        a valid arb leg, just with reversed signs. Each injection is logged so
        an operator can audit which Polymarket↔Limitless pairs the solver was
        treating as equivalent on a given run and whether polarity was flipped."""
        if not self._exchange_matcher:
            return

        # Native-partition clusters never have YES/NO outcomes by definition
        # (their outcomes are candidate names, team names, etc.). Cross-exchange
        # matching doesn't apply — short-circuit quietly.
        if cluster.constraint_source == "native_partition":
            self._limitless_skip_native_partition += 1
            logger.debug(
                "Limitless equivalence N/A for native_partition cluster",
                cluster_id=cluster.cluster_id,
            )
            return

        mapped = self._exchange_matcher.mapped_pairs
        for m in cluster.markets:
            # Defensive guard: never look up Limitless equivalents for a market
            # whose market_id is empty. Historically this caused a cache-poisoning
            # cascade where mapped.get("") returned a phantom Limitless slug and
            # every cluster got the same bogus injection. See parser fix at
            # polymarket_client.py _parse_market for the upstream root cause.
            if not m.market_id:
                logger.warning(
                    "Skipping Limitless injection: market_id is empty",
                    cluster_id=cluster.cluster_id,
                    cluster_source=cluster.constraint_source,
                    question=m.question[:80],
                )
                continue

            l_id = mapped.get(m.market_id)
            if not l_id:
                continue

            if l_id not in market_ids:
                market_ids.append(l_id)
            market_exchanges[l_id] = "limitless"
            market_titles[l_id] = f"{m.question} (Limitless)"
            market_urls[l_id] = f"https://limitless.exchange/markets/{l_id}"

            pm_yes_out = get_yes_outcome(m)
            pm_no_out = get_no_outcome(m)
            if pm_yes_out is None or pm_no_out is None:
                # 3-way moneyline fallback: when the LLM accepted a sports
                # 3-way PM ↔ 2-way LM pair it also persisted a draw_rule +
                # home_outcome_name so the solver can bind synthetic
                # equivalences. If that projection is present we emit it
                # here instead of skipping.
                draw_rule, home_name = self._exchange_matcher.get_pair_projection(
                    m.market_id
                )
                if draw_rule in ("draw_is_no", "double_chance") and home_name:
                    emitted = self._emit_moneyline_equivalence(
                        cluster=cluster,
                        pm_market=m,
                        l_id=l_id,
                        draw_rule=draw_rule,
                        home_outcome_name=home_name,
                        stored_constraints=stored_constraints,
                        token_labels=token_labels,
                    )
                    if emitted:
                        continue
                self._limitless_skip_polarity_mismatch += 1
                logger.info(
                    "Skipping Limitless equivalence: outcomes don't match yes/no",
                    cluster_source=cluster.constraint_source,
                    cluster_id=cluster.cluster_id,
                    market_id=m.market_id,
                    outcome_names=[o.name for o in m.outcomes],
                )
                continue
            pm_yes = pm_yes_out.token_id or pm_yes_out.outcome_id
            pm_no = pm_no_out.token_id or pm_no_out.outcome_id

            l_yes = limitless_yes_token(l_id)
            l_no = limitless_no_token(l_id)

            polarity_aligned = self._exchange_matcher.get_pair_polarity(m.market_id)
            # When polarity is aligned, PM-YES ≡ LM-YES (token_a=l_yes, token_b=l_no).
            # When inverted, PM-YES ≡ LM-NO — swap the Limitless bindings so the
            # coefficient attached to PM-YES lands on the correct Limitless token.
            lm_for_pm_yes = l_yes if polarity_aligned else l_no
            lm_for_pm_no = l_no if polarity_aligned else l_yes

            token_labels[l_yes] = TokenLabel(
                market_id=l_id,
                market_title=f"{m.question} (Limitless)",
                outcome_name="Yes",
                exchange="limitless",
            )
            token_labels[l_no] = TokenLabel(
                market_id=l_id,
                market_title=f"{m.question} (Limitless)",
                outcome_name="No",
                exchange="limitless",
            )

            for c in stored_constraints:
                if pm_yes in c.coefficients:
                    c.coefficients[lm_for_pm_yes] = c.coefficients[pm_yes]
                if pm_no in c.coefficients:
                    c.coefficients[lm_for_pm_no] = c.coefficients[pm_no]

            logger.info(
                "Cross-exchange equivalence injected",
                cluster_id=cluster.cluster_id,
                polymarket_market_id=m.market_id,
                limitless_id=l_id,
                polarity_aligned=polarity_aligned,
                pm_yes=pm_yes,
                pm_no=pm_no,
                lm_bound_to_pm_yes=lm_for_pm_yes,
                lm_bound_to_pm_no=lm_for_pm_no,
            )

    def _emit_moneyline_equivalence(
        self,
        *,
        cluster: MarketCluster,
        pm_market: Market,
        l_id: str,
        draw_rule: str,
        home_outcome_name: str,
        stored_constraints: list[StoredConstraint],
        token_labels: dict[str, TokenLabel],
    ) -> bool:
        """Emit solver equivalence constraints for a 3-way Polymarket moneyline
        matched to a 2-way Limitless binary.

        Encodes two pin-to-zero equalities (each as a pair of >= inequalities)
        so the solver treats the combined Polymarket outcomes as the same
        security as the Limitless YES/NO legs:

        - ``draw_is_no``:   LM_yes ≡ PM_home ;       LM_no ≡ PM_draw + PM_away
        - ``double_chance``: LM_yes ≡ PM_home + PM_draw ; LM_no ≡ PM_away

        Returns True when both equalities were emitted (the caller should skip
        the binary-path injection below). Returns False when we couldn't
        identify home / draw / away outcomes, in which case the caller logs
        the skip as before.
        """
        from polyquant.agents.logic_architect import stable_constraint_id

        home = draw = away = None
        home_norm = home_outcome_name.strip().lower()
        for outcome in pm_market.outcomes:
            name = (outcome.name or "").strip().lower()
            if not name:
                continue
            if home is None and name == home_norm:
                home = outcome
                continue
            if draw is None and name in {"draw", "tie", "drawn", "d", "x"}:
                draw = outcome
                continue
            # First remaining outcome after home/draw is treated as away.
            if away is None and outcome is not home and outcome is not draw:
                away = outcome
        if home is None or draw is None or away is None:
            logger.warning(
                "Moneyline equivalence: couldn't resolve home/draw/away outcomes",
                cluster_id=cluster.cluster_id,
                market_id=pm_market.market_id,
                home_name=home_outcome_name,
                outcomes=[o.name for o in pm_market.outcomes],
            )
            return False

        home_tok = home.token_id or home.outcome_id
        draw_tok = draw.token_id or draw.outcome_id
        away_tok = away.token_id or away.outcome_id
        l_yes = limitless_yes_token(l_id)
        l_no = limitless_no_token(l_id)
        if not all([home_tok, draw_tok, away_tok]):
            logger.warning(
                "Moneyline equivalence: missing token ids on outcomes",
                cluster_id=cluster.cluster_id,
                market_id=pm_market.market_id,
            )
            return False

        token_labels[l_yes] = TokenLabel(
            market_id=l_id,
            market_title=f"{pm_market.question} (Limitless)",
            outcome_name="Yes",
            exchange="limitless",
        )
        token_labels[l_no] = TokenLabel(
            market_id=l_id,
            market_title=f"{pm_market.question} (Limitless)",
            outcome_name="No",
            exchange="limitless",
        )

        if draw_rule == "draw_is_no":
            yes_terms = {home_tok: 1.0}
            no_terms = {draw_tok: 1.0, away_tok: 1.0}
        elif draw_rule == "double_chance":
            yes_terms = {home_tok: 1.0, draw_tok: 1.0}
            no_terms = {away_tok: 1.0}
        else:
            # Defensive — the caller already filtered to these two rules.
            return False

        def _emit_equality(lm_token: str, pm_terms: dict[str, float], label: str) -> None:
            forward_coefs: dict[str, float] = dict(pm_terms)
            forward_coefs[lm_token] = -1.0
            reverse_coefs = {k: -v for k, v in forward_coefs.items()}

            for direction, coefs in (("fwd", forward_coefs), ("rev", reverse_coefs)):
                cid = stable_constraint_id(
                    source_cluster_id=cluster.cluster_id,
                    coefficients=coefs,
                    rhs=0.0,
                    prefix=f"moneyline_{label}_{direction}",
                )
                stored_constraints.append(
                    StoredConstraint(
                        constraint_id=cid,
                        description=(
                            f"[MONEYLINE {draw_rule}] {label} leg equivalence ({direction})"
                        ),
                        coefficients=coefs,
                        rhs=0.0,
                        confidence=0.95,
                        reasoning=(
                            f"Sports 3-way ↔ 2-way projection under draw_rule="
                            f"{draw_rule}: pins the {label} Limitless leg to the "
                            "matching combination of Polymarket outcomes."
                        ),
                        source_markets=[pm_market.market_id, l_id],
                    )
                )

        _emit_equality(l_yes, yes_terms, "yes")
        _emit_equality(l_no, no_terms, "no")

        logger.info(
            "Moneyline equivalence emitted",
            cluster_id=cluster.cluster_id,
            polymarket_market_id=pm_market.market_id,
            limitless_id=l_id,
            draw_rule=draw_rule,
            home=home_outcome_name,
        )
        return True

    @staticmethod
    def _validate_token_id_shapes(stored_constraints: list[StoredConstraint]) -> None:
        """Warn when a coefficient key doesn't match a recognized token_id
        shape (Polymarket decimal or Limitless suffixed). Anything that slips
        past these patterns is almost certainly an unmapped placeholder that
        the solver will silently ignore."""
        for c in stored_constraints:
            invalid_keys = [
                k for k in c.coefficients
                if not (_POLY_TOKEN_RE.match(k) or _LIMITLESS_TOKEN_RE.match(k))
            ]
            if invalid_keys:
                logger.warning(
                    "Constraint has unrecognized token_id format",
                    constraint_id=c.constraint_id,
                    invalid_keys=invalid_keys,
                )


