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
)
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

        Handles NegRisk, native_partition, and cross_market_partition clusters.
        The constraint shape is fully determined by the cluster's market set —
        no semantic reasoning is needed.

        Cross-exchange equivalencies are still injected on top, so Limitless
        matches land on these clusters too.
        """
        from polyquant.agents.logic_architect import build_partition_constraint

        logic_constraint = build_partition_constraint(cluster)
        if logic_constraint is None:
            logger.warning(
                "Mechanical cluster produced no constraint — skipping",
                cluster_id=cluster.cluster_id,
                constraint_source=cluster.constraint_source,
                market_count=len(cluster.markets),
            )
            return None

        stored_constraints: list[StoredConstraint] = [
            StoredConstraint(
                constraint_id=logic_constraint.constraint_id,
                description=logic_constraint.description,
                coefficients=logic_constraint.coefficients,
                rhs=logic_constraint.rhs,
                confidence=logic_constraint.confidence,
                reasoning=logic_constraint.reasoning,
                source_markets=logic_constraint.source_markets,
            )
        ]
        stored_dependencies: list[StoredDependency] = []

        market_ids, market_exchanges, market_titles = self._build_market_metadata(cluster)
        self._inject_limitless_equivalencies(
            cluster, stored_constraints, market_ids, market_exchanges, market_titles
        )
        self._validate_token_id_shapes(stored_constraints)

        manifest = ConstraintManifest(
            cluster_id=cluster.cluster_id,
            topic=cluster.topic,
            market_ids=market_ids,
            market_exchanges=market_exchanges,
            market_titles=market_titles,
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
        # all emit a single LogicalConstraint with known structure. Skip
        # LogicArchitect and Validator entirely — they'd just re-derive what
        # we already know, at real Pro-tier quota cost.
        if cluster.constraint_source in (
            "negrisk",
            "native_partition",
            "cross_market_partition",
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

                validated = await self._validator.validate(analysis)

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
        market_ids, market_exchanges, market_titles = self._build_market_metadata(cluster)
        self._inject_limitless_equivalencies(
            cluster, stored_constraints, market_ids, market_exchanges, market_titles
        )
        self._validate_token_id_shapes(stored_constraints)

        manifest = ConstraintManifest(
            cluster_id=cluster.cluster_id,
            topic=cluster.topic,
            market_ids=market_ids,
            market_exchanges=market_exchanges,
            market_titles=market_titles,
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
    def _build_market_metadata(
        cluster: MarketCluster,
    ) -> tuple[list[str], dict[str, str], dict[str, str]]:
        market_ids = [m.market_id for m in cluster.markets]
        market_exchanges = {m.market_id: "polymarket" for m in cluster.markets}
        market_titles = {m.market_id: m.question for m in cluster.markets}
        return market_ids, market_exchanges, market_titles

    def _inject_limitless_equivalencies(
        self,
        cluster: MarketCluster,
        stored_constraints: list[StoredConstraint],
        market_ids: list[str],
        market_exchanges: dict[str, str],
        market_titles: dict[str, str],
    ) -> None:
        """Add Limitless YES/NO token coefficients alongside their Polymarket
        counterparts so the solver treats matched pairs as the same security.

        Polarity is resolved by exact outcome name (never by index). Each
        injection is logged so an operator can audit which Polymarket↔Limitless
        pairs the solver was treating as equivalent on a given run."""
        if not self._exchange_matcher:
            return

        mapped = self._exchange_matcher.mapped_pairs
        for m in cluster.markets:
            l_id = mapped.get(m.market_id)
            if not l_id:
                continue

            if l_id not in market_ids:
                market_ids.append(l_id)
            market_exchanges[l_id] = "limitless"
            market_titles[l_id] = f"{m.question} (Limitless)"

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

            l_yes = limitless_yes_token(l_id)
            l_no = limitless_no_token(l_id)

            for c in stored_constraints:
                if pm_yes in c.coefficients:
                    c.coefficients[l_yes] = c.coefficients[pm_yes]
                if pm_no in c.coefficients:
                    c.coefficients[l_no] = c.coefficients[pm_no]

            logger.info(
                "Cross-exchange equivalence injected",
                cluster_id=cluster.cluster_id,
                polymarket_market_id=m.market_id,
                limitless_id=l_id,
                pm_yes=pm_yes,
                pm_no=pm_no,
                l_yes=l_yes,
                l_no=l_no,
            )

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


