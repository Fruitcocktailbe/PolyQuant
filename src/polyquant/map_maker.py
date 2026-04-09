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
from datetime import datetime
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
from polyquant.utils import config, get_logger
from polyquant.utils.cache import cache

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
        limit: int = 500,
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
        start_time = datetime.utcnow()

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
            logger.info("Phase 1: Discovery - Scanning markets...")
            
            if not self._discovery:
                raise RuntimeError("Discovery agent not initialized")
            
            clusters = await self._discovery.scan_markets(
                limit=limit,
                min_liquidity=min_liquidity,
                skip_processed=skip_processed,
            )
            
            # Record in monitor for UI
            from polyquant.api.server import monitor
            asyncio.create_task(monitor.update_status(
                clusters=[{
                    "id": c.cluster_id,
                    "topic": c.topic,
                    "count": len(c.markets),
                    "status": "pending_analysis"
                } for c in clusters]
            ))

            logger.info(f"Discovered {len(clusters)} clusters")
            results["discovery"] = {
                "clusters_found": len(clusters),
                "total_markets": sum(len(c.markets) for c in clusters),
            }
            
            if not clusters:
                results["status"] = "complete"
                results["outcome"] = "no_clusters"
                return results
            
            # ================================================================
            # PHASE 2 & 3: REASONING + VALIDATION (per cluster)
            # ================================================================
            logger.info("Phase 2-3: Reasoning & Validation...")

            manifests_saved = 0
            total_constraints = 0
            total_dependencies = 0
            cache_hits = 0

            total_clusters = len(clusters)
            for idx, cluster in enumerate(clusters, 1):
                # Progress logging
                logger.info(
                    f"Processing cluster {idx}/{total_clusters}",
                    cluster_id=cluster.cluster_id,
                    topic=cluster.topic,
                )

                manifest = await self._analyze_cluster(cluster)

                if manifest:
                    # Check if this was a cache hit
                    if hasattr(manifest, '_from_cache') and manifest._from_cache:
                        cache_hits += 1

                    if manifest.constraint_count > 0:
                        if self._store:
                            await self._store.save_manifest(manifest)
                        manifests_saved += 1
                        total_constraints += manifest.constraint_count
                        total_dependencies += manifest.dependency_count

                # Progress percentage
                progress_pct = (idx / total_clusters) * 100
                logger.info(f"Progress: {progress_pct:.1f}% complete")
            
            results["analysis"] = {
                "manifests_saved": manifests_saved,
                "total_constraints": total_constraints,
                "total_dependencies": total_dependencies,
                "cache_hits": cache_hits,
                "cache_hit_rate": f"{(cache_hits / total_clusters * 100):.1f}%" if total_clusters > 0 else "0%",
            }
            
            results["status"] = "complete"
            results["outcome"] = "success"
            
        except Exception as e:
            logger.error("Map building failed", error=str(e))
            results["status"] = "error"
            results["error"] = str(e)
        
        # Record timing
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        results["elapsed_seconds"] = elapsed
        
        logger.info(
            "Map building complete",
            status=results["status"],
            elapsed=elapsed,
        )
        
        return results
    
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
                manifest = ConstraintManifest(
                    cluster_id=cluster.cluster_id,
                    topic=cluster.topic,
                    market_ids=[m.market_id for m in cluster.markets],
                    constraints=[
                        StoredConstraint(**c) for c in cached_result.get("constraints", [])
                    ],
                    dependencies=[
                        StoredDependency(**d) for d in cached_result.get("dependencies", [])
                    ],
                )
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
                logger.warning(
                    "Validation failed",
                    cluster_id=cluster.cluster_id,
                    issues=len(validated.issues),
                )
                # Still save partial constraints that passed
            
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
                    h = await self._polymarket.get_history(mid)
                    return [float(p.get("p", 0)) for p in h]
                
                # CorrelationAgent might need to be async or we fetch here
                # Let's assume we can pass the provider and it handles it
                # Logic: analyze_pairs(cluster.markets, history_provider)
                signals = await self._correlation_agent.analyze_pairs(
                    cluster.markets, 
                    history_provider
                )
                correlations = [s.model_dump(mode="json") for s in signals]

            # Create manifest
            manifest = ConstraintManifest(
                cluster_id=cluster.cluster_id,
                topic=cluster.topic,
                market_ids=[m.market_id for m in cluster.markets],
                constraints=stored_constraints,
                dependencies=stored_dependencies,
                correlations=correlations,
            )

            # Cache the result for future runs (5 minute TTL)
            cache_data = {
                "constraints": [
                    {
                        "constraint_id": c.constraint_id,
                        "description": c.description,
                        "coefficients": c.coefficients,
                        "rhs": c.rhs,
                        "confidence": c.confidence,
                        "reasoning": c.reasoning,
                        "source_markets": c.source_markets,
                    }
                    for c in stored_constraints
                ],
                "dependencies": [
                    {
                        "source_market_id": d.source_market_id,
                        "source_outcome": d.source_outcome,
                        "target_market_id": d.target_market_id,
                        "target_outcome": d.target_outcome,
                        "relationship": d.relationship,
                        "confidence": d.confidence,
                    }
                    for d in stored_dependencies
                ],
            }
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
    print("""
    ===============================================================
                      PolyQuant Map Maker
              Offline Constraint Analysis Engine
    ===============================================================
    """)
    
    async with MapMaker() as map_maker:
        result = await map_maker.build_map(
            limit=500,
            min_liquidity=1000,
        )
        
        print("\n" + "=" * 60)
        print("Map Building Result:")
        print("=" * 60)
        
        for key, value in result.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
