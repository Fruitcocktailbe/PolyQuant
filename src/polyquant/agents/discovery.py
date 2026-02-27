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

import google.generativeai as genai
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
    
    # System prompt for Gemini's clustering task
    # System prompt for Gemini's clustering task
    # OPTIMIZED for "Event Clustering" to find conflicting/correlated markets
    CLUSTERING_PROMPT = """You are an expert at analyzing prediction markets and identifying "Event Clusters".

YOUR GOAL:
Group markets that are about the **SAME underlying real-world event**, even if they are phrased differently.
We want to find markets that might CONFLICT or CORRELATE with each other.

### 1. WHAT IS AN EVENT CLUSTER?
An event cluster is a set of markets whose outcomes depend on the same future reality.
- **Good Cluster (Same Event)**: "Will Trump win 2024?" + "Will a Republican win 2024?" + "Winner of 2024 US Election"
- **Good Cluster (Dependent Events)**: "Will BTC hit 100k?" + "Will ETH hit 10k?" (Crypto Market Cycle)
- **Bad Cluster (Just a Topic)**: "Will Trump win?" + "Will Biden have ice cream?" (Same person, unrelated events)

### 2. INPUT DATA
You will be given a list of markets. Each has:
- `ID`: Unique identifier
- `Question`: The main question
- `Volume/Liquidity`: Use this to prioritize! High volume markets are the "anchors" of a cluster.

### 3. YOUR TASK
1. Scan the list for related markets.
2. Group them into clusters.
3. For each cluster, identify **Potential Logical Dependencies**.
   - *Example*: "If Market A resolves YES, Market B MUST resolve NO" (Mutually Exclusive)
   - *Example*: "If Market A resolves YES, Market B MUST resolve YES" (Subset/Implication)

### 4. OUTPUT JSON
Return a JSON object with this EXACT structure:
{
    "clusters": [
        {
            "topic": "Short accurate description of the event (e.g. 'US Election 2024')",
            "market_ids": ["id1", "id2", "id3"],
            "potential_dependencies": [
                "Market id1 (Trump Win) implies Market id2 (GOP Win)",
                "Market id1 and Market id3 are mutually exclusive"
            ],
            "confidence": 0.9  // How sure are you these are related? (0.0 to 1.0)
        }
    ]
}

*CRITICAL*: Do not include markets in a cluster if they are only loosely related by topic but have no logical connection. We want ARBITRAGE opportunities, not just categories.
"""

    def __init__(self):
        """Initialize the Discovery Agent."""
        self._polymarket: PolymarketClient | None = None
        self._genai_model = None
        self._processed_markets: set[str] = set()
        
        logger.info("DiscoveryAgent initialized")
    
    async def __aenter__(self) -> "DiscoveryAgent":
        """Async context manager - initialize clients."""
        # Initialize Polymarket client
        self._polymarket = PolymarketClient()
        await self._polymarket.__aenter__()
        
        # Configure Gemini
        api_key = config.gemini_api_key.get_secret_value()
        if not api_key or "your-" in api_key:
            logger.warning("Gemini API key not set - running in No-LLM mode")
            self._genai_model = None
        else:
            genai.configure(api_key=api_key)
            self._genai_model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
        
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
        
        A zombie market has:
        - Extreme prices (< 0.02 or > 0.98) suggesting resolution
        - All outcomes near 0 or 1 (already resolved)
        
        Args:
            market: Market to check
            
        Returns:
            True if zombie (should be filtered), False if valid
        """
        for outcome in market.outcomes:
            # Extreme prices suggest resolution or broken market
            if outcome.price < 0.02 or outcome.price > 0.98:
                return True
        return False

    
    async def scan_markets(
        self,
        limit: int = 100,
        min_liquidity: float = 1000.0,
        skip_processed: bool = True,
        start_offset: int = 0,
    ) -> list[MarketCluster]:
        """
        Scan Polymarket for markets and cluster them by topic.
        
        This is the main entry point for the Discovery Agent. It:
        1. Fetches active markets from Polymarket
        2. Filters by liquidity and processed status
        3. Uses Gemini to cluster by topic
        4. Returns clusters for the Logic Architect
        
        Args:
            limit: Maximum number of markets to fetch
            min_liquidity: Minimum liquidity threshold in dollars
            skip_processed: Skip markets we've already analyzed
            start_offset: offset to start scanning from
            
        Returns:
            List of MarketCluster objects
        """
        if not self._polymarket:
            raise RuntimeError("DiscoveryAgent not initialized. Use 'async with discovery:'")
        
        logger.info(
            "Scanning markets",
            limit=limit,
            offset=start_offset,
            min_liquidity=min_liquidity,
        )
        
        all_markets: list[Market] = []
        offset = start_offset
        batch_size = 100  # API usually limits per request
        
        # Step 1: Fetch markets from Polymarket (with pagination)
        while len(all_markets) < limit:
            # Calculate how many more to fetch
            remaining = limit - len(all_markets)
            fetch_limit = min(batch_size, remaining)
            
            batch = await self._polymarket.get_active_markets(
                limit=fetch_limit,
                offset=offset,
                min_liquidity=min_liquidity,
            )
            
            if not batch:
                break
            
            # Filter out zombie markets (extreme prices = resolution artifacts)
            valid_markets = [m for m in batch if not self._is_zombie_market(m)]
            zombie_count = len(batch) - len(valid_markets)
            if zombie_count > 0:
                logger.debug("Filtered zombie markets", count=zombie_count)
                
            all_markets.extend(valid_markets)
            offset += len(batch)
            
            # Optimization: If we got fewer than requested, we likely hit the end
            if len(batch) < fetch_limit:
                break
        
        if not all_markets:
            logger.info("No markets found matching criteria")
            return []
        
        # Step 2: Filter out already-processed markets (checking Redis)
        markets_to_process = []
        
        if skip_processed:
            for m in all_markets:
                # Check local cache first
                if m.market_id in self._processed_markets:
                    continue
                    
                # Check Redis cache
                if await cache.is_market_processed(m.market_id):
                    self._processed_markets.add(m.market_id) # Update local cache
                    continue
                    
                markets_to_process.append(m)
        else:
            markets_to_process = all_markets
            
        if not markets_to_process:
            logger.info("All scanned markets already processed")
            return []
        
        logger.info(f"Found {len(markets_to_process)} new markets to analyze")
        
        # Step 3: Optimization - Group NegRisk markets automatically
        negrisk_groups: dict[str, list[Market]] = {}
        other_markets: list[Market] = []
        
        negrisk_count: int = 0
        for m in markets_to_process:
            if m.negrisk and m.group_id:
                if m.group_id not in negrisk_groups:
                    negrisk_groups[m.group_id] = []
                negrisk_groups[m.group_id].append(m)
                negrisk_count += 1
            else:
                other_markets.append(m)
        
        logger.info(
            "NegRisk Grouping Debug", 
            total_markets=len(markets_to_process), 
            negrisk_found=negrisk_count,
            groups_formed=len(negrisk_groups)
        )
                
        clusters: list[MarketCluster] = []
        
        # Process NegRisk groups (High Priority)
        for group_id, group_markets in negrisk_groups.items():
            total_price = 0.0
            for m in group_markets:
                 if m.outcomes:
                     total_price += m.outcomes[0].price
            
            cluster_id = f"negrisk_{group_id}"
            clusters.append(
                MarketCluster(
                    cluster_id=cluster_id,
                    topic=f"NegRisk Group {group_id} (Sum: {total_price:.2f})",
                    markets=group_markets,
                    potential_dependencies=[],
                )
            )
            
        # Step 4: Cluster remaining markets using Gemini
        if other_markets:
            logger.info(f"Clustering {len(other_markets)} remaining markets with Gemini...")
            generated_clusters = await self._cluster_markets(other_markets)
            clusters.extend(generated_clusters)
        
        # Step 5: Mark markets as processed in Redis
        for market in markets_to_process:
            self._processed_markets.add(market.market_id)
            await cache.mark_market_processed(market.market_id)
        
        logger.info(
            "Market scan complete",
            clusters_found=len(clusters),
            markets_processed=len(markets_to_process),
        )
        
        return clusters
    
    async def _cluster_markets(self, markets: list[Market]) -> list[MarketCluster]:
        """
        Use Gemini to cluster markets by topic.
        """
        if not self._genai_model:
            logger.info("Gemini not initialized, skipping clustering")
            return [
                MarketCluster(
                    topic="Uncategorized (No AI)",
                    markets=markets,
                    potential_dependencies=[],
                )
            ]
        
        # Format markets for the prompt
        market_descriptions = "\n".join([
            f"ID: {m.market_id}\nQuestion: {m.question}\nDescription: {m.description[:200] if m.description else 'N/A'}\n"
            for m in markets
        ])
        
        # Create market lookup for quick access
        market_lookup = {m.market_id: m for m in markets}
        
        logger.debug("Calling Gemini for market clustering")

        try:
            # Use asyncio.to_thread to avoid blocking the event loop
            response = await asyncio.to_thread(
                self._genai_model.generate_content,
                f"{self.CLUSTERING_PROMPT}\n\nMarkets to analyze:\n{market_descriptions}"
            )
            
            # Parse response
            response_text = response.text
            result = json.loads(response_text)
            
        except Exception as e:
            logger.error(f"Gemini clustering failed: {e}")
            # Fallback: return all markets as a single cluster
            return [
                MarketCluster(
                    topic="Uncategorized",
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
            
            if cluster_markets:  # Only include non-empty clusters
                clusters.append(
                    MarketCluster(
                        topic=cluster_data.get("topic", "Unknown"),
                        markets=cluster_markets,
                        potential_dependencies=cluster_data.get("potential_dependencies", []),
                    )
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
    min_liquidity: float = 1000.0,
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
