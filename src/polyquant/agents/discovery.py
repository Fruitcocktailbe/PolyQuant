"""
Discovery Agent for PolyQuant 2.0 - Phase 1

The Discovery Agent is the first phase in the PolyQuant pipeline. Its job is
to continuously scan Polymarket for new markets and identify potential
cross-market dependencies that might indicate arbitrage opportunities.

RESPONSIBILITIES:
-----------------
1. Monitor Polymarket for newly created markets
2. Scan news feeds and Twitter for relevant events
3. Identify markets that might be logically related
4. Pass promising market pairs to the Logic Architect (Phase 2)

HOW IT WORKS:
-------------
1. Fetch active markets from Polymarket API
2. For each market, use GPT-4o to analyze:
   - What real-world event does this market represent?
   - What other markets might be logically connected?
   - Are there any news events that could affect multiple markets?
3. Group markets into clusters that share logical connections
4. Send clusters to Logic Architect for formal dependency analysis

DESIGN DECISIONS:
-----------------
- Uses GPT-4o for its strong reasoning and multimodal capabilities
- Caches results to avoid re-analyzing unchanged markets
- Runs as an async loop for continuous monitoring
- Implements rate limiting to stay within API limits

USAGE:
------
    agent = DiscoveryAgent()
    
    # Run a single scan
    clusters = await agent.scan_markets()
    
    # Or run continuously
    async for cluster in agent.monitor():
        print(f"Found {len(cluster.markets)} related markets")
"""

import asyncio
from datetime import datetime, timedelta
from typing import AsyncIterator

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from polyquant.data import Market, PolymarketClient, get_polymarket_client
from polyquant.utils import config, get_logger

logger = get_logger(__name__)


class MarketCluster(BaseModel):
    """
    A group of markets that may have logical dependencies.
    
    The Discovery Agent groups markets that share common topics or events,
    allowing the Logic Architect to analyze them for formal dependencies.
    
    Attributes:
        cluster_id: Unique identifier for tracking
        markets: List of markets in this cluster
        topic: Common topic or theme (e.g., "2024 US Election")
        keywords: Extracted keywords for categorization
        confidence: How confident the agent is these are related (0-1)
        created_at: When this cluster was identified
    """
    cluster_id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()))
    markets: list[Market] = Field(default_factory=list)
    topic: str = Field(default="Unknown")
    keywords: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DiscoveryAgent:
    """
    Phase 1: Market Discovery and Initial Clustering
    
    The Discovery Agent scans Polymarket for new opportunities and clusters
    related markets together for deeper analysis.
    
    Architecture:
    - Uses GPT-4o for semantic understanding of market topics
    - Async HTTP for efficient API calls
    - In-memory cache for processed markets (TODO: Redis for persistence)
    
    Example:
        agent = DiscoveryAgent()
        
        async with agent:
            # Single scan
            clusters = await agent.scan_markets()
            for cluster in clusters:
                print(f"Topic: {cluster.topic}, Markets: {len(cluster.markets)}")
            
            # Continuous monitoring
            async for cluster in agent.monitor(interval_seconds=60):
                await process_cluster(cluster)
    """
    
    # System prompt for GPT-4o - explains the task and expected output
    CLUSTERING_PROMPT = """You are a market analysis agent for Polymarket prediction markets.

Your task is to analyze market questions and identify logical groupings.

For each market question, determine:
1. The core topic/event (e.g., "2024 US Presidential Election")
2. Key entities involved (e.g., "Trump", "Biden", "Pennsylvania")
3. Potential logical dependencies with other markets

OUTPUT FORMAT (JSON):
{
    "topic": "Main topic or event",
    "keywords": ["keyword1", "keyword2", ...],
    "related_market_indices": [0, 2, 5],  // indices of related markets from input
    "reasoning": "Brief explanation of why these are related"
}

Focus on finding markets where:
- One outcome logically implies something about another market
- Events share common underlying factors
- Resolution of one market provides information about another"""

    def __init__(self):
        """Initialize the Discovery Agent with API clients."""
        self._openai: AsyncOpenAI | None = None
        self._polymarket: PolymarketClient | None = None
        self._processed_markets: set[str] = set()  # Cache of already-analyzed market IDs
        self._last_scan: datetime | None = None
        
        logger.info("DiscoveryAgent initialized")
    
    async def __aenter__(self) -> "DiscoveryAgent":
        """Async context manager - initialize API clients."""
        self._openai = AsyncOpenAI(api_key=config.openai_api_key.get_secret_value())
        self._polymarket = PolymarketClient()
        await self._polymarket.__aenter__()
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager - cleanup clients."""
        if self._polymarket:
            await self._polymarket.__aexit__(*args)
        # OpenAI client doesn't need explicit cleanup
    
    async def scan_markets(
        self,
        limit: int = 100,
        min_liquidity: float = 1000.0,
        skip_processed: bool = True,
    ) -> list[MarketCluster]:
        """
        Perform a single scan of Polymarket and cluster related markets.
        
        This is the main method for batch processing. For continuous
        monitoring, use the monitor() async generator instead.
        
        Algorithm:
        1. Fetch active markets from Polymarket
        2. Filter out already-processed markets (if skip_processed=True)
        3. Use GPT-4o to analyze and cluster markets by topic
        4. Return list of MarketCluster objects
        
        Args:
            limit: Maximum markets to fetch from API
            min_liquidity: Minimum liquidity threshold in USD
            skip_processed: Whether to skip previously analyzed markets
            
        Returns:
            List of MarketCluster objects with related markets grouped
        """
        if not self._polymarket:
            raise RuntimeError("Agent not initialized. Use 'async with agent:' context.")
        
        logger.info(
            "Starting market scan",
            limit=limit,
            min_liquidity=min_liquidity,
            skip_processed=skip_processed,
        )
        
        # Step 1: Fetch markets from Polymarket
        markets = await self._polymarket.get_active_markets(
            limit=limit,
            min_liquidity=min_liquidity,
        )
        
        logger.info("Fetched markets", count=len(markets))
        
        # Step 2: Filter out already-processed markets
        if skip_processed:
            new_markets = [
                m for m in markets
                if m.market_id not in self._processed_markets
            ]
            logger.info(
                "Filtered to new markets",
                total=len(markets),
                new=len(new_markets),
            )
            markets = new_markets
        
        if not markets:
            logger.info("No new markets to analyze")
            return []
        
        # Step 3: Cluster markets using GPT-4o
        clusters = await self._cluster_markets(markets)
        
        # Step 4: Mark markets as processed
        for market in markets:
            self._processed_markets.add(market.market_id)
        
        self._last_scan = datetime.utcnow()
        
        logger.info(
            "Scan complete",
            markets_analyzed=len(markets),
            clusters_found=len(clusters),
        )
        
        return clusters
    
    async def monitor(
        self,
        interval_seconds: int = 60,
        max_iterations: int | None = None,
    ) -> AsyncIterator[MarketCluster]:
        """
        Continuously monitor Polymarket for new opportunities.
        
        This is an async generator that yields MarketCluster objects as
        they are discovered. Use this for production deployment.
        
        Args:
            interval_seconds: Seconds between scans
            max_iterations: Stop after this many scans (None = infinite)
            
        Yields:
            MarketCluster objects as they are discovered
            
        Example:
            async for cluster in agent.monitor(interval_seconds=30):
                await send_to_logic_architect(cluster)
        """
        iteration = 0
        
        logger.info(
            "Starting continuous monitoring",
            interval=interval_seconds,
            max_iterations=max_iterations,
        )
        
        while max_iterations is None or iteration < max_iterations:
            try:
                clusters = await self.scan_markets()
                
                for cluster in clusters:
                    yield cluster
                    
            except Exception as e:
                logger.error(
                    "Error during scan",
                    error=str(e),
                    iteration=iteration,
                )
                # Continue monitoring despite errors
            
            iteration += 1
            
            # Wait before next scan
            logger.debug("Waiting before next scan", seconds=interval_seconds)
            await asyncio.sleep(interval_seconds)
    
    async def _cluster_markets(self, markets: list[Market]) -> list[MarketCluster]:
        """
        Use GPT-4o to cluster markets by topic and potential dependencies.
        
        This is the core AI component of the Discovery Agent. It analyzes
        market questions to find logical groupings.
        
        Args:
            markets: List of markets to analyze
            
        Returns:
            List of MarketCluster objects
        """
        if not self._openai or not markets:
            return []
        
        # Prepare market data for the prompt
        market_descriptions = "\n".join(
            f"{i}. [{m.market_id}] {m.question}"
            for i, m in enumerate(markets)
        )
        
        logger.debug("Calling GPT-4o for clustering", market_count=len(markets))
        
        try:
            response = await self._openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.CLUSTERING_PROMPT},
                    {
                        "role": "user",
                        "content": f"Analyze these markets and group related ones:\n\n{market_descriptions}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000,
            )
            
            # Parse the response
            import json
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            
            # Handle both single cluster and multiple clusters in response
            if "clusters" in result:
                cluster_datas = result["clusters"]
            else:
                cluster_datas = [result]
            
            clusters = []
            for cluster_data in cluster_datas:
                # Get the markets in this cluster
                indices = cluster_data.get("related_market_indices", [])
                cluster_markets = [
                    markets[i] for i in indices
                    if 0 <= i < len(markets)
                ]
                
                if cluster_markets:
                    cluster = MarketCluster(
                        markets=cluster_markets,
                        topic=cluster_data.get("topic", "Unknown"),
                        keywords=cluster_data.get("keywords", []),
                        confidence=cluster_data.get("confidence", 0.5),
                    )
                    clusters.append(cluster)
            
            logger.debug("Clustering complete", clusters_found=len(clusters))
            return clusters
            
        except Exception as e:
            logger.error("GPT-4o clustering failed", error=str(e))
            
            # Fallback: return each market as its own cluster
            return [
                MarketCluster(markets=[m], topic=m.question[:50])
                for m in markets
            ]
    
    def reset_cache(self) -> None:
        """Clear the processed markets cache to re-analyze all markets."""
        self._processed_markets.clear()
        logger.info("Market cache cleared")


# Convenience function for quick scans
async def discover_markets(
    limit: int = 100,
    min_liquidity: float = 1000.0,
) -> list[MarketCluster]:
    """
    Convenience function to perform a single market scan.
    
    Creates and manages the agent lifecycle automatically.
    
    Args:
        limit: Maximum markets to fetch
        min_liquidity: Minimum liquidity in USD
        
    Returns:
        List of MarketCluster objects
        
    Example:
        clusters = await discover_markets(limit=50)
        for cluster in clusters:
            print(f"Found: {cluster.topic}")
    """
    async with DiscoveryAgent() as agent:
        return await agent.scan_markets(limit=limit, min_liquidity=min_liquidity)
