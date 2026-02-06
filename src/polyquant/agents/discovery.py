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

import json
from datetime import datetime
from typing import Any

import google.generativeai as genai
from pydantic import BaseModel, Field

from polyquant.data import Market, PolymarketClient
from polyquant.utils import config, get_logger

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
    CLUSTERING_PROMPT = """You are an expert at analyzing prediction markets and identifying logical relationships.

Given a list of prediction markets, your task is to:
1. Group them into clusters by topic (e.g., "US Elections", "Sports", "Crypto")
2. For each cluster, identify potential logical dependencies between markets

A logical dependency exists when the outcome of one market constrains or implies something about another.
Examples:
- "Will Trump win?" and "Will a Republican win?" - if Trump wins, Republican wins
- "Will BTC hit 100K?" and "Will BTC hit 50K?" - if 100K, then 50K must also happen

Return your analysis as JSON with this structure:
{
    "clusters": [
        {
            "topic": "Topic description",
            "market_ids": ["id1", "id2"],
            "potential_dependencies": [
                "If market X outcome A happens, then market Y must have outcome B"
            ]
        }
    ]
}

Focus on clusters where there are likely logical constraints between markets."""

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
        genai.configure(api_key=config.gemini_api_key.get_secret_value())
        self._genai_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager - cleanup."""
        if self._polymarket:
            await self._polymarket.__aexit__(*args)
    
    async def scan_markets(
        self,
        limit: int = 100,
        min_liquidity: float = 1000.0,
        skip_processed: bool = True,
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
            
        Returns:
            List of MarketCluster objects
        """
        if not self._polymarket:
            raise RuntimeError("DiscoveryAgent not initialized. Use 'async with discovery:'")
        
        logger.info(
            "Scanning markets",
            limit=limit,
            min_liquidity=min_liquidity,
        )
        
        # Step 1: Fetch markets from Polymarket
        markets = await self._polymarket.get_active_markets(
            limit=limit,
            min_liquidity=min_liquidity,
        )
        
        if not markets:
            logger.info("No markets found matching criteria")
            return []
        
        # Step 2: Filter out already-processed markets
        if skip_processed:
            markets = [
                m for m in markets
                if m.market_id not in self._processed_markets
            ]
            
            if not markets:
                logger.info("All markets already processed")
                return []
        
        logger.info(f"Found {len(markets)} markets to analyze")
        
        # Step 3: Use Gemini to cluster markets
        clusters = await self._cluster_markets(markets)
        
        # Step 4: Mark markets as processed
        for market in markets:
            self._processed_markets.add(market.market_id)
        
        logger.info(
            "Market scan complete",
            clusters_found=len(clusters),
            markets_processed=len(markets),
        )
        
        return clusters
    
    async def _cluster_markets(self, markets: list[Market]) -> list[MarketCluster]:
        """
        Use Gemini to cluster markets by topic.
        """
        if not self._genai_model:
            raise RuntimeError("Gemini not initialized")
        
        # Format markets for the prompt
        market_descriptions = "\n".join([
            f"ID: {m.market_id}\nQuestion: {m.question}\nDescription: {m.description[:200] if m.description else 'N/A'}\n"
            for m in markets
        ])
        
        # Create market lookup for quick access
        market_lookup = {m.market_id: m for m in markets}
        
        logger.debug("Calling Gemini for market clustering")
        
        try:
            response = self._genai_model.generate_content(
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
