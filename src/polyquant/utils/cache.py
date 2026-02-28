"""
Redis Cache Manager for PolyQuant 2.0

This module handles all Redis interactions, providing a clean interface for:
1. De-duplicating processed markets (saving LLM costs)
2. Persisting Kill Switch state (safety)
3. Caching optimization results (performance)

USAGE:
------
    from polyquant.utils.cache import cache
    
    # Check if market already analyzed
    if await cache.is_market_processed("market_123"):
        continue
        
    # Save safety state
    await cache.save_kill_switch_state(capital=10000, drawdown=0.05)
"""

import json
from datetime import timedelta
from typing import Any, Optional

import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError

from polyquant.utils import config, get_logger

logger = get_logger(__name__)


class RedisCache:
    """
    Async Redis wrapper for PolyQuant state management.
    """
    
    def __init__(self):
        self._redis_url = config.redis_url
        self._client: redis.Redis | None = None
        self._is_connected = False
        
        # Key prefixes
        self.MARKET_PREFIX = "polyquant:market:"
        self.KILL_SWITCH_KEY = "polyquant:kill_switch:state"
        self.SOLVER_PREFIX = "polyquant:solver:"
        self.LLM_RESULT_PREFIX = "polyquant:llm:"
        self.MANIFEST_VERSION_KEY = "polyquant:manifest:version"
        
    async def connect(self) -> bool:
        """Establish Redis connection."""
        if self._is_connected:
            return True
            
        try:
            logger.info("Connecting to Redis", url=self._redis_url)
            self._client = redis.from_url(
                self._redis_url, 
                encoding="utf-8", 
                decode_responses=True,
                socket_timeout=5.0
            )
            await self._client.ping()
            self._is_connected = True
            logger.info("Redis connected successfully")
            return True
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection failed: {e}")
            self._is_connected = False
            return False
            
    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._is_connected = False
            
    async def is_market_processed(self, market_id: str) -> bool:
        """Check if a market has already been processed by the Discovery Agent."""
        if not self._is_connected or not self._client:
            return False
            
        key = f"{self.MARKET_PREFIX}{market_id}"
        return await self._client.exists(key)
        
    async def mark_market_processed(self, market_id: str, ttl_hours: int = 24) -> None:
        """Mark a market as processed to avoid re-analysis."""
        if not self._is_connected or not self._client:
            return

        key = f"{self.MARKET_PREFIX}{market_id}"
        # Set a value (timestamp) and expiry
        # Markets change, so we should re-analyze them periodically (e.g. 24h)
        await self._client.setex(key, timedelta(hours=ttl_hours), "processed")

    async def are_markets_processed(self, market_ids: list[str]) -> dict[str, bool]:
        """
        Batch check if markets have been processed.

        This is significantly faster than calling is_market_processed() N times.
        Uses Redis pipelining for efficiency.

        Args:
            market_ids: List of market IDs to check

        Returns:
            Dictionary mapping market_id -> is_processed

        Example:
            results = await cache.are_markets_processed(["m1", "m2", "m3"])
            # {"m1": True, "m2": False, "m3": True}
        """
        if not self._is_connected or not self._client:
            return {mid: False for mid in market_ids}

        if not market_ids:
            return {}

        try:
            # Use pipeline for batch operations
            pipe = self._client.pipeline()
            keys = [f"{self.MARKET_PREFIX}{mid}" for mid in market_ids]

            for key in keys:
                pipe.exists(key)

            results = await pipe.execute()

            # Map results back to market IDs
            return {
                market_id: bool(result)
                for market_id, result in zip(market_ids, results)
            }
        except Exception as e:
            logger.error(f"Batch market check failed: {e}")
            return {mid: False for mid in market_ids}

    async def mark_markets_processed(
        self,
        market_ids: list[str],
        ttl_hours: int = 24
    ) -> None:
        """
        Batch mark multiple markets as processed.

        Uses Redis pipelining for efficiency.

        Args:
            market_ids: List of market IDs to mark
            ttl_hours: Time-to-live in hours (default 24)
        """
        if not self._is_connected or not self._client:
            return

        if not market_ids:
            return

        try:
            pipe = self._client.pipeline()
            ttl = timedelta(hours=ttl_hours)

            for market_id in market_ids:
                key = f"{self.MARKET_PREFIX}{market_id}"
                pipe.setex(key, ttl, "processed")

            await pipe.execute()
            logger.debug(f"Marked {len(market_ids)} markets as processed")
        except Exception as e:
            logger.error(f"Batch market marking failed: {e}")
        
    async def save_kill_switch_state(self, state: dict[str, Any]) -> None:
        """Persist critical safety state."""
        if not self._is_connected or not self._client:
            return
            
        try:
            await self._client.set(self.KILL_SWITCH_KEY, json.dumps(state))
        except Exception as e:
            logger.error(f"Failed to save kill switch state: {e}")
            
    async def load_kill_switch_state(self) -> dict[str, Any] | None:
        """Load persisted safety state."""
        if not self._is_connected or not self._client:
            return None
            
        try:
            data = await self._client.get(self.KILL_SWITCH_KEY)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to load kill switch state: {e}")
            return None
    
    # ========== LLM RESULT CACHING (Performance) ==========

    async def get_llm_result(self, cache_key: str) -> dict[str, Any] | None:
        """
        Retrieve cached LLM analysis result.

        Args:
            cache_key: Hash of the input (e.g., cluster hash)

        Returns:
            Cached result dict or None if not found
        """
        if not self._is_connected or not self._client:
            return None

        try:
            key = f"{self.LLM_RESULT_PREFIX}{cache_key}"
            data = await self._client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Failed to get LLM result: {e}")
            return None

    async def set_llm_result(
        self,
        cache_key: str,
        result: dict[str, Any],
        ttl_seconds: int = 300
    ) -> None:
        """
        Cache LLM analysis result.

        Args:
            cache_key: Hash of the input
            result: Result dictionary to cache
            ttl_seconds: Time-to-live in seconds (default 5 minutes)
        """
        if not self._is_connected or not self._client:
            return

        try:
            key = f"{self.LLM_RESULT_PREFIX}{cache_key}"
            await self._client.setex(
                key,
                timedelta(seconds=ttl_seconds),
                json.dumps(result)
            )
        except Exception as e:
            logger.error(f"Failed to cache LLM result: {e}")

    # ========== SOLVER RESULT CACHING (Performance) ==========

    async def get_solver_result(self, cache_key: str) -> dict[str, Any] | None:
        """
        Retrieve cached solver result.

        Args:
            cache_key: Hash of inputs (order book state)

        Returns:
            Cached solver result or None if not found
        """
        if not self._is_connected or not self._client:
            return None

        try:
            key = f"{self.SOLVER_PREFIX}{cache_key}"
            data = await self._client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Failed to get solver result: {e}")
            return None

    async def set_solver_result(
        self,
        cache_key: str,
        result: dict[str, Any],
        ttl_seconds: int = 60
    ) -> None:
        """
        Cache solver result.

        Args:
            cache_key: Hash of inputs
            result: Solver result to cache
            ttl_seconds: Time-to-live in seconds (default 1 minute for fast-moving markets)
        """
        if not self._is_connected or not self._client:
            return

        try:
            key = f"{self.SOLVER_PREFIX}{cache_key}"
            await self._client.setex(
                key,
                timedelta(seconds=ttl_seconds),
                json.dumps(result)
            )
        except Exception as e:
            logger.error(f"Failed to cache solver result: {e}")

    # ========== MANIFEST VERSIONING (Tracking) ==========

    async def get_manifest_version(self) -> int:
        """Get current manifest version number."""
        if not self._is_connected or not self._client:
            return 0

        try:
            version = await self._client.get(self.MANIFEST_VERSION_KEY)
            return int(version) if version else 0
        except Exception as e:
            logger.warning(f"Failed to get manifest version: {e}")
            return 0

    async def increment_manifest_version(self) -> int:
        """
        Increment and return new manifest version.

        Returns:
            New version number
        """
        if not self._is_connected or not self._client:
            return 0

        try:
            new_version = await self._client.incr(self.MANIFEST_VERSION_KEY)
            return int(new_version)
        except Exception as e:
            logger.error(f"Failed to increment manifest version: {e}")
            return 0

    # ========== HEARTBEAT (Operational Safety) ==========

    HEARTBEAT_KEY = "polyquant:heartbeat"
    
    async def set_heartbeat(self) -> None:
        """
        Update heartbeat timestamp using monotonic time.

        Call this every loop iteration. If the timestamp stops updating,
        it indicates the process may be hung.

        Note: Uses monotonic time for consistency with PriceCache.
        For cross-process monitoring, use wall-clock time instead.
        """
        if not self._is_connected or not self._client:
            return

        import time
        try:
            # Use monotonic time for consistency and performance
            ts = str(time.monotonic())
            await self._client.set(self.HEARTBEAT_KEY, ts)
        except Exception as e:
            logger.warning(f"Failed to set heartbeat: {e}")

    async def get_heartbeat_age(self) -> float | None:
        """
        Get seconds since last heartbeat.

        Returns:
            Age in seconds, or None if no heartbeat or not connected

        Note: Uses monotonic time - only valid within the same process.
        """
        if not self._is_connected or not self._client:
            return None

        import time
        try:
            ts = await self._client.get(self.HEARTBEAT_KEY)
            if ts:
                # Calculate age using monotonic time
                return time.monotonic() - float(ts)
            return None
        except Exception as e:
            logger.warning(f"Failed to get heartbeat: {e}")
            return None

# Global instance
cache = RedisCache()
