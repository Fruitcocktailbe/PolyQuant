#!/usr/bin/env python
"""
Watchdog for PolyQuant 2.0

This script runs as a SEPARATE process and monitors the main bot's
liveness by checking the heartbeat timestamp in Redis.

If the heartbeat is older than the threshold, the main bot is likely
hung (infinite loop, deadlock, etc.) and needs to be terminated.

USAGE:
------
    # Run in a separate terminal
    python -m polyquant.utils.watchdog
    
    # Or as a background service
    nohup python -m polyquant.utils.watchdog &
"""

import asyncio
import os
import sys
import signal

# Add src to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from polyquant.utils.cache import cache
from polyquant.utils import get_logger

logger = get_logger(__name__)

# Configuration
HEARTBEAT_THRESHOLD_SECONDS = 60  # Alert if heartbeat older than this
CHECK_INTERVAL_SECONDS = 10       # How often to check


async def main():
    """Main watchdog loop."""
    logger.info(
        "Watchdog starting",
        threshold_seconds=HEARTBEAT_THRESHOLD_SECONDS,
        check_interval=CHECK_INTERVAL_SECONDS,
    )
    
    # Connect to Redis
    connected = await cache.connect()
    if not connected:
        logger.error("Cannot start watchdog: Redis not available")
        sys.exit(1)
    
    logger.info("Watchdog connected to Redis, monitoring heartbeat...")
    
    consecutive_failures = 0
    
    while True:
        try:
            age = await cache.get_heartbeat_age()
            
            if age is None:
                logger.debug("No heartbeat found (bot may not have started)")
                consecutive_failures = 0
            elif age > HEARTBEAT_THRESHOLD_SECONDS:
                consecutive_failures += 1
                logger.warning(
                    "Heartbeat stale!",
                    age_seconds=round(age, 1),
                    threshold=HEARTBEAT_THRESHOLD_SECONDS,
                    consecutive_failures=consecutive_failures,
                )
                
                if consecutive_failures >= 3:
                    logger.critical(
                        "ALERT: Bot appears hung. Consider terminating.",
                        age_seconds=round(age, 1),
                    )
                    # TODO: Add actual termination logic
                    # - Find and kill the polyquant.main process
                    # - Cancel all open orders via API
            else:
                logger.debug("Heartbeat OK", age_seconds=round(age, 1))
                consecutive_failures = 0
                
        except Exception as e:
            logger.error("Watchdog error", error=str(e))
        
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    print("PolyQuant Watchdog - Monitoring bot liveness...")
    print(f"Threshold: {HEARTBEAT_THRESHOLD_SECONDS}s | Check interval: {CHECK_INTERVAL_SECONDS}s")
    print("Press Ctrl+C to stop")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWatchdog stopped.")
