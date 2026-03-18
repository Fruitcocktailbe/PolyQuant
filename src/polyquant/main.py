"""
PolyQuant 2.0 - Liquidity Vacuum Scanner

This is the main entry point for the PolyQuant Momentum Scanner.
It runs continuously to:
1. Discover markets from Polymarket (Gamma API) where YES < $0.10
2. Connect to real-time price feeds (WebSocket)
3. Detect momentum and volume spikes
4. Execute trades when opportunities arise

USAGE:
------
    # Start real-time scanning/trading
    python -m polyquant.main

    # Or programmatically
    from polyquant.main import run_navigator

    async def main():
        await run_navigator()
"""

import asyncio
import argparse
import signal
from typing import Any

from polyquant.utils import get_logger

logger = get_logger(__name__)


async def run_navigator(limit: int = 0) -> None:
    """
    Run the Navigator for real-time momentum scanning.

    This runs continuously, monitoring markets via WebSocket and 
    executing trades when strong signals are found.
    """
    from polyquant.navigator import Navigator

    logger.info("Starting Navigator (Scanner)...")

    navigator: Navigator | None = None

    def signal_handler(sig: int, frame: Any) -> None:
        print("\nShutdown signal received...")
        if navigator:
            navigator.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        navigator = Navigator(market_limit=limit)
        async with navigator:
            await navigator.run()
    except KeyboardInterrupt:
        logger.info("Navigator stopped by user")
    except Exception as e:
        logger.error("Navigator failed", error=str(e), exc_info=True)
        raise


async def main() -> None:
    """Main entry point for PolyQuant Scanner."""
    parser = argparse.ArgumentParser(
        description="PolyQuant 2.0 - Real-Time Momentum Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m polyquant.main              # Run full scan
  python -m polyquant.main --limit 50   # Quick test with 50 markets
        """,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of Tier-1 markets to process (0 = all)",
    )

    args = parser.parse_args()

    print(f"""
    ===============================================================
                          PolyQuant 2.0
               Real-Time Momentum Scanning System
    ===============================================================
    """)

    await run_navigator(limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())
