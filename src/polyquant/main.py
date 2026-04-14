"""
PolyQuant 2.0 - Main Entry Point

This is the main entry point for PolyQuant's two-mode architecture:

1. MAP MODE (Offline - "Slow Brain"):
   - Discovers markets from Polymarket
   - Analyzes logical dependencies using LLMs
   - Validates constraints
   - Generates and persists Constraint Manifests
   - Run periodically (e.g., hourly)

2. TRADE MODE (Online - "Fast Brain"):
   - Loads pre-computed Constraint Manifests
   - Connects to real-time price feeds (WebSocket)
   - Detects arbitrage opportunities using solvers
   - Executes trades with <50ms latency
   - Runs continuously

USAGE:
------
    # Build constraint map (run first, or periodically)
    python -m polyquant.main map

    # Start real-time trading
    python -m polyquant.main trade

    # Or programmatically
    from polyquant.main import run_map_maker, run_navigator

    async def main():
        # Build map first
        await run_map_maker()

        # Then trade
        await run_navigator()
"""

import asyncio
import argparse
import signal
import sys
from typing import Any

from polyquant.utils import get_logger

logger = get_logger(__name__)


async def run_map_maker(
    limit: int = 0,
    min_liquidity: float = 1000,
    force: bool = False,
) -> dict[str, Any]:
    """
    Run the Map Maker to build constraint manifests.

    This performs offline analysis of market structures and saves
    the results to disk for the Navigator to use.

    Args:
        limit: Maximum number of markets to analyze.
        min_liquidity: Minimum liquidity threshold for markets.
        force: Whether to force a re-scan of already processed markets.

    Returns:
        dict: Results summary with cluster count, constraint count, etc.
    """
    from polyquant.map_maker import MapMaker
    from polyquant.api.server import app, monitor, setup_web_logging
    import uvicorn

    logger.info("Starting Map Maker...")
    
    # Start Sidecar UI Server so Dashboard can watch MapMaker progress
    config_uv = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config_uv)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.5)  # Give uvicorn a moment to bind to port
    logger.info("🌐 API server started on http://0.0.0.0:8000 — Dashboard will now show scanning progress")
    setup_web_logging()
    await monitor.update_status(status="MAPPING")

    try:
        async with MapMaker() as map_maker:
            result = await map_maker.build_map(
                limit=limit,
                min_liquidity=min_liquidity,
                skip_processed=not force,
            )

            print("\n" + "=" * 60)
            print("Map Building Result:")
            print("=" * 60)
            for key, value in result.items():
                print(f"  {key}: {value}")
            
            # Keep server alive for a few seconds to let final updates flush to UI
            await monitor.update_status(status="MAPPING_COMPLETE")
            await asyncio.sleep(5)
            server.should_exit = True
            await server_task

            return result
    except Exception as e:
        logger.error("Map Maker failed", error=str(e), exc_info=True)
        try:
            server.should_exit = True
            await asyncio.wait_for(server_task, timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            pass
        return {"status": "failed", "error": str(e)}


async def run_navigator() -> None:
    """
    Run the Navigator for real-time trading.

    This loads pre-computed constraints and runs continuously,
    monitoring markets and executing trades when opportunities arise.
    """
    from polyquant.navigator import Navigator

    logger.info("Starting Navigator...")

    navigator: Navigator | None = None

    def signal_handler(sig: int, frame: Any) -> None:
        print("\nShutdown signal received...")
        if navigator:
            navigator.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        navigator = Navigator()
        async with navigator:
            await navigator.run()
    except KeyboardInterrupt:
        logger.info("Navigator stopped by user")
    except Exception as e:
        logger.error("Navigator failed", error=str(e), exc_info=True)
        raise


async def main() -> None:
    """Main entry point for PolyQuant."""
    parser = argparse.ArgumentParser(
        description="PolyQuant 2.0 - Autonomous Arbitrage Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  map       Run the Map Maker (offline constraint analysis)
  trade     Run the Navigator (real-time trading)

Typical Workflow:
  1. Run 'map' mode to build constraint manifests (first time or periodically)
  2. Run 'trade' mode to start real-time trading

Examples:
  python -m polyquant.main map      # Build constraint map
  python -m polyquant.main trade    # Start trading
        """,
    )

    parser.add_argument(
        "mode",
        choices=["map", "trade"],
        nargs="?",
        default="map",
        help="Execution mode (default: map)",
    )

    # Map Maker arguments
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum events to scan in 'map' mode (default: 0 = all events above --min-liquidity)",
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=1000.0,
        help="Minimum liquidity threshold in 'map' mode (default: 1000)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-scan of already processed markets in 'map' mode",
    )

    args = parser.parse_args()

    print(f"""
    ===============================================================
                          PolyQuant 2.0
            Autonomous Arbitrage Extraction System
                        Mode: {args.mode.upper()}
    ===============================================================
    """)

    if args.mode == "map":
        result = await run_map_maker(
            limit=args.limit,
            min_liquidity=args.min_liquidity,
            force=args.force,
        )
        if result.get("status") in ("failed", "error"):
            sys.exit(1)
    elif args.mode == "trade":
        await run_navigator()
    else:
        # This shouldn't happen due to argparse choices, but just in case
        logger.error(f"Unknown mode: {args.mode}")
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
