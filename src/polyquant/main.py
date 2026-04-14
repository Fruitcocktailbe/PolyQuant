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
import contextlib
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
    from polyquant.api.server import monitor, start_api_server

    logger.info("Starting Map Maker...")

    # Start Sidecar UI Server so Dashboard can watch MapMaker progress
    server, server_task = await start_api_server()
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


async def run_supervisor() -> None:
    """
    Run the Navigator and MapMaker together in one process.

    Owns a single uvicorn server for the whole lifetime so MapMaker and
    Navigator share the in-memory monitor and one dashboard at port 8000.
    MapMaker is invoked on a timer (config.map_interval_seconds); Navigator
    hot-reloads new manifests from disk as they land. On shutdown, cancels
    the timer, stops Navigator, and drains uvicorn.

    Note on shared state: MapMaker drives monitor.state.pipeline_stage
    (DISCOVERY/LOGIC/MATCHING/COMPLETE) during each cycle. Navigator sets
    pipeline_stage=COMPLETE once at startup; after that, MapMaker owns it.
    """
    from polyquant.api.server import monitor, start_api_server
    from polyquant.map_maker import MapMaker
    from polyquant.navigator import Navigator
    from polyquant.utils.config import config

    logger.info("Starting supervisor (Navigator + periodic MapMaker)...")

    # Own the dashboard server for the whole supervisor lifetime. start_api_server
    # is idempotent, so Navigator.__aenter__'s own call becomes a no-op.
    server, server_task = await start_api_server()

    async def map_maker_timer() -> None:
        if not config.map_run_on_start:
            await asyncio.sleep(config.map_interval_seconds)
        while True:
            try:
                logger.info(
                    "Supervisor: starting MapMaker cycle",
                    limit=config.map_limit,
                    min_liquidity=config.map_min_liquidity,
                )
                await monitor.update_status(status="MAPPING")
                async with MapMaker() as mm:
                    result = await mm.build_map(
                        limit=config.map_limit,
                        min_liquidity=config.map_min_liquidity,
                        skip_processed=True,
                    )
                logger.info("Supervisor: MapMaker cycle finished", result=result)
                await monitor.update_status(status="MAPPING_COMPLETE")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Supervisor: MapMaker cycle failed", error=str(e), exc_info=True)
                await monitor.update_status(status="MAPPING_FAILED")
            await asyncio.sleep(config.map_interval_seconds)

    navigator: Navigator | None = None
    map_task: asyncio.Task[None] = asyncio.create_task(
        map_maker_timer(), name="map_maker_timer"
    )

    def _graceful(sig: int, frame: Any) -> None:
        print("\nShutdown signal received...")
        if navigator is not None:
            navigator.stop()
        map_task.cancel()

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    try:
        navigator = Navigator()
        async with navigator:
            await navigator.run()
    except KeyboardInterrupt:
        logger.info("Supervisor stopped by user")
    except Exception as e:
        logger.error("Supervisor failed", error=str(e), exc_info=True)
        raise
    finally:
        map_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await map_task
        server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


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
  map       Run the Map Maker once (offline constraint analysis)
  trade     Run the Navigator alone (real-time trading; assumes a map exists)
  run       Supervisor: Navigator + periodic MapMaker in one process
            (recommended for production — shares one dashboard server)

Typical Workflow:
  1. Run 'run' mode for continuous operation (supervisor manages map refreshes).
  2. Use 'map' / 'trade' for one-shot debugging.

Examples:
  python -m polyquant.main run      # Continuous supervisor (production)
  python -m polyquant.main map      # Build constraint map (one-shot)
  python -m polyquant.main trade    # Start trading (one-shot)
        """,
    )

    parser.add_argument(
        "mode",
        choices=["map", "trade", "run"],
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
    elif args.mode == "run":
        await run_supervisor()
    else:
        # This shouldn't happen due to argparse choices, but just in case
        logger.error(f"Unknown mode: {args.mode}")
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
