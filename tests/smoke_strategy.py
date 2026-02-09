
import asyncio
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from polyquant.agents.correlation import CorrelationAgent
from polyquant.agents.microstructure import MicrostructureAgent
from polyquant.data.polymarket_client import PolymarketClient
from polyquant.data.market_models import OrderBook, OrderLevel, MicrostructureSignal
from polyquant.map_maker import MapMaker
from polyquant.navigator import Navigator, ExecutionGuard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmokeTest")

async def main():
    logger.info("Testing imports...")

    # 1. Test Agent Instantiation
    corr = CorrelationAgent()
    micro = MicrostructureAgent()
    logger.info("Agents instantiated successfully.")

    # 2. Test Microstructure Logic
    ob = OrderBook(
        outcome_id="test_outcome",
        bids=[OrderLevel(price=0.5, size=100)],
        asks=[OrderLevel(price=0.52, size=50)]
    )
    signal = micro.analyze(ob)
    logger.info(f"Microstructure Analysis: Spread={signal.spread}, Imbalance={signal.imbalance}")
    assert abs(signal.spread - 0.02) < 0.0001
    assert abs(signal.imbalance - (1/3)) < 0.0001

    # 3. Test New Architecture (MapMaker + Navigator)
    # We assume .env is present or defaults work?

    logger.info("Testing MapMaker initialization...")
    map_maker = MapMaker()
    logger.info("MapMaker initialized.")

    logger.info("Testing Navigator initialization...")
    navigator = Navigator()
    logger.info("Navigator initialized.")

    logger.info("Testing ExecutionGuard initialization...")
    guard = ExecutionGuard()
    logger.info("ExecutionGuard initialized.")

    logger.info("Smoke Test PASSED")

if __name__ == "__main__":
    asyncio.run(main())
