"""
WebSocket Verification Script for PolyQuant

This script verifies that we can connect to Polymarket's WebSocket API
and receive real-time order book updates.

USAGE:
------
    python scripts/verify_ws.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polyquant.data.polymarket_client import PolymarketWSClient
from polyquant.data.market_models import OrderBook
from polyquant.utils import get_logger

logger = get_logger(__name__)

# Trump 2024 Election Winner (high volume market for testing)
# Token ID for "Yes"
TEST_TOKEN_ID = "21742633143463906290569050155826241533067272736897614950488156847949938836455"

async def main():
    print("=" * 50)
    print("PolyQuant WebSocket Verification")
    print("=" * 50)
    
    client = PolymarketWSClient()
    
    # Store updates to verify we got them
    updates_received = 0
    
    async def on_update(book: OrderBook):
        nonlocal updates_received
        updates_received += 1
        
        mid = book.mid_price
        spread = book.spread
        
        print(f"[{updates_received}] Update for {book.outcome_id[:10]}...")
        print(f"    Mid Price: {mid:.4f}" if mid else "    Mid Price: N/A")
        print(f"    Spread:    {spread:.4f}" if spread else "    Spread:    N/A")
        print(f"    Bids: {len(book.bids)} | Asks: {len(book.asks)}")
        
        if updates_received >= 3:
            print("\n✅ Verification Successful: Received 3 updates.")
            print("   (Ctrl+C to exit if it doesn't close automatically)")
            
            # Cancel the task to exit
            # In a real app we'd have a clean shutdown method
            sys.exit(0)

    try:
        print("1. Connecting to WebSocket...")
        await client.connect()
        print("   ✅ Connected.")
        
        print(f"2. Subscribing to token: {TEST_TOKEN_ID[:10]}...")
        await client.subscribe([TEST_TOKEN_ID], on_update)
        print("   ✅ Subscribed. Waiting for updates...")
        
        # Wait for updates
        await asyncio.sleep(30)
        
        if updates_received == 0:
            print("\n❌ Verification Failed: Timed out waiting for updates.")
            print("   Check your internet connection or the Token ID.")
            
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
