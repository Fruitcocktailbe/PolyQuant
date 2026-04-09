"""Quick test to verify PolymarketClient.get_history() returns enough data points."""
import asyncio
from polyquant.data.polymarket_client import PolymarketClient

async def test():
    async with PolymarketClient() as client:
        markets, _ = await client.get_active_markets(limit=5, min_liquidity=5000)
        print(f"\nFetched {len(markets)} markets with >$5k liquidity\n")
        
        for m in markets:
            history = await client.get_history(m.market_id)
            print(f"  {m.question[:60]}")
            print(f"    Market ID: {m.market_id[:20]}...")
            print(f"    Data points: {len(history)}")
            if history:
                print(f"    First: {history[0]}")
                print(f"    Last:  {history[-1]}")
            else:
                print(f"    ⚠️  No history returned!")
            print()
        
        # Also try with a token_id (outcome-level) 
        if markets and markets[0].outcomes:
            token_id = markets[0].outcomes[0].token_id
            if token_id:
                print(f"\n--- Testing with token_id instead of market_id ---")
                history2 = await client.get_history(token_id)
                print(f"    Token ID: {token_id[:20]}...")
                print(f"    Data points: {len(history2)}")
                if history2:
                    print(f"    First: {history2[0]}")
                    print(f"    Last:  {history2[-1]}")

if __name__ == "__main__":
    asyncio.run(test())
