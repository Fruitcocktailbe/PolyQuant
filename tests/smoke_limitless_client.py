import asyncio
from polyquant.data.limitless_client import LimitlessClient

async def test():
    async with LimitlessClient() as client:
        markets = await client.get_markets()
        print(f"\nSUCCESS: Fetched {len(markets)} Limitless markets!")

if __name__ == "__main__":
    asyncio.run(test())
