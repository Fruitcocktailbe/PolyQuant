import asyncio
import json
from polyquant.data.limitless_client import LimitlessClient

async def main():
    c = LimitlessClient()
    await c.__aenter__()
    markets = await c.get_markets(1)
    if markets:
        print(json.dumps(markets[0], indent=2))
    else:
        print("No markets")
    await c.__aexit__()

if __name__ == "__main__":
    asyncio.run(main())
