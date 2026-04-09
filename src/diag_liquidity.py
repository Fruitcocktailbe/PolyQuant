import httpx
import asyncio
import json

async def check():
    async with httpx.AsyncClient(base_url='https://gamma-api.polymarket.com') as client:
        # Fetch a larger batch to see what's actually out there
        resp = await client.get('/markets', params={'limit': 100, 'active': 'true'})
        data = resp.json()
        
        total = len(data)
        liq_1k = [m for m in data if float(m.get('liquidity', 0)) >= 1000]
        liq_0 = [m for m in data if float(m.get('liquidity', 0)) == 0]
        
        print(f"Total markets returned from API: {total}")
        print(f"Markets with liquidity >= 1000: {len(liq_1k)}")
        print(f"Markets with liquidity == 0: {len(liq_0)}")
        
        if liq_1k:
            print("\nExample markets >= 1000:")
            for m in liq_1k[:3]:
                print(f"- {m.get('question')} (Liquidity: {m.get('liquidity')})")

if __name__ == "__main__":
    asyncio.run(check())
