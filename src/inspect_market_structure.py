import httpx
import asyncio
import json

async def inspect():
    async with httpx.AsyncClient(base_url='https://gamma-api.polymarket.com') as client:
        # Fetch a few markets
        resp = await client.get('/markets', params={'limit': 5, 'active': 'true'})
        data = resp.json()
        
        if data:
            print("Keys in market object:")
            print(json.dumps(list(data[0].keys()), indent=2))
            
            print("\nFull object sample:")
            print(json.dumps(data[0], indent=2))
            
            # Check for specific interesting fields
            for m in data:
                if 'type' in m or 'market_type' in m:
                    print(f"\nMarket Type: {m.get('type') or m.get('market_type')}")
                if 'group' in m or 'group_id' in m:
                    print(f"Group: {m.get('group') or m.get('group_id')}")

if __name__ == "__main__":
    asyncio.run(inspect())
