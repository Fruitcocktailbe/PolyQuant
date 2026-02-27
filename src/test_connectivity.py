import httpx
import asyncio

async def test_connectivity():
    url = "https://gamma-api.polymarket.com/markets?limit=1&active=true"
    print(f"Connecting to {url}...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("Connection Successful!")
                try:
                    data = response.json()
                    print(f"Data received: {len(data)} markets")
                except Exception as e:
                    print(f"Failed to parse JSON: {e}")
            else:
                print(f"Connection Failed: {response.text}")
    except Exception as e:
        print(f"Network Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_connectivity())
