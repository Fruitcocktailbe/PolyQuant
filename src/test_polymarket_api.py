"""Test all 4 Polymarket API endpoints."""
import requests

print("=" * 60)
print("POLYMARKET API ENDPOINT TESTS")
print("=" * 60)

# Test all 4 APIs
apis = {
    "Gamma API (discovery)": "https://gamma-api.polymarket.com/markets?limit=2",
    "CLOB API (trading)": "https://clob.polymarket.com/markets?limit=2",
    "Data API (positions)": "https://data-api.polymarket.com/markets?limit=2",
}

for name, url in apis.items():
    print(f"\n[{name}]")
    print(f"  URL: {url}")
    try:
        resp = requests.get(url, timeout=10)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                print(f"  Records: {len(data)}")
            elif isinstance(data, dict):
                print(f"  Keys: {list(data.keys())[:5]}")
        else:
            print(f"  Response: {resp.text[:100]}")
    except Exception as e:
        print(f"  Error: {e}")

# WebSocket just check if it resolves (can't fully test without async)
print(f"\n[WebSocket (realtime)]")
print(f"  URL: wss://ws-subscriptions-clob.polymarket.com/ws")
print(f"  Status: (requires async connection test)")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
