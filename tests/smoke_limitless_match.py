import asyncio
from polyquant.agents.exchange_matcher import ExchangeMatcher

async def test():
    matcher = ExchangeMatcher()
    _, stats = await matcher.run_matching_pipeline()
    print(f"Limitless Fetched: {stats['limitless_fetched']}")
    print(stats)

if __name__ == "__main__":
    asyncio.run(test())
