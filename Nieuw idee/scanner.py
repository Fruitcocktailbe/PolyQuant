#!/usr/bin/env python3
"""
Real-Time Polymarket Scanner — "Liquidity Vacuum" Strategy
===========================================================
Polls the Gamma / CLOB APIs every 60 seconds, applies a three-tier
filter funnel, and logs STRONG SIGNAL alerts to scanner_alerts.txt.

Usage:
    pip install aiohttp
    python scanner.py              # full scan
    python scanner.py --limit 50   # quick test with 50 markets
"""

import argparse
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

# ─────────────────────────── Configuration ────────────────────────────

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Tier-1
MAX_YES_PRICE = 0.10

# Tier-2 thresholds
VOLUME_SPIKE_MULT = 2.5        # current vol > 2.5× avg vol
MOMENTUM_THRESHOLD = 0.02      # 2 % price change in 90 s

# Tier-3
ORDERBOOK_IMBALANCE = 0.70     # 70 % on one side
EMA_PERIOD = 200               # minutes
EMA_DEVIATION = 0.12           # 12 % deviation from expected price
SENTIMENT_OFFSET = 0.0         # tune manually or via external feed

# Timing
SCAN_INTERVAL = 60             # seconds between scans
MARKET_DISCOVERY_INTERVAL = 300  # seconds between full market refreshes
PRICE_POLL_INTERVAL = 10       # seconds for real-time price updates
WARMUP_HISTORY_MINUTES = 200

# Rate-limit guard (simple token-bucket style)
GAMMA_MAX_REQ_PER_10S = 280    # stay under 300
CLOB_MAX_REQ_PER_10S = 900     # stay under 1000

# Output
ALERT_FILE = Path(__file__).parent / "scanner_alerts.txt"

# ─────────────────────────── Logging ──────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scanner")

# ─────────────────────────── Data models ──────────────────────────────

@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class MarketInfo:
    condition_id: str
    slug: str
    question: str
    token_id: str  # YES token
    price: float = 0.0


@dataclass
class MarketTracker:
    """Keeps a rolling window of 1-min candles + real-time snapshots."""
    info: MarketInfo
    candles: deque = field(default_factory=lambda: deque(maxlen=WARMUP_HISTORY_MINUTES))
    recent_prices: deque = field(default_factory=lambda: deque(maxlen=10))  # last ~10 snapshots for 90 s window
    ema: float = 0.0
    warmed_up: bool = False

    # ── EMA helpers ──────────────────────────────────────────────────
    def compute_ema(self) -> float:
        """Compute EMA(200) from stored candles."""
        if len(self.candles) < 2:
            self.ema = self.candles[-1].close if self.candles else 0.0
            return self.ema
        k = 2 / (EMA_PERIOD + 1)
        ema = self.candles[0].close
        for c in list(self.candles)[1:]:
            ema = c.close * k + ema * (1 - k)
        self.ema = ema
        return self.ema

    def expected_price(self) -> float:
        return self.ema * (1 + SENTIMENT_OFFSET)

    # ── Volume helpers ───────────────────────────────────────────────
    def avg_volume_10d(self) -> float:
        """Rough 10-day avg volume from stored candles (best-effort)."""
        vols = [c.volume for c in self.candles]
        if not vols:
            return 0.0
        return sum(vols) / len(vols)

    def latest_volume(self) -> float:
        if self.candles:
            return self.candles[-1].volume
        return 0.0

    # ── Momentum (90 s) ─────────────────────────────────────────────
    def momentum_90s(self) -> float:
        """Price change over ~90 seconds using recent_prices deque."""
        if len(self.recent_prices) < 2:
            return 0.0
        now_price = self.recent_prices[-1][1]
        # find price closest to 90 s ago
        target_ts = time.time() - 90
        oldest = self.recent_prices[0]
        for ts, px in self.recent_prices:
            if ts <= target_ts:
                oldest = (ts, px)
        if oldest[1] == 0:
            return 0.0
        return (now_price - oldest[1]) / oldest[1]


# ─────────────────────────── Rate limiter ─────────────────────────────

class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_requests: int, window: float = 10.0):
        self.max_requests = max_requests
        self.window = window
        self.timestamps: deque = deque()

    async def acquire(self):
        now = time.monotonic()
        # drop expired
        while self.timestamps and self.timestamps[0] < now - self.window:
            self.timestamps.popleft()
        if len(self.timestamps) >= self.max_requests:
            wait = self.window - (now - self.timestamps[0]) + 0.1
            log.warning("Rate limit approaching — sleeping %.1f s", wait)
            await asyncio.sleep(wait)
        self.timestamps.append(time.monotonic())


# ─────────────────────────── Scanner ──────────────────────────────────

class PolymarketScanner:

    def __init__(self, market_limit: int = 0):
        self.trackers: dict[str, MarketTracker] = {}
        self.gamma_limiter = RateLimiter(GAMMA_MAX_REQ_PER_10S)
        self.clob_limiter = RateLimiter(CLOB_MAX_REQ_PER_10S)
        self.session: Optional[aiohttp.ClientSession] = None
        self._last_discovery = 0.0
        self._cycle_count = 0
        self._market_limit = market_limit  # 0 = no limit

    # ── HTTP helpers ─────────────────────────────────────────────────

    async def _get(self, url: str, limiter: RateLimiter, params: dict | None = None) -> dict | list | None:
        await limiter.acquire()
        try:
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 429:
                    retry = int(resp.headers.get("Retry-After", "10"))
                    log.warning("HTTP 429 (rate-limited) — backing off %d s", retry)
                    await asyncio.sleep(retry)
                    return None
                if resp.status != 200:
                    return None
                return await resp.json()
        except asyncio.TimeoutError:
            log.error("Request timeout: %s", url.split("/")[-1])
            return None
        except Exception as exc:
            log.error("Request failed (%s): %s", url.split("/")[-1], exc)
            return None

    # ── Tier 1: Market discovery ─────────────────────────────────────

    async def discover_markets(self) -> list[MarketInfo]:
        """Fetch all active markets and filter price < MAX_YES_PRICE."""
        markets: list[MarketInfo] = []
        total_fetched = 0
        offset = 0
        limit = 100
        while True:
            data = await self._get(
                f"{GAMMA_BASE}/markets",
                self.gamma_limiter,
                params={"limit": limit, "offset": offset, "active": "true", "closed": "false"},
            )
            if not data:
                break
            total_fetched += len(data)
            for m in data:
                try:
                    prices = m.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        import json
                        prices = json.loads(prices)
                    yes_price = float(prices[0]) if prices else 1.0

                    if yes_price >= MAX_YES_PRICE:
                        continue

                    tokens = m.get("clobTokenIds", "[]")
                    if isinstance(tokens, str):
                        import json
                        tokens = json.loads(tokens)
                    yes_token = tokens[0] if tokens else ""

                    markets.append(MarketInfo(
                        condition_id=m.get("conditionId", m.get("id", "")),
                        slug=m.get("slug", "unknown"),
                        question=m.get("question", ""),
                        token_id=yes_token,
                        price=yes_price,
                    ))
                except (IndexError, ValueError, KeyError):
                    continue
            if self._market_limit and len(markets) >= self._market_limit:
                markets = markets[:self._market_limit]
                break
            if len(data) < limit:
                break
            offset += limit
        log.info("Tier-1: %d candidates from %d markets (YES < $%.2f)%s",
                 len(markets), total_fetched, MAX_YES_PRICE,
                 f" [LIMITED to {self._market_limit}]" if self._market_limit else "")
        return markets

    # ── Tier 2: Volume spike + Momentum ──────────────────────────────

    async def tier2_check(self, tracker: MarketTracker) -> bool:
        """Quick volume & momentum filter (cheap API calls)."""
        avg_vol = tracker.avg_volume_10d()
        cur_vol = tracker.latest_volume()
        vol_ok = avg_vol > 0 and cur_vol > VOLUME_SPIKE_MULT * avg_vol

        mom = tracker.momentum_90s()
        mom_ok = abs(mom) > MOMENTUM_THRESHOLD

        return vol_ok or mom_ok

    # ── Tier 3: Order book imbalance + EMA deviation ─────────────────

    async def tier3_check(self, tracker: MarketTracker) -> Optional[dict]:
        """Heavy check: order book imbalance + EMA deviation."""
        # Fetch order book
        book = await self._get(
            f"{CLOB_BASE}/book",
            self.clob_limiter,
            params={"token_id": tracker.info.token_id},
        )
        imbalance = 0.0
        if book:
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            bid_vol = sum(float(b.get("size", 0)) for b in bids)
            ask_vol = sum(float(a.get("size", 0)) for a in asks)
            total = bid_vol + ask_vol
            if total > 0:
                imbalance = max(bid_vol, ask_vol) / total

        if imbalance < ORDERBOOK_IMBALANCE:
            return None

        # EMA deviation
        tracker.compute_ema()
        expected = tracker.expected_price()
        if expected == 0:
            return None
        deviation = abs(tracker.info.price - expected) / expected

        if deviation < EMA_DEVIATION:
            return None

        return {
            "imbalance": imbalance,
            "ema": tracker.ema,
            "expected": expected,
            "deviation": deviation,
            "vol_mult": (tracker.latest_volume() / tracker.avg_volume_10d())
                        if tracker.avg_volume_10d() > 0 else 0,
        }

    # ── Warm-up ──────────────────────────────────────────────────────

    async def warm_up_tracker(self, tracker: MarketTracker):
        """Download ~200 minutes of history for a market."""
        data = await self._get(
            f"{CLOB_BASE}/prices-history",
            self.clob_limiter,
            params={
                "market": tracker.info.condition_id,
                "interval": "1m",
                "fidelity": WARMUP_HISTORY_MINUTES,
            },
        )
        if not data:
            history = []
        elif isinstance(data, list):
            history = data
        else:
            history = data.get("history", [])

        for point in (history or []):
            try:
                tracker.candles.append(Candle(
                    timestamp=float(point.get("t", 0)),
                    open=float(point.get("o", point.get("p", 0))),
                    high=float(point.get("h", point.get("p", 0))),
                    low=float(point.get("l", point.get("p", 0))),
                    close=float(point.get("c", point.get("p", 0))),
                    volume=float(point.get("v", 0)),
                ))
            except (ValueError, TypeError):
                continue
        if tracker.candles:
            tracker.compute_ema()
            tracker.warmed_up = True

    # ── Alert logging ────────────────────────────────────────────────

    def log_alert(self, tracker: MarketTracker, details: dict):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = (
            f"[{ts}] - {tracker.info.slug} - "
            f"${tracker.info.price:.4f} - "
            f"VOL×{details['vol_mult']:.1f} - "
            f"EMA_DEV {details['deviation']:.1%} - "
            f"IMBAL {details['imbalance']:.0%} - "
            f"STRONG SIGNAL"
        )
        log.info("🚨  %s", line)
        with open(ALERT_FILE, "a") as f:
            f.write(line + "\n")

    # ── Main scan cycle ──────────────────────────────────────────────

    async def scan_once(self):
        self._cycle_count += 1
        cycle_start = time.time()
        now = cycle_start

        log.info("── Cycle #%d ──────────────────────────────────────────",
                 self._cycle_count)

        # Refresh market list periodically
        if now - self._last_discovery > MARKET_DISCOVERY_INTERVAL:
            markets = await self.discover_markets()
            self._last_discovery = now

            # Add new trackers, prune stale ones
            new_count = 0
            cold_count = 0
            total_new = sum(1 for m in markets if m.condition_id not in self.trackers)
            processed = 0
            seen = set()
            for m in markets:
                seen.add(m.condition_id)
                if m.condition_id not in self.trackers:
                    t = MarketTracker(info=m)
                    await self.warm_up_tracker(t)
                    processed += 1
                    if t.warmed_up:
                        self.trackers[m.condition_id] = t
                        new_count += 1
                    else:
                        cold_count += 1
                    if processed % 100 == 0 or processed == total_new:
                        log.info("Warming up: %d/%d done (%d with data, %d empty)",
                                 processed, total_new, new_count, cold_count)
                else:
                    self.trackers[m.condition_id].info.price = m.price

            stale = [k for k in self.trackers if k not in seen]
            for k in stale:
                del self.trackers[k]
            log.info("Trackers: +%d new, -%d stale, %d skipped (no data), %d active",
                     new_count, len(stale), cold_count, len(self.trackers))

        # Update recent prices for all tracked markets
        for tracker in self.trackers.values():
            tracker.recent_prices.append((time.time(), tracker.info.price))

        # Tier-2 filter
        tier2_passed: list[MarketTracker] = []
        for tracker in self.trackers.values():
            if await self.tier2_check(tracker):
                tier2_passed.append(tracker)

        # Tier-3 deep analysis
        signals = 0
        if tier2_passed:
            for tracker in tier2_passed:
                details = await self.tier3_check(tracker)
                if details:
                    signals += 1
                    self.log_alert(tracker, details)

        elapsed = time.time() - cycle_start
        log.info("Cycle #%d done in %.1fs | %d tracked | T2: %d | signals: %d",
                 self._cycle_count, elapsed, len(self.trackers),
                 len(tier2_passed), signals)

    # ── Entry point ──────────────────────────────────────────────────

    async def run(self):
        log.info("Polymarket Liquidity Vacuum Scanner%s",
                 f" [TEST MODE: limit={self._market_limit}]" if self._market_limit else "")
        log.info("  Scan: %ds | Discovery: %ds | Alerts: %s",
                 SCAN_INTERVAL, MARKET_DISCOVERY_INTERVAL, ALERT_FILE)

        self.session = aiohttp.ClientSession(
            headers={"User-Agent": "PolyScanner/1.0"}
        )
        try:
            while True:
                try:
                    await self.scan_once()
                except Exception as exc:
                    log.exception("Scan cycle error: %s", exc)
                await asyncio.sleep(SCAN_INTERVAL)
        except asyncio.CancelledError:
            log.info("Scanner stopped.")
        finally:
            await self.session.close()


# ─────────────────────────── Main ─────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket Liquidity Vacuum Scanner")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of Tier-1 markets to process (0 = all)")
    args = parser.parse_args()
    try:
        asyncio.run(PolymarketScanner(market_limit=args.limit).run())
    except KeyboardInterrupt:
        log.info("Shut down by user (Ctrl+C).")
