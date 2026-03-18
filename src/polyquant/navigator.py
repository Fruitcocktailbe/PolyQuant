"""
Navigator (Scanner) - Real-time momentum trading engine for PolyQuant 2.0.

This replaces the old arbitrage Navigator.
It implements a "Liquidity Vacuum" strategy:
1. Discover markets where YES price < 0.10.
2. Fetch 200m history for EMA.
3. Subscribe to real-time WebSockets to monitor prices.
4. Execute trades when volume spikes, momentum is high, and order book is imbalanced.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from polyquant.data.market_models import OrderBook, OrderSide, ProposedTrade, TimeInForce
from polyquant.data.price_cache import PriceCache
from polyquant.risk import KillSwitch, PositionSizer
from polyquant.data.polymarket_client import PolymarketClient
from polyquant.data.trade_store import TradeStore
from polyquant.execution.executor import TradeExecutor
from polyquant.execution.rust_client import RustClient
from polyquant.api.server import monitor, app, set_trade_store, setup_web_logging
from polyquant.utils import config, get_logger
import uvicorn

logger = get_logger(__name__)

# --- Strategy Constants ---
MAX_YES_PRICE = 0.10
VOLUME_SPIKE_MULT = 2.5        # current vol > 2.5x avg vol
MOMENTUM_THRESHOLD = 0.02      # 2% price change in 90s
ORDERBOOK_IMBALANCE = 0.70     # 70% on one side
EMA_PERIOD = 200               # minutes
EMA_DEVIATION = 0.12           # 12% deviation from expected price
SENTIMENT_OFFSET = 0.0
SCAN_INTERVAL = 60             # seconds
MARKET_DISCOVERY_INTERVAL = 300 # seconds
WARMUP_HISTORY_MINUTES = 200

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
    recent_prices: deque = field(default_factory=lambda: deque(maxlen=10))  # 10 snapshots (90s window)
    ema: float = 0.0
    warmed_up: bool = False

    def compute_ema(self) -> float:
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

    def avg_volume_10d(self) -> float:
        vols = [c.volume for c in self.candles]
        if not vols:
            return 0.0
        return sum(vols) / len(vols)

    def latest_volume(self) -> float:
        return self.candles[-1].volume if self.candles else 0.0

    def momentum_90s(self) -> float:
        if len(self.recent_prices) < 2:
            return 0.0
        now_price = self.recent_prices[-1][1]
        target_ts = time.time() - 90
        oldest = self.recent_prices[0]
        for ts, px in self.recent_prices:
            if ts <= target_ts:
                oldest = (ts, px)
        if oldest[1] == 0:
            return 0.0
        return (now_price - oldest[1]) / oldest[1]


class Navigator:
    """
    Real-Time Momentum Scanner executing the Liquidity Vacuum Strategy.
    """

    def __init__(self, market_limit: int = 0):
        self._polymarket: PolymarketClient | None = None
        self._price_cache: PriceCache | None = None
        self._kill_switch: KillSwitch | None = None
        self._position_sizer: PositionSizer | None = None
        self._trade_store: TradeStore | None = None
        self._rust_client: RustClient | None = None
        self._trade_executor: TradeExecutor | None = None

        self._is_running = False
        self._server_task: asyncio.Task | None = None
        self._market_limit = market_limit

        self.trackers: dict[str, MarketTracker] = {}
        self._last_discovery = 0.0
        self._cycle_count = 0

        logger.info("Scanner Navigator initialized", market_limit=market_limit)

    async def __aenter__(self) -> "Navigator":
        """Initialize components."""
        logger.info("Starting Scanner components...")

        config_uv = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
        self._uvicorn_server = uvicorn.Server(config_uv)
        self._server_task = asyncio.create_task(self._uvicorn_server.serve())
        await asyncio.sleep(0.5)
        setup_web_logging()
        await monitor.update_status(status="STARTING")

        self._polymarket = PolymarketClient()
        await self._polymarket.__aenter__()

        self._trade_store = TradeStore()
        set_trade_store(self._trade_store)

        try:
            recent_trades = await self._trade_store.get_recent_trades(limit=50)
            monitor.state.trades_executed = recent_trades
        except Exception as e:
            logger.warning(f"Could not load recent trades for UI: {e}")

        self._rust_client = RustClient()
        self._trade_executor = TradeExecutor(
            rust_client=self._rust_client,
            trade_store=self._trade_store,
            paper_mode=(config.trading_mode == "paper")
        )

        self._kill_switch = KillSwitch(initial_capital=10000, on_trigger=self._on_kill_switch_trigger)
        await self._kill_switch.load_state()

        self._position_sizer = PositionSizer(capital=10000)
        self._price_cache = PriceCache(stale_threshold_seconds=5.0)

        if self._polymarket:
            async def on_ws_update(token_id: str, book: OrderBook):
                await self._price_cache.update(token_id, book)
                
                # Update recent prices immediately on WS tick (for momentum calculation)
                for tracker in self.trackers.values():
                    if tracker.info.token_id == token_id:
                        mid = book.mid_price
                        if mid is not None:
                            tracker.info.price = float(mid)
                            tracker.recent_prices.append((time.time(), float(mid)))

            self._ws_update_callback = on_ws_update

        self._is_running = True
        await monitor.update_status(status="ONLINE", active_solvers=0)
        return self

    async def _on_kill_switch_trigger(self, reason: str) -> None:
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
        if self._polymarket:
            await self._polymarket.cancel_all_orders()

    def stop(self) -> None:
        self._is_running = False

    async def __aexit__(self, *args: Any) -> None:
        logger.info("Shutting down Scanner...")
        self._is_running = False

        if getattr(self, "_uvicorn_server", None) is not None:
            self._uvicorn_server.should_exit = True
            if self._server_task is not None:
                try:
                    await asyncio.wait_for(self._server_task, timeout=2.0)
                except Exception:
                    pass

        if self._trade_store:
            await self._trade_store.close()
        if self._polymarket:
            await self._polymarket.__aexit__(*args)

    # --- Scanner Logic ---

    async def discover_markets(self) -> list[MarketInfo]:
        """Fetch Tier-1 active markets < $0.10 from Gamma API."""
        markets: list[MarketInfo] = []
        limit = 100
        offset = 0

        while True:
            response = await self._polymarket._retry_get(
                self._polymarket._gamma_client,
                "/markets",
                params={"limit": limit, "offset": offset, "active": "true", "closed": "false"}
            )
            data = response.json()
            if not data:
                break

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

                    if not yes_token:
                        continue

                    markets.append(MarketInfo(
                        condition_id=m.get("conditionId", m.get("id", "")),
                        slug=m.get("slug", "unknown"),
                        question=m.get("question", ""),
                        token_id=yes_token,
                        price=yes_price,
                    ))
                except Exception:
                    continue

            if self._market_limit and len(markets) >= self._market_limit:
                markets = markets[:self._market_limit]
                break

            if len(data) < limit:
                break
            offset += limit

        logger.info(f"Tier-1: Discovered {len(markets)} candidate markets (YES < ${MAX_YES_PRICE})")
        return markets

    async def warm_up_tracker(self, tracker: MarketTracker):
        """Tier 2/3 Prep: Download ~200 minutes of history via CLOB API."""
        try:
            response = await self._polymarket._retry_get(
                self._polymarket._clob_client,
                "/prices-history",
                params={
                    "market": tracker.info.condition_id,
                    "interval": "1m",
                    "fidelity": WARMUP_HISTORY_MINUTES,
                }
            )
            data = response.json()
            history = data if isinstance(data, list) else data.get("history", [])

            for point in (history or []):
                tracker.candles.append(Candle(
                    timestamp=float(point.get("t", 0)),
                    open=float(point.get("o", point.get("p", 0))),
                    high=float(point.get("h", point.get("p", 0))),
                    low=float(point.get("l", point.get("p", 0))),
                    close=float(point.get("c", point.get("p", 0))),
                    volume=float(point.get("v", 0)),
                ))
            if tracker.candles:
                tracker.compute_ema()
                tracker.warmed_up = True
        except Exception as e:
            logger.warning(f"Warm up failed for {tracker.info.slug}: {e}")

    async def tier2_check(self, tracker: MarketTracker) -> bool:
        avg_vol = tracker.avg_volume_10d()
        cur_vol = tracker.latest_volume()
        vol_ok = avg_vol > 0 and cur_vol > VOLUME_SPIKE_MULT * avg_vol
        mom = tracker.momentum_90s()
        mom_ok = abs(mom) > MOMENTUM_THRESHOLD
        return vol_ok or mom_ok

    async def tier3_check(self, tracker: MarketTracker) -> Optional[dict]:
        # Use WebSocket cached order book instead of making HTTP call
        book = self._price_cache.get(tracker.info.token_id) if self._price_cache else None
        
        imbalance = 0.0
        if book:
            bid_vol = float(book.total_bid_depth())
            ask_vol = float(book.total_ask_depth())
            total = bid_vol + ask_vol
            if total > 0:
                imbalance = max(bid_vol, ask_vol) / total
        else:
            # Fallback if WS not populated yet
            ob = await self._polymarket.get_order_book(tracker.info.token_id)
            if ob:
                bid_vol = float(ob.total_bid_depth())
                ask_vol = float(ob.total_ask_depth())
                total = bid_vol + ask_vol
                if total > 0:
                    imbalance = max(bid_vol, ask_vol) / total

        if imbalance < ORDERBOOK_IMBALANCE:
            return None

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
            "vol_mult": (tracker.latest_volume() / tracker.avg_volume_10d()) if tracker.avg_volume_10d() > 0 else 0,
        }

    async def execute_signal(self, tracker: MarketTracker, details: dict):
        """Execute the trade when a Strong Signal is found."""
        if self._kill_switch and not self._kill_switch.is_trading_allowed():
            logger.warning("Kill switch active - bypassing execution.")
            return

        # Determine side based on deviation. If current price < expected, BUY YES.
        side = OrderSide.BUY if tracker.info.price < details["expected"] else OrderSide.SELL
        
        # Position sizing
        size_usdc = self._position_sizer.calculate_total_size(Decimal("1.0"))  # Using max size for now
        shares = Decimal(str(size_usdc)) / Decimal(str(max(tracker.info.price, 0.01)))
        
        trade = ProposedTrade(
            outcome_id=tracker.info.token_id,
            side=side,
            size=Decimal(str(int(shares))),  # Whole shares
            limit_price=Decimal(str(tracker.info.price)),
            time_in_force=TimeInForce.FOK,
            reason=f"Volatility/Momentum Spikes: Volx{details['vol_mult']:.1f}, Dev {details['deviation']:.1%}"
        )

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        logger.info(f"🚨 STRONG SIGNAL [{ts}] - {tracker.info.slug} - Executing {trade.side.value} it at {trade.limit_price}")
        
        # Execute
        result = await self._trade_executor.execute_trade(trade)
        await monitor.update_status(last_action=f"Trade {trade.side.value} on {tracker.info.slug}")

    async def scan_once(self):
        self._cycle_count += 1
        now = time.time()
        logger.info(f"── Cycle #{self._cycle_count} ──────────────────────────────────────────")

        # 1. Market Discovery
        if now - self._last_discovery > MARKET_DISCOVERY_INTERVAL:
            markets = await self.discover_markets()
            self._last_discovery = now

            new_tokens = []
            seen = set()
            for m in markets:
                seen.add(m.condition_id)
                if m.condition_id not in self.trackers:
                    t = MarketTracker(info=m)
                    await self.warm_up_tracker(t)
                    if t.warmed_up:
                        self.trackers[m.condition_id] = t
                        new_tokens.append(m.token_id)
                else:
                    self.trackers[m.condition_id].info.price = m.price

            # Prune stale
            stale = [k for k in self.trackers if k not in seen]
            for k in stale:
                del self.trackers[k]

            # Subscribe to WS for new tokens
            if new_tokens and getattr(self._polymarket, "ws_client", None):
                logger.info(f"WebSocket Subscribing to {len(new_tokens)} new tokens...")
                await self._polymarket.ws_client.subscribe(new_tokens, self._ws_update_callback)

        # 2. Check Signals
        tier2_passed = []
        for tracker in self.trackers.values():
            if await self.tier2_check(tracker):
                tier2_passed.append(tracker)

        signals = 0
        if tier2_passed:
            for tracker in tier2_passed:
                details = await self.tier3_check(tracker)
                if details:
                    signals += 1
                    await self.execute_signal(tracker, details)

        logger.info(f"Cycle #{self._cycle_count} | {len(self.trackers)} Tracked | T2 Passed: {len(tier2_passed)} | Signals: {signals}")

    async def run(self):
        logger.info("Initializing WebSocket client...")
        if not getattr(self._polymarket, "ws_client", None):
            from polyquant.data.polymarket_client import PolymarketWSClient
            self._polymarket.ws_client = PolymarketWSClient()
            await self._polymarket.ws_client.connect()

        logger.info("Starting Scan Loop...")
        while self._is_running:
            try:
                await self.scan_once()
            except Exception as exc:
                logger.exception("Scan cycle error: %s", exc)
            await asyncio.sleep(SCAN_INTERVAL)
