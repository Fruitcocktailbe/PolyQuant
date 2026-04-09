"""
Navigator - Real-time trading engine.

The Navigator is the "fast brain" of PolyQuant. It runs continuously to:
1. Load the pre-computed Constraint Map from disk.
2. Connect to Polymarket's WebSocket for real-time prices.
3. Run the Solver (Frank-Wolfe) on price updates.
4. Execute trades when profitable opportunities are found.

The Navigator does NOT use LLMs. All logic is pre-computed by the Map Maker.

SPEED REQUIREMENTS:
-------------------
- Tick-to-Decision: <10ms (excluding network)
- Decision-to-Execution: <30ms (WebSocket submission)
- Total Latency Target: <50ms

USAGE:
------
    # Run as a script
    python -m polyquant.navigator
    
    # Or programmatically
    from polyquant.navigator import Navigator
    
    async def run():
        navigator = Navigator()
        await navigator.run()
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any

from polyquant.data import PolymarketClient, OrderBook
from polyquant.data.constraint_store import ConstraintStore, ConstraintManifest
from polyquant.data.price_cache import PriceCache
from polyquant.execution.executor import TradeExecutor
from polyquant.risk import KillSwitch, PositionSizer
from polyquant.solver import ArbitrageDetector, SCIPSolver
from polyquant.agents import MicrostructureAgent
from polyquant.api.server import monitor, app, set_trade_store
from polyquant.utils import config, get_logger
from polyquant.utils.profiling import timed_operation, async_timed, print_latency_report  # Week 4
import uvicorn

logger = get_logger(__name__)


class LatencyTracker:
    """
    Tracks latency metrics for the Navigator's trading loop.

    This class maintains a rolling window of latency measurements
    to detect performance degradation and feed the KillSwitch.

    Metrics tracked:
    - tick_to_decision: Time from price update to opportunity detection
    - decision_to_execution: Time from detection to order submission
    - total_latency: End-to-end latency
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize the LatencyTracker.

        Args:
            window_size: Number of samples to keep for rolling statistics
        """
        self._window_size = window_size
        self._tick_to_decision: list[float] = []
        self._decision_to_execution: list[float] = []
        self._total_latency: list[float] = []

    def record_tick_to_decision(self, latency_ms: float) -> None:
        """Record time from price update to opportunity detection."""
        self._tick_to_decision.append(latency_ms)
        if len(self._tick_to_decision) > self._window_size:
            self._tick_to_decision.pop(0)

    def record_decision_to_execution(self, latency_ms: float) -> None:
        """Record time from detection to execution."""
        self._decision_to_execution.append(latency_ms)
        if len(self._decision_to_execution) > self._window_size:
            self._decision_to_execution.pop(0)

    def record_total_latency(self, latency_ms: float) -> None:
        """Record end-to-end latency."""
        self._total_latency.append(latency_ms)
        if len(self._total_latency) > self._window_size:
            self._total_latency.pop(0)

    def get_average_total_latency(self) -> float:
        """Get average total latency over the window."""
        if not self._total_latency:
            return 0.0
        return sum(self._total_latency) / len(self._total_latency)

    def get_p95_total_latency(self) -> float:
        """Get 95th percentile total latency."""
        if not self._total_latency:
            return 0.0
        sorted_latencies = sorted(self._total_latency)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive latency statistics."""
        return {
            "tick_to_decision_avg": (
                sum(self._tick_to_decision) / len(self._tick_to_decision)
                if self._tick_to_decision else 0.0
            ),
            "decision_to_execution_avg": (
                sum(self._decision_to_execution) / len(self._decision_to_execution)
                if self._decision_to_execution else 0.0
            ),
            "total_latency_avg": self.get_average_total_latency(),
            "total_latency_p95": self.get_p95_total_latency(),
            "samples": len(self._total_latency),
        }


class ExecutionGuard:
    """
    Fast, in-memory constraint checker.
    
    This replaces the slow LLM-based Validator for real-time trading.
    It loads the ConstraintManifest from disk and checks trades
    against the pre-computed rules in microseconds.
    """
    
    def __init__(self):
        """Initialize the ExecutionGuard."""
        self._manifests: dict[str, ConstraintManifest] = {}
        self._constraint_matrix: dict[str, Any] = {}  # Pre-computed matrix
        
        logger.info("ExecutionGuard initialized")
    
    def load_manifests(self, manifests: list[ConstraintManifest]) -> None:
        """
        Load constraint manifests into memory.
        
        Args:
            manifests: List of ConstraintManifests to load.
        """
        for manifest in manifests:
            self._manifests[manifest.cluster_id] = manifest
            
            # Pre-compute constraint matrix for O(1) lookups
            for constraint in manifest.constraints:
                for outcome_id, coeff in constraint.coefficients.items():
                    if outcome_id not in self._constraint_matrix:
                        self._constraint_matrix[outcome_id] = []
                    
                    data_item: dict[str, Any] = {
                        "constraint_id": constraint.constraint_id,
                        "coefficient": coeff,
                        "rhs": constraint.rhs,
                    }
                    self._constraint_matrix[outcome_id].append(data_item)
        
        logger.info(
            "Loaded manifests into ExecutionGuard",
            manifest_count=len(manifests),
            total_constraints=sum(m.constraint_count for m in manifests),
        )
    
    def check_trade(
        self,
        outcome_id: str,
        side: str,  # "buy" or "sell"
        size: float,
        price: float,
    ) -> tuple[bool, str]:
        """
        Check if a trade is valid against the loaded constraints.

        This is the HOT PATH. Must complete in <1ms.

        Performs fast pre-flight checks before execution. The ArbitrageDetector
        has already validated the trade respects constraints using the Frank-Wolfe
        solver. This is a last-mile sanity check for basic validity.

        Args:
            outcome_id: The outcome being traded.
            side: "buy" or "sell".
            size: Trade size.
            price: Trade price.

        Returns:
            Tuple of (is_valid, reason).
        """
        # ========== BASIC VALIDATION (Must be fast!) ==========

        # Check 1: Validate price bounds (0 < price < 1)
        if price <= 0.0 or price >= 1.0:
            logger.warning(
                "Trade rejected: Invalid price",
                outcome_id=outcome_id,
                price=price,
                reason="Price must be in range (0, 1)"
            )
            return False, "invalid_price"

        # Check 2: Validate size (must be positive)
        if size <= 0:
            logger.warning(
                "Trade rejected: Invalid size",
                outcome_id=outcome_id,
                size=size,
                reason="Size must be positive"
            )
            return False, "invalid_size"

        # Check 3: Warn on extreme prices (likely resolved or broken market)
        if price < 0.02:
            logger.warning(
                "Trade warning: Extremely low price",
                outcome_id=outcome_id,
                price=price,
                reason="Price < 0.02 suggests market may be resolved or illiquid"
            )
            # Allow but warn - position sizer will likely reject anyway

        if price > 0.98:
            logger.warning(
                "Trade warning: Extremely high price",
                outcome_id=outcome_id,
                price=price,
                reason="Price > 0.98 suggests market may be resolved or illiquid"
            )
            # Allow but warn

        # Check 4: Validate side
        if side not in ("buy", "sell"):
            logger.warning(
                "Trade rejected: Invalid side",
                outcome_id=outcome_id,
                side=side,
                reason="Side must be 'buy' or 'sell'"
            )
            return False, "invalid_side"

        # ========== CONSTRAINT VALIDATION ==========

        # Check 5: Is the outcome in our constraint matrix?
        if outcome_id not in self._constraint_matrix:
            # No constraints on this outcome - this is unusual but not necessarily wrong
            # It might be an outcome that's not part of any constrained cluster
            logger.debug(
                "Trade has no constraints",
                outcome_id=outcome_id,
                reason="Outcome not found in constraint matrix"
            )
            return True, "no_constraints"

        # Check 6: Validate against constraints
        # Get all constraints involving this outcome
        constraints = self._constraint_matrix[outcome_id]

        # For each constraint, we need to check:
        # sum(coefficient_i * position_i) >= rhs
        #
        # However, we don't have current positions here (this is a pre-trade check).
        # The ArbitrageDetector has already validated this trade respects constraints.
        # So we do basic sanity checks:

        for constraint in constraints:
            coeff = constraint["coefficient"]
            rhs = constraint["rhs"]

            # Sanity check 1: If this is essentially a zero coefficient
            if abs(coeff) < 0.0001:
                # This outcome doesn't actually affect this constraint
                continue

            # Sanity check 2: If coefficient and RHS suggest impossible situation
            # For example: if coefficient is 1.0 and rhs is 2.0 (impossible for single outcome)
            if len(constraints) == 1 and coeff > 0 and rhs > 1.0:
                logger.warning(
                    "Suspicious constraint",
                    outcome_id=outcome_id,
                    constraint_id=constraint["constraint_id"],
                    coefficient=coeff,
                    rhs=rhs,
                    reason="Single outcome constraint with RHS > 1.0"
                )
                # Don't reject - might be multi-outcome constraint we're seeing partially

        # ========== PASS ==========

        # If we reached here, all basic checks passed
        # The ArbitrageDetector has already done the heavy lifting (LP validation)
        # This was just a last-mile sanity check

        logger.debug(
            "Trade passed ExecutionGuard",
            outcome_id=outcome_id,
            side=side,
            size=size,
            price=price,
            constraints_checked=len(constraints)
        )

        return True, "passed"
    
    def get_cluster_for_outcome(self, outcome_id: str) -> str | None:
        """Get the cluster ID that contains this outcome."""
        for cluster_id, manifest in self._manifests.items():
            for constraint in manifest.constraints:
                if outcome_id in constraint.coefficients:
                    return cluster_id
        return None


class Navigator:
    """
    Real-time trading engine.
    
    The Navigator loads pre-computed constraints and trades
    against live market data. It uses:
    - ExecutionGuard for fast constraint checking.
    - ArbitrageDetector for profit calculation.
    - SCIPSolver for optimal trade sizing.
    
    Example:
        async with Navigator() as navigator:
            await navigator.run()
    """
    
    def __init__(self):
        """Initialize the Navigator."""
        self._polymarket: PolymarketClient | None = None
        self._store: ConstraintStore | None = None
        self._guard: ExecutionGuard | None = None
        self._executor: TradeExecutor | None = None
        self._arbitrage_detector: ArbitrageDetector | None = None
        self._solver: SCIPSolver | None = None
        self._kill_switch: KillSwitch | None = None
        self._position_sizer: PositionSizer | None = None
        self._microstructure_agent: MicrostructureAgent | None = None

        # Arbitrage improvement modules
        self._bayesian_updater = None   # BayesianUpdater (phantom arb prevention)
        self._correlation_engine = None  # CorrelationEngine (leader-laggard pairs)
        self._previous_prices: dict[str, float] = {}  # For correlation delta tracking

        # Safety modules
        self._trade_store = None  # TradeStore (ACID persistence)
        self._balance_refresh_task: asyncio.Task | None = None  # Background balance refresh
        
        self._is_running = False
        self._server_task: asyncio.Task | None = None

        # Metrics
        self._ticks_processed = 0
        self._opportunities_found = 0
        self._trades_executed = 0

        # Latency tracking
        self._latency_tracker = LatencyTracker(window_size=100)

        # Missing attributes reported by IDE
        self._price_cache: PriceCache | None = None
        self._mock_task: asyncio.Task | None = None
        self._ws_update_callback: Any | None = None

        logger.info("Navigator initialized")
    
    async def __aenter__(self) -> "Navigator":
        """Initialize all components."""
        logger.info("Starting Navigator...")
        
        # Start Sidecar UI Server
        config_uv = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
        server = uvicorn.Server(config_uv)
        self._server_task = asyncio.create_task(server.serve())
        await monitor.update_status(status="STARTING")
        
        # Initialize Polymarket client
        self._polymarket = PolymarketClient()
        await self._polymarket.__aenter__()
        
        # Load constraint store
        self._store = ConstraintStore()
        manifests = await self._store.load_all_manifests()
        
        # Initialize execution guard with pre-computed constraints
        self._guard = ExecutionGuard()
        self._guard.load_manifests(manifests)

        # Bayesian Updater: load dependency graph from manifests
        from polyquant.agents.bayesian_updater import BayesianUpdater
        self._bayesian_updater = BayesianUpdater()
        self._bayesian_updater.load_dependencies(manifests)
        
        # Initialize solver components
        self._solver = SCIPSolver()
        self._arbitrage_detector = ArbitrageDetector()
        self._microstructure_agent = MicrostructureAgent()

        # Correlation Engine: pairs are scanned during MapMaker,
        # Navigator only checks for live signals (fast, no IO)
        from polyquant.agents.correlation import CorrelationEngine
        self._correlation_engine = CorrelationEngine()

        # Initialize TradeStore (SQLite, ACID, WAL mode)
        from polyquant.data.trade_store import TradeStore
        self._trade_store = TradeStore()
        await self._trade_store.initialize()
        set_trade_store(self._trade_store)  # Share with API server for /api/trades

        # Fetch USDC balance (cold path, cached locally)
        cached_balance = Decimal(str(config.initial_capital))
        if config.trading_mode == "live" and self._polymarket:
            try:
                cached_balance = await self._polymarket.get_usdc_balance()
            except Exception:
                logger.warning("Balance fetch failed, using config.initial_capital")

        # Initialize trade executor (paper or live based on config)
        self._executor = TradeExecutor(
            client=self._polymarket,
            trading_mode=config.trading_mode,
            trade_store=self._trade_store,
            cached_balance=cached_balance,
            kill_switch=self._kill_switch,
        )

        # Start background balance refresh (every 60s, never on hot path)
        if config.trading_mode == "live":
            self._balance_refresh_task = asyncio.create_task(
                self._refresh_balance_loop()
            )
        
        # Initialize risk management (using config-driven capital)
        self._kill_switch = KillSwitch(
            initial_capital=config.initial_capital,
            on_trigger=self._on_kill_switch_trigger,
        )
        await self._kill_switch.load_state()
        self._position_sizer = PositionSizer(capital=config.initial_capital)
        
        self._is_running = True
        await monitor.update_status(status="ONLINE")
        
        # Initialize price cache for low-latency access
        self._price_cache = PriceCache(stale_threshold_seconds=2.0)
        
        # Subscribe cache to WebSocket updates
        # NOTE: This connects the WS client (in PolymarketClient) to our local cache
        if self._polymarket and self._polymarket._ws_client:
            # We need to bridge the WS client callback to our cache update
            # The WS client expects a callback(book: OrderBook)
            async def on_ws_update(book: OrderBook):
                await self._price_cache.update(book.outcome_id, book)
            
            # This will be registered when we subscribe to specific tokens
            self._ws_update_callback = on_ws_update
            
        logger.info(
            "Navigator started",
            manifests_loaded=len(manifests),
        )
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Cleanup all components."""
        logger.info("Shutting down Navigator...")
        
        self._is_running = False
        
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        if self._polymarket:
            await self._polymarket.__aexit__(*args)

        # Cancel balance refresh
        if self._balance_refresh_task:
            self._balance_refresh_task.cancel()
            try:
                await self._balance_refresh_task
            except asyncio.CancelledError:
                pass

        # Close TradeStore
        if self._trade_store:
            await self._trade_store.close()
            
        if self._price_cache:
            self._price_cache.clear()
        
        logger.info(
            "Navigator shutdown complete",
            ticks_processed=self._ticks_processed,
            opportunities_found=self._opportunities_found,
            trades_executed=self._trades_executed,
        )
    
    async def _refresh_balance_loop(self) -> None:
        """
        Background task: refresh the executor's cached USDC balance every 60s.

        This ensures the balance check stays accurate without ever touching
        the hot path. Runs as a fire-and-forget asyncio.Task.
        """
        while self._is_running:
            try:
                await asyncio.sleep(60)
                if self._polymarket and self._executor:
                    new_balance = await self._polymarket.get_usdc_balance()
                    if new_balance > Decimal("0"):
                        self._executor._cached_balance = new_balance
                        logger.debug(
                            "Balance refreshed",
                            balance=str(new_balance),
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Balance refresh failed", error=str(e))

    async def run(self, max_ticks: int | None = None) -> None:
        """
        Run the Navigator's real-time trading loop.
        
        Architecture:
        1. Subscribe to WebSocket updates for all relevant markets.
        2. Event loop waits for price updates (via PriceCache callbacks).
        3. On update, trigger arbitrage detection.
        4. Execute profitable trades immediately.
        
        Args:
            max_ticks: Maximum number of ticks to process (None = infinite).
        """
        logger.info("Starting trading loop", max_ticks=max_ticks)
        
        # Get all markets we have constraints for
        if not self._store:
            raise RuntimeError("ConstraintStore not initialized")
        
        cluster_ids = await self._store.list_clusters()
        
        if not cluster_ids:
            logger.warning("No constraint manifests found. Run Map Maker first.")
            await monitor.update_status(status="NO_CONSTRAINTS")
            return
        
        # Load all market IDs from manifests
        market_ids = []
        for cluster_id in cluster_ids:
            manifest = await self._store.load_manifest(cluster_id)
            if manifest:
                market_ids.extend(manifest.market_ids)
        
        if not market_ids:
            logger.warning("No markets found in manifests")
            return
        
        logger.info(f"Monitoring {len(market_ids)} markets")
        
        # 1. Initial snapshot fetch (to populate cache before WS takes over)
        logger.info("Fetching initial order book snapshots...")
        initial_books = await self._fetch_order_books(market_ids)
        for token_id, book in initial_books.items():
            await self._price_cache.update(token_id, book)
            
        # 2. Subscribe to WebSocket updates
        # We need token IDs, not market IDs, for subscription
        token_ids = list(initial_books.keys())
        if self._polymarket and token_ids:
            logger.info(f"Subscribing to {len(token_ids)} tokens via WebSocket")
            # Create WS client if not exists (it should be in PolymarketClient)
            # For now, simplistic access
            if not getattr(self._polymarket, "ws_client", None):
                 # Initialize WS client if missing (add to PolymarketClient later)
                 from polyquant.data.polymarket_client import PolymarketWSClient
                 self._polymarket.ws_client = PolymarketWSClient()
                 await self._polymarket.ws_client.connect()
            
            await self._polymarket.ws_client.subscribe(
                token_ids, 
                self._ws_update_callback
            )
            
        # 3. Main Event Loop
        tick_count = 0
        while self._is_running and (max_ticks is None or tick_count < max_ticks):
            # Event-driven architecture: Wait for price updates instead of polling
            # This saves ~5ms per tick and eliminates CPU waste
            await self._price_cache.wait_for_update()

            tick_start = datetime.utcnow()

            # Check connection health (Fix 8: hardened via PolymarketClient API)
            if self._polymarket and not self._polymarket.is_ws_healthy():
                logger.warning(
                    "Trading blocked: Stale WebSocket connection",
                    reason="No price updates received for >30 seconds"
                )
                if self._kill_switch:
                    self._kill_switch.record_api_error()

                await asyncio.sleep(1)
                continue

            # Check kill switch
            if self._kill_switch and not self._kill_switch.can_trade():
                logger.warning("Trading blocked by kill switch")
                await asyncio.sleep(1)
                continue

            # Collect stale tokens (Fix 8: hardened via PolymarketClient API)
            stale_tokens: set[str] = (
                self._polymarket.get_stale_tokens() if self._polymarket else set()
            )

            try:
                # Get fresh order books from cache (O(1) access)
                current_books = self._price_cache.get_all()

                if not current_books:
                     # No fresh data yet, wait for next update
                     continue

                # Timestamp: Start opportunity detection
                detect_start = datetime.utcnow()

                # Check for arbitrage opportunities
                opportunities = await self._detect_opportunities(
                    current_books, stale_tokens=stale_tokens
                )

                # Record tick-to-decision latency
                detect_elapsed = (datetime.utcnow() - detect_start).total_seconds() * 1000
                self._latency_tracker.record_tick_to_decision(detect_elapsed)

                # Execute trades
                if opportunities:
                    exec_start = datetime.utcnow()

                    for opp in opportunities:
                        await self._execute_opportunity(opp)

                    # Record decision-to-execution latency
                    exec_elapsed = (datetime.utcnow() - exec_start).total_seconds() * 1000
                    self._latency_tracker.record_decision_to_execution(exec_elapsed)

                tick_count += 1
                self._ticks_processed = tick_count

                # Record total latency
                tick_elapsed = (datetime.utcnow() - tick_start).total_seconds() * 1000
                self._latency_tracker.record_total_latency(tick_elapsed)

                # Check latency and log warnings
                if tick_elapsed > 50:
                    logger.warning(
                        "Tick exceeded latency target",
                        tick=tick_count,
                        elapsed_ms=tick_elapsed,
                        latency_stats=self._latency_tracker.get_stats(),
                    )

                    # Feed high latency to kill switch
                    if self._kill_switch:
                        await self._kill_switch.record_latency(tick_elapsed)

                # Periodic latency logging (every 100 ticks)
                if tick_count % 100 == 0:
                    stats = self._latency_tracker.get_stats()
                    logger.info(
                        "Latency statistics",
                        tick=tick_count,
                        **stats,
                    )
                
            except Exception as e:
                logger.error("Tick failed", error=str(e))

            # No sleep needed - event-driven architecture handles timing
    
    async def _fetch_order_books(
        self,
        market_ids: list[str],
    ) -> dict[str, OrderBook]:
        """Fetch order books for all markets."""
        if not self._polymarket:
            return {}

        results = {}
        # Simple implementation: fetch one by one
        # In production, we'd use WebSocket updates
        for mid in market_ids:
            # We need the outcomes to get token_ids
            # This is slow, but better than nothing for now
            # TODO: Cache market objects to avoid re-fetching metadata
            market = await self._polymarket._gamma_client.get(f"/markets/{mid}")
            if market.status_code == 200:
                data = market.json()
                for token in data.get("tokens", []):
                    tid = token.get("token_id")
                    if tid:
                        ob = await self._polymarket.get_order_book(tid)
                        if ob:
                            results[tid] = ob
        return results
    
    async def _detect_opportunities(
        self,
        order_books: dict[str, OrderBook],
        stale_tokens: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Detect arbitrage opportunities in the current order books.

        Pipeline per cluster:
        1. Skip if any token has stale WS data (sequence gap)
        2. Microstructure analysis (imbalance warnings)
        3. Bayesian price adjustment (phantom arb prevention)
        4. Frank-Wolfe solver via ArbitrageDetector
        5. Correlation signal check (leader-laggard pairs)

        Args:
            order_books: Dictionary of token_id -> OrderBook
            stale_tokens: Set of token_ids with WS sequence gaps

        Returns:
            List of opportunity dictionaries
        """
        if not self._guard or not self._arbitrage_detector:
            return []

        stale_tokens = stale_tokens or set()
        opportunities = []

        # Build current mid-prices for Bayesian + Correlation (O(n) once)
        current_mid_prices: dict[str, float] = {}
        for token_id, book in order_books.items():
            mid = book.mid_price if hasattr(book, 'mid_price') else 0.0
            if mid and mid > 0:
                current_mid_prices[token_id] = mid

        # Group order books by cluster
        cluster_books: dict[str, dict[str, OrderBook]] = {}
        for token_id, book in order_books.items():
            cluster_id = self._guard.get_cluster_for_outcome(token_id)
            if cluster_id:
                if cluster_id not in cluster_books:
                    cluster_books[cluster_id] = {}
                cluster_books[cluster_id][token_id] = book

        # ── Per-cluster detection ──
        for cluster_id, books in cluster_books.items():
            if not books:
                continue

            try:
                # 0. Skip clusters with stale WS data (sequence gaps)
                if stale_tokens and any(tid in stale_tokens for tid in books):
                    logger.debug(
                        "Skipping cluster with stale WS data",
                        cluster_id=cluster_id,
                    )
                    continue

                # 1. Microstructure Analysis
                if self._microstructure_agent:
                    for tid, book in books.items():
                        signal = self._microstructure_agent.analyze(book)
                        if abs(signal.imbalance) > 0.5:
                            logger.info(
                                "Critical imbalance detected",
                                cluster_id=cluster_id,
                                token_id=tid,
                                imbalance=signal.imbalance
                            )

                # 2. Bayesian price adjustment (phantom arb prevention)
                if self._bayesian_updater:
                    cluster_prices = {
                        tid: current_mid_prices.get(tid, 0.0)
                        for tid in books
                    }
                    adjusted_prices, adjustments = (
                        self._bayesian_updater.adjust_prices(cluster_prices)
                    )
                    if adjustments:
                        for adj in adjustments:
                            logger.debug(adj.reason)

                # 3. Run Arbitrage Detector (using manifest)
                if not self._guard:
                    continue
                manifest = self._guard._manifests.get(cluster_id)
                if not manifest:
                    continue

                try:
                    arb_opportunity = await self._arbitrage_detector.detect(
                        validated=manifest,
                        order_books=cluster_books,
                        min_profit=config.fw_min_profit
                    )
                except Exception as e:
                    logger.error(f"ArbitrageDetector failed: {e}", cluster_id=cluster_id)
                    arb_opportunity = None

                if arb_opportunity:
                    opp_data = {
                        "cluster_id": cluster_id,
                        "source": "constraint",
                        "expected_profit": float(arb_opportunity.expected_profit),
                        "trades": [
                            {
                                "outcome_id": t.outcome_id,
                                "side": t.side.value,
                                "size": float(t.size),
                                "limit_price": float(t.limit_price),
                            }
                            for t in arb_opportunity.trades
                        ],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    opportunities.append(opp_data)
                    
                    # Update monitor
                    from polyquant.api.server import monitor
                    asyncio.create_task(monitor.update_status(
                        opportunities=opportunities[-10:]
                    ))

            except Exception as e:
                logger.error(
                    "Opportunity detection failed for cluster",
                    cluster_id=cluster_id,
                    error=str(e),
                )

        # ── Correlation-based signals (cross-cluster, after constraint detection) ──
        if (self._correlation_engine and
            self._previous_prices and current_mid_prices):
            try:
                corr_signals = self._correlation_engine.check_for_signals(
                    current_prices=current_mid_prices,
                    previous_prices=self._previous_prices,
                )
                for sig in corr_signals:
                    # Fix 7: Generate executable trades from correlation signals
                    if sig.leader_move > 0:
                        # Leader went UP → laggard should follow UP → BUY laggard
                        side = "buy"
                        limit_price = min(sig.expected_laggard_price, 0.99)
                    else:
                        # Leader went DOWN → laggard should follow DOWN → SELL laggard
                        side = "sell"
                        limit_price = max(sig.expected_laggard_price, 0.01)

                    # Conservative sizing proportional to deviation strength
                    base_size = 50.0  # $50 base for correlation trades
                    size = base_size * min(sig.deviation_sigma / 2.0, 2.0)

                    opportunities.append({
                        "cluster_id": f"corr_{sig.pair.leader_id[:8]}",
                        "source": "correlation",
                        "expected_profit": abs(sig.expected_laggard_move) * size,
                        "trades": [{
                            "outcome_id": sig.pair.laggard_id,
                            "market_id": "",
                            "side": side,
                            "size": size,
                            "limit_price": limit_price,
                            "priority": 5,  # Lower priority than constraint arb
                        }],
                        "leader": sig.pair.leader_question[:60],
                        "laggard": sig.pair.laggard_question[:60],
                        "deviation_sigma": sig.deviation_sigma,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
            except Exception as e:
                logger.error("Correlation signal check failed", error=str(e))

        # Update previous prices for next tick's correlation delta
        self._previous_prices = current_mid_prices

        return opportunities
    
    def _calculate_execution_costs(
        self,
        trades: list[Any],
        order_books: dict[str, "OrderBook"] | None = None,
    ) -> dict[str, float]:
        """
        Calculate total execution costs: fees + gas + VWAP slippage.

        P0-1 (C1/C2): Polymarket taker fees + Polygon gas.
        P0-2 (C3): VWAP slippage from order book depth.

        Returns:
            Dict with cost breakdown and total.
        """
        num_legs = len(trades)
        total_notional = sum(
            float(t.size * t.limit_price) if hasattr(t, 'size') else
            float(t.get("size", 0) * t.get("limit_price", 0))
            for t in trades
        )

        # Taker fee: applied per leg on notional
        taker_fee = config.polymarket_taker_fee_pct * total_notional

        # Gas: per transaction on Polygon
        gas_cost = config.polygon_gas_per_tx * num_legs

        # VWAP slippage: estimate from order books if available
        slippage_cost = 0.0
        if order_books and self._price_cache:
            from polyquant.data import OrderSide
            for t in trades:
                outcome_id = t.outcome_id if hasattr(t, 'outcome_id') else t.get("outcome_id", "")
                size = Decimal(str(t.size if hasattr(t, 'size') else t.get("size", 0)))
                side_val = t.side if hasattr(t, 'side') else OrderSide(t.get("side", "buy"))
                limit_price = float(t.limit_price if hasattr(t, 'limit_price') else t.get("limit_price", 0))

                ob = self._price_cache.get(outcome_id)
                if ob and ob.mid_price and size > 0:
                    vwap = ob.get_vwap(side_val, size)
                    if vwap is not None:
                        slippage = abs(float(vwap) - float(ob.mid_price)) * float(size)
                        slippage_cost += slippage

        total_cost = taker_fee + gas_cost + slippage_cost

        return {
            "taker_fee": round(taker_fee, 4),
            "gas_cost": round(gas_cost, 4),
            "slippage_cost": round(slippage_cost, 4),
            "total_cost": round(total_cost, 4),
            "total_notional": round(total_notional, 4),
            "num_legs": num_legs,
        }

    def _pre_execution_price_check(
        self, opportunity: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        P0-5 (C9): Re-validate prices from cache before execution.

        Prices may have drifted since detection. Re-check that the
        arbitrage signal still exists using fresh cached prices.

        Returns:
            (is_valid, reason)
        """
        if not self._price_cache:
            return True, "no_cache"

        trades = opportunity.get("trades", [])
        if not trades:
            return True, "no_trades"

        cluster_id = opportunity.get("cluster_id", "")

        # Check each trade leg has a fresh price close to expected
        for td in trades:
            outcome_id = td.get("outcome_id", "") if isinstance(td, dict) else td.outcome_id
            expected_price = float(td.get("limit_price", 0) if isinstance(td, dict) else td.limit_price)

            ob = self._price_cache.get(outcome_id)
            if not ob or not ob.mid_price:
                continue  # No fresh data — allow (WS staleness filter catches this)

            current_mid = float(ob.mid_price)
            drift_pct = abs(current_mid - expected_price) / max(expected_price, 0.01)

            # If price drifted more than VWAP slippage limit, abort
            if drift_pct > config.vwap_slippage_limit:
                logger.warning(
                    "Pre-execution price drift exceeded limit",
                    cluster_id=cluster_id,
                    outcome_id=outcome_id,
                    expected=expected_price,
                    current=current_mid,
                    drift_pct=f"{drift_pct:.2%}",
                    limit=f"{config.vwap_slippage_limit:.2%}",
                )
                return False, f"price_drift_{outcome_id}"

        return True, "prices_valid"

    async def _execute_opportunity(self, opportunity: Any) -> None:
        """
        Execute a trading opportunity through the TradeExecutor.

        P0 Safety checks applied before execution:
        1. Pre-execution price validation (C9)
        2. Fee + gas + slippage cost deduction (C1/C2/C3)
        3. Net profit must exceed costs

        In paper mode this simulates fills; in live mode it submits
        real orders via the CLOB API.
        """
        self._opportunities_found += 1

        if not self._executor:
            logger.warning("No executor available, skipping opportunity")
            return

        # Build an OptimizationResult-like object from the opportunity dict
        trades_data = opportunity.get("trades", [])
        if not trades_data:
            logger.debug("Opportunity has no trades", opportunity=opportunity)
            return

        # ── P0-5: Pre-execution price validation ──
        price_valid, price_reason = self._pre_execution_price_check(opportunity)
        if not price_valid:
            logger.info(
                "Opportunity aborted: price drift",
                cluster_id=opportunity.get("cluster_id"),
                reason=price_reason,
            )
            return

        # Convert dicts back to ProposedTrade objects
        from polyquant.data import ProposedTrade, OrderSide
        from polyquant.solver.scip_solver import OptimizationResult

        proposed_trades = []
        for td in trades_data:
            proposed_trades.append(
                ProposedTrade(
                    market_id=td.get("market_id", ""),  # May not be set; CLOB uses outcome_id
                    outcome_id=td["outcome_id"],
                    side=OrderSide(td["side"]),
                    size=td["size"],
                    limit_price=td["limit_price"],
                    priority=td.get("priority", 1),
                )
            )

        # ── P0-1/2: Calculate execution costs (fees + gas + VWAP slippage) ──
        costs = self._calculate_execution_costs(proposed_trades)
        gross_profit = float(opportunity.get("expected_profit", 0.0))
        net_profit = gross_profit - costs["total_cost"]

        if net_profit <= 0:
            logger.info(
                "Opportunity unprofitable after costs — skipping",
                cluster_id=opportunity.get("cluster_id"),
                gross_profit=f"${gross_profit:.2f}",
                taker_fee=f"${costs['taker_fee']:.2f}",
                gas_cost=f"${costs['gas_cost']:.2f}",
                slippage=f"${costs['slippage_cost']:.2f}",
                total_cost=f"${costs['total_cost']:.2f}",
                net_profit=f"${net_profit:.2f}",
            )
            return

        logger.debug(
            "Profit after costs",
            gross=f"${gross_profit:.2f}",
            costs=f"${costs['total_cost']:.2f}",
            net=f"${net_profit:.2f}",
        )

        # Fix 2: Apply PositionSizer constraints (Kelly + exposure caps)
        if self._position_sizer and self._price_cache:
            sized_trades = []
            for trade in proposed_trades:
                ob = self._price_cache.get(trade.outcome_id)
                if ob:
                    pos = self._position_sizer.calculate_for_trade(
                        trade=trade, order_book=ob, probability=trade.limit_price,
                    )
                    if pos.is_positive_ev and pos.recommended_size > 0:
                        # Cap trade size to Kelly-recommended maximum
                        trade.size = min(trade.size, pos.recommended_size)
                        sized_trades.append(trade)
                    else:
                        logger.info(
                            "PositionSizer rejected trade",
                            outcome_id=trade.outcome_id,
                            reason=pos.limited_by,
                        )
                else:
                    sized_trades.append(trade)  # No book data = pass through
            proposed_trades = sized_trades

            if not proposed_trades:
                logger.info("All trades rejected by PositionSizer")
                return

        opt_result = OptimizationResult(
            success=True,
            trades=proposed_trades,
            expected_profit=net_profit,  # Use NET profit (after costs)
        )

        # Execute via TradeExecutor (paper or live)
        exec_result = await self._executor.execute_atomic(opt_result)

        # Fix 1: Record PnL to KillSwitch for drawdown tracking
        if self._kill_switch:
            if exec_result.success:
                # Successful arb: book the NET profit (after fees)
                await self._kill_switch.record_pnl(net_profit)
            else:
                # Failed execution with unwind: estimate slippage loss
                # H8: Use actual order book spread if available, fallback 3%
                unwind_spread_pct = 0.03  # Default fallback
                if self._price_cache and exec_result.fills:
                    spreads = []
                    for fill in exec_result.fills:
                        ob = self._price_cache.get(fill.trade.outcome_id)
                        if ob and ob.spread is not None:
                            spreads.append(float(ob.spread))
                    if spreads:
                        # Use worst spread + 1% safety margin
                        unwind_spread_pct = max(spreads) + 0.01

                unwind_loss = float(exec_result.total_filled) * unwind_spread_pct
                if unwind_loss > 0:
                    await self._kill_switch.record_pnl(-unwind_loss)

        if exec_result.success:
            self._trades_executed += len(exec_result.fills)
            logger.info(
                "Opportunity executed",
                cluster_id=opportunity.get("cluster_id"),
                fills=exec_result.trade_count,
                total_notional=exec_result.total_filled,
                net_profit=f"${net_profit:.2f}",
                trading_mode=config.trading_mode,
            )

            # Push to dashboard
            try:
                await monitor.update_status(
                    last_trade={
                        "cluster_id": opportunity.get("cluster_id"),
                        "fills": exec_result.trade_count,
                        "notional": round(exec_result.total_filled, 2),
                        "net_profit": round(net_profit, 2),
                        "costs": costs,
                        "mode": config.trading_mode,
                    }
                )
            except Exception:
                pass
        else:
            logger.warning(
                "Opportunity execution failed",
                cluster_id=opportunity.get("cluster_id"),
                reason=exec_result.reason,
            )
    
    def _on_kill_switch_trigger(self, event: Any) -> None:
        """Handle kill switch trigger — cancel all orders and halt."""
        logger.critical("KILL SWITCH TRIGGERED — cancelling all CLOB orders")
        monitor.trigger_kill_switch()
        self._is_running = False

        # Fix 3: Cancel all open orders on the exchange
        if self._polymarket:
            asyncio.create_task(self._polymarket.cancel_all_orders())


async def main() -> None:
    """Main entry point for the Navigator."""
    print("""
    ===============================================================
                       PolyQuant Navigator
                 Real-Time Trading Engine
    ===============================================================
    """)
    
    async with Navigator() as navigator:
        await navigator.run()


if __name__ == "__main__":
    asyncio.run(main())
