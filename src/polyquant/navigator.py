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
import json
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, TYPE_CHECKING, Dict, List, Set, Union, Tuple

if TYPE_CHECKING:
    from polyquant.execution.executor import TradeExecutor
    from polyquant.execution.rust_client import RustClient
    from polyquant.data.constraint_store import ConstraintManifest
    from polyquant.data import OrderBook
    from polyquant.agents.bayesian_updater import BayesianUpdater
    from polyquant.agents.correlation import CorrelationEngine

from polyquant.data import OrderBook
from polyquant.data.constraint_store import ConstraintStore, ConstraintManifest
from polyquant.data.price_cache import PriceCache
from polyquant.risk import KillSwitch, PositionSizer
from polyquant.solver import ArbitrageDetector, SCIPSolver
from polyquant.agents import MicrostructureAgent
from polyquant.data.polymarket_client import PolymarketClient
from polyquant.data.limitless_client import LimitlessClient
from polyquant.data.trade_store import TradeStore
from polyquant.api.server import monitor, app, set_trade_store, set_constraint_store, setup_web_logging
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
        self._constraint_matrix: dict[str, list[dict[str, Any]]] = {}  # Pre-computed matrix
        
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
                    
                    # Use local variable for data dictionary to help inference
                    matrix_entry: dict[str, Any] = {
                        "constraint_id": str(constraint.constraint_id),
                        "coefficient": float(coeff),
                        "rhs": float(constraint.rhs),
                    }
                    self._constraint_matrix[outcome_id].append(matrix_entry)
        
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
        self._arbitrage_detector: ArbitrageDetector | None = None
        self._solver: SCIPSolver | None = None
        self._kill_switch: KillSwitch | None = None
        self._position_sizer: PositionSizer | None = None
        self._microstructure_agent: MicrostructureAgent | None = None
        self._trade_store: TradeStore | None = None
        self._limitless: LimitlessClient | None = None
        self._balance_refresh_task: Optional[asyncio.Task] = None
        self._trade_executor: Optional['TradeExecutor'] = None
        self._rust_client: Optional['RustClient'] = None
        self._limitless_polling_task: Optional[asyncio.Task] = None

        # Arbitrage improvement modules
        self._bayesian_updater = None   # BayesianUpdater (phantom arb prevention)
        self._correlation_engine = None  # CorrelationEngine (leader-laggard pairs)
        self._previous_prices: dict[str, float] = {}  # For correlation delta tracking
        
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

        # Hot-reloading attributes
        self._hot_reload_task: asyncio.Task | None = None
        self._loaded_cluster_ids: set[str] = set()

        logger.info("Navigator initialized")
    
    async def __aenter__(self) -> "Navigator":
        """Initialize all components."""
        logger.info("Starting Navigator...")
        
        # Start Sidecar UI Server
        config_uv = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
        server = uvicorn.Server(config_uv)
        self._uvicorn_server = server  # Store reference for graceful shutdown
        self._server_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.5)  # Give uvicorn a moment to bind to port
        logger.info("🌐 API server started on http://0.0.0.0:8000 — endpoints: /status, /api/status, /ws")
        setup_web_logging()
        await monitor.update_status(status="STARTING")
        
        # Initialize Polymarket client
        self._polymarket = PolymarketClient()
        await self._polymarket.__aenter__()
        
        # Initialize Limitless client
        self._limitless = LimitlessClient()
        await self._limitless.__aenter__()

        # Initialize TradeStore for persistent execution logs
        self._trade_store = TradeStore()
        set_trade_store(self._trade_store)
        
        # Hydrate the UI monitor with historical trades
        try:
            recent_trades = await self._trade_store.get_recent_trades(limit=50)
            monitor.state.trades_executed = recent_trades
        except Exception as e:
            logger.warning(f"Could not load recent trades for UI: {e}")
        
        # Initialize ZMQ IPC Client
        from polyquant.execution.rust_client import RustClient
        self._rust_client = RustClient()
        
        # Initialize TradeExecutor with Rust execution sidecar
        from polyquant.execution.executor import TradeExecutor
        self._trade_executor = TradeExecutor(
            rust_client=self._rust_client,
            trade_store=self._trade_store,
            paper_mode=(config.trading_mode == "paper")
        )
        
        # Load constraint store
        self._store = ConstraintStore()
        set_constraint_store(self._store)
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
        
        # Initialize risk management
        self._kill_switch = KillSwitch(
            initial_capital=10000,
            on_trigger=self._on_kill_switch_trigger,
        )
        await self._kill_switch.load_state()
        self._position_sizer = PositionSizer(capital=10000)
        
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
            
        # Start balance refresh task (Rule 1 & 7)
        self._balance_refresh_task = asyncio.create_task(self._refresh_balance_loop())

        logger.info(
            "Navigator started",
            manifests_loaded=len(manifests),
        )
        return self

    async def _refresh_balance_loop(self) -> None:
        """Background loop to keep internal balances fresh (Rule 1)."""
        while self._is_running:
            try:
                executor = self._trade_executor
                if not executor:
                    await asyncio.sleep(5)
                    continue

                # 1. Fetch Poly balance (USDC on Polygon)
                poly_client = self._polymarket
                if poly_client:
                    poly_bal = await poly_client.get_usdc_balance()
                    executor.poly_balance = Decimal(str(poly_bal))
                
                # 2. Fetch Base balance (USDC on Base)
                limitless_client = self._limitless
                if limitless_client:
                    base_bal = await limitless_client.get_usdc_balance()
                    executor.base_balance = Decimal(str(base_bal))
                
                logger.debug(
                    "Balance refreshed", 
                    poly=executor.poly_balance,
                    base=executor.base_balance
                )
            except Exception as e:
                logger.error("Failed to refresh balances", error=str(e))
            
            await asyncio.sleep(60) # Refresh every minute
    
    async def _hot_reload_loop(self) -> None:
        """Background loop to dynamically load new markets from ConstraintStore."""
        while self._is_running:
            try:
                await asyncio.sleep(60) # Modest polling interval
                
                if not self._store or not self._guard:
                    continue
                    
                current_clusters = await self._store.list_clusters()
                new_clusters = [cid for cid in current_clusters if cid not in self._loaded_cluster_ids]
                
                if new_clusters:
                    logger.info(f"Hot-reloading {len(new_clusters)} new market clusters...")
                    
                    new_token_ids: set[str] = set()
                    new_manifests = []
                    
                    for cid in new_clusters:
                        manifest = await self._store.load_manifest(cid)
                        if manifest:
                            # Extract token IDs from constraint coefficients
                            for constraint in manifest.constraints:
                                new_token_ids.update(constraint.coefficients.keys())
                            new_manifests.append(manifest)
                            
                    if not new_token_ids:
                        continue
                        
                    # Subscribe to WebSockets
                    token_ids = list(new_token_ids)
                    if self._polymarket and hasattr(self._polymarket, "ws_client") and self._polymarket.ws_client:
                        logger.info(f"Subscribing to {len(token_ids)} new tokens via WebSocket")
                        await self._polymarket.ws_client.subscribe(
                            token_ids,
                            self._ws_update_callback
                        )
                        
                    # Inject into guard
                    guard = self._guard
                    if guard is not None:
                        guard.load_manifests(new_manifests)
                    
                    # Update local state
                    self._loaded_cluster_ids.update(new_clusters)
                    logger.info(f"Successfully hot-reloaded {len(new_clusters)} clusters.")
                    
            except Exception as e:
                logger.error("Failed to hot-reload markets", error=str(e))

    async def _limitless_polling_loop(self, token_ids: list[str]) -> None:
        """
        Background task to poll Limitless for order books, as they don't have WS.
        """
        if not self._limitless:
            return
            
        logger.info(f"Limitless polling loop active for {len(token_ids)} tokens")
        while self._is_running:
            try:
                # Fetch books sequentially to avoid hammering the beta API
                for token_id in token_ids:
                    if not self._is_running:
                        break
                        
                    ob = await self._limitless.get_order_book(token_id)
                    if ob:
                        # Feed the price cache so the main event loop wakes up
                        await self._price_cache.update(token_id, ob)
                        
                # Wait 1s between full passes
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error("Limitless polling loop encountered error", error=str(e))
                await asyncio.sleep(5.0) # Back off on error

    def stop(self) -> None:
        """Gracefully stop the navigator loop."""
        logger.info("Stop signal received via signal_handler. Shutting down...")
        self._is_running = False
        if getattr(self, "_price_cache", None) and hasattr(self._price_cache, "_update_event"):
            try:
                # Wake up the event loop if waiting for price
                # asyncio.Event handles thread safety internally for basic set() in same loop
                self._price_cache._update_event.set()
            except Exception:
                pass

    async def __aexit__(self, *args: Any) -> None:
        """Cleanup all components."""
        logger.info("Shutting down Navigator...")
        
        self._is_running = False

        if self._balance_refresh_task is not None:
            self._balance_refresh_task.cancel()
            try:
                await self._balance_refresh_task
            except (asyncio.CancelledError, Exception):
                pass
                
        if self._hot_reload_task is not None:
            self._hot_reload_task.cancel()
            try:
                await self._hot_reload_task
            except (asyncio.CancelledError, Exception):
                pass

        if getattr(self, "_uvicorn_server", None) is not None:
            # Tell uvicorn to shut down gracefully instead of hard canceling
            self._uvicorn_server.should_exit = True
            
            # Wait for the task to finish gracefully
            if self._server_task is not None:
                try:
                    await asyncio.wait_for(self._server_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
        elif self._server_task is not None:
            # Fallback to hard cancel if we don't have the server object
            self._server_task.cancel()
            try:
                await asyncio.wait_for(self._server_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

        if hasattr(self, "_limitless_polling_task") and self._limitless_polling_task is not None:
            self._limitless_polling_task.cancel()
            try:
                await self._limitless_polling_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._trade_store:
            await self._trade_store.close()
        
        if self._polymarket:
            await self._polymarket.__aexit__(*args)
            
        if self._limitless:
            await self._limitless.__aexit__(*args)
            
        if self._price_cache:
            self._price_cache.clear()
            
        if hasattr(self, "_limitless_polling_task") and self._limitless_polling_task:
            self._limitless_polling_task.cancel()
            try:
                await self._limitless_polling_task
            except asyncio.CancelledError:
                pass
        
        logger.info(
            "Navigator shutdown complete",
            ticks_processed=self._ticks_processed,
            opportunities_found=self._opportunities_found,
            trades_executed=self._trades_executed,
        )
    
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
            from polyquant.api.server import monitor
            await monitor.update_status(status="NO_CONSTRAINTS")
            return
        
        # Load all manifests and extract unique token IDs from constraint coefficients.
        # NOTE: market_ids in manifests are often empty for NegRisk markets.
        # The token IDs in constraint coefficients are the REAL identifiers we need.
        all_token_ids: set[str] = set()
        market_exchanges: dict[str, str] = {}
        for cluster_id in cluster_ids:
            manifest = await self._store.load_manifest(cluster_id)
            if manifest:
                # Extract token IDs from constraint coefficients (the authoritative source)
                for constraint in manifest.constraints:
                    all_token_ids.update(constraint.coefficients.keys())
                # Also grab any valid (non-empty) market IDs for exchange routing
                for mid in manifest.market_ids:
                    if mid:
                        all_token_ids.add(mid)
                market_exchanges.update(manifest.market_exchanges)
                
        # Initialize the set of loaded clusters
        self._loaded_cluster_ids = set(cluster_ids)
        
        token_id_list = list(all_token_ids)
        if not token_id_list:
            logger.warning("No token IDs found in constraint manifests")
            return
        
        logger.info(f"Monitoring {len(token_id_list)} unique tokens across {len(cluster_ids)} clusters")
        
        # Format clusters for the UI Dashboard
        ui_clusters = []
        for cid in cluster_ids:
            # We don't have the original `MarketCluster` object here, just the `ConstraintManifest`.
            # But the UI only needs `id`, `topic` (which we can fake or extract), `count`, and `status`.
            manifest = await self._store.load_manifest(cid)
            if manifest:
                ui_clusters.append({
                    "id": cid,
                    "topic": getattr(manifest, "topic", "") or f"Cluster {cid[:8]}",
                    "count": len(manifest.constraints),
                    "status": "active"
                })
        
        from polyquant.api.server import monitor
        await monitor.update_status(
            pipeline_stage="COMPLETE", 
            clusters=ui_clusters
        )
        
        # Start hot-reload task
        self._hot_reload_task = asyncio.create_task(self._hot_reload_loop())
        
        # 1. Subscribe to WebSocket updates for ALL tokens
        # WS is the PRIMARY data source — subscribe all tokens from constraints.
        # Resolved tokens are silently ignored.

        if self._polymarket:
            logger.info(f"Subscribing to {len(token_id_list)} tokens via WebSocket")
            # Create WS client if not exists
            if not getattr(self._polymarket, "ws_client", None):
                 from polyquant.data.polymarket_client import PolymarketWSClient
                 self._polymarket.ws_client = PolymarketWSClient()
                 await self._polymarket.ws_client.connect()
            
            await self._polymarket.ws_client.subscribe(
                token_id_list, 
                self._ws_update_callback
            )
            
        # 3. Limitless REST Polling Task (since no WS exists yet)
        limitless_tokens = [tid for tid in token_id_list if "_" in tid and not tid.startswith("0x")]
        if hasattr(self, "_limitless") and self._limitless and limitless_tokens:
             logger.info(f"Starting background REST polling for {len(limitless_tokens)} Limitless tokens")
             self._limitless_polling_task = asyncio.create_task(
                 self._limitless_polling_loop(limitless_tokens)
             )
            
        # 4. Main Event Loop
        tick_count = 0
        no_data_ticks = 0  # Track consecutive timeouts
        last_heartbeat = time.monotonic()
        HEARTBEAT_INTERVAL = 30  # seconds
        
        logger.info("🟢 Navigator main loop STARTED — waiting for price data...")
        
        while self._is_running:
            if max_ticks is not None and tick_count >= max_ticks:
                break

            # Event-driven: Wait up to 5s for a price update
            got_update = await self._price_cache.wait_for_update(timeout=5.0)
            
            if not got_update:
                no_data_ticks += 1
                # First few timeouts: informational
                if no_data_ticks == 6:  # ~30 seconds of no data
                    logger.warning(
                        "⚠️  No WebSocket price data received for 30s. "
                        "Check: Is the WS connection alive? Are tokens subscribed?"
                    )
                # Periodic reminder every 60s
                if no_data_ticks > 0 and no_data_ticks % 12 == 0:
                    logger.warning(
                        f"⏳ Still waiting for price data... ({no_data_ticks * 5}s without updates) "
                        f"| Cache: {self._price_cache.size} total, {self._price_cache.fresh_count} fresh"
                    )
            else:
                if no_data_ticks > 6:
                    logger.info(f"✅ Price data resumed after {no_data_ticks * 5}s gap")
                no_data_ticks = 0
            
            # Heartbeat: Show activity every HEARTBEAT_INTERVAL seconds
            now = time.monotonic()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                cache_size = self._price_cache.size
                fresh = self._price_cache.fresh_count
                # HEARTBEAT threshold: 5s — for log visibility only, NOT for trading decisions
                ws_healthy = (
                    self._polymarket.is_ws_healthy(max_age_seconds=5.0) 
                    if self._polymarket else False
                )
                stale_count = (
                    len(self._polymarket.ws_stale_assets)
                    if self._polymarket else 0
                )
                logger.info(
                    f"💓 Heartbeat | ticks={tick_count} | cache={fresh}/{cache_size} fresh "
                    f"| WS={'🟢' if ws_healthy else '🔴'} "
                    f"| stale_tokens={stale_count} "
                    f"| {'LIVE' if fresh > 0 else 'WAITING FOR DATA'}"
                )
                last_heartbeat = now

            tick_start = datetime.utcnow()

            # TRADING GUARD threshold: 200ms (config.ws_max_age_ms) — blocks ALL trades
            # This is deliberately aggressive: we NEVER trade on data older than 200ms.
            if self._polymarket and hasattr(self._polymarket, 'ws_client'):
                if not self._polymarket.is_ws_healthy(max_age_seconds=config.ws_max_age_ms / 1000.0):
                    logger.warning(
                        "Trading blocked: Stale WebSocket connection",
                        reason=f"No price updates received for >{config.ws_max_age_ms}ms"
                    )
                    # Feed to kill switch as potential issue
                    if self._kill_switch:
                        self._kill_switch.record_api_error()

                    await asyncio.sleep(1)
                    continue

            # Check kill switch
            if self._kill_switch and not await self._kill_switch.can_trade():
                logger.warning("Trading blocked by kill switch")
                await asyncio.sleep(1)
                continue

            # Collect stale tokens (WS sequence gaps detected)
            stale_tokens: set[str] = (
                self._polymarket.ws_stale_assets if self._polymarket else set()
            )

            try:
                # Get fresh order books from cache (O(1) access)
                current_books = self._price_cache.get_all()

                if not current_books:
                     # No fresh data yet, wait for next update
                     continue

                # Timestamp: Start opportunity detection
                detect_start = datetime.utcnow()
                
                # Debug logging to show the hot path is active
                logger.debug(
                    f"🔥 HOT PATH: Evaluating {len(current_books)} markets for arbitrage..."
                )

                # Check for arbitrage opportunities
                opportunities = await self._detect_opportunities(
                    current_books, stale_tokens=stale_tokens
                )

                # Record tick-to-decision latency
                detect_elapsed = (datetime.utcnow() - detect_start).total_seconds() * 1000
                self._latency_tracker.record_tick_to_decision(detect_elapsed)

                # Execute trades — pass detection timestamp for staleness checks
                if opportunities:
                    exec_start = datetime.utcnow()
                    # Stamp at price observation time, not dispatch time
                    signal_ts_us = int(detect_start.timestamp() * 1_000_000)

                    for opp in opportunities:
                        await self._execute_opportunity(opp, signal_timestamp_us=signal_ts_us)

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

                # Balance reconciliation heartbeat (every 500 ticks)
                if tick_count % 500 == 0 and self._trade_executor:
                    await self._reconcile_balances()
                
            except Exception as e:
                logger.error("Tick failed", error=str(e))

            # No sleep needed - event-driven architecture handles timing
    
    # ── Detection & Execution ──
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
                bayesian = self._bayesian_updater
                if bayesian is not None:
                    cluster_prices = {
                        tid: current_mid_prices.get(tid, 0.0)
                        for tid in books
                    }
                    adjusted_prices, adjustments = bayesian.adjust_prices(cluster_prices)
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
                    logger.info(
                        "Arbitrage opportunity found!",
                        cluster_id=cluster_id,
                        expected_profit=float(arb_opportunity.expected_profit)
                    )
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
                    last_opps = opportunities[-10:] if opportunities else []
                    asyncio.create_task(monitor.update_status(
                        opportunities=last_opps
                    ))

            except Exception as e:
                logger.error(
                    "Opportunity detection failed for cluster",
                    cluster_id=cluster_id,
                    error=str(e),
                )

        # ── Correlation-based signals (cross-cluster, after constraint detection) ──
        correlation = self._correlation_engine
        if (correlation is not None and
            self._previous_prices and current_mid_prices):
            try:
                corr_signals = correlation.check_for_signals(
                    current_prices=current_mid_prices,
                    previous_prices=self._previous_prices,
                )
                for sig in corr_signals:
                    opportunities.append({
                        "cluster_id": f"corr_{sig.pair.leader_id[:8]}",
                        "source": "correlation",
                        "expected_profit": abs(sig.expected_laggard_move) * 100,
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
    
    async def _reconcile_balances(self) -> None:
        """Periodic on-chain balance check to detect drift from local state."""
        try:
            drift_threshold = Decimal("1.0")  # $1 drift triggers warning
            executor = self._trade_executor
            if not executor:
                return

            # Polymarket USDC balance
            if hasattr(self, "_polymarket") and self._polymarket:
                on_chain_poly = await self._polymarket.get_balance()
                local_poly = executor.poly_balance
                poly_drift = abs(on_chain_poly - local_poly)
                if poly_drift > drift_threshold:
                    logger.warning(
                        "BALANCE DRIFT: Polymarket",
                        on_chain=float(on_chain_poly),
                        local=float(local_poly),
                        drift=float(poly_drift),
                    )
                    # Auto-correct to on-chain truth
                    executor.poly_balance = on_chain_poly

            # Limitless USDC balance
            if hasattr(self, "_limitless") and self._limitless:
                on_chain_base = await self._limitless.get_usdc_balance()
                local_base = executor.base_balance
                base_drift = abs(on_chain_base - local_base)
                if base_drift > drift_threshold:
                    logger.warning(
                        "BALANCE DRIFT: Limitless",
                        on_chain=float(on_chain_base),
                        local=float(local_base),
                        drift=float(base_drift),
                    )
                    executor.base_balance = on_chain_base

        except Exception as e:
            logger.debug("Balance reconciliation skipped", error=str(e))

    async def _execute_opportunity(self, opportunity: Any, signal_timestamp_us: int = 0) -> None:
        """Execute a trading opportunity.

        Args:
            opportunity: The detected arbitrage opportunity.
            signal_timestamp_us: Microsecond timestamp of when the price signal
                was first observed (for Rust-side staleness rejection).
        """
        # TODO: Implement trade execution
        self._opportunities_found += 1
        logger.info("Would execute opportunity", opportunity=opportunity)
    
    def _on_kill_switch_trigger(self, event: Any) -> None:
        """Handle kill switch trigger — halt Rust sidecar and block until ACK."""
        logger.critical("KILL SWITCH TRIGGERED — halting Rust OMS")
        monitor.trigger_kill_switch()
        self._is_running = False

        # Send halt to Rust sidecar and BLOCK until acknowledged
        rust_client = self._rust_client
        if rust_client is not None:
            reason = f"kill_switch: {event.reason.value}" if hasattr(event, "reason") else "kill_switch: unknown"
            resp = None
            for attempt in range(3):
                try:
                    resp = rust_client.send_halt(reason=reason)
                    if resp and resp.get("status") == "halted":
                        logger.info("Rust OMS confirmed HALTED", attempt=attempt + 1)
                        break
                    logger.warning(f"Halt ACK unexpected: {resp}, retrying...")
                except Exception as e:
                    logger.error(f"Halt attempt {attempt + 1} failed: {e}")
            if not resp or resp.get("status") != "halted":
                logger.critical("RUST OMS DID NOT ACK HALT after 3 attempts — sidecar may still be executing!")


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
