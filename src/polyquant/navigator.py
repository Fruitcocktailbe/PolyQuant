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
from polyquant.risk import KillSwitch, PositionSizer
from polyquant.solver import ArbitrageDetector, SCIPSolver
from polyquant.api.server import monitor, app
from polyquant.utils import config, get_logger
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
                    self._constraint_matrix[outcome_id].append({
                        "constraint_id": constraint.constraint_id,
                        "coefficient": coeff,
                        "rhs": constraint.rhs,
                    })
        
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
        
        Args:
            outcome_id: The outcome being traded.
            side: "buy" or "sell".
            size: Trade size.
            price: Trade price.
            
        Returns:
            Tuple of (is_valid, reason).
        """
        # Check 1: Is the outcome in our constraint matrix?
        if outcome_id not in self._constraint_matrix:
            # No constraints on this outcome, allow the trade
            return True, "no_constraints"
        
        # Check 2: Validate against all constraints involving this outcome
        # For now, we just verify the trade doesn't violate basic rules
        # A full check would evaluate the LP against current prices
        
        # Simplified check: Ensure we're not trading resolved outcomes
        # (This would be expanded with proper constraint checking)
        
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
        
        self._is_running = False
        self._server_task: asyncio.Task | None = None

        # Metrics
        self._ticks_processed = 0
        self._opportunities_found = 0
        self._trades_executed = 0

        # Latency tracking
        self._latency_tracker = LatencyTracker(window_size=100)

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
        
        # Initialize solver components
        self._solver = SCIPSolver()
        self._arbitrage_detector = ArbitrageDetector()
        
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
            
        if self._price_cache:
            self._price_cache.clear()
        
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
            # In a real event-driven system, we'd wait for a signal.
            # Here, we poll the cache which is updated by the background WS task.
            # This separates the "IO thread" (WS) from the "Compute thread" (Solver).
            
            tick_start = datetime.utcnow()
            
            # Check kill switch
            if self._kill_switch and not await self._kill_switch.can_trade():
                logger.warning("Trading blocked by kill switch")
                await asyncio.sleep(1)
                continue
            
            try:
                # Get fresh order books from cache (O(1) access)
                # Only process if we have enough data for a cluster
                current_books = self._price_cache.get_all()

                if not current_books:
                     await asyncio.sleep(0.01) # fast spin
                     continue

                # Timestamp: Start opportunity detection
                detect_start = datetime.utcnow()

                # Check for arbitrage opportunities
                opportunities = await self._detect_opportunities(current_books)

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
            
            # Ultra-low latency sleep (yield to WS task)
            await asyncio.sleep(0.001)
    
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
    ) -> list[dict[str, Any]]:
        """
        Detect arbitrage opportunities in the current order books.

        This method:
        1. Groups order books by cluster (using ExecutionGuard)
        2. For each cluster, runs the Frank-Wolfe solver
        3. Returns profitable opportunities

        Args:
            order_books: Dictionary of token_id -> OrderBook

        Returns:
            List of opportunity dictionaries, each containing:
                - cluster_id: Which cluster this opportunity belongs to
                - target_prices: Optimal prices from solver
                - current_prices: Current market prices
                - profit: Expected profit in USD
                - trades: List of trades to execute
        """
        if not self._guard or not self._arbitrage_detector:
            return []

        opportunities = []

        # Group order books by cluster
        cluster_books: dict[str, dict[str, OrderBook]] = {}
        for token_id, book in order_books.items():
            cluster_id = self._guard.get_cluster_for_outcome(token_id)
            if cluster_id:
                if cluster_id not in cluster_books:
                    cluster_books[cluster_id] = {}
                cluster_books[cluster_id][token_id] = book

        # For each cluster, check for arbitrage
        for cluster_id, books in cluster_books.items():
            if not books:
                continue

            try:
                # Get constraint manifest for this cluster
                manifest = self._guard._manifests.get(cluster_id)
                if not manifest:
                    continue

                # Run solver (STUB - actual implementation would call ArbitrageDetector)
                # For now, just structure the call
                # opportunity = await self._arbitrage_detector.find_arbitrage(
                #     manifest=manifest,
                #     order_books=books,
                # )

                # STUB: No actual opportunities detected
                opportunity = None

                if opportunity:
                    opportunities.append({
                        "cluster_id": cluster_id,
                        "target_prices": opportunity.get("target_prices", {}),
                        "current_prices": {
                            tid: book.mid_price() for tid, book in books.items()
                        },
                        "profit": opportunity.get("profit", 0.0),
                        "trades": opportunity.get("trades", []),
                    })

            except Exception as e:
                logger.error(
                    "Opportunity detection failed for cluster",
                    cluster_id=cluster_id,
                    error=str(e),
                )

        return opportunities
    
    async def _execute_opportunity(self, opportunity: Any) -> None:
        """Execute a trading opportunity."""
        # TODO: Implement trade execution
        self._opportunities_found += 1
        logger.info("Would execute opportunity", opportunity=opportunity)
    
    def _on_kill_switch_trigger(self, event: Any) -> None:
        """Handle kill switch trigger."""
        logger.critical("KILL SWITCH TRIGGERED")
        monitor.trigger_kill_switch()
        self._is_running = False


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
