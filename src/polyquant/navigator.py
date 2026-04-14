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
    from polyquant.agents.correlation import CorrelationEngine, CorrelationSignal

from polyquant.data import OrderBook, OrderSide, ProposedTrade
from polyquant.data.constraint_store import ConstraintStore, ConstraintManifest
from polyquant.data.price_cache import PriceCache
from polyquant.risk import KillSwitch, PositionSizer
from polyquant.solver import ArbitrageDetector, SCIPSolver
from polyquant.solver.scip_solver import OptimizationResult
from polyquant.data import ArbitrageOpportunity
from polyquant.data.market_models import MarketDependency
from polyquant.agents import MicrostructureAgent
from polyquant.agents.validator import ValidatedResult
from polyquant.agents.logic_architect import LogicalConstraint, AnalysisResult
from polyquant.data.polymarket_client import PolymarketClient
from polyquant.data.limitless_client import LimitlessClient
from polyquant.data.trade_store import TradeStore
from polyquant.api.server import monitor, set_trade_store, set_constraint_store, start_api_server
from polyquant.utils import config, get_logger
from polyquant.utils.market_utils import extract_market_id
from polyquant.utils.profiling import timed_operation, async_timed, print_latency_report  # Week 4

logger = get_logger(__name__)


def _manifest_to_validated(manifest: ConstraintManifest) -> ValidatedResult:
    """Adapter: wrap a persisted manifest as the ValidatedResult the solver expects.

    On-disk manifests are MapMaker's post-validation output, but the solver's
    ValidatedResult contract predates the persistence layer. StoredConstraint and
    StoredDependency are near-identical to LogicalConstraint and MarketDependency,
    so this is a direct field copy with no logic.
    """
    logical_constraints = [
        LogicalConstraint(
            constraint_id=c.constraint_id,
            description=c.description,
            coefficients=dict(c.coefficients),
            rhs=c.rhs,
            confidence=c.confidence,
            source_markets=list(c.source_markets),
            reasoning=c.reasoning,
        )
        for c in manifest.constraints
    ]
    logical_deps = [
        MarketDependency(
            source_market_id=d.source_market_id,
            source_outcome=d.source_outcome,
            target_market_id=d.target_market_id,
            target_outcome=d.target_outcome,
            relationship=d.relationship,
            confidence=d.confidence,
        )
        for d in manifest.dependencies
    ]
    return ValidatedResult(
        original=AnalysisResult(
            cluster_id=manifest.cluster_id,
            dependencies=logical_deps,
            constraints=logical_constraints,
        ),
        is_valid=True,
        validated_constraints=logical_constraints,
        validated_dependencies=logical_deps,
        market_exchanges=dict(manifest.market_exchanges),
    )


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
        # Guards _manifests + _constraint_matrix against hot-reload races.
        # Detection holds it across a tick; hot-reload waits one tick.
        self._manifests_lock = asyncio.Lock()

        logger.info("ExecutionGuard initialized")
    
    def load_manifests(self, manifests: list[ConstraintManifest]) -> None:
        """
        Load constraint manifests into memory.

        Idempotent: re-loading a cluster_id replaces its prior constraint
        matrix entries instead of appending. This lets the hot-reload loop
        re-inject updated manifests without doubling up rules.

        Args:
            manifests: List of ConstraintManifests to load.
        """
        for manifest in manifests:
            # If this cluster_id was already loaded, strip its old matrix
            # entries before adding the new ones.
            if manifest.cluster_id in self._manifests:
                old_constraint_ids = {
                    str(c.constraint_id)
                    for c in self._manifests[manifest.cluster_id].constraints
                }
                for outcome_id in list(self._constraint_matrix.keys()):
                    self._constraint_matrix[outcome_id] = [
                        e for e in self._constraint_matrix[outcome_id]
                        if e["constraint_id"] not in old_constraint_ids
                    ]

            self._manifests[manifest.cluster_id] = manifest

            for constraint in manifest.constraints:
                for outcome_id, coeff in constraint.coefficients.items():
                    if outcome_id not in self._constraint_matrix:
                        self._constraint_matrix[outcome_id] = []

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
        self._previous_prices_ts: float | None = None  # monotonic ts of last snapshot
        # Skip correlation deltas if WS gap exceeds this — prevents treating
        # cross-outage moves as one-tick signals.
        self._correlation_max_gap_s: float = config.correlation_max_signal_gap_s

        # token_id → exchange name (polymarket|limitless). Rebuilt whenever
        # manifests are loaded. Used by correlation trade construction.
        self._token_to_exchange: dict[str, str] = {}

        # Pending correlation trade outcomes awaiting delayed resolution.
        # key: internal uid, value: dict(pair_leader_id, laggard_id, direction,
        # expected_laggard_price, submit_ts_monotonic).
        self._pending_corr_outcomes: dict[str, dict[str, Any]] = {}
        self._correlation_resolver_task: asyncio.Task | None = None


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
        self._loaded_manifest_mtimes: dict[str, float] = {}

        logger.info("Navigator initialized")
    
    async def __aenter__(self) -> "Navigator":
        """Initialize all components."""
        logger.info("Starting Navigator...")
        
        # Start Sidecar UI Server
        self._uvicorn_server, self._server_task = await start_api_server()
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

        # Build token_id → exchange map used by correlation trade construction.
        # Rebuilt on every hot-reload so it stays in sync with manifests.
        self._rebuild_token_exchange_map(manifests)

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
        
        # Initialize risk management. Kill switch state MUST be restored from
        # Redis before trading starts — otherwise a crash-restart could resume
        # with an in-memory is_triggered=False and bypass a prior halt.
        #
        # Redis connection is explicitly established here (not lazily) so a
        # connection failure fails fast with a clear error instead of silently
        # degrading to "no persisted state" mode.
        from polyquant.utils.cache import cache as _cache
        try:
            cache_connected = await _cache.connect()
        except Exception as e:
            logger.critical(
                "REDIS CONNECT FAILED - REFUSING TO START",
                error_type=type(e).__name__,
                error_message=str(e),
                redis_url=config.redis_url,
                exc_info=True,
            )
            raise SystemExit(
                f"Redis connect raised {type(e).__name__}: {e}. "
                f"URL: {config.redis_url}. Refusing to start trading."
            ) from e
        if not cache_connected:
            logger.critical(
                "REDIS CONNECT RETURNED FALSE - REFUSING TO START",
                redis_url=config.redis_url,
            )
            raise SystemExit(
                f"Redis connect returned False. URL: {config.redis_url}. "
                f"Check Redis service and REDIS_URL. Refusing to start trading."
            )

        self._kill_switch = KillSwitch(
            initial_capital=10000,
            on_trigger=self._on_kill_switch_trigger,
        )
        try:
            await self._kill_switch.load_state()
        except Exception as e:
            logger.critical(
                "KILL SWITCH STATE LOAD FAILED - REFUSING TO START",
                error_type=type(e).__name__,
                error_message=str(e),
                redis_url=config.redis_url,
                exc_info=True,
            )
            raise SystemExit(
                f"Kill switch state load failed ({type(e).__name__}: {e}). "
                f"Redis at {config.redis_url} reachable but state load errored. "
                f"Refusing to start trading."
            ) from e
        self._position_sizer = PositionSizer(capital=10000)
        
        self._is_running = True
        await monitor.update_status(status="ONLINE", active_solvers=1)
        
        # Initialize price cache for low-latency access
        self._price_cache = PriceCache(stale_threshold_seconds=2.0)

        # Wire the cache into the executor so it can recompute net profit
        # against fresh VWAP right before dispatch. Constructed late because
        # TradeExecutor is built earlier in __aenter__ and PriceCache only
        # exists once the WS subscription stack is ready.
        if self._trade_executor is not None:
            self._trade_executor._price_cache = self._price_cache
        
        # Subscribe cache to WebSocket updates
        # NOTE: This connects the WS client (in PolymarketClient) to our local cache
        # We need to bridge the WS client callback to our cache update
        # The WS client expects a callback(book: OrderBook)
        async def on_ws_update(book: OrderBook):
            if self._price_cache:
                # All updates arriving via this callback come from the
                # Polymarket WS subscription. Limitless updates are pushed
                # through the polling loop below and tagged separately.
                await self._price_cache.update(book.outcome_id, book, exchange="polymarket")
        
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
        """
        Background loop to dynamically load new and updated manifests from
        ConstraintStore.

        Picks up two cases:
        1. NEW cluster_ids (e.g. fresh LLM clusters with timestamp ids).
        2. UPDATED manifests (e.g. NegRisk clusters which reuse stable
           cluster_ids across MapMaker runs). Detected via on-disk mtime.
        """
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Modest polling interval

                if not self._store or not self._guard:
                    continue

                current_mtimes = self._store.list_clusters_with_mtimes()
                changed_cluster_ids: list[str] = []
                for cid, mtime in current_mtimes.items():
                    prev_mtime = self._loaded_manifest_mtimes.get(cid)
                    if prev_mtime is None or mtime > prev_mtime:
                        changed_cluster_ids.append(cid)

                if not changed_cluster_ids:
                    continue

                new_count = sum(1 for cid in changed_cluster_ids if cid not in self._loaded_cluster_ids)
                updated_count = len(changed_cluster_ids) - new_count
                logger.info(
                    f"Hot-reloading {len(changed_cluster_ids)} clusters "
                    f"({new_count} new, {updated_count} updated)"
                )

                new_token_ids: set[str] = set()
                changed_manifests = []
                for cid in changed_cluster_ids:
                    manifest = await self._store.load_manifest(cid)
                    if manifest:
                        for constraint in manifest.constraints:
                            new_token_ids.update(constraint.coefficients.keys())
                        changed_manifests.append(manifest)

                if not changed_manifests:
                    continue

                # Subscribe to any token IDs we don't already follow.
                # Re-subscribing to existing tokens is a harmless no-op.
                token_ids = list(new_token_ids)
                if (
                    token_ids
                    and self._polymarket
                    and hasattr(self._polymarket, "ws_client")
                    and self._polymarket.ws_client
                ):
                    logger.info(f"Subscribing to {len(token_ids)} tokens via WebSocket")
                    await self._polymarket.ws_client.subscribe(
                        token_ids,
                        self._ws_update_callback,
                    )

                # Inject into guard (under lock to avoid races with detection).
                # load_manifests is idempotent: re-loading a cluster_id replaces
                # its prior matrix entries.
                guard = self._guard
                if guard is not None:
                    async with guard._manifests_lock:
                        guard.load_manifests(changed_manifests)
                        # Rebuild token→exchange map from the full manifest set
                        # so changed clusters pick up new exchange attribution
                        # without stale entries from prior reloads.
                        all_manifests = list(guard._manifests.values())
                        self._rebuild_token_exchange_map(all_manifests)

                # Update local state
                self._loaded_cluster_ids.update(cid for cid in changed_cluster_ids)
                self._loaded_manifest_mtimes.update(
                    {cid: current_mtimes[cid] for cid in changed_cluster_ids}
                )
                logger.info(f"Successfully hot-reloaded {len(changed_manifests)} clusters.")

            except Exception as e:
                logger.error("Failed to hot-reload markets", error=str(e))

    async def _build_correlation_opportunity(
        self,
        sig: "CorrelationSignal",
        order_books: dict[str, OrderBook],
    ) -> dict | None:
        """Construct an executable opportunity dict from a correlation signal.

        Returns None when: execution is disabled, the laggard book is missing
        or one-sided, sizing returns zero, or VWAP can't be computed. A non-None
        return can be handed straight to `_execute_opportunity` like any other
        ArbitrageOpportunity-based dict.

        Shape matches the dicts produced in `_tick` so the execution path is
        a single code path regardless of signal source.
        """
        if not config.correlation_execution_enabled:
            return None

        laggard_id = sig.pair.laggard_id
        ob = order_books.get(laggard_id)
        if ob is None:
            return None

        current_p = float(sig.current_laggard_price)
        expected_p = float(sig.expected_laggard_price)
        if current_p <= 0 or expected_p <= 0:
            return None

        # Direction: BUY if we expect the laggard to rise toward the leader,
        # SELL (short via buying the opposite outcome, in line with fw_solver's
        # convention) if we expect it to fall.
        if expected_p > current_p:
            side = OrderSide.BUY
            top_price = float(ob.best_ask) if ob.best_ask else 0.0
            top_size = float(ob.asks[0].size) if ob.asks else 0.0
            if top_price <= 0 or top_size <= 0:
                return None
            # Sanity: target must still represent a profit after fees/slippage.
            if expected_p <= top_price:
                return None
            # Kelly odds convention used by fw_solver: odds = (1/price) - 1.
            odds = (1.0 / top_price) - 1.0
        else:
            side = OrderSide.SELL
            top_price = float(ob.best_bid) if ob.best_bid else 0.0
            top_size = float(ob.bids[0].size) if ob.bids else 0.0
            if top_price <= 0 or top_size <= 0:
                return None
            if expected_p >= top_price:
                return None
            # For SELL: odds = bid / (1 - bid), matching fw_solver's Kelly SELL path.
            if top_price >= 1.0:
                return None
            odds = top_price / (1.0 - top_price)

        # Conservative effective probability: lean heavily on historical accuracy
        # (which defaults to 0.5 on untested pairs), nudge upward only when the
        # deviation is large AND the underlying correlation is tight. Hard-cap
        # at the configured ceiling so no single signal can size up catastrophically.
        avg_r = (float(sig.pair.correlation_7d) + float(sig.pair.correlation_30d)) / 2.0
        accuracy_prior = max(0.5, float(sig.pair.accuracy))
        sigma_confidence = min(1.0, float(sig.deviation_sigma) / 4.0)
        effective_probability = accuracy_prior + (sigma_confidence * avg_r * 0.1)
        effective_probability = min(config.correlation_max_probability, effective_probability)
        effective_probability = max(0.5, effective_probability)

        sizer = self._position_sizer
        if sizer is None:
            return None

        depth_usd = top_size * top_price
        size_result = sizer.calculate_size(
            probability=effective_probability,
            odds=odds,
            order_book_depth=depth_usd,
        )
        # Apply correlation-specific additional shrinkage on top of kelly_fraction.
        raw_shares = float(size_result.recommended_size) * config.correlation_kelly_fraction
        if raw_shares <= 0:
            return None

        vwap = ob.get_vwap(side, Decimal(str(raw_shares)))
        if vwap is None:
            return None

        # Guard: don't trade if the realistic fill price no longer beats the target.
        if side == OrderSide.BUY and float(vwap) >= expected_p:
            return None
        if side == OrderSide.SELL and float(vwap) <= expected_p:
            return None

        # Expected profit, discounted by pair accuracy to reflect statistical
        # (not guaranteed) nature of the signal.
        price_gap = abs(expected_p - float(vwap))
        raw_profit = Decimal(str(price_gap * raw_shares))
        expected_profit = raw_profit * Decimal(str(max(0.5, float(sig.pair.accuracy))))

        exchange_name = self._token_to_exchange.get(laggard_id, "polymarket")
        market_id = extract_market_id(laggard_id)
        reason = f"correlation:{sig.pair.leader_id}"

        trade = ProposedTrade(
            market_id=market_id,
            outcome_id=laggard_id,
            side=side,
            size=Decimal(str(raw_shares)),
            limit_price=vwap,
            exchange=exchange_name,
            reason=reason,
        )

        arb_opportunity = ArbitrageOpportunity(
            markets=[market_id],
            trades=[trade],
            expected_profit=expected_profit,
            confidence=float(sig.pair.accuracy) if sig.pair.accuracy > 0 else 0.5,
        )

        cluster_id = f"correlation:{sig.pair.leader_id}->{sig.pair.laggard_id}"
        return {
            "cluster_id": cluster_id,
            "source": "correlation",
            "expected_profit": float(expected_profit),
            "arb_object": arb_opportunity,
            "trades": [
                {
                    "outcome_id": trade.outcome_id,
                    "side": trade.side.value,
                    "size": float(trade.size),
                    "limit_price": float(trade.limit_price),
                    "exchange": trade.exchange,
                }
            ],
            "timestamp": datetime.utcnow().isoformat(),
            # Hand-off metadata so the tick loop can register an outcome tracker.
            "_correlation_meta": {
                "pair_leader_id": sig.pair.leader_id,
                "laggard_id": laggard_id,
                "direction": side.value,
                "expected_laggard_price": expected_p,
                "current_laggard_price": current_p,
            },
        }

    async def _correlation_resolver_loop(self) -> None:
        """Background task: resolve pending correlation trade outcomes.

        For each pending entry older than config.correlation_outcome_delay_minutes,
        fetch the current laggard mid-price, decide whether the signal was correct
        (laggard moved in the expected direction by at least half the expected gap),
        and feed the verdict back to the CorrelationEngine so pair.accuracy stays
        calibrated. Entries whose laggard price can't be fetched are retried
        until a 4× overall deadline, then dropped with a warning.
        """
        check_interval_s = 30.0
        delay_s = config.correlation_outcome_delay_minutes * 60.0
        drop_after_s = delay_s * 4.0

        while self._is_running:
            try:
                await asyncio.sleep(check_interval_s)
            except asyncio.CancelledError:
                return

            if not self._pending_corr_outcomes:
                continue

            engine = self._correlation_engine
            if engine is None:
                continue

            now = time.monotonic()
            ripe_uids = [
                uid for uid, meta in self._pending_corr_outcomes.items()
                if (now - meta["submit_ts_monotonic"]) >= delay_s
            ]
            for uid in ripe_uids:
                meta = self._pending_corr_outcomes[uid]
                age = now - meta["submit_ts_monotonic"]
                try:
                    current_price = await self._fetch_current_price(meta["laggard_id"])
                except Exception as e:
                    logger.debug(
                        "Correlation outcome fetch failed; will retry",
                        uid=uid, error=str(e),
                    )
                    current_price = None

                if current_price is None:
                    if age >= drop_after_s:
                        logger.warning(
                            "Correlation outcome dropped: price unavailable past deadline",
                            uid=uid,
                            laggard_id=meta["laggard_id"],
                        )
                        self._pending_corr_outcomes.pop(uid, None)
                    continue

                expected_p = float(meta["expected_laggard_price"])
                entry_p = float(meta["current_laggard_price"])
                actual_p = float(current_price)
                target_gap = expected_p - entry_p  # signed
                actual_gap = actual_p - entry_p

                # "Correct" if the laggard moved at least halfway toward
                # the expected target in the predicted direction.
                if target_gap == 0:
                    was_correct = False
                elif (target_gap > 0 and actual_gap >= target_gap * 0.5) or \
                     (target_gap < 0 and actual_gap <= target_gap * 0.5):
                    was_correct = True
                else:
                    was_correct = False

                try:
                    engine.record_signal_outcome(
                        pair_leader_id=meta["pair_leader_id"],
                        was_correct=was_correct,
                    )
                    logger.info(
                        "Correlation outcome resolved",
                        uid=uid,
                        pair_leader_id=meta["pair_leader_id"],
                        was_correct=was_correct,
                        entry_price=entry_p,
                        expected_price=expected_p,
                        actual_price=actual_p,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to record correlation outcome", uid=uid, error=str(e)
                    )
                finally:
                    self._pending_corr_outcomes.pop(uid, None)

    async def _fetch_current_price(self, token_id: str) -> float | None:
        """Fetch current mid-price for a token via the appropriate exchange client.

        Returns None when the book is unavailable or lacks a mid.
        """
        exchange = self._token_to_exchange.get(token_id, "polymarket")
        try:
            if exchange == "polymarket" and self._polymarket:
                ob = await self._polymarket.get_order_book(token_id)
            elif exchange == "limitless" and self._limitless:
                ob = await self._limitless.get_order_book(token_id)
            else:
                return None
        except Exception:
            return None

        if ob is None:
            return None
        mid = getattr(ob, "mid_price", None)
        if mid is None or float(mid) <= 0:
            return None
        return float(mid)

    def _rebuild_token_exchange_map(
        self, manifests: list["ConstraintManifest"]
    ) -> None:
        """Rebuild self._token_to_exchange from the given manifests.

        Walks every constraint's coefficients to collect token_ids, then
        resolves each to an exchange label via manifest.market_exchanges
        (keyed by market_id, so we strip the token suffix first).

        Unknown tokens default to "polymarket" — matches the fallback used
        in fw_solver's per-cluster exchange lookup.
        """
        mapping: dict[str, str] = {}
        for manifest in manifests:
            me = dict(manifest.market_exchanges or {})
            for constraint in manifest.constraints:
                for token_id in constraint.coefficients.keys():
                    market_id = extract_market_id(token_id)
                    exchange_info = me.get(market_id, "polymarket")
                    # market_exchanges may use "limitless:<slug>" form; collapse to "limitless".
                    if exchange_info.startswith("limitless"):
                        mapping[token_id] = "limitless"
                    else:
                        mapping[token_id] = "polymarket"
        self._token_to_exchange = mapping
        logger.debug(
            "Token→exchange map rebuilt",
            token_count=len(mapping),
            manifest_count=len(manifests),
        )

    async def _limitless_polling_loop(
        self, token_ids: list[str], matched_token_ids: set[str] | None = None
    ) -> None:
        """
        Background task to poll Limitless for order books.

        Two-tier polling:
        - Fast tier (250ms): tokens with active cross-exchange Polymarket pairs
        - Slow tier (1s): all other Limitless tokens
        
        Rate budget: Limitless allows 100 req/sec per user.
        Worst case at 250ms with 20 matched tokens = ~80 req/sec (safe).
        """
        if not self._limitless:
            return

        matched = matched_token_ids or set()
        fast_tokens = [tid for tid in token_ids if tid in matched]
        slow_tokens = [tid for tid in token_ids if tid not in matched]

        logger.info(
            f"Limitless polling loop active",
            fast_tier=f"{len(fast_tokens)} tokens @ 250ms",
            slow_tier=f"{len(slow_tokens)} tokens @ 1s",
        )

        fast_counter = 0  # Track cycles to interleave slow tokens

        while self._is_running:
            try:
                # Fast tier: poll matched tokens every cycle (250ms)
                for token_id in fast_tokens:
                    if not self._is_running:
                        break
                    ob = await self._limitless.get_order_book(token_id)
                    if ob:
                        await self._price_cache.update(token_id, ob, exchange="limitless")

                # Slow tier: poll unmatched tokens every 4th cycle (~1s)
                fast_counter += 1
                if fast_counter >= 4 and slow_tokens:
                    fast_counter = 0
                    for token_id in slow_tokens:
                        if not self._is_running:
                            break
                        ob = await self._limitless.get_order_book(token_id)
                        if ob:
                            await self._price_cache.update(token_id, ob, exchange="limitless")

                # 250ms between fast cycles
                await asyncio.sleep(0.25)

            except Exception as e:
                logger.error("Limitless polling loop encountered error", error=str(e))
                await asyncio.sleep(5.0)  # Back off on error



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

        if self._correlation_resolver_task is not None:
            self._correlation_resolver_task.cancel()
            try:
                await self._correlation_resolver_task
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
            logger.warning("No constraint manifests yet — entering WAITING mode. Hot-reload will pick them up.")
            await monitor.update_status(status="WAITING_FOR_CONSTRAINTS")

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

        # Initialize the set of loaded clusters and seed mtime tracking so the
        # hot-reload loop only fires on subsequent on-disk changes.
        self._loaded_cluster_ids = set(cluster_ids)
        self._loaded_manifest_mtimes = self._store.list_clusters_with_mtimes()

        token_id_list = list(all_token_ids)
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

        # Start correlation outcome resolver (no-op if no signals ever land)
        self._correlation_resolver_task = asyncio.create_task(
            self._correlation_resolver_loop()
        )
        
        # 1. Subscribe to WebSocket updates for ALL tokens
        # WS is the PRIMARY data source — subscribe all tokens from constraints.
        # Resolved tokens are silently ignored.

        if self._polymarket and token_id_list:
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
            
        # 3. Limitless REST Polling Task (since no WS exists yet).
        # Build an authoritative token_id -> exchange map from manifests, replacing the
        # earlier `"_" in tid` string heuristic with a deterministic lookup that matches
        # what the solvers already do (see scip_solver.py:328, fw_solver.py:734).
        token_exchange: dict[str, str] = {}
        for _cid in cluster_ids:
            _manifest = await self._store.load_manifest(_cid)
            if not (_manifest and _manifest.market_exchanges):
                continue
            for tid in token_id_list:
                mid = extract_market_id(tid)
                if mid in _manifest.market_exchanges:
                    token_exchange[tid] = _manifest.market_exchanges[mid]

        limitless_tokens = [tid for tid, exch in token_exchange.items() if exch.startswith("limitless")]
        matched_limitless_tokens: set[str] = set(limitless_tokens)

        unrouted = [tid for tid in token_id_list if tid not in token_exchange]
        if unrouted:
            logger.warning(
                "Token IDs with no exchange mapping in manifests",
                count=len(unrouted),
                sample=unrouted[:5],
            )

        if hasattr(self, "_limitless") and self._limitless and limitless_tokens:
             logger.info(f"Starting background REST polling for {len(limitless_tokens)} Limitless tokens ({len(matched_limitless_tokens)} fast-tier)")
             self._limitless_polling_task = asyncio.create_task(
                 self._limitless_polling_loop(limitless_tokens, matched_limitless_tokens)
             )
            
        # 4. Main Event Loop
        tick_count = 0
        no_data_ticks = 0  # Track consecutive timeouts
        last_heartbeat = time.monotonic()
        HEARTBEAT_INTERVAL = 30  # seconds
        was_stale = False
        
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
                    if not was_stale:
                        logger.warning(
                            "Trading blocked: Stale WebSocket connection",
                            reason=f"No price updates received for >{config.ws_max_age_ms}ms"
                        )
                        was_stale = True
                    continue
                else:
                    if was_stale:
                        logger.info("✅ Data resumed, trading sequence re-armed")
                        was_stale = False

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

        # Snapshot manifests under lock so hot-reload can't change cluster
        # attribution mid-tick. The lock is held only for the snapshot copy
        # (no awaits inside) — hot-reload waits only microseconds.
        async with self._guard._manifests_lock:
            manifest_snapshot: dict[str, ConstraintManifest] = dict(self._guard._manifests)
            outcome_to_cluster: dict[str, str] = {}
            for cid, m in manifest_snapshot.items():
                for constraint in m.constraints:
                    for oid in constraint.coefficients:
                        outcome_to_cluster[oid] = cid

        # Group order books by cluster (using snapshot lookup)
        cluster_books: dict[str, dict[str, OrderBook]] = {}
        for token_id, book in order_books.items():
            cluster_id = outcome_to_cluster.get(token_id)
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

                # 3. Run Arbitrage Detector (using snapshot manifest)
                manifest = manifest_snapshot.get(cluster_id)
                if not manifest:
                    continue

                try:
                    validated = _manifest_to_validated(manifest)
                    arb_opportunity = await self._arbitrage_detector.detect(
                        validated=validated,
                        order_books=books,
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
                        "arb_object": arb_opportunity,  # Keep typed object for executor
                        "trades": [
                            {
                                "outcome_id": t.outcome_id,
                                "side": t.side.value,
                                "size": float(t.size),
                                "limit_price": float(t.limit_price),
                                "exchange": t.exchange,
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

        # ── Correlation-based signals (cross-cluster) ──
        # Signals are always logged. Execution is gated behind
        # config.correlation_execution_enabled inside _build_correlation_opportunity,
        # so flipping the flag is a one-place change.
        correlation = self._correlation_engine
        prev_prices = self._previous_prices
        prev_ts = self._previous_prices_ts
        now_ts = time.monotonic()
        if (correlation is not None and prev_prices and current_mid_prices
                and prev_ts is not None
                and (now_ts - prev_ts) <= self._correlation_max_gap_s):
            try:
                corr_signals = correlation.check_for_signals(
                    current_prices=current_mid_prices,
                    previous_prices=prev_prices,
                )
                for sig in corr_signals:
                    logger.info(
                        "Correlation signal observed",
                        leader=sig.pair.leader_question[:60],
                        laggard=sig.pair.laggard_question[:60],
                        deviation_sigma=sig.deviation_sigma,
                        expected_laggard_move=float(sig.expected_laggard_move),
                        execution_enabled=config.correlation_execution_enabled,
                    )
                    corr_opp = await self._build_correlation_opportunity(
                        sig, order_books
                    )
                    if corr_opp is not None:
                        opportunities.append(corr_opp)
            except Exception as e:
                logger.error("Correlation signal check failed", error=str(e))

        # Update previous prices for next tick's correlation delta
        self._previous_prices = current_mid_prices
        self._previous_prices_ts = now_ts

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

    async def _has_live_depth(self, trades: list[Any], cluster_id: str = "") -> bool:
        """
        Min-viable live depth check.

        For each trade leg, fetch the live order book and confirm there is at
        least one resting order on the side we want to hit. Blocks obvious
        zero-fill cases. Does NOT attempt to size against depth or enforce a
        numeric threshold — that's intentionally left to a follow-up
        (solver-native sizing).

        This is the only place "current liquidity" gates execution. MapMaker
        is structural-only; live conditions live here.
        """
        for trade in trades:
            exchange = getattr(trade, "exchange", None)
            token_id = trade.outcome_id
            try:
                if exchange == "polymarket" and self._polymarket:
                    ob = await self._polymarket.get_order_book(token_id)
                elif exchange == "limitless" and self._limitless:
                    ob = await self._limitless.get_order_book(token_id)
                else:
                    logger.warning(
                        "Depth check: unknown exchange or client unavailable, skipping trade",
                        exchange=exchange,
                        token_id=token_id,
                        cluster_id=cluster_id,
                    )
                    return False
            except Exception as e:
                logger.warning(
                    "Depth check failed; skipping trade",
                    token_id=token_id,
                    exchange=exchange,
                    error=str(e),
                )
                return False

            if ob is None:
                logger.info(
                    "Depth check: no order book, skipping trade",
                    token_id=token_id,
                    exchange=exchange,
                    cluster_id=cluster_id,
                )
                return False

            side_value = trade.side.value if hasattr(trade.side, "value") else trade.side
            book_side = ob.asks if str(side_value).upper() == "BUY" else ob.bids
            if not book_side or float(book_side[0].size) <= 0:
                logger.info(
                    "Depth check: zero depth on target side, skipping trade",
                    token_id=token_id,
                    side=side_value,
                    exchange=exchange,
                    cluster_id=cluster_id,
                )
                return False

        return True

    async def _execute_opportunity(self, opportunity: Any, signal_timestamp_us: int = 0) -> None:
        """Execute a trading opportunity through the TradeExecutor.

        Converts the detected ArbitrageOpportunity into an OptimizationResult,
        runs ExecutionGuard pre-flight checks on every leg, then dispatches
        to TradeExecutor.execute_atomic() for paper or live execution.

        Args:
            opportunity: Dict with keys: cluster_id, source, expected_profit,
                arb_object (ArbitrageOpportunity), trades (list of dicts).
            signal_timestamp_us: Microsecond timestamp of when the price signal
                was first observed (for Rust-side staleness rejection).
        """
        self._opportunities_found += 1

        # ── Extract the typed ArbitrageOpportunity ──
        arb: ArbitrageOpportunity | None = opportunity.get("arb_object") if isinstance(opportunity, dict) else None
        if not arb or not arb.trades:
            logger.warning(
                "Opportunity has no executable trades (missing arb_object or empty trades)",
                cluster_id=opportunity.get("cluster_id") if isinstance(opportunity, dict) else "unknown",
            )
            return

        cluster_id = opportunity.get("cluster_id", "unknown")

        logger.info(
            "Executing opportunity",
            cluster_id=cluster_id,
            expected_profit=float(arb.expected_profit),
            trade_count=len(arb.trades),
            trades=[
                {
                    "outcome_id": t.outcome_id,
                    "side": t.side.value,
                    "size": float(t.size),
                    "price": float(t.limit_price),
                    "exchange": t.exchange,
                }
                for t in arb.trades
            ],
        )

        # ── Pre-flight: ExecutionGuard constraint checks ──
        if self._guard:
            for trade in arb.trades:
                valid, reason = self._guard.check_trade(
                    outcome_id=trade.outcome_id,
                    side=trade.side.value,
                    size=float(trade.size),
                    price=float(trade.limit_price),
                )
                if not valid:
                    logger.warning(
                        "Trade rejected by ExecutionGuard",
                        reason=reason,
                        outcome_id=trade.outcome_id,
                        cluster_id=cluster_id,
                    )
                    return

        # ── Pre-flight: per-exchange staleness dead-man switch ──
        # Every exchange involved in this trade must have produced an update
        # within the cache staleness window. Checking per-token is not enough:
        # if Polymarket is ticking but the Limitless polling loop is stuck,
        # a Limitless leg's stale book would be padded by the Polymarket
        # update's global event signal.
        if self._price_cache:
            exchanges_in_trade = {t.exchange for t in arb.trades if t.exchange}
            for exch in exchanges_in_trade:
                if self._price_cache.exchange_stale(exch):
                    logger.warning(
                        "Trade rejected: exchange feed stale",
                        exchange=exch,
                        cluster_id=cluster_id,
                        stale_threshold_s=self._price_cache._stale_threshold,
                    )
                    return

        # ── Pre-flight: min-viable live depth check ──
        # Block trades where any leg has zero depth on the side we want to hit.
        # This is the only place "current liquidity" gates execution — MapMaker
        # emits constraints regardless of snapshot conditions.
        if not await self._has_live_depth(arb.trades, cluster_id=cluster_id):
            return

        # ── Wrap in OptimizationResult (what TradeExecutor expects) ──
        opt_result = OptimizationResult(
            success=True,
            trades=arb.trades,
            expected_profit=arb.expected_profit,
            status="arbitrage_detected",
        )

        # ── Execute atomically via TradeExecutor ──
        if not self._trade_executor:
            logger.error("TradeExecutor not initialized — cannot execute")
            return

        result = await self._trade_executor.execute_atomic(
            opt_result,
            signal_timestamp_us=signal_timestamp_us,
        )

        if result.success:
            self._trades_executed += result.trade_count
            logger.info(
                "✅ Trade executed successfully",
                cluster_id=cluster_id,
                fills=result.trade_count,
                total_filled=float(result.total_filled),
                expected_profit=float(arb.expected_profit),
            )

            # Register correlation outcome-tracking metadata so the resolver
            # loop can update pair.accuracy once the laggard has had time to react.
            corr_meta = (
                opportunity.get("_correlation_meta")
                if isinstance(opportunity, dict)
                else None
            )
            if corr_meta:
                uid = f"{corr_meta['pair_leader_id']}:{time.monotonic_ns()}"
                self._pending_corr_outcomes[uid] = {
                    **corr_meta,
                    "submit_ts_monotonic": time.monotonic(),
                }

            # Feed PnL to KillSwitch for drawdown tracking
            if self._kill_switch:
                # Approximate PnL: expected_profit is the best estimate until
                # we have proper mark-to-market from fill prices vs fair value
                pnl = float(arb.expected_profit)
                await self._kill_switch.record_pnl(pnl)

                # Record each fill for toxic flow detection
                for _ in result.fills:
                    self._kill_switch.record_fill()
        else:
            logger.warning(
                "❌ Trade execution failed",
                cluster_id=cluster_id,
                reason=result.reason,
                fills_before_failure=result.trade_count,
            )
    
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
