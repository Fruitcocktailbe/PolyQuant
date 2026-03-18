"""
Trade Execution for PolyQuant 2.0

This module handles atomic execution of multi-leg arbitrage trades.

SAFETY FEATURES:
----------------
1. Illiquid legs executed FIRST (fail fast, no exposure)
2. IOC orders by default (no hanging orders)
3. Unwind logic if later legs fail (reverse filled trades)

Week 4 Enhancement: Parallel execution of independent legs (~10ms improvement)

USAGE:
------
    executor = TradeExecutor(client=polymarket_client)

    result = await executor.execute_atomic(optimization_result)

    if result.success:
        print(f"All {len(result.fills)} trades filled")
    else:
        print(f"Aborted: {result.reason}")
"""

import asyncio
from decimal import Decimal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import groupby
from typing import Any

from polyquant.data import ProposedTrade, OrderSide
from polyquant.utils import get_logger, config

logger = get_logger(__name__)

# Unwind retry parameters
UNWIND_MAX_RETRIES = 3
UNWIND_SPREAD_WIDENING = [Decimal("0.03"), Decimal("0.06"), Decimal("0.09")]  # 3% → 6% → 9%


@dataclass
class Fill:
    """A single filled order."""
    trade: ProposedTrade
    filled_size: Decimal
    filled_price: Decimal
    fill_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: str = ""
    fill_quality: float = 0.0  # (filled_price - midpoint) / spread
    
    @property
    def notional(self) -> Decimal:
        return self.filled_size * self.filled_price


@dataclass
class ExecutionResult:
    """Result of an atomic execution attempt."""
    success: bool
    fills: list[Fill] = field(default_factory=list)
    reason: str = ""
    total_filled: Decimal = Decimal("0")
    total_failed: Decimal = Decimal("0")
    
    @property
    def trade_count(self) -> int:
        return len(self.fills)


class TradeExecutor:
    """
    Executes multi-leg trades atomically.
    
    Strategy:
    1. Sort trades by priority (illiquid first)
    2. Execute sequentially
    3. If any leg fails, unwind all previous fills
    4. Use IOC to prevent hanging orders
    """
    def __init__(self, rust_client: Any, trade_store: Any = None, paper_mode: bool = False):
        """
        Initialize executor.
        
        Args:
            rust_client: RustClient for ZMQ ultra-low latency execution sidecar
            trade_store: TradeStore for persistent storage
            paper_mode: If True, simulate fills locally without dispatching to Rust
        """
        self._rust_client = rust_client
        self._trade_store = trade_store
        self._paper_mode = paper_mode

        # Dual Balance Tracking
        self.poly_balance: Decimal = Decimal("0")
        self.base_balance: Decimal = Decimal("0")

        # Fill quality tracking
        self._fill_quality_window: list[float] = []  # Rolling window
        self._fill_quality_window_size: int = 20  # Last 20 fills
        self._fill_quality_alert_threshold: float = 0.5  # Warn if avg > 0.5
        self._fill_quality_alert_active: bool = False
        
        logger.info(
            "TradeExecutor initialized",
            paper_mode=self._paper_mode,
        )
    
    async def execute_trade(self, trade: ProposedTrade) -> ExecutionResult:
        """
        Execute a single directional trade.

        Returns:
            ExecutionResult with fills or failure reason
        """
        if not trade:
            return ExecutionResult(
                success=False,
                reason="No trade to execute",
            )
            
        result_trades = [trade]

        # Single-pass pre-flight checks: balance, in-flight capital, VWAP slippage
        if not self._paper_mode:
            poly_buy_cost = Decimal("0")
            base_buy_cost = Decimal("0")
            for t in result_trades:
                if t.side.value == "BUY":
                    if t.exchange == "polymarket":
                        poly_buy_cost += t.notional_value
                    elif t.exchange == "limitless":
                        base_buy_cost += t.notional_value

            if poly_buy_cost > self.poly_balance:
                return ExecutionResult(
                    success=False,
                    reason=f"Insufficient Polymarket balance: {self.poly_balance} < {poly_buy_cost}",
                )
            if base_buy_cost > self.base_balance:
                return ExecutionResult(
                    success=False,
                    reason=f"Insufficient Limitless balance: {self.base_balance} < {base_buy_cost}",
                )

            total_in_flight = poly_buy_cost + base_buy_cost
            if total_in_flight > Decimal(str(config.max_in_flight_capital)):
                return ExecutionResult(
                    success=False,
                    reason=f"Exceeds max in-flight capital: {total_in_flight} > {config.max_in_flight_capital}",
                )

            # P0: VWAP slippage enforcement — reject if any trade exceeds limit
            slippage_limit = Decimal(str(config.vwap_slippage_limit))
            if slippage_limit > 0:
                for t in result.trades:
                    if hasattr(t, "vwap_slippage") and t.vwap_slippage is not None:
                        if t.vwap_slippage > slippage_limit:
                            return ExecutionResult(
                                success=False,
                                reason=f"VWAP slippage {t.vwap_slippage} exceeds limit {slippage_limit} on {t.outcome_id}",
                            )

        # Sort by priority (lower = first = illiquid)
        sorted_trades = sorted(result_trades, key=lambda t: t.priority)

        # Week 4: Group by priority for parallel execution
        priority_groups = [
            list(group) for _, group in groupby(sorted_trades, key=lambda t: t.priority)
        ]

        logger.info(
            "Starting atomic execution with batching",
            trade_count=len(sorted_trades),
            priority_groups=len(priority_groups),
            group_sizes=[len(g) for g in priority_groups],
        )

        filled: list[Fill] = []
        total_filled_notional = Decimal("0")
        total_failed_notional = Decimal("0")
        leg_counter = 0

        # Execute each priority group sequentially
        for group_idx, priority_group in enumerate(priority_groups):
            logger.debug(
                f"Executing priority group {group_idx}",
                trades_in_group=len(priority_group),
                priority=priority_group[0].priority if priority_group else None,
            )

            # Week 4: Execute trades within group in parallel (independent)
            group_fills = await self._execute_batch(priority_group, leg_counter)

            # Check if any trade in the batch failed
            if len(group_fills) != len(priority_group):
                # At least one leg failed - unwind everything
                failed_trades = [t for t in priority_group if not any(f.trade == t for f in group_fills)]
                total_failed_notional += sum((t.notional_value for t in failed_trades), Decimal("0"))
                logger.warning(
                    "Batch execution failed, unwinding",
                    group_idx=group_idx,
                    failed_count=len(failed_trades),
                    fills_to_unwind=len(filled),
                )
                unwind_ok = await self._unwind(filled)
                reason = f"Group {group_idx} failed: {failed_trades[0].outcome_id if failed_trades else 'unknown'}"
                if not unwind_ok:
                    reason += " | UNWIND FAILED — positions may be unhedged"
                return ExecutionResult(
                    success=False,
                    reason=reason,
                    fills=filled,
                    total_filled=total_filled_notional,
                    total_failed=total_failed_notional,
                )

            filled.extend(group_fills)
            total_filled_notional += sum(f.notional for f in group_fills)
            leg_counter += len(priority_group)

            logger.debug(
                f"Priority group {group_idx} complete",
                fills=len(group_fills),
                total_fills_so_far=len(filled),
            )

        # Track fill quality for all fills
        for f in filled:
            await self._track_fill_quality(f)

        logger.info(
            "Atomic execution complete",
            fills=len(filled),
            total_notional=total_filled_notional,
        )
        
        # Record fills to persistent storage
        # Record fills to persistent storage (awaited for durability) and UI
        if filled:
            if self._trade_store:
                try:
                    await self._trade_store.record_fills(filled)
                except Exception as e:
                    logger.error("Failed to persist fills to trade store", error=str(e))

            # UI Monitor Broadcast (best-effort, non-blocking)
            from polyquant.api.server import monitor
            for f in filled:
                fill_data = {
                    "order_id": f.order_id,
                    "filled_size": float(f.filled_size),
                    "filled_price": float(f.filled_price),
                    "fill_time": f.fill_time.isoformat(),
                    "trade": {
                        "outcome_id": f.trade.outcome_id,
                        "side": f.trade.side.value,
                        "exchange": f.trade.exchange,
                    }
                }
                asyncio.create_task(monitor.record_trade(fill_data))

        # Update local balances — only deduct confirmed fills
        self.poly_balance -= sum((f.notional for f in filled if f.trade.exchange == "polymarket"), Decimal("0"))
        self.base_balance -= sum((f.notional for f in filled if f.trade.exchange == "limitless"), Decimal("0"))

        return ExecutionResult(
            success=True,
            fills=filled,
            total_filled=total_filled_notional,
            total_failed=total_failed_notional,
        )

    async def _track_fill_quality(self, fill: Fill) -> None:
        """
        Track fill quality and push UI alert if consistently poor.

        Fill quality = (filled_price - midpoint) / spread
        High values (> 0.5) mean we are consistently being filled far
        from fair value — a sign that faster competitors are correcting
        mispricing before our orders execute.
        """
        quality = fill.fill_quality

        # Add to rolling window
        self._fill_quality_window.append(quality)
        if len(self._fill_quality_window) > self._fill_quality_window_size:
            self._fill_quality_window.pop(0)

        # Only alert after enough data
        if len(self._fill_quality_window) < 10:
            return

        avg_quality = sum(self._fill_quality_window) / len(self._fill_quality_window)

        if avg_quality > self._fill_quality_alert_threshold:
            if not self._fill_quality_alert_active:
                self._fill_quality_alert_active = True
                logger.warning(
                    "FILL QUALITY ALERT: Consistently poor fill quality detected",
                    avg_fill_quality=f"{avg_quality:.3f}",
                    window_size=len(self._fill_quality_window),
                    threshold=self._fill_quality_alert_threshold,
                    recommendation="Latency may be too high — faster competitors are "
                                   "correcting mispricing before your orders execute. "
                                   "Consider upgrading to Private RPC or reducing position size.",
                )

                # Push alert to UI dashboard
                try:
                    from polyquant.api.server import monitor
                    import asyncio
                    asyncio.create_task(monitor.update_status(
                        fill_quality_alert={
                            "level": "WARNING",
                            "message": (
                                f"Poor fill quality: {avg_quality:.2f} avg over last "
                                f"{len(self._fill_quality_window)} fills. You may be "
                                f"arriving too late — faster competitors are correcting "
                                f"mispricing before your orders execute."
                            ),
                            "metric": round(avg_quality, 3),
                            "threshold": self._fill_quality_alert_threshold,
                            "recommendation": "Consider reducing position size or upgrading to Private RPC.",
                        }
                    ))
                except Exception:
                    pass  # Don't crash on UI alert failure
        else:
            if self._fill_quality_alert_active:
                self._fill_quality_alert_active = False
                logger.info(
                    "Fill quality recovered",
                    avg_fill_quality=f"{avg_quality:.3f}",
                )

    async def _execute_batch(
        self,
        trades: list[ProposedTrade],
        start_leg_idx: int = 0,
    ) -> list[Fill]:
        """
        Execute a batch of trades via the Rust OMS Sidecar.

        Args:
            trades: List of trades to execute
            start_leg_idx: Starting leg index for logging

        Returns:
            List of successful fills (may be shorter than trades if some failed)
        """
        if self._paper_mode:
            logger.debug("Simulating batch execution in paper mode")
            return [
                Fill(
                    trade=t,
                    filled_size=t.size,
                    filled_price=t.limit_price,
                    order_id=f"paper_sim_{start_leg_idx}_{i}"
                )
                for i, t in enumerate(trades)
            ]

        if not getattr(self, "_rust_client", None):
            raise RuntimeError("CRITICAL: RustClient is required for live execution.")

        logger.info("Dispatching to Rust OMS Sidecar", trades=len(trades))

        resp = self._rust_client.dispatch_trades(trades)

        if not resp:
            logger.error("No response from Rust OMS (timeout or connection error)")
            return []

        status = resp.get("status", "")

        if status == "halted":
            logger.critical("Rust OMS is HALTED — cannot execute trades")
            return []

        if status == "rejected":
            rust_errors = resp.get("errors", [])
            for err in rust_errors:
                logger.warning(
                    "Trade rejected by Rust OMS",
                    outcome_id=err.get("trade", {}).get("outcome_id"),
                    error=err.get("error"),
                )
            return []

        # Log any errors from partial failures
        rust_errors = resp.get("errors", [])
        for err in rust_errors:
            err_trade = err.get("trade", {})
            logger.error(
                "Trade failed in Rust OMS",
                outcome_id=err_trade.get("outcome_id"),
                exchange=err_trade.get("exchange"),
                error=err.get("error"),
            )

        # Parse actual fills from Rust response
        fills = []
        for rust_fill in resp.get("fills", []):
            rust_trade = rust_fill.get("trade", {})
            outcome_id = rust_trade.get("outcome_id", "")

            # Match back to original ProposedTrade by outcome_id
            matching_trade = next(
                (t for t in trades if t.outcome_id == outcome_id), None
            )
            if matching_trade is None:
                logger.warning("Rust returned fill for unknown outcome_id", outcome_id=outcome_id)
                continue

            filled_size = Decimal(rust_fill.get("filled_size", "0"))
            filled_price = Decimal(rust_fill.get("filled_price", "0"))
            order_id = rust_fill.get("order_id", "")

            if filled_size <= 0:
                logger.warning("Zero-fill from Rust OMS", outcome_id=outcome_id)
                continue

            fills.append(Fill(
                trade=matching_trade,
                filled_size=filled_size,
                filled_price=filled_price,
                order_id=order_id,
            ))

        if len(fills) != len(trades):
            logger.warning(
                "Partial fill from Rust OMS",
                expected=len(trades),
                actual=len(fills),
                errors=len(rust_errors),
            )

        return fills
    
    async def _unwind(self, fills: list[Fill]) -> bool:
        """
        Reverse all filled trades to return to neutral.

        Retries up to UNWIND_MAX_RETRIES times with widening spread
        (3% → 6% → 9%). If all retries fail, triggers kill switch.

        Args:
            fills: List of fills to unwind

        Returns:
            True if all fills were successfully unwound, False otherwise.
        """
        if not fills:
            return True

        logger.warning(
            "HANGING LEG: Starting unwind with retry logic",
            fills_to_reverse=len(fills),
        )

        if self._paper_mode:
            for fill in reversed(fills):
                reverse_side = OrderSide.SELL if fill.trade.side == OrderSide.BUY else OrderSide.BUY
                logger.debug("Paper unwind", outcome_id=fill.trade.outcome_id, side=reverse_side.value)
            return True

        if not (hasattr(self, "_rust_client") and self._rust_client):
            logger.error("CRITICAL: Cannot unwind — RustClient not available")
            return False

        remaining_fills = list(fills)

        for attempt in range(UNWIND_MAX_RETRIES):
            spread_widen = UNWIND_SPREAD_WIDENING[attempt]
            logger.warning(
                f"Unwind attempt {attempt + 1}/{UNWIND_MAX_RETRIES}",
                spread_widen=float(spread_widen),
                remaining=len(remaining_fills),
            )

            unwind_trades = []
            for fill in reversed(remaining_fills):
                reverse_side = OrderSide.SELL if fill.trade.side == OrderSide.BUY else OrderSide.BUY

                # Widen limit price to improve fill probability
                original_price = fill.filled_price
                if reverse_side == OrderSide.SELL:
                    # Selling: lower the limit price to be more aggressive
                    widened_price = original_price * (Decimal("1") - spread_widen)
                else:
                    # Buying back: raise the limit price
                    widened_price = original_price * (Decimal("1") + spread_widen)

                unwind_trades.append(ProposedTrade(
                    outcome_id=fill.trade.outcome_id,
                    side=reverse_side,
                    size=Decimal(str(fill.filled_size)),
                    limit_price=widened_price,
                    exchange=fill.trade.exchange,
                    reason=f"unwind_attempt_{attempt + 1}",
                ))

            resp = self._rust_client.dispatch_trades(unwind_trades)

            if not resp:
                logger.error(f"Unwind attempt {attempt + 1}: no response from Rust OMS")
                await asyncio.sleep(0.1 * (attempt + 1))
                continue

            status = resp.get("status", "")
            unwind_fills = resp.get("fills", [])
            unwind_errors = resp.get("errors", [])

            # Determine which fills were successfully unwound
            unwound_ids = {f.get("trade", {}).get("outcome_id") for f in unwind_fills}
            remaining_fills = [f for f in remaining_fills if f.trade.outcome_id not in unwound_ids]

            # Restore balances for successful unwinds
            for uf in unwind_fills:
                filled_size = Decimal(uf.get("filled_size", "0"))
                filled_price = Decimal(uf.get("filled_price", "0"))
                notional = filled_size * filled_price
                exchange = uf.get("trade", {}).get("exchange", "")
                if exchange == "polymarket":
                    self.poly_balance += notional
                elif exchange == "limitless":
                    self.base_balance += notional

            if not remaining_fills:
                logger.info(f"Unwind complete on attempt {attempt + 1}")
                return True

            logger.warning(
                f"Unwind attempt {attempt + 1} partial",
                unwound=len(unwound_ids),
                remaining=len(remaining_fills),
                errors=len(unwind_errors),
            )

            if attempt < UNWIND_MAX_RETRIES - 1:
                await asyncio.sleep(0.2 * (attempt + 1))

        # All retries exhausted — trigger kill switch
        logger.critical(
            "UNWIND FAILED after all retries — triggering kill switch",
            remaining_positions=len(remaining_fills),
            positions=[f.trade.outcome_id for f in remaining_fills],
        )

        if hasattr(self, "_kill_switch") and self._kill_switch:
            await self._kill_switch.trigger(reason="unwind_failure")
        elif hasattr(self, "_rust_client") and self._rust_client:
            self._rust_client.send_halt(reason="unwind_failure_all_retries_exhausted")

        return False
