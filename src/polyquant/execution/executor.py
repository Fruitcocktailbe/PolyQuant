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
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from itertools import groupby
from typing import Any

from polyquant.data import ProposedTrade, OrderSide, OrderBook
from polyquant.solver.scip_solver import OptimizationResult
from polyquant.utils import config, get_logger

logger = get_logger(__name__)


@dataclass
class Fill:
    """A single filled order."""
    trade: ProposedTrade
    filled_size: Decimal
    filled_price: Decimal
    fill_time: datetime = field(default_factory=datetime.utcnow)
    order_id: str = ""
    fill_quality: float = 0.0  # (filled_price - midpoint) / spread — ratio, not currency
    
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

    Modes:
    - paper: Simulates fills instantly at limit price (no API calls)
    - live:  Submits real orders via PolymarketClient.place_order
    """
    
    def __init__(
        self,
        client: Any = None,
        trading_mode: str = "paper",
        trade_store: Any = None,
        cached_balance: Decimal = Decimal("0"),
        kill_switch: Any = None,
    ):
        """
        Initialize executor.

        Args:
            client: PolymarketClient for order submission.
            trading_mode: 'paper' for simulation, 'live' for real orders.
            trade_store: TradeStore for atomic fill persistence.
            cached_balance: Pre-fetched USDC balance (updated locally after fills).
            kill_switch: KillSwitch instance — triggered on unwind failure.
        """
        self._client = client
        self._trading_mode = trading_mode
        self._paper_mode = trading_mode == "paper"
        self._trade_store = trade_store
        self._cached_balance = cached_balance
        self._kill_switch = kill_switch

        # Fill quality tracking
        self._fill_quality_window: list[float] = []  # Rolling window
        self._fill_quality_window_size: int = 20  # Last 20 fills
        self._fill_quality_alert_threshold: float = 0.5  # Warn if avg > 0.5
        self._fill_quality_alert_active: bool = False
        
        if not self._paper_mode and self._client is None:
            raise ValueError(
                "PolymarketClient is required for live trading mode. "
                "Pass a client instance or set trading_mode='paper'."
            )
        
        logger.info(
            "TradeExecutor initialized",
            trading_mode=self._trading_mode,
            paper_mode=self._paper_mode,
            cached_balance=str(self._cached_balance),
        )

    @staticmethod
    def _calculate_fill_quality(
        filled_price: float, order_book: OrderBook | None
    ) -> float:
        """
        Calculate fill quality: (FilledPrice - Midpoint) / Spread.

        High values (> 0.5) mean we are consistently filled far from
        fair value — a sign of adverse selection.
        """
        if not order_book:
            return 0.0
        mid = order_book.mid_price
        spread = order_book.spread
        if mid is None or spread is None or spread == 0:
            return 0.0
        return abs(filled_price - mid) / spread
    
    async def execute_atomic(self, result: OptimizationResult) -> ExecutionResult:
        """
        Execute all trades atomically or none.

        Week 4: Parallel execution of independent legs within priority groups.

        Strategy:
        1. Group trades by priority (lower = illiquid, must execute first)
        2. Within a group (same priority), execute in parallel (independent)
        3. Between groups, execute sequentially (dependencies)

        Args:
            result: Optimization result with trades to execute

        Returns:
            ExecutionResult with fills or failure reason
        """
        if not result.success or not result.trades:
            return ExecutionResult(
                success=False,
                reason="No trades to execute",
            )

        # ── RULE 1: Pre-trade balance check ──
        required_notional = sum(
            (t.notional_value for t in result.trades), Decimal("0")
        )
        if not self._paper_mode and self._cached_balance < required_notional:
            logger.warning(
                "TRADE BLOCKED: Insufficient balance",
                cached_balance=str(self._cached_balance),
                required=str(required_notional),
                shortfall=str(required_notional - self._cached_balance),
            )
            return ExecutionResult(
                success=False,
                reason=(
                    f"Insufficient balance: have {self._cached_balance}, "
                    f"need {required_notional}"
                ),
            )

        # Sort by priority (lower = first = illiquid)
        sorted_trades = sorted(result.trades, key=lambda t: t.priority)

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
                logger.warning(
                    "Batch execution failed, unwinding",
                    group_idx=group_idx,
                    failed_count=len(failed_trades),
                    fills_to_unwind=len(filled),
                )
                await self._unwind(filled)
                return ExecutionResult(
                    success=False,
                    reason=f"Group {group_idx} failed: {failed_trades[0].outcome_id if failed_trades else 'unknown'}",
                    fills=filled,
                    total_failed=sum((t.notional_value for t in failed_trades), Decimal("0")),
                )

            filled.extend(group_fills)
            leg_counter += len(priority_group)

            logger.debug(
                f"Priority group {group_idx} complete",
                fills=len(group_fills),
                total_fills_so_far=len(filled),
            )

        total = sum((f.notional for f in filled), Decimal("0"))

        # Track fill quality for all fills
        for f in filled:
            await self._track_fill_quality(f)

        # Update cached balance (local, off-hot-path)
        self._cached_balance -= total

        # ── RULE 2 & 4: Persist fills atomically (fire-and-forget) ──
        if self._trade_store:
            try:
                await self._trade_store.record_fills(filled)
            except Exception:
                logger.error("TradeStore write failed (non-fatal)")

        logger.info(
            "Atomic execution complete",
            fills=len(filled),
            total_notional=str(total),
            remaining_balance=str(self._cached_balance),
        )

        return ExecutionResult(
            success=True,
            fills=filled,
            total_filled=total,
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
        Execute a batch of trades.

        P1-H5: In live mode, execute sequentially to preserve atomicity.
        If leg N fails, we know exactly which legs succeeded and need
        unwinding. Parallel execution risks partial fills where both
        succeed/fail non-deterministically.

        In paper mode, parallel execution is safe (no real orders).

        Args:
            trades: List of trades to execute
            start_leg_idx: Starting leg index for logging

        Returns:
            List of successful fills (may be shorter than trades if some failed)
        """
        if len(trades) == 1:
            fill = await self._submit_order(trades[0])
            if fill:
                logger.debug(
                    "Single trade filled",
                    leg=start_leg_idx,
                    outcome_id=trades[0].outcome_id,
                    size=fill.filled_size,
                    price=fill.filled_price,
                )
                return [fill]
            return []

        # Paper mode: parallel is safe and faster
        if self._paper_mode:
            fill_tasks = [self._submit_order(trade) for trade in trades]
            fill_results = await asyncio.gather(*fill_tasks, return_exceptions=True)
            return [r for r in fill_results if isinstance(r, Fill)]

        # Live mode: sequential execution for atomicity (H5)
        logger.debug(
            f"Executing {len(trades)} trades sequentially (live mode)",
            start_leg=start_leg_idx,
        )

        successful_fills = []
        for i, trade in enumerate(trades):
            try:
                fill = await self._submit_order(trade)
            except Exception as e:
                logger.error(
                    "Trade execution raised exception",
                    leg=start_leg_idx + i,
                    outcome_id=trade.outcome_id,
                    error=str(e),
                )
                break

            if fill is None:
                logger.warning(
                    "Trade execution failed",
                    leg=start_leg_idx + i,
                    outcome_id=trade.outcome_id,
                )
                break

            logger.debug(
                "Sequential trade filled",
                leg=start_leg_idx + i,
                outcome_id=trade.outcome_id,
                size=fill.filled_size,
                price=fill.filled_price,
            )
            successful_fills.append(fill)

        return successful_fills
    
    async def _submit_order(self, trade: ProposedTrade) -> Fill | None:
        """
        Submit a single order via API or simulate in paper mode.
        
        Returns:
            Fill if successful, None if failed/rejected
        """
        if self._paper_mode:
            # Simulate fill in paper mode
            logger.debug(
                "Paper fill",
                outcome_id=trade.outcome_id,
                side=trade.side.value,
                size=trade.size,
                price=trade.limit_price,
            )
            return Fill(
                trade=trade,
                filled_size=trade.size,
                filled_price=trade.limit_price,
                order_id=f"paper_{trade.outcome_id[:8]}_{datetime.utcnow().timestamp():.0f}",
            )
        
        # ── LIVE EXECUTION via PolymarketClient ──
        try:
            result = await self._client.place_order(trade)

            status = result.get("status", "unknown")
            order_id = result.get("order_id", "")

            if status == "error":
                reason = result.get("reason", "unknown")
                logger.error(
                    "Live order rejected",
                    outcome_id=trade.outcome_id,
                    reason=reason,
                )
                return None

            # ── FOK Fill Confirmation ──
            # Check the CLOB's raw response for actual match status.
            # FOK orders are either fully matched or fully rejected.
            raw = result.get("raw_response", {})
            clob_status = raw.get("status", "unknown")

            if clob_status == "matched":
                # FOK succeeded — fully filled
                filled_size = Decimal(str(raw.get("matchedAmount", trade.size)))
                logger.info(
                    "FOK order matched",
                    order_id=order_id,
                    outcome_id=trade.outcome_id,
                    side=trade.side.value,
                    filled_size=filled_size,
                    price=trade.limit_price,
                )
                return Fill(
                    trade=trade,
                    filled_size=filled_size,
                    filled_price=trade.limit_price,
                    order_id=order_id or "",
                )
            elif clob_status == "delayed":
                # Sent to matcher but not confirmed — poll for confirmation
                logger.warning(
                    "Order delayed — polling for confirmation",
                    order_id=order_id,
                    outcome_id=trade.outcome_id,
                )
                return await self._confirm_order(trade, order_id)
            else:
                # Any other status = rejection or unknown
                logger.warning(
                    "FOK order not matched",
                    order_id=order_id,
                    clob_status=clob_status,
                    outcome_id=trade.outcome_id,
                )
                return None
            
        except Exception as e:
            logger.error(
                "Order submission failed",
                error=str(e),
                outcome_id=trade.outcome_id,
            )
            return None
    
    async def _confirm_order(
        self, trade: ProposedTrade, order_id: str
    ) -> Fill | None:
        """
        Poll CLOB for order confirmation, cancel if timeout.

        When CLOB returns "delayed", the order may still fill. We poll
        every config.order_confirm_poll_interval_ms for up to
        config.order_confirm_timeout_ms. If it confirms, return Fill.
        If timeout, explicitly cancel to prevent ghost orders.

        Args:
            trade: The original trade.
            order_id: The CLOB order ID to poll.

        Returns:
            Fill if order matched, None if cancelled/timed out.
        """
        timeout_ms = config.order_confirm_timeout_ms
        poll_ms = config.order_confirm_poll_interval_ms
        elapsed_ms = 0

        while elapsed_ms < timeout_ms:
            await asyncio.sleep(poll_ms / 1000.0)
            elapsed_ms += poll_ms

            try:
                status_resp = await self._client.get_order_status(order_id)
                status = status_resp.get("status", "unknown")

                if status == "matched":
                    filled_size = Decimal(
                        str(status_resp.get("matchedAmount", trade.size))
                    )
                    logger.info(
                        "Delayed order confirmed matched",
                        order_id=order_id,
                        outcome_id=trade.outcome_id,
                        elapsed_ms=elapsed_ms,
                        filled_size=filled_size,
                    )
                    return Fill(
                        trade=trade,
                        filled_size=filled_size,
                        filled_price=trade.limit_price,
                        order_id=order_id,
                    )

                if status in ("cancelled", "expired", "error"):
                    logger.info(
                        "Delayed order resolved as not filled",
                        order_id=order_id,
                        status=status,
                        elapsed_ms=elapsed_ms,
                    )
                    return None

                # Still "delayed" or "live" — keep polling
                logger.debug(
                    "Order still pending",
                    order_id=order_id,
                    status=status,
                    elapsed_ms=elapsed_ms,
                )

            except Exception as e:
                logger.warning(
                    "Order status poll failed",
                    order_id=order_id,
                    error=str(e),
                    elapsed_ms=elapsed_ms,
                )

        # Timeout — explicitly cancel to prevent ghost fill
        logger.warning(
            "Order confirmation timed out — cancelling ghost order",
            order_id=order_id,
            outcome_id=trade.outcome_id,
            timeout_ms=timeout_ms,
        )
        try:
            await self._client.cancel_order(order_id)
        except Exception as e:
            logger.error(
                "GHOST ORDER: Cancel failed — order may fill unhedged",
                order_id=order_id,
                outcome_id=trade.outcome_id,
                error=str(e),
            )

        return None

    async def _unwind(self, fills: list[Fill]) -> None:
        """
        Reverse all filled trades to return to neutral.

        P0-4: Retries each leg up to 3 times with widening price.
        If ANY leg fails after all retries, triggers the kill switch
        to halt trading immediately — never leaves unhedged positions
        while continuing to trade.

        Args:
            fills: List of fills to unwind
        """
        if not fills:
            return

        MAX_RETRIES = 3
        unwind_failed = False

        logger.info("Starting unwind", fills_to_reverse=len(fills))

        for fill in reversed(fills):
            reverse_side = OrderSide.SELL if fill.trade.side == OrderSide.BUY else OrderSide.BUY

            if self._paper_mode:
                logger.debug(
                    "Paper unwind",
                    outcome_id=fill.trade.outcome_id,
                    side=reverse_side.value,
                    size=fill.filled_size,
                )
                continue

            # ── LIVE UNWIND with retries ──
            leg_success = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    # Widen price more aggressively on each retry
                    widen_pct = Decimal(str(0.03 * attempt))  # 3%, 6%, 9%

                    if reverse_side == OrderSide.BUY:
                        unwind_price = min(
                            fill.filled_price * (1 + widen_pct), Decimal("0.99")
                        )
                    else:
                        unwind_price = max(
                            fill.filled_price * (1 - widen_pct), Decimal("0.01")
                        )

                    from polyquant.data import TimeInForce
                    unwind_trade = ProposedTrade(
                        outcome_id=fill.trade.outcome_id,
                        side=reverse_side,
                        size=fill.filled_size,
                        limit_price=unwind_price,
                        priority=0,
                        time_in_force=TimeInForce.FAK,
                    )
                    result = await self._client.place_order(unwind_trade)
                    status = result.get("status", "unknown")

                    if status != "error":
                        # Check for actual match via confirmation polling
                        raw = result.get("raw_response", {})
                        clob_status = raw.get("status", "unknown")
                        if clob_status in ("matched", "live"):
                            logger.info(
                                "Unwind leg succeeded",
                                outcome_id=fill.trade.outcome_id,
                                side=reverse_side.value,
                                size=fill.filled_size,
                                attempt=attempt,
                                order_id=result.get("order_id"),
                            )
                            leg_success = True
                            break
                        elif clob_status == "delayed":
                            # Poll for confirmation
                            order_id = result.get("order_id", "")
                            if order_id:
                                confirmed = await self._confirm_order(
                                    unwind_trade, order_id
                                )
                                if confirmed:
                                    leg_success = True
                                    break

                    logger.warning(
                        "Unwind attempt failed",
                        outcome_id=fill.trade.outcome_id,
                        attempt=attempt,
                        max_retries=MAX_RETRIES,
                        status=status,
                    )

                except Exception as e:
                    logger.error(
                        "Unwind attempt exception",
                        outcome_id=fill.trade.outcome_id,
                        attempt=attempt,
                        error=str(e),
                    )

                # Brief pause before retry
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(0.5 * attempt)

            if not leg_success:
                unwind_failed = True
                logger.critical(
                    "UNWIND FAILED after all retries",
                    outcome_id=fill.trade.outcome_id,
                    side=reverse_side.value,
                    size=str(fill.filled_size),
                    retries=MAX_RETRIES,
                )

        # P0-4: If ANY unwind leg failed, trigger kill switch immediately
        if unwind_failed and self._kill_switch:
            from polyquant.risk.kill_switch import TriggerReason
            logger.critical(
                "KILL SWITCH TRIGGERED: Unwind failure — halting all trading"
            )
            await self._kill_switch.trigger(
                reason="Unwind failed after retries — unhedged position exists",
                trigger_type=TriggerReason.UNWIND_FAILURE,
            )

        logger.info(
            "Unwind complete",
            all_succeeded=not unwind_failed,
        )
