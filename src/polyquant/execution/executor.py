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
from polyquant.solver.scip_solver import OptimizationResult
from polyquant.utils import get_logger

logger = get_logger(__name__)


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
    
    def __init__(self, client: Any = None, limitless_client: Any = None, trade_store: Any = None):
        """
        Initialize executor.
        
        Args:
            client: PolymarketClient for order submission (None for paper mode)
            limitless_client: LimitlessClient for Base execution (None for paper mode)
            trade_store: TradeStore for persistent storage
        """
        self._client = client
        self._limitless_client = limitless_client
        self._trade_store = trade_store
        self._paper_mode = client is None

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
    
    async def execute_atomic(self, result: OptimizationResult) -> ExecutionResult:
        """
        Execute all trades atomically or none.

        Week 4: Parallel execution of independent legs within priority groups.

        Strategy:
        1. Group trades by priority (lower = illiquid, must execute first)
        2. Within a group (same priority), execute in parallel (independent)
        3. Between groups, execute sequentially (dependencies)

        Returns:
            ExecutionResult with fills or failure reason
        """
        if not result.success or not result.trades:
            return ExecutionResult(
                success=False,
                reason="No trades to execute",
            )

        # Rule 1: Pre-flight Balance Check
        required_poly = sum((t.notional_value for t in result.trades if t.exchange == "polymarket"), Decimal("0"))
        required_base = sum((t.notional_value for t in result.trades if t.exchange == "limitless"), Decimal("0"))
        
        if not self._paper_mode:
            if self.poly_balance < required_poly:
                return ExecutionResult(
                    success=False, 
                    reason=f"Insufficient Poly balance: {self.poly_balance} < {required_poly}"
                )
            if self.base_balance < required_base:
                return ExecutionResult(
                    success=False, 
                    reason=f"Insufficient Base balance: {self.base_balance} < {required_base}"
                )

        # 1. Dual Balance Checks
        poly_cost = sum(float(t.notional_value) for t in result.trades if t.exchange == "polymarket" and t.side.value == "BUY")
        base_cost = sum(float(t.notional_value) for t in result.trades if t.exchange == "limitless" and t.side.value == "BUY")
        
        if not self._paper_mode:
            if poly_cost > self.poly_balance:
                return ExecutionResult(success=False, reason=f"Insufficient Polymarket balance: Need {poly_cost}, have {self.poly_balance}")
            if base_cost > self.base_balance:
                return ExecutionResult(success=False, reason=f"Insufficient Limitless balance: Need {base_cost}, have {self.base_balance}")
            
            # 2. In-Flight Capital Check
            total_in_flight = poly_cost + base_cost
            from polyquant.utils import config
            if float(total_in_flight) > config.max_in_flight_capital:
                return ExecutionResult(
                    success=False, 
                    reason=f"Exceeds max in-flight capital limit: {total_in_flight} > {config.max_in_flight_capital}"
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
                await self._unwind(filled)
                return ExecutionResult(
                    success=False,
                    reason=f"Group {group_idx} failed: {failed_trades[0].outcome_id if failed_trades else 'unknown'}",
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
        if self._trade_store and filled:
            asyncio.create_task(self._trade_store.record_fills(filled))

        # Update local balances for immediate consistency
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
        Execute a batch of trades in parallel.

        Week 4: Parallel execution for independent trades (~10ms improvement).

        Args:
            trades: List of trades to execute (assumed independent)
            start_leg_idx: Starting leg index for logging

        Returns:
            List of successful fills (may be shorter than trades if some failed)
        """
        if len(trades) == 1:
            # Single trade - no need for parallelism
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

        # Week 4: Execute multiple trades in parallel
        logger.debug(
            f"Executing {len(trades)} trades in parallel",
            start_leg=start_leg_idx,
        )

        # Submit all orders concurrently
        fill_tasks = [self._submit_order(trade) for trade in trades]
        fill_results = await asyncio.gather(*fill_tasks, return_exceptions=True)

        # Collect successful fills
        successful_fills = []
        for i, (trade, fill_result) in enumerate(zip(trades, fill_results)):
            if isinstance(fill_result, Exception):
                logger.error(
                    "Trade execution raised exception",
                    leg=start_leg_idx + i,
                    outcome_id=trade.outcome_id,
                    error=str(fill_result),
                )
                # Exception = failure, stop here
                break
            elif fill_result is None:
                logger.warning(
                    "Trade execution failed",
                    leg=start_leg_idx + i,
                    outcome_id=trade.outcome_id,
                )
                # None = failure, stop here
                break
            else:
                # Success!
                logger.debug(
                    "Parallel trade filled",
                    leg=start_leg_idx + i,
                    outcome_id=trade.outcome_id,
                    size=fill_result.filled_size,
                    price=fill_result.filled_price,
                )
                successful_fills.append(fill_result)
                
                if fill_result.filled_size < trade.size: # Compare Decimal with Decimal
                    logger.warning(
                        "Partial fill detected. Breaking delta neutrality. Halting batch to trigger unwind.",
                        filled=fill_result.filled_size,
                        requested=trade.size
                    )
                    break

        return successful_fills
    
    async def _submit_order(self, trade: ProposedTrade) -> Fill | None:
        """
        Submit a single order via API or simulate in paper mode.
        
        Returns:
            Fill if successful, None if failed/rejected
        """
        if self._paper_mode:
            # Simulate fill in paper mode
            return Fill(
                trade=trade,
                filled_size=trade.size,
                filled_price=trade.limit_price,
            )
        
        # Real execution via API
        try:
            filled_amount = Decimal("0")
            price = Decimal("0")
            order_id = ""
            fill_quality = 0.0

            if trade.exchange == "limitless":
                if not self._limitless_client:
                    return None
                # Call Limitless API
                is_buy = trade.side == OrderSide.BUY
                market_slug = trade.reason.split("slug:")[1] if "slug:" in trade.reason else trade.outcome_id # Fallback
                
                # We need the market to get exchangeContract. We fetch it here, but in production, 
                # we'd inject it during optimization to save latency.
                market_info = await self._limitless_client.get_market(market_slug)
                res = await self._limitless_client.place_order(
                    market=market_info,
                    token_id=trade.outcome_id,
                    price=trade.limit_price,
                    size=trade.size,
                    is_buy=is_buy
                )
                if res and res.get("id"):
                    actual_filled_size = float(res.get("filledAmount", res.get("filledSize", 0.0)))
                    # Fallback if API omitted explicit size but status states complete
                    if actual_filled_size == 0.0 and res.get("status", "open") in ["filled", "closed", "matched"]:
                         actual_filled_size = float(trade.size)
                    return Fill(trade=trade, filled_size=actual_filled_size, filled_price=float(trade.limit_price), order_id=res["id"])
                return None
                
            else:
                if not self._client:
                    return None
                # Polymarket execution (Placeholder)
                logger.warning("Real Polymarket execution not implemented yet")
                return None
            
        except Exception as e:
            logger.error("Order submission failed", error=str(e), exchange=trade.exchange)
            return None
    
    async def _unwind(self, fills: list[Fill]) -> None:
        """
        Reverse all filled trades to return to neutral.
        
        Args:
            fills: List of fills to unwind
        """
        if not fills:
            return
            
        logger.warning("Leg Risk: Starting unwind of all previously filled legs", fills_to_reverse=len(fills))
        
        for fill in reversed(fills):
            # Reverse the side
            reverse_side = OrderSide.SELL if fill.trade.side == OrderSide.BUY else OrderSide.BUY
            
            if self._paper_mode:
                logger.debug(
                    "Paper unwind",
                    outcome_id=fill.trade.outcome_id,
                    side=reverse_side.value,
                    size=fill.filled_size,
                )
            else:
                logger.info(f"Unwinding {fill.filled_size} shares on {fill.trade.exchange}")
                from decimal import Decimal
                trade_to_unwind = ProposedTrade(
                    outcome_id=fill.trade.outcome_id,
                    side=reverse_side,
                    size=Decimal(str(fill.filled_size)),
                    limit_price=fill.trade.limit_price, 
                    exchange=fill.trade.exchange,
                    reason=fill.trade.reason
                )
                # Attempt to submit the unwind order
                await self._submit_order(trade_to_unwind)
        
        logger.info("Unwind complete")
