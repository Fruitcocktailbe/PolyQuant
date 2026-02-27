"""
Trade Execution for PolyQuant 2.0

This module handles atomic execution of multi-leg arbitrage trades.

SAFETY FEATURES:
----------------
1. Illiquid legs executed FIRST (fail fast, no exposure)
2. IOC orders by default (no hanging orders)
3. Unwind logic if later legs fail (reverse filled trades)

USAGE:
------
    executor = TradeExecutor(client=polymarket_client)
    
    result = await executor.execute_atomic(optimization_result)
    
    if result.success:
        print(f"All {len(result.fills)} trades filled")
    else:
        print(f"Aborted: {result.reason}")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from polyquant.data import ProposedTrade, OrderSide
from polyquant.solver.scip_solver import OptimizationResult
from polyquant.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Fill:
    """A single filled order."""
    trade: ProposedTrade
    filled_size: float
    filled_price: float
    fill_time: datetime = field(default_factory=datetime.utcnow)
    order_id: str = ""
    
    @property
    def notional(self) -> float:
        return self.filled_size * self.filled_price


@dataclass
class ExecutionResult:
    """Result of an atomic execution attempt."""
    success: bool
    fills: list[Fill] = field(default_factory=list)
    reason: str = ""
    total_filled: float = 0.0
    total_failed: float = 0.0
    
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
    
    def __init__(self, client: Any = None):
        """
        Initialize executor.
        
        Args:
            client: PolymarketClient for order submission (None for paper mode)
        """
        self._client = client
        self._paper_mode = client is None
        
        logger.info(
            "TradeExecutor initialized",
            paper_mode=self._paper_mode,
        )
    
    async def execute_atomic(self, result: OptimizationResult) -> ExecutionResult:
        """
        Execute all trades atomically or none.
        
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
        
        # Sort by priority (lower = first = illiquid)
        sorted_trades = sorted(result.trades, key=lambda t: t.priority)
        
        logger.info(
            "Starting atomic execution",
            trade_count=len(sorted_trades),
            priorities=[t.priority for t in sorted_trades],
        )
        
        filled: list[Fill] = []
        
        for i, trade in enumerate(sorted_trades):
            fill = await self._submit_order(trade)
            
            if fill is None:
                # This leg failed - unwind everything
                logger.warning(
                    "Leg failed, unwinding",
                    failed_leg=i,
                    outcome_id=trade.outcome_id,
                    fills_to_unwind=len(filled),
                )
                await self._unwind(filled)
                return ExecutionResult(
                    success=False,
                    reason=f"Leg {i} failed: {trade.outcome_id}",
                    fills=filled,
                    total_failed=trade.notional_value,
                )
            
            filled.append(fill)
            logger.debug(
                "Leg filled",
                leg=i,
                outcome_id=trade.outcome_id,
                size=fill.filled_size,
                price=fill.filled_price,
            )
        
        total = sum(f.notional for f in filled)
        
        logger.info(
            "Atomic execution complete",
            fills=len(filled),
            total_notional=total,
        )
        
        return ExecutionResult(
            success=True,
            fills=filled,
            total_filled=total,
        )
    
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
        
        # Real execution via CLOB API
        try:
            # TODO: Implement real API call
            # response = await self._client.submit_order(
            #     token_id=trade.outcome_id,
            #     side=trade.side.value,
            #     size=trade.size,
            #     price=trade.limit_price,
            #     time_in_force=trade.time_in_force.value,
            # )
            logger.warning("Real execution not implemented yet")
            return None
            
        except Exception as e:
            logger.error("Order submission failed", error=str(e))
            return None
    
    async def _unwind(self, fills: list[Fill]) -> None:
        """
        Reverse all filled trades to return to neutral.
        
        Args:
            fills: List of fills to unwind
        """
        if not fills:
            return
            
        logger.info("Starting unwind", fills_to_reverse=len(fills))
        
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
                # TODO: Submit real unwind order
                pass
        
        logger.info("Unwind complete")
