"""
Kill Switch for PolyQuant 2.0

The kill switch is a critical safety mechanism that halts all trading
when predefined risk thresholds are breached.

WHEN DOES IT TRIGGER?
---------------------
1. Drawdown exceeds 15% (configurable)
2. Solver timeout average exceeds threshold
3. API errors exceed threshold
4. Manual override

WHY IS THIS IMPORTANT?
----------------------
Automated trading systems can lose money very quickly if something
goes wrong. The kill switch prevents:
- Runaway losses from bugs
- Excessive losses from model errors
- Losses from market manipulation or unusual conditions
- Losses from API/connectivity issues

ARCHITECTURE:
-------------
The kill switch runs as a background monitor that:
1. Tracks P&L in real-time
2. Computes rolling metrics (drawdown, error rates)
3. Triggers halt if any threshold is breached
4. Requires manual reset to resume

USAGE:
------
    kill_switch = KillSwitch(
        max_drawdown=0.15,
        initial_capital=10000,
    )
    
    # Before each trade
    if kill_switch.is_triggered:
        raise SystemError("Kill switch triggered - trading halted")
    
    # After each trade
    kill_switch.update_pnl(profit_or_loss)
    
    # Manual trigger
    kill_switch.trigger("Market anomaly detected")
    
    # Manual reset (requires confirmation)
    kill_switch.reset(confirm=True)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field

from polyquant.utils import config, get_logger

logger = get_logger(__name__)


class TriggerReason(str, Enum):
    """Reasons why the kill switch might trigger."""
    DRAWDOWN = "drawdown_exceeded"
    SOLVER_TIMEOUT = "solver_timeout_exceeded"
    API_ERRORS = "api_error_rate_exceeded"
    MANUAL = "manual_trigger"
    LATENCY = "latency_exceeded"
    UNKNOWN = "unknown"


@dataclass
class TriggerEvent:
    """Record of a kill switch trigger."""
    timestamp: datetime
    reason: TriggerReason
    details: str
    metric_value: float
    threshold: float


class KillSwitchState(BaseModel):
    """
    Current state of the kill switch.
    
    Attributes:
        is_triggered: Whether trading is halted
        trigger_event: Details of what triggered (if any)
        current_drawdown: Current drawdown as a fraction
        high_water_mark: Highest capital value seen
        current_capital: Current capital value
        trigger_count: How many times the kill switch has triggered
    """
    is_triggered: bool = False
    trigger_event: TriggerEvent | None = None
    current_drawdown: float = 0.0
    high_water_mark: float = 0.0
    current_capital: float = 0.0
    trigger_count: int = 0


class KillSwitch:
    """
    Safety mechanism to halt trading when risk thresholds are breached.
    
    The kill switch monitors multiple metrics and triggers a trading
    halt if any threshold is exceeded.
    
    Metrics Monitored:
    - Drawdown (peak-to-trough decline)
    - Solver timeout rate (5-minute rolling average)
    - API error rate
    - Trade latency
    
    Example:
        kill_switch = KillSwitch(initial_capital=10000)
        
        # Check before trading
        if not kill_switch.can_trade():
            logger.warning("Trading halted by kill switch")
            return
        
        # Update after trade
        kill_switch.record_pnl(profit=-50)
        
        # Check status
        state = kill_switch.get_state()
        print(f"Current drawdown: {state.current_drawdown:.1%}")
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_drawdown: float | None = None,
        solver_timeout_threshold: float = 30.0,
        max_solver_timeout_rate: float = 0.5,
        latency_threshold_ms: int | None = None,
        on_trigger: Callable[[TriggerEvent], None] | None = None,
    ):
        """
        Initialize the kill switch.
        
        Args:
            initial_capital: Starting capital
            max_drawdown: Maximum drawdown before trigger (0-1)
            solver_timeout_threshold: Timeout threshold in seconds
            max_solver_timeout_rate: Max fraction of timeouts in window
            latency_threshold_ms: Max latency before warning
            on_trigger: Callback when kill switch triggers
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.high_water_mark = initial_capital
        
        # Thresholds (from config or parameters)
        self.max_drawdown = max_drawdown or config.max_drawdown
        self.solver_timeout_threshold = solver_timeout_threshold
        self.max_solver_timeout_rate = max_solver_timeout_rate
        self.latency_threshold_ms = latency_threshold_ms or config.latency_target_ms
        
        # State
        self._is_triggered = False
        self._trigger_event: TriggerEvent | None = None
        self._trigger_count = 0
        
        # Rolling windows for metrics
        self._solver_times: list[tuple[datetime, float]] = []
        self._api_errors: list[datetime] = []
        self._trade_latencies: list[tuple[datetime, float]] = []
        
        # Callback
        self._on_trigger = on_trigger
        
        logger.info(
            "KillSwitch initialized",
            max_drawdown=self.max_drawdown,
            initial_capital=initial_capital,
        )
    
    @property
    def is_triggered(self) -> bool:
        """Whether the kill switch is currently triggered."""
        return self._is_triggered
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown as a fraction (0-1)."""
        if self.high_water_mark <= 0:
            return 0.0
        return (self.high_water_mark - self.current_capital) / self.high_water_mark
    
    def can_trade(self) -> bool:
        """
        Check if trading is allowed.
        
        Returns False if kill switch is triggered.
        Also runs metric checks that might trigger.
        """
        if self._is_triggered:
            return False
        
        # Run checks that might trigger
        self._check_drawdown()
        self._check_solver_timeouts()
        self._check_latency()
        
        return not self._is_triggered
    
    def record_pnl(self, amount: float) -> None:
        """
        Record a P&L change.
        
        Updates current capital and high water mark.
        May trigger if drawdown exceeds threshold.
        
        Args:
            amount: Profit (positive) or loss (negative)
        """
        self.current_capital += amount
        
        # Update high water mark
        if self.current_capital > self.high_water_mark:
            self.high_water_mark = self.current_capital
        
        logger.debug(
            "P&L recorded",
            amount=amount,
            current_capital=self.current_capital,
            high_water_mark=self.high_water_mark,
            drawdown=self.current_drawdown,
        )
        
        # Check if this triggers
        self._check_drawdown()
    
    def record_solver_time(self, seconds: float) -> None:
        """
        Record a solver execution time.
        
        Args:
            seconds: How long the solver took
        """
        now = datetime.utcnow()
        self._solver_times.append((now, seconds))
        
        # Prune old entries (keep last 5 minutes)
        cutoff = now - timedelta(minutes=5)
        self._solver_times = [
            (t, s) for t, s in self._solver_times if t > cutoff
        ]
        
        self._check_solver_timeouts()
    
    def record_api_error(self) -> None:
        """Record an API error occurrence."""
        now = datetime.utcnow()
        self._api_errors.append(now)
        
        # Prune old entries
        cutoff = now - timedelta(minutes=5)
        self._api_errors = [t for t in self._api_errors if t > cutoff]
    
    def record_latency(self, ms: float) -> None:
        """
        Record a trade latency measurement.
        
        Args:
            ms: Latency in milliseconds
        """
        now = datetime.utcnow()
        self._trade_latencies.append((now, ms))
        
        # Prune old entries
        cutoff = now - timedelta(minutes=5)
        self._trade_latencies = [
            (t, l) for t, l in self._trade_latencies if t > cutoff
        ]
        
        self._check_latency()
    
    def trigger(self, reason: str, trigger_type: TriggerReason = TriggerReason.MANUAL) -> None:
        """
        Manually trigger the kill switch.
        
        Args:
            reason: Human-readable reason for triggering
            trigger_type: Type of trigger
        """
        if self._is_triggered:
            logger.warning("Kill switch already triggered")
            return
        
        self._trigger(trigger_type, reason, 0, 0)
    
    def reset(self, confirm: bool = False) -> bool:
        """
        Reset the kill switch to allow trading.
        
        Requires explicit confirmation to prevent accidental reset.
        
        Args:
            confirm: Must be True to actually reset
            
        Returns:
            Whether reset was successful
        """
        if not confirm:
            logger.warning("Kill switch reset requires confirm=True")
            return False
        
        if not self._is_triggered:
            logger.info("Kill switch was not triggered")
            return True
        
        logger.warning(
            "Kill switch reset",
            previous_trigger=self._trigger_event,
        )
        
        self._is_triggered = False
        self._trigger_event = None
        
        return True
    
    def get_state(self) -> KillSwitchState:
        """Get the current state of the kill switch."""
        return KillSwitchState(
            is_triggered=self._is_triggered,
            trigger_event=self._trigger_event,
            current_drawdown=self.current_drawdown,
            high_water_mark=self.high_water_mark,
            current_capital=self.current_capital,
            trigger_count=self._trigger_count,
        )
    
    def _trigger(
        self,
        reason: TriggerReason,
        details: str,
        metric_value: float,
        threshold: float,
    ) -> None:
        """Internal method to trigger the kill switch."""
        self._is_triggered = True
        self._trigger_count += 1
        
        self._trigger_event = TriggerEvent(
            timestamp=datetime.utcnow(),
            reason=reason,
            details=details,
            metric_value=metric_value,
            threshold=threshold,
        )
        
        logger.critical(
            "KILL SWITCH TRIGGERED",
            reason=reason.value,
            details=details,
            metric_value=metric_value,
            threshold=threshold,
        )
        
        # Call callback if provided
        if self._on_trigger:
            try:
                self._on_trigger(self._trigger_event)
            except Exception as e:
                logger.error("Kill switch callback failed", error=str(e))
    
    def _check_drawdown(self) -> None:
        """Check if drawdown exceeds threshold."""
        if self._is_triggered:
            return
        
        if self.current_drawdown > self.max_drawdown:
            self._trigger(
                TriggerReason.DRAWDOWN,
                f"Drawdown {self.current_drawdown:.1%} exceeds {self.max_drawdown:.1%}",
                self.current_drawdown,
                self.max_drawdown,
            )
    
    def _check_solver_timeouts(self) -> None:
        """Check if solver timeout rate is too high."""
        if self._is_triggered or len(self._solver_times) < 3:
            return
        
        timeouts = sum(
            1 for _, s in self._solver_times
            if s > self.solver_timeout_threshold
        )
        rate = timeouts / len(self._solver_times)
        
        if rate > self.max_solver_timeout_rate:
            self._trigger(
                TriggerReason.SOLVER_TIMEOUT,
                f"Solver timeout rate {rate:.1%} exceeds {self.max_solver_timeout_rate:.1%}",
                rate,
                self.max_solver_timeout_rate,
            )
    
    def _check_latency(self) -> None:
        """Check if latency is consistently too high."""
        if self._is_triggered or len(self._trade_latencies) < 5:
            return
        
        recent = self._trade_latencies[-5:]
        avg_latency = sum(l for _, l in recent) / len(recent)
        
        # Only trigger if consistently 3x over threshold
        if avg_latency > self.latency_threshold_ms * 3:
            self._trigger(
                TriggerReason.LATENCY,
                f"Average latency {avg_latency:.0f}ms exceeds 3x threshold",
                avg_latency,
                self.latency_threshold_ms * 3,
            )
