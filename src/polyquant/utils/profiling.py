"""
Performance Profiling and Instrumentation for PolyQuant 2.0

Week 4: Production observability with timing spans and metrics.

This module provides lightweight profiling decorators and context managers
for measuring performance in production. Can be integrated with OpenTelemetry,
Prometheus, or used standalone.

USAGE:
------
    from polyquant.utils.profiling import timed_operation, LatencyTracker

    # Method decorator
    @timed_operation("arbitrage_detection")
    async def detect_opportunities(self, ...):
        # Your code here
        pass

    # Context manager
    async def some_function():
        async with timed_operation("database_query"):
            result = await db.query(...)

    # Manual tracking
    tracker = LatencyTracker()
    tracker.record("tick_latency", 25.5)  # 25.5ms

    # Get statistics
    stats = tracker.get_stats("tick_latency")
    print(f"p95: {stats.p95}ms")
"""

import asyncio
import functools
import time
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Callable, TypeVar, ParamSpec

import numpy as np

from polyquant.utils import get_logger

logger = get_logger(__name__)

P = ParamSpec('P')
R = TypeVar('R')


@dataclass
class LatencyStats:
    """Statistics for a timed operation."""
    operation: str
    count: int
    mean: float
    median: float
    p95: float
    p99: float
    min: float
    max: float
    total: float

    def __str__(self) -> str:
        return (
            f"{self.operation}: "
            f"count={self.count}, "
            f"mean={self.mean:.2f}ms, "
            f"p95={self.p95:.2f}ms, "
            f"p99={self.p99:.2f}ms"
        )


class LatencyTracker:
    """
    Tracks latency measurements for operations.

    Singleton for collecting timing data across the application. Single-loop
    use only — no concurrent-safety primitives.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._measurements = defaultdict(list)
            cls._instance._enabled = True
        return cls._instance

    def enable(self):
        """Enable latency tracking."""
        self._enabled = True
        logger.info("Latency tracking enabled")

    def disable(self):
        """Disable latency tracking (for performance)."""
        self._enabled = False
        logger.info("Latency tracking disabled")

    def record(self, operation: str, latency_ms: float):
        """
        Record a latency measurement.

        Args:
            operation: Name of the operation (e.g., "tick_to_decision")
            latency_ms: Latency in milliseconds
        """
        if not self._enabled:
            return

        self._measurements[operation].append(latency_ms)

        # Log slow operations (>100ms)
        if latency_ms > 100:
            logger.warning(
                f"Slow operation detected: {operation}",
                latency_ms=latency_ms,
            )

    def get_stats(self, operation: str) -> LatencyStats | None:
        """
        Get statistics for an operation.

        Args:
            operation: Name of the operation

        Returns:
            LatencyStats or None if no measurements
        """
        measurements = self._measurements.get(operation, [])
        if not measurements:
            return None

        arr = np.array(measurements)

        return LatencyStats(
            operation=operation,
            count=len(measurements),
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            total=float(np.sum(arr)),
        )

    def get_all_stats(self) -> dict[str, LatencyStats]:
        """
        Get statistics for all tracked operations.

        Returns:
            Dict mapping operation name to LatencyStats
        """
        return {
            operation: self.get_stats(operation)
            for operation in self._measurements.keys()
            if self.get_stats(operation) is not None
        }

    def reset(self):
        """Clear all measurements."""
        self._measurements.clear()
        logger.info("Latency measurements reset")

    def print_summary(self):
        """Print a summary of all tracked operations."""
        stats = self.get_all_stats()

        if not stats:
            logger.info("No latency measurements recorded")
            return

        logger.info(f"\n{'='*80}")
        logger.info("LATENCY SUMMARY")
        logger.info(f"{'='*80}")

        # Sort by p95 latency (slowest first)
        sorted_stats = sorted(stats.values(), key=lambda s: s.p95, reverse=True)

        for stat in sorted_stats:
            logger.info(str(stat))

        logger.info(f"{'='*80}\n")


# Global tracker instance
_tracker = LatencyTracker()


def timed_operation(operation_name: str):
    """
    Decorator to automatically track function/method latency.

    Works with both sync and async functions.

    Args:
        operation_name: Name to use for tracking

    Example:
        @timed_operation("arbitrage_detection")
        async def detect(self, ...):
            # Function code
            pass
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start = time.monotonic()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    _tracker.record(operation_name, elapsed_ms)

            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start = time.monotonic()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    _tracker.record(operation_name, elapsed_ms)

            return sync_wrapper  # type: ignore

    return decorator


@asynccontextmanager
async def async_timed(operation_name: str):
    """
    Async context manager for timing operations.

    Example:
        async with async_timed("database_query"):
            result = await db.query(...)
    """
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000
        _tracker.record(operation_name, elapsed_ms)


@contextmanager
def sync_timed(operation_name: str):
    """
    Sync context manager for timing operations.

    Example:
        with sync_timed("file_write"):
            file.write(data)
    """
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000
        _tracker.record(operation_name, elapsed_ms)


# Convenience functions
def get_tracker() -> LatencyTracker:
    """Get the global latency tracker."""
    return _tracker


def print_latency_report():
    """Print a latency report for all tracked operations."""
    _tracker.print_summary()


def reset_measurements():
    """Reset all latency measurements."""
    _tracker.reset()
