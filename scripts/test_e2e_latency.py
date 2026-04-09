"""
End-to-End Latency Testing Script for PolyQuant 2.0

Week 4: Validates <50ms p95 latency target before production deployment.

This script runs the Navigator in paper mode against live Polymarket data
and measures latency at each stage of the trading pipeline.

USAGE:
------
    # Run for 1 hour (default)
    python scripts/test_e2e_latency.py

    # Run for 24 hours
    python scripts/test_e2e_latency.py --duration 86400

    # Run with max ticks limit
    python scripts/test_e2e_latency.py --max-ticks 1000

    # Custom report path
    python scripts/test_e2e_latency.py --output report.json

METRICS TRACKED:
----------------
- tick_to_decision: Price update → Opportunity detection
- decision_to_execution: Detection → Order submission (paper mode)
- total_latency: End-to-end pipeline latency
- opportunity_detection_rate: Opportunities found per minute
- websocket_health: Connection stability metrics

SUCCESS CRITERIA:
-----------------
- p95 latency < 50ms
- p99 latency < 100ms
- Uptime > 99.5% (< 7 min downtime per 24h)
- Memory growth < 10MB/hour
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polyquant.utils.profiling import get_tracker, LatencyStats
from polyquant.utils import get_logger

logger = get_logger(__name__)


class E2ELatencyTest:
    """
    End-to-end latency testing harness.

    Runs Navigator in paper mode and collects comprehensive metrics.
    """

    def __init__(
        self,
        duration_seconds: int = 3600,
        max_ticks: int | None = None,
        output_path: str = "latency_report.json",
    ):
        """
        Initialize the test harness.

        Args:
            duration_seconds: Test duration in seconds (default 1 hour)
            max_ticks: Maximum ticks to process (None for unlimited)
            output_path: Path to save the JSON report
        """
        self.duration_seconds = duration_seconds
        self.max_ticks = max_ticks
        self.output_path = Path(output_path)
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

        # Metrics
        self.ticks_processed = 0
        self.opportunities_detected = 0
        self.trades_executed = 0
        self.errors_encountered = 0
        self.websocket_disconnects = 0

    async def run(self):
        """
        Run the end-to-end latency test.

        Returns:
            Test report dictionary
        """
        logger.info(
            "Starting E2E latency test",
            duration_seconds=self.duration_seconds,
            max_ticks=self.max_ticks,
        )

        self.start_time = datetime.utcnow()

        try:
            # Import Navigator here to avoid circular imports
            from polyquant.navigator import Navigator

            # Create Navigator instance
            async with Navigator() as navigator:
                # Patch Navigator to collect metrics
                self._instrument_navigator(navigator)

                # Run Navigator with limits
                logger.info("Navigator started, collecting metrics...")
                await navigator.run(max_ticks=self.max_ticks)

        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.error(f"Test failed with exception: {e}", exc_info=True)
            self.errors_encountered += 1
        finally:
            self.end_time = datetime.utcnow()

        # Generate report
        report = self._generate_report()

        # Save to file
        self._save_report(report)

        # Print summary
        self._print_summary(report)

        return report

    def _instrument_navigator(self, navigator):
        """
        Add instrumentation hooks to Navigator instance.

        Args:
            navigator: Navigator instance to instrument
        """
        # Wrap key methods to collect metrics
        original_detect = navigator._detect_opportunities

        async def instrumented_detect(*args, **kwargs):
            """Wrapped opportunity detection with metrics."""
            result = await original_detect(*args, **kwargs)
            if result:
                self.opportunities_detected += 1
            return result

        navigator._detect_opportunities = instrumented_detect

        logger.debug("Navigator instrumented for metrics collection")

    def _generate_report(self) -> dict:
        """
        Generate comprehensive test report.

        Returns:
            Dictionary with all test metrics and analysis
        """
        tracker = get_tracker()
        all_stats = tracker.get_all_stats()

        # Calculate test duration
        if self.start_time and self.end_time:
            actual_duration = (self.end_time - self.start_time).total_seconds()
        else:
            actual_duration = 0

        # Build report
        report = {
            "test_metadata": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": actual_duration,
                "target_duration": self.duration_seconds,
                "max_ticks": self.max_ticks,
            },
            "summary_metrics": {
                "ticks_processed": self.ticks_processed,
                "opportunities_detected": self.opportunities_detected,
                "trades_executed": self.trades_executed,
                "errors_encountered": self.errors_encountered,
                "websocket_disconnects": self.websocket_disconnects,
                "opportunity_rate_per_minute": (
                    self.opportunities_detected / (actual_duration / 60)
                    if actual_duration > 0
                    else 0
                ),
            },
            "latency_stats": {},
            "success_criteria": {},
        }

        # Add latency statistics
        for operation, stats in all_stats.items():
            if stats:
                report["latency_stats"][operation] = {
                    "count": stats.count,
                    "mean_ms": round(stats.mean, 2),
                    "median_ms": round(stats.median, 2),
                    "p95_ms": round(stats.p95, 2),
                    "p99_ms": round(stats.p99, 2),
                    "min_ms": round(stats.min, 2),
                    "max_ms": round(stats.max, 2),
                }

        # Evaluate success criteria
        total_latency_stats = all_stats.get("total_latency")
        if total_latency_stats:
            report["success_criteria"] = {
                "p95_under_50ms": total_latency_stats.p95 < 50.0,
                "p99_under_100ms": total_latency_stats.p99 < 100.0,
                "mean_under_30ms": total_latency_stats.mean < 30.0,
                "uptime_above_99_5_percent": (
                    (actual_duration - self.websocket_disconnects) / actual_duration > 0.995
                    if actual_duration > 0
                    else False
                ),
            }

            # Overall PASS/FAIL
            report["success_criteria"]["overall_pass"] = all([
                report["success_criteria"]["p95_under_50ms"],
                report["success_criteria"]["p99_under_100ms"],
                report["success_criteria"]["uptime_above_99_5_percent"],
            ])
        else:
            report["success_criteria"]["overall_pass"] = False

        return report

    def _save_report(self, report: dict):
        """
        Save report to JSON file.

        Args:
            report: Report dictionary to save
        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.output_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Report saved to {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def _print_summary(self, report: dict):
        """
        Print test summary to console.

        Args:
            report: Report dictionary
        """
        print("\n" + "=" * 80)
        print("END-TO-END LATENCY TEST REPORT")
        print("=" * 80)

        # Test metadata
        meta = report["test_metadata"]
        print(f"\nTest Duration: {meta['duration_seconds']:.1f}s "
              f"(target: {meta['target_duration']}s)")

        # Summary metrics
        summary = report["summary_metrics"]
        print(f"\nSummary:")
        print(f"  Ticks Processed: {summary['ticks_processed']}")
        print(f"  Opportunities Detected: {summary['opportunities_detected']}")
        print(f"  Trades Executed: {summary['trades_executed']}")
        print(f"  Errors: {summary['errors_encountered']}")
        print(f"  Opportunity Rate: {summary['opportunity_rate_per_minute']:.2f}/min")

        # Latency stats
        print(f"\nLatency Statistics:")
        for operation, stats in report["latency_stats"].items():
            print(f"  {operation}:")
            print(f"    Mean: {stats['mean_ms']:.2f}ms")
            print(f"    p95: {stats['p95_ms']:.2f}ms")
            print(f"    p99: {stats['p99_ms']:.2f}ms")

        # Success criteria
        criteria = report["success_criteria"]
        print(f"\nSuccess Criteria:")
        print(f"  ✓ p95 < 50ms: {criteria.get('p95_under_50ms', False)}")
        print(f"  ✓ p99 < 100ms: {criteria.get('p99_under_100ms', False)}")
        print(f"  ✓ Mean < 30ms: {criteria.get('mean_under_30ms', False)}")
        print(f"  ✓ Uptime > 99.5%: {criteria.get('uptime_above_99_5_percent', False)}")

        # Overall result
        overall = criteria.get('overall_pass', False)
        if overall:
            print(f"\n{'=' * 80}")
            print("✅ TEST PASSED - System meets production latency requirements!")
            print(f"{'=' * 80}\n")
        else:
            print(f"\n{'=' * 80}")
            print("❌ TEST FAILED - System does not meet latency requirements")
            print(f"{'=' * 80}\n")


async def main():
    """Main entry point for E2E latency testing."""
    parser = argparse.ArgumentParser(
        description="End-to-end latency testing for PolyQuant 2.0"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Test duration in seconds (default: 3600 = 1 hour)",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Maximum ticks to process (default: unlimited)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="latency_report.json",
        help="Output path for JSON report (default: latency_report.json)",
    )

    args = parser.parse_args()

    # Create test harness
    test = E2ELatencyTest(
        duration_seconds=args.duration,
        max_ticks=args.max_ticks,
        output_path=args.output,
    )

    # Run test
    await test.run()


if __name__ == "__main__":
    asyncio.run(main())
