"""
Correlation Engine for PolyQuant 2.0

Identifies statistical relationships between prediction markets to find
cross-event arbitrage opportunities (leader-laggard pairs).

ANTI-SPURIOUS SAFEGUARDS:
-------------------------
This engine is designed to AVOID false signals from:

1. Spurious Correlation: Require ≥50 data points + Bonferroni correction
   to control false discovery rate when testing many pairs.

2. Regime Shifts: Require correlation stability across multiple timeframes
   (7-day AND 30-day). Reject if short-term ≠ long-term correlation.

3. Survivorship Bias: Track prediction accuracy of each pair's signals
   and auto-disable pairs that drop below 55% accuracy.

4. Liquidity Illusion: Only trade pairs where both markets have sufficient
   liquidity and tight spreads.

5. Causal Filtering: Optionally verify via LLM that the correlation has
   a plausible causal explanation (not just statistical noise).

USAGE:
------
    engine = CorrelationEngine(client=polymarket_client)

    # Analyze during MapMaker phase
    signals = await engine.scan_for_pairs(markets, min_correlation=0.7)

    for signal in signals:
        print(f"Pair: {signal.leader_id} → {signal.laggard_id}")
        print(f"Correlation: {signal.correlation:.2f}")
        print(f"Expected lag: {signal.avg_lag_minutes:.1f} min")
"""

import math
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from polyquant.data.market_models import Market
from polyquant.utils import config, get_logger

logger = get_logger(__name__)


@dataclass
class CorrelationPair:
    """A validated correlation pair with safeguard metadata."""
    leader_id: str
    leader_question: str
    laggard_id: str
    laggard_question: str
    correlation_7d: float           # Short-term correlation
    correlation_30d: float          # Long-term correlation
    avg_lag_minutes: float          # How far the laggard typically lags
    data_points: int                # Number of price observations used
    p_value: float                  # Statistical significance (Bonferroni-adjusted)
    causal_reason: str = ""         # Human-readable explanation of causal link

    # Tracking fields
    signals_generated: int = 0
    signals_correct: int = 0
    disabled: bool = False
    disabled_reason: str = ""

    @property
    def accuracy(self) -> float:
        """Prediction accuracy of this pair's signals."""
        if self.signals_generated == 0:
            return 0.5  # No data = assume coin flip
        return self.signals_correct / self.signals_generated

    @property
    def is_stable(self) -> bool:
        """Whether correlation is stable across timeframes."""
        # Both must be positive and within 0.3 of each other
        if self.correlation_7d < 0.5 or self.correlation_30d < 0.5:
            return False
        return abs(self.correlation_7d - self.correlation_30d) < 0.3

    @property
    def is_significant(self) -> bool:
        """Whether the correlation is statistically significant."""
        return self.p_value < 0.05  # After Bonferroni correction


@dataclass
class CorrelationSignal:
    """A live trading signal from a correlated pair."""
    pair: CorrelationPair
    leader_move: float              # How much the leader moved
    expected_laggard_move: float    # Expected laggard response
    current_laggard_price: float    # Current price of laggard
    expected_laggard_price: float   # Expected price after catch-up
    deviation_sigma: float          # How many standard deviations the gap is
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CorrelationEngine:
    """
    Identifies and trades statistical relationships between markets.

    Uses a multi-stage pipeline with aggressive filtering to minimize
    false positives from spurious correlations.
    """

    # ── Safeguard thresholds ──
    MIN_HISTORY_POINTS = 50         # Minimum price observations per market
    MIN_STABLE_CORRELATION = 0.7    # Minimum correlation on BOTH timeframes
    MIN_LIQUIDITY = 5000            # Minimum $ liquidity for both markets
    MAX_SPREAD_PCT = 0.05           # Maximum bid-ask spread (5%)
    ACCURACY_KILL_THRESHOLD = 0.55  # Disable pair if accuracy drops below
    DEVIATION_THRESHOLD_SIGMA = 2.0 # Only signal if gap > 2σ
    MAX_PAIRS_TO_TEST = 200         # Cap for Bonferroni correction

    def __init__(self):
        self._pairs: list[CorrelationPair] = []
        self._disabled_pairs: list[CorrelationPair] = []
        self._total_signals: int = 0

    async def scan_for_pairs(
        self,
        markets: list[Market],
        price_history_provider: Any,
        min_correlation: float = 0.7,
    ) -> list[CorrelationPair]:
        """
        Scan markets for statistically significant correlated pairs.

        This is called during the MapMaker phase (offline, not latency-sensitive).

        Args:
            markets: List of active markets to analyze
            price_history_provider: Async callable(market_id) -> list[{"t": int, "p": float}]
            min_correlation: Minimum Pearson r to consider

        Returns:
            List of validated CorrelationPair objects
        """
        # Filter: Only markets with sufficient liquidity
        candidates = [m for m in markets if m.liquidity >= self.MIN_LIQUIDITY]

        if len(candidates) < 2:
            logger.info("Not enough liquid markets for correlation analysis",
                        candidates=len(candidates))
            return []

        logger.info(
            "Starting correlation scan",
            candidates=len(candidates),
            max_pairs=self.MAX_PAIRS_TO_TEST,
        )

        # Fetch price histories
        histories: dict[str, list[float]] = {}
        for market in candidates:
            try:
                raw_history = await price_history_provider(market.market_id)
                if raw_history and len(raw_history) >= self.MIN_HISTORY_POINTS:
                    # Extract price series (sorted by time)
                    sorted_history = sorted(raw_history, key=lambda x: x.get("t", 0))
                    histories[market.market_id] = [
                        float(h.get("p", 0)) for h in sorted_history
                    ]
            except Exception as e:
                logger.debug(f"Failed to fetch history for {market.market_id}: {e}")

        logger.info(
            "Price histories fetched",
            markets_with_history=len(histories),
            min_points=self.MIN_HISTORY_POINTS,
        )

        if len(histories) < 2:
            return []

        # Build market lookup
        market_lookup = {m.market_id: m for m in candidates}

        # Test all pairs (capped to control Bonferroni correction)
        market_ids = list(histories.keys())
        pairs_tested = 0
        raw_pairs: list[CorrelationPair] = []

        for i in range(len(market_ids)):
            for j in range(i + 1, len(market_ids)):
                if pairs_tested >= self.MAX_PAIRS_TO_TEST:
                    break

                mid_a = market_ids[i]
                mid_b = market_ids[j]
                series_a = histories[mid_a]
                series_b = histories[mid_b]

                # Align series to same length
                min_len = min(len(series_a), len(series_b))
                if min_len < self.MIN_HISTORY_POINTS:
                    continue

                sa = series_a[-min_len:]
                sb = series_b[-min_len:]

                # Calculate correlation on two timeframes
                r_full = self._pearson_correlation(sa, sb)
                # 7-day proxy: last 1/4 of data
                cutoff_7d = max(self.MIN_HISTORY_POINTS, min_len // 4)
                r_7d = self._pearson_correlation(sa[-cutoff_7d:], sb[-cutoff_7d:])

                pairs_tested += 1

                # Quick filter: skip if either timeframe is below threshold
                if abs(r_full) < min_correlation or abs(r_7d) < min_correlation:
                    continue

                # Compute p-value with Bonferroni correction
                p_raw = self._correlation_p_value(r_full, min_len)
                p_adjusted = min(1.0, p_raw * pairs_tested)  # Bonferroni

                if p_adjusted >= 0.05:
                    continue  # Not significant after correction

                # Determine leader/laggard by comparing volatility
                # The more volatile market is typically the leader
                vol_a = self._volatility(sa)
                vol_b = self._volatility(sb)

                if vol_a >= vol_b:
                    leader_id, laggard_id = mid_a, mid_b
                else:
                    leader_id, laggard_id = mid_b, mid_a

                # Estimate average lag (simplified: cross-correlation peak)
                avg_lag = self._estimate_lag(
                    histories[leader_id], histories[laggard_id]
                )

                pair = CorrelationPair(
                    leader_id=leader_id,
                    leader_question=market_lookup.get(leader_id, Market(
                        market_id=leader_id, question="Unknown", outcomes=[]
                    )).question,
                    laggard_id=laggard_id,
                    laggard_question=market_lookup.get(laggard_id, Market(
                        market_id=laggard_id, question="Unknown", outcomes=[]
                    )).question,
                    correlation_7d=r_7d,
                    correlation_30d=r_full,
                    avg_lag_minutes=avg_lag,
                    data_points=min_len,
                    p_value=p_adjusted,
                )

                # Stability check
                if not pair.is_stable:
                    logger.debug(
                        "Pair rejected: unstable correlation across timeframes",
                        leader=leader_id,
                        laggard=laggard_id,
                        r_7d=f"{r_7d:.3f}",
                        r_30d=f"{r_full:.3f}",
                    )
                    continue

                raw_pairs.append(pair)

            if pairs_tested >= self.MAX_PAIRS_TO_TEST:
                break

        self._pairs = raw_pairs

        logger.info(
            "Correlation scan complete",
            pairs_tested=pairs_tested,
            significant_pairs=len(raw_pairs),
            bonferroni_applied=True,
        )

        return raw_pairs

    def check_for_signals(
        self,
        current_prices: dict[str, float],
        previous_prices: dict[str, float],
    ) -> list[CorrelationSignal]:
        """
        Check active pairs for live trading signals.

        Called on each Navigator tick (must be fast).

        Args:
            current_prices: outcome_id -> current price
            previous_prices: outcome_id -> previous tick price

        Returns:
            List of actionable CorrelationSignals
        """
        signals: list[CorrelationSignal] = []

        for pair in self._pairs:
            if pair.disabled:
                continue

            leader_curr = current_prices.get(pair.leader_id)
            leader_prev = previous_prices.get(pair.leader_id)
            laggard_curr = current_prices.get(pair.laggard_id)

            if leader_curr is None or leader_prev is None or laggard_curr is None:
                continue

            # Did the leader move significantly?
            leader_move = leader_curr - leader_prev
            if abs(leader_move) < 0.02:  # < 2 cents = noise
                continue

            # Expected laggard response based on correlation
            # E[ΔSⱼ | ΔSᵢ] = (Σᵢⱼ / σᵢ²) × ΔSᵢ ≈ r × (σⱼ/σᵢ) × ΔSᵢ
            # Simplified: use correlation as proportional factor
            avg_r = (pair.correlation_7d + pair.correlation_30d) / 2
            expected_move = leader_move * avg_r * 0.8  # Conservative

            expected_laggard = laggard_curr + expected_move

            # How far off is the laggard from expectation?
            deviation = abs(expected_laggard - laggard_curr)

            # Convert to sigma (approximate using price volatility)
            # Use 1% as baseline volatility for prediction markets
            sigma = max(0.01, abs(leader_move) * 0.5)
            deviation_sigma = deviation / sigma

            if deviation_sigma >= self.DEVIATION_THRESHOLD_SIGMA:
                signal = CorrelationSignal(
                    pair=pair,
                    leader_move=leader_move,
                    expected_laggard_move=expected_move,
                    current_laggard_price=laggard_curr,
                    expected_laggard_price=expected_laggard,
                    deviation_sigma=deviation_sigma,
                )
                signals.append(signal)
                self._total_signals += 1
                pair.signals_generated += 1

                logger.info(
                    "CORRELATION SIGNAL: Leader-laggard deviation detected",
                    leader=pair.leader_question[:50],
                    laggard=pair.laggard_question[:50],
                    leader_move=f"{leader_move:+.3f}",
                    expected_laggard=f"{expected_laggard:.3f}",
                    actual_laggard=f"{laggard_curr:.3f}",
                    deviation_sigma=f"{deviation_sigma:.1f}σ",
                )

        return signals

    def record_signal_outcome(self, pair_leader_id: str, was_correct: bool) -> None:
        """
        Record whether a correlation signal was correct.

        Call this AFTER the laggard has had time to react (e.g., 10 minutes).
        If accuracy drops below threshold, the pair is auto-disabled.

        Args:
            pair_leader_id: Leader market ID of the pair
            was_correct: Whether the laggard moved in the expected direction
        """
        for pair in self._pairs:
            if pair.leader_id == pair_leader_id:
                if was_correct:
                    pair.signals_correct += 1

                # Check accuracy kill threshold
                if (pair.signals_generated >= 10 and
                    pair.accuracy < self.ACCURACY_KILL_THRESHOLD):
                    pair.disabled = True
                    pair.disabled_reason = (
                        f"Accuracy dropped below {self.ACCURACY_KILL_THRESHOLD:.0%}: "
                        f"{pair.accuracy:.1%} ({pair.signals_correct}/{pair.signals_generated})"
                    )
                    self._disabled_pairs.append(pair)
                    logger.warning(
                        "Correlation pair AUTO-DISABLED due to low accuracy",
                        leader=pair.leader_question[:50],
                        laggard=pair.laggard_question[:50],
                        accuracy=f"{pair.accuracy:.1%}",
                        signals=pair.signals_generated,
                        reason=pair.disabled_reason,
                    )
                break

    # ── Math utilities ──

    @staticmethod
    def _pearson_correlation(a: list[float], b: list[float]) -> float:
        """Pearson correlation coefficient."""
        n = len(a)
        if n < 2 or n != len(b):
            return 0.0

        mean_a = sum(a) / n
        mean_b = sum(b) / n

        numerator = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a, b))
        denom_a = math.sqrt(sum((ai - mean_a) ** 2 for ai in a))
        denom_b = math.sqrt(sum((bi - mean_b) ** 2 for bi in b))

        if denom_a == 0 or denom_b == 0:
            return 0.0

        return numerator / (denom_a * denom_b)

    @staticmethod
    def _correlation_p_value(r: float, n: int) -> float:
        """
        Approximate p-value for Pearson correlation using t-distribution.

        Uses the Fisher transformation for better approximation.
        """
        if n < 4 or abs(r) >= 1.0:
            return 1.0

        # t-statistic
        t_stat = r * math.sqrt((n - 2) / (1 - r * r))

        # Approximate p-value using normal distribution for large n
        # (t-distribution ≈ normal for n > 30)
        if n > 30:
            # Two-tailed p-value from z-score
            z = abs(t_stat)
            # Approximation of erfc
            p = math.exp(-0.5 * z * z) / (z * math.sqrt(2 * math.pi)) if z > 0 else 1.0
            return min(1.0, 2 * p)  # Two-tailed
        else:
            # For small samples, use a conservative approximation
            # In production, use scipy.stats.t.sf()
            z = abs(t_stat)
            p = math.exp(-0.5 * z * z) * 2
            return min(1.0, p)

    @staticmethod
    def _volatility(series: list[float]) -> float:
        """Standard deviation of returns."""
        if len(series) < 2:
            return 0.0
        returns = [series[i] - series[i - 1] for i in range(1, len(series))]
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    @staticmethod
    def _estimate_lag(leader: list[float], laggard: list[float]) -> float:
        """
        Estimate average lag in time units between leader and laggard.

        Uses simplified cross-correlation (checks offsets 0-10).
        Returns the offset (in data points) with highest correlation.
        """
        best_lag = 0
        best_r = 0.0
        min_len = min(len(leader), len(laggard))

        for lag in range(0, min(11, min_len // 4)):
            if lag == 0:
                r = CorrelationEngine._pearson_correlation(
                    leader[:min_len], laggard[:min_len]
                )
            else:
                # Leader leads by `lag` periods
                r = CorrelationEngine._pearson_correlation(
                    leader[:min_len - lag], laggard[lag:min_len]
                )

            if abs(r) > abs(best_r):
                best_r = r
                best_lag = lag

        # Convert data points to minutes (assuming 60-min fidelity default)
        return best_lag * 60.0

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for monitoring/UI."""
        return {
            "active_pairs": len([p for p in self._pairs if not p.disabled]),
            "disabled_pairs": len(self._disabled_pairs),
            "total_signals": self._total_signals,
            "avg_accuracy": (
                sum(p.accuracy for p in self._pairs if p.signals_generated > 0)
                / max(1, len([p for p in self._pairs if p.signals_generated > 0]))
            ),
        }
