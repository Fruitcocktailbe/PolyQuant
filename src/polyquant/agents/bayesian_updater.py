"""
Bayesian Conditional Updater for PolyQuant 2.0

When Market A moves significantly, correlated markets (B, C, ...)
should adjust their fair values. Without this, the Solver may
detect "phantom arbitrage" — mispricing that only exists because
the correlated market hasn't reacted yet in the order book.

HOW IT WORKS:
-------------
1. The LogicArchitect extracts dependency relationships during MapMaker:
   - IMPLIES: If A=YES then B=YES  → P(B) >= P(A)
   - EXCLUDES: If A=YES then B=NO  → P(A) + P(B) <= 1
   - PARTITION: Sum of all outcomes = 1

2. When the Navigator sees a significant price move in Market A,
   the Bayesian Updater recalculates the expected fair values for
   all linked markets.

3. The adjusted prices (theta_adjusted) replace the raw order book
   prices before being passed to the Frank-Wolfe solver.

WHY THIS MATTERS:
-----------------
Example: "Trump wins" at $0.60, "Republican wins" at $0.55.
Constraint: IMPLIES (Trump → Republican), so P(Rep) >= P(Trump).
Solver sees a violation and reports arb. BUT if "Trump wins" just
dropped from $0.65 → $0.60 in the last 500ms, "Republican wins"
likely hasn't reacted yet. The Bayesian fair value of "Republican"
should be ~$0.58, which doesn't violate the constraint. No real arb.

USAGE:
------
    updater = BayesianUpdater()
    updater.load_dependencies(manifests)

    # On each tick:
    adjusted = updater.adjust_prices(current_prices, previous_prices)
    # Pass adjusted prices to solver instead of raw prices
"""

import logging
from typing import Any

from polyquant.data.constraint_store import ConstraintManifest, StoredDependency
from polyquant.utils import get_logger

logger = get_logger(__name__)


class PriceAdjustment:
    """Result of a Bayesian price adjustment."""
    __slots__ = ("outcome_id", "raw_price", "adjusted_price", "reason")

    def __init__(self, outcome_id: str, raw_price: float, adjusted_price: float, reason: str):
        self.outcome_id = outcome_id
        self.raw_price = raw_price
        self.adjusted_price = adjusted_price
        self.reason = reason


class BayesianUpdater:
    """
    Adjusts order book prices for correlated markets to eliminate
    phantom arbitrage caused by price lag.

    The updater maintains a dependency graph extracted from the
    ConstraintManifest. On each tick, it checks if any market has
    moved significantly. If so, it adjusts linked markets' fair
    values before passing them to the Solver.
    """

    # Minimum price move (absolute) to trigger Bayesian adjustment.
    # Below this, the move is noise and adjustment would add instability.
    MOVE_THRESHOLD = 0.03  # 3 cents / 3 percentage points

    # Maximum adjustment factor to prevent over-correction.
    # An adjustment > 20% of original price is likely a model error.
    MAX_ADJUSTMENT_PCT = 0.20

    def __init__(self):
        # Dependency graph: outcome_id -> list of (linked_outcome_id, relationship, strength)
        self._dep_graph: dict[str, list[tuple[str, str, float]]] = {}
        # Previous tick's prices for delta calculation
        self._previous_prices: dict[str, float] = {}
        # Track adjustment count for monitoring
        self._adjustment_count: int = 0
        self._phantom_prevented_count: int = 0

    def load_dependencies(self, manifests: list[ConstraintManifest]) -> None:
        """
        Build the dependency graph from constraint manifests.

        Args:
            manifests: List of validated ConstraintManifests from the store.
        """
        self._dep_graph.clear()

        for manifest in manifests:
            for dep in manifest.dependencies:
                # Build bidirectional links
                source_key = dep.source_outcome
                target_key = dep.target_outcome
                confidence = dep.confidence

                if source_key not in self._dep_graph:
                    self._dep_graph[source_key] = []
                if target_key not in self._dep_graph:
                    self._dep_graph[target_key] = []

                # Source → Target link
                self._dep_graph[source_key].append(
                    (target_key, dep.relationship, confidence)
                )

                # Reverse link (with inverted relationship for reasoning)
                reverse_rel = self._invert_relationship(dep.relationship)
                self._dep_graph[target_key].append(
                    (source_key, reverse_rel, confidence)
                )

        total_links = sum(len(v) for v in self._dep_graph.values())
        logger.info(
            "Bayesian dependency graph loaded",
            outcomes_tracked=len(self._dep_graph),
            total_links=total_links,
            manifests=len(manifests),
        )

    def adjust_prices(
        self,
        current_prices: dict[str, float],
    ) -> tuple[dict[str, float], list[PriceAdjustment]]:
        """
        Adjust prices for correlated markets based on recent moves.

        This is called on every tick BEFORE the Solver runs.

        Args:
            current_prices: Dict of outcome_id -> current order book mid-price.

        Returns:
            Tuple of:
                - adjusted_prices: Dict of outcome_id -> adjusted price
                - adjustments: List of PriceAdjustment objects for logging/UI
        """
        adjusted = dict(current_prices)
        adjustments: list[PriceAdjustment] = []

        if not self._previous_prices:
            # First tick — no deltas to compute. Store and pass through.
            self._previous_prices = dict(current_prices)
            return adjusted, adjustments

        # Find outcomes that moved significantly
        movers: list[tuple[str, float]] = []
        for outcome_id, price in current_prices.items():
            prev = self._previous_prices.get(outcome_id)
            if prev is not None:
                delta = price - prev
                if abs(delta) >= self.MOVE_THRESHOLD:
                    movers.append((outcome_id, delta))

        # For each significant mover, adjust linked outcomes
        for mover_id, delta in movers:
            linked = self._dep_graph.get(mover_id, [])

            for linked_id, relationship, confidence in linked:
                if linked_id not in current_prices:
                    continue  # Not in current tick's data

                raw_price = current_prices[linked_id]
                adj = self._calculate_adjustment(
                    mover_id=mover_id,
                    mover_delta=delta,
                    mover_price=current_prices[mover_id],
                    linked_id=linked_id,
                    linked_price=raw_price,
                    relationship=relationship,
                    confidence=confidence,
                )

                if adj is not None:
                    # Apply adjustment
                    adjusted[linked_id] = adj.adjusted_price
                    adjustments.append(adj)
                    self._adjustment_count += 1

        # Update previous prices for next tick
        self._previous_prices = dict(current_prices)

        if adjustments:
            logger.info(
                "Bayesian price adjustments applied",
                movers=len(movers),
                adjustments=len(adjustments),
                total_adjustments=self._adjustment_count,
                phantoms_prevented=self._phantom_prevented_count,
            )

        return adjusted, adjustments

    def _calculate_adjustment(
        self,
        mover_id: str,
        mover_delta: float,
        mover_price: float,
        linked_id: str,
        linked_price: float,
        relationship: str,
        confidence: float,
    ) -> PriceAdjustment | None:
        """
        Calculate the Bayesian adjustment for a linked outcome.

        The adjustment depends on the relationship type:
        - IMPLIES: If A drops, B should drop proportionally
        - EXCLUDES: If A drops, B should rise proportionally
        - PARTITION: Adjustment is distributed across all linked outcomes

        Returns:
            PriceAdjustment or None if no adjustment needed.
        """
        # Scale adjustment by confidence (0-1)
        # Low confidence dependencies get weaker adjustments
        adjustment_strength = confidence * 0.5  # Conservative: max 50% pass-through

        if relationship == "implies":
            # A implies B: A goes down → B should go down
            # A goes up → B should go up (but B >= A, so only if B < A after move)
            expected_delta = mover_delta * adjustment_strength
            new_price = linked_price + expected_delta

        elif relationship == "excludes":
            # A excludes B: A goes up → B should go down
            expected_delta = -mover_delta * adjustment_strength
            new_price = linked_price + expected_delta

        elif relationship == "partition":
            # Partition: sum must be ~1. If A goes up by Δ, the rest must
            # go down by ~Δ/(N-1). But we don't know N here, so use a
            # conservative proportional adjustment.
            expected_delta = -mover_delta * adjustment_strength * 0.5
            new_price = linked_price + expected_delta

        elif relationship == "reverse_implies":
            # B implies A (reverse): A drops → B might drop (weaker signal)
            expected_delta = mover_delta * adjustment_strength * 0.3
            new_price = linked_price + expected_delta

        else:
            return None

        # Safety: clamp to valid probability range
        new_price = max(0.01, min(0.99, new_price))

        # Safety: cap maximum adjustment
        max_adj = linked_price * self.MAX_ADJUSTMENT_PCT
        if abs(new_price - linked_price) > max_adj:
            direction = 1 if new_price > linked_price else -1
            new_price = linked_price + direction * max_adj

        # Only report if adjustment is meaningful (> 0.5 cents)
        if abs(new_price - linked_price) < 0.005:
            return None

        # Check if this adjustment would have prevented a phantom opportunity
        if relationship == "implies" and mover_delta < 0:
            # Mover dropped, linked hasn't moved yet → classic phantom
            if linked_price > mover_price:  # Constraint would appear violated
                self._phantom_prevented_count += 1
                reason = (
                    f"Phantom prevention: {mover_id} dropped {mover_delta:+.3f}, "
                    f"{linked_id} hasn't reacted. Adjusting from "
                    f"{linked_price:.3f} → {new_price:.3f}"
                )
                logger.debug(reason)
            else:
                reason = (
                    f"Bayesian adjustment ({relationship}): "
                    f"{mover_id} Δ={mover_delta:+.3f} → "
                    f"{linked_id} {linked_price:.3f} → {new_price:.3f}"
                )
        else:
            reason = (
                f"Bayesian adjustment ({relationship}): "
                f"{mover_id} Δ={mover_delta:+.3f} → "
                f"{linked_id} {linked_price:.3f} → {new_price:.3f}"
            )

        return PriceAdjustment(
            outcome_id=linked_id,
            raw_price=linked_price,
            adjusted_price=new_price,
            reason=reason,
        )

    @staticmethod
    def _invert_relationship(relationship: str) -> str:
        """Invert a dependency relationship for reverse links."""
        inversions = {
            "implies": "reverse_implies",
            "excludes": "excludes",       # Mutual exclusion is symmetric
            "partition": "partition",      # Partition is symmetric
        }
        return inversions.get(relationship, relationship)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for monitoring/UI."""
        return {
            "outcomes_tracked": len(self._dep_graph),
            "total_links": sum(len(v) for v in self._dep_graph.values()),
            "adjustments_made": self._adjustment_count,
            "phantoms_prevented": self._phantom_prevented_count,
        }
