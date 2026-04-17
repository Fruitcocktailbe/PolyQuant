"""Unit tests for the deterministic ladder + conditional-subset detectors
introduced in the map-maker audit PR.

These modules don't depend on any LLM or network so they can (and should)
run fully offline — the solver relies on their output to never silently
miss classic Polymarket arbitrage structures.
"""

from __future__ import annotations

from decimal import Decimal

from polyquant.agents.ladder_detector import (
    build_conditional_constraints,
    build_ladder_constraints,
    detect_conditional_subsets,
    detect_monotonic_ladders,
)
from polyquant.data.market_models import Market, Outcome


def _binary_market(
    market_id: str,
    question: str,
    *,
    yes_token: str | None = None,
    no_token: str | None = None,
    event_slug: str = "evt",
    liquidity: float = 10_000.0,
    conditional_parent_id: str | None = None,
) -> Market:
    return Market(
        market_id=market_id,
        question=question,
        description="",
        outcomes=[
            Outcome(
                outcome_id=f"{market_id}_yes",
                name="Yes",
                price=Decimal("0.5"),
                token_id=yes_token or f"{market_id}_yes_tok",
            ),
            Outcome(
                outcome_id=f"{market_id}_no",
                name="No",
                price=Decimal("0.5"),
                token_id=no_token or f"{market_id}_no_tok",
            ),
        ],
        event_slug=event_slug,
        liquidity=liquidity,
        conditional_parent_id=conditional_parent_id,
    )


# --------------------------------------------------------- monotonic ladder


def test_ladder_detector_groups_matching_threshold_template():
    markets = [
        _binary_market("m1", "Will BTC reach $100,000 by EOY 2026?"),
        _binary_market("m2", "Will BTC reach $120,000 by EOY 2026?"),
        _binary_market("m3", "Will BTC reach $150,000 by EOY 2026?"),
        # Different event/year shouldn't fold in
        _binary_market("m4", "Will BTC reach $100,000 by EOY 2027?", event_slug="evt2"),
    ]
    clusters = detect_monotonic_ladders(markets)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.constraint_source == "monotonic_ladder"
    assert {m.market_id for m in cluster.markets} == {"m1", "m2", "m3"}
    # Rungs sorted ascending by threshold
    thresholds = [m.question for m in cluster.markets]
    assert "100,000" in thresholds[0]
    assert "120,000" in thresholds[1]
    assert "150,000" in thresholds[2]


def test_ladder_detector_skips_ladders_below_min_rungs():
    markets = [
        _binary_market("m1", "Will BTC reach $100k?"),
        _binary_market("m2", "Will BTC reach $120k?"),
    ]
    assert detect_monotonic_ladders(markets) == []


def test_build_ladder_constraints_emits_chained_subset():
    markets = [
        _binary_market("m1", "Will BTC reach $100,000?", yes_token="tok_100"),
        _binary_market("m2", "Will BTC reach $120,000?", yes_token="tok_120"),
        _binary_market("m3", "Will BTC reach $150,000?", yes_token="tok_150"),
    ]
    cluster = detect_monotonic_ladders(markets)[0]
    constraints = build_ladder_constraints(cluster)
    # Two adjacent pairs → two constraints
    assert len(constraints) == 2
    # Each constraint must be `low_yes - high_yes >= 0`
    low_high_seen = []
    for cons in constraints:
        assert cons.rhs == 0.0
        items = sorted(cons.coefficients.items())
        positives = [tok for tok, coef in cons.coefficients.items() if coef > 0]
        negatives = [tok for tok, coef in cons.coefficients.items() if coef < 0]
        assert len(positives) == 1 and len(negatives) == 1
        low_high_seen.append((positives[0], negatives[0]))
        assert len(items) == 2
    # Ordering: low threshold token is positive, high threshold is negative,
    # for both adjacent pairs.
    expected = {("tok_100", "tok_120"), ("tok_120", "tok_150")}
    assert set(low_high_seen) == expected


def test_ladder_constraints_skip_when_yes_outcome_missing():
    markets = [
        _binary_market("m1", "Will BTC reach $100,000?"),
        _binary_market("m2", "Will BTC reach $120,000?"),
        _binary_market("m3", "Will BTC reach $150,000?"),
    ]
    # Strip the YES outcome from the middle rung — constraint involving it
    # should silently skip rather than crash.
    markets[1] = Market(
        market_id="m2",
        question="Will BTC reach $120,000?",
        description="",
        outcomes=[Outcome(outcome_id="m2_no", name="No", price=Decimal("0.5"))],
        event_id="evt",
        liquidity=10_000.0,
    )
    # The detector itself requires binary polarity, so m2 is filtered out at
    # cluster formation. With only 2 rungs left (< min_rungs=3), no cluster
    # should be emitted at all.
    assert detect_monotonic_ladders(markets) == []


# --------------------------------------------------------- conditional subset


def test_conditional_subset_groups_children_with_parent():
    parent = _binary_market("parent", "Will Harris be nominated?")
    child1 = _binary_market(
        "child1",
        "Will Harris win the general election?",
        conditional_parent_id="parent",
    )
    child2 = _binary_market(
        "child2",
        "Will Harris win by >5 points?",
        conditional_parent_id="parent",
    )
    orphan = _binary_market("other", "Unrelated question?")  # no parent

    clusters = detect_conditional_subsets([parent, child1, child2, orphan])
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.constraint_source == "conditional_subset"
    ids = [m.market_id for m in cluster.markets]
    # Parent is first by convention
    assert ids[0] == "parent"
    assert set(ids[1:]) == {"child1", "child2"}


def test_conditional_subset_skipped_when_parent_absent():
    child = _binary_market(
        "child1",
        "Will the vote pass the House?",
        conditional_parent_id="missing_parent",
    )
    assert detect_conditional_subsets([child]) == []


def test_build_conditional_constraints_emits_subset_inequalities():
    parent = _binary_market("p", "Will X be nominated?", yes_token="p_yes")
    child = _binary_market(
        "c", "Will X be elected?", yes_token="c_yes", conditional_parent_id="p",
    )
    cluster = detect_conditional_subsets([parent, child])[0]
    constraints = build_conditional_constraints(cluster)
    assert len(constraints) == 1
    cons = constraints[0]
    assert cons.rhs == 0.0
    # P(parent) - P(child) >= 0
    assert cons.coefficients == {"p_yes": 1.0, "c_yes": -1.0}
