# Map-Maker Logic Review & Improvements

## Context

The user asked for a first-principles review of the Map Maker's arbitrage-discovery logic, before reading the code. Specifically: (1) intra-Polymarket arbitrage across related markets, and (2) cross-exchange arbitrage between Polymarket and Limitless. The goal is to compare an ideal design against the current implementation and identify concrete changes worth making.

Polymarket's native hierarchy is **Tag → Event → Market**. An Event groups related markets (e.g., "2024 US Presidential Election"); each Market is one tradable binary outcome with its own order book. **NegRisk Events** are a special class where 3+ markets are *mutually exclusive and exhaustive*, so Σ YES prices must equal $1.00, and a NO token in any market atomically converts to 1 YES in every other market via the Neg Risk Adapter contract.

## Ideal Map-Maker Taxonomy (research-derived)

Four distinct arbitrage classes should be captured:

1. **Intra-market trivial arb** — YES + NO < $1 in a single binary market. Rare, HFT-captured.
2. **Intra-event structural arb (NegRisk / partition)** — Σ YES across an exhaustive condition set ≠ 1.0. Per the *Unravelling the Probabilistic Forest* paper (arxiv 2508.03474), NegRisk rebalancing is **8.6% of opportunities but 73% of profits** — the single highest-value class.
3. **Inter-event logical arb (combinatorial)** — logical implications between markets in *different* events. Classic: "Trump wins 2028" ≤ "Republican wins 2028". Requires LLM + topic clustering to discover.
4. **Cross-exchange arb (Polymarket ↔ Limitless)** — same underlying outcome listed on both exchanges. Matching must verify resolution source, resolution date, settlement criteria, and question semantics; opportunities last seconds, so mismatched resolution = silent loss.

## What the Code Already Does Well

- **Event-first discovery.** [discovery.py:337-443](src/polyquant/agents/discovery.py#L337-L443) uses the Gamma `/events` endpoint, not raw `/markets`, so Polymarket's native hierarchy is respected.
- **NegRisk as a first-class mechanical cluster.** [discovery.py:415-476](src/polyquant/agents/discovery.py#L415-L476) detects `negRiskMarketID`, emits `[PARTITION] Σ=1.0` unconditionally, and bypasses LLM/Validator entirely via [map_maker.py:682-687](src/polyquant/map_maker.py#L682-L687). Zero quota cost, deterministic. This captures the highest-value arb class correctly.
- **Cross-market partition detection.** [partition_detector.py](src/polyquant/agents/partition_detector.py) finds separate binary markets that implicitly partition one real-world event (e.g., per-candidate "Will X win?" markets) via a 4-layer graph + LLM pipeline.
- **Cross-exchange matcher is fully wired.** [exchange_matcher.py:161-664](src/polyquant/agents/exchange_matcher.py#L161-L664) runs a four-layer funnel (structural pre-filter → MiniLM embeddings → Gemini LLM verify → persistent cache), and [map_maker.py:902-1000](src/polyquant/map_maker.py#L902-L1000) injects matches as **coefficient aliases** on existing constraints — elegant, lets the solver treat Polymarket YES and Limitless YES as the same security.
- **LLM prompt is structurally correct.** [logic_architect.py:244-387](src/polyquant/agents/logic_architect.py#L244-L387) explicitly forbids filtering on liquidity/spread, uses price only as a *structural sanity check*, and defines the four relationship types (MUTUALLY_EXCLUSIVE, PARTITION, SUBSET, CAUSAL_GROUP).
- **Content-addressed caching** on `SHA256(topic+market_ids)` at [map_maker.py:544-569](src/polyquant/map_maker.py#L544-L569) avoids re-paying LLM cost on unchanged clusters, while cross-exchange equivalencies are re-injected every run so newly-discovered Limitless pairs always land.

## Gaps Worth Changing (prioritized)

### Gap 1 — **Mechanical clusters are siloed from cross-event LLM analysis** *(highest impact)*

At [discovery.py:557-567](src/polyquant/agents/discovery.py#L557-L567) the residual list for the LLM stage is built by *excluding* any market already consumed by NegRisk / native-partition / cross-market-partition:

```python
for m in ev["markets"]:
    if (m.market_id not in auto_clustered_market_ids
        and m.market_id not in cross_clustered_ids):
        all_llm_markets.append(m)
```

**Consequence:** The YES tokens inside a NegRisk event (e.g. "Trump wins 2024", "Harris wins 2024") never reach the LLM, so the LogicArchitect cannot discover inter-event implications involving them. A classic combinatorial arb like **"Trump wins 2024 ≤ Republican wins 2024"** — where both legs live inside separate NegRisk events — is structurally unreachable today. The paper calls this out as a distinct category from NegRisk rebalancing; PolyQuant captures the latter but not the former.

**Fix:** Keep mechanical partition constraints as-is (they are correct and cheap), but **also** pass mechanical-cluster markets into a second LLM-clustering pass whose *only* job is to discover inter-event SUBSET / MUTUALLY_EXCLUSIVE relationships between high-signal markets drawn from different events. Cross-event implications get emitted as additional constraints on a synthetic "logical link" cluster that references both source events. The NegRisk Σ=1 constraint is unaffected.

Files:
- [src/polyquant/agents/discovery.py](src/polyquant/agents/discovery.py) — add a second residual list containing representative YES-tokens from mechanical clusters (typically the top-K by liquidity/volume per event to bound LLM cost).
- [src/polyquant/map_maker.py](src/polyquant/map_maker.py) — new cluster type `cross_event_logical` that runs the full LogicArchitect/Validator path but only emits SUBSET/MUTUALLY_EXCLUSIVE constraints (not partitions, to avoid duplicating NegRisk work).
- [src/polyquant/agents/logic_architect.py](src/polyquant/agents/logic_architect.py) — small prompt variant for this pass that explicitly tells the model "these markets come from *different* events; look for implications *between* events, ignore partition structure — that is already handled."

### Gap 2 — **Cross-exchange resolution-source verification is LLM-only**

[exchange_matcher.py:56-86](src/polyquant/agents/exchange_matcher.py#L56-L86) passes `resolution_source` and `end_date` strings into the LLM prompt and lets Gemini decide. There is no structural canonicalization. If Polymarket resolves on "AP call" and Limitless resolves on "NYT projection", and both descriptions are terse, the model can plausibly say *match*. A false positive here is not a missed profit — it is an **executed leg that never offsets**, i.e. real capital loss.

**Fix:** Add a structural pre-check alongside the existing prefilter:
1. Canonicalize `resolution_source` strings (lowercase, strip punctuation, map known aliases: `"ap" == "associated press"`, `"cg" == "coingecko"`, etc.).
2. **Fast path**: if canonicalized sources match *exactly*, auto-link the pair on source alone — no doubt about the oracle.
3. **Slow path**: if sources differ, still allow the LLM to verify, but require an explicit `"resolution_sources_match": true/false` field in the response schema. The LLM may judge that two differently-worded sources reference the same underlying real-world event (e.g. "AP official call" and "network projections" for an election); if it confirms equivalence, the pair is accepted.
4. Require `end_date` to be within a configurable delta (default: ±24h for daily-resolution markets, ±1h for minute-resolution ones — today's ±1 *week* bucket is far too loose for time-sensitive markets).

Files:
- [src/polyquant/agents/match_prefilter.py](src/polyquant/agents/match_prefilter.py) — add source canonicalization and tightened expiry delta.
- [src/polyquant/agents/exchange_matcher.py](src/polyquant/agents/exchange_matcher.py) — update `LLM_VERIFY_PROMPT` (line 56) to return structured `resolution_sources_match` and `timeframes_match` flags. Reject if either is false, independent of `is_match`.

### Gap 3 — **Standalone binary markets have no explicit YES + NO = 1 constraint**

A non-NegRisk binary market's YES and NO tokens are complements by CTF split/merge mechanics but PolyQuant emits no manifest constraint for them. The solver therefore cannot see `z[YES] + z[NO] == 1` as a structural identity, and cannot exploit `YES + NO < $1` arb within a single market (rare but legitimate).

**Fix:** In [discovery.py](src/polyquant/agents/discovery.py) emit a trivial `native_partition` cluster for every standalone binary market (2 outcomes, not part of a larger group). Mechanical, zero LLM cost, handled by existing `build_partition_constraint()`.

### Gap 4 — **Tags are fetched but unused**

[polymarket_client.py:411-416](src/polyquant/data/polymarket_client.py#L411) reads tag labels into event metadata, but nothing downstream filters, groups, or weights by tag. Two concrete uses would pay for themselves:

1. **Category skip-list** in config (e.g. `excluded_tags: ["Sports"]`) — a user running PolyQuant on a thin-liquidity VPS can skip whole domains without code changes.
2. **Tag prior for cross-market partition detection** — markets sharing a tag are far more likely to be genuine partition candidates; this cuts the keyword-graph search space in [partition_detector.py](src/polyquant/agents/partition_detector.py).

### Gap 5 — **Limitless-only arbitrage is invisible AND Limitless is only matched against *clustered* Polymarket events**

Two sub-issues:

**5a.** The map builds clusters exclusively from Polymarket events. Limitless markets enter only as *aliases* injected onto existing Polymarket clusters. If an arbitrage exists purely inside Limitless (or between two Limitless markets) with no Polymarket counterpart, it is never discovered.

**5b.** *(critical — flagged by user)* The cross-exchange matcher currently only compares Limitless markets against Polymarket markets that **made it into a cluster**. But many Polymarket events never cluster — they get filtered by liquidity floor, zombie detection, or simply don't match any clustering tier. Those unclustered Polymarket events can still have legitimate Limitless counterparts, and those pairs are silently lost today.

**Fix:** Mirror the event-based discovery loop against [limitless_client.py](src/polyquant/data/limitless_client.py), AND ensure the cross-exchange candidate pool on the Polymarket side is **every fetched Polymarket event**, not just those that ended up in clusters. Unclustered Polymarket events that successfully match a Limitless market should spawn a new cross-exchange cluster (two-market equivalence pair) so the Navigator can still trade the arb.

## Scope (confirmed with user)

All 5 gaps. Gap 1 uses **top-K = 3** representatives per mechanical cluster for the cross-event LLM pass.

Recommended order of execution is correctness-before-capability: **Gap 2 → Gap 3 → Gap 4 → Gap 1 → Gap 5**. This way every intermediate state leaves the pipeline in a shippable position, the highest-risk change (Gap 1's new LLM path) lands on top of a cleaned-up foundation, and Gap 5 (Limitless-first discovery) is last because it is the largest surface-area change and benefits from Gap 2's tightened matching.

## Execution Plan (all 5 gaps)

### Step 1 — Gap 2 (correctness, low risk)
1. Add `canonicalize_resolution_source()` helper in [match_prefilter.py](src/polyquant/agents/match_prefilter.py) with a small alias table (`"ap" == "associated press"`, `"cg" == "coingecko"`, strip punctuation, lowercase). ~30 lines.
2. Extend `_expiry_week_bucket()` into `_expiry_delta_hours()` with configurable tolerance. Default: 24h for daily-resolution markets.
3. Update `LLM_VERIFY_PROMPT` in [exchange_matcher.py:56-86](src/polyquant/agents/exchange_matcher.py#L56) to require `resolution_sources_match` and `timeframes_match` booleans in the response JSON.
4. In the decision site around line 620-650, reject the pair if either flag is false, regardless of `is_match`.
5. Bump the cache schema version so old cached decisions (made with the looser prompt) are re-verified once. Invalidate existing `.polyquant/constraints/market_pairs.json` on version mismatch.

### Step 2 — Gap 3 (cheap mechanical addition)
1. In [discovery.py](src/polyquant/agents/discovery.py) Phase 2, for any binary YES/NO market not already absorbed by NegRisk or cross-market partition, emit a trivial `native_partition` cluster. ~15 lines.
2. Verify `build_partition_constraint()` handles the 2-outcome case without change — it should, since the function already operates on outcome arrays of any length.
3. Regression check: total NegRisk cluster count must be unchanged.

### Step 3 — Gap 4 (tag-aware discovery)
1. Add `excluded_tags: list[str] = Field(default_factory=list)` to [utils/config.py](src/polyquant/utils/config.py) Settings.
2. In [discovery.py:337-443](src/polyquant/agents/discovery.py#L337) (`scan_markets` loop over events), filter out any event whose tag set intersects `excluded_tags` before clustering.
3. Pass event-tag labels into [partition_detector.py](src/polyquant/agents/partition_detector.py) as a prior: prefer pairs sharing at least one tag when building the keyword-overlap graph (Layer 1). This cuts the search space without changing correctness.
4. Expose tag stats in the map-maker report so the user can see which tags were skipped.

### Step 4 — Gap 1 (new capability, highest risk)
1. Add `_select_mechanical_cluster_representatives(clusters, k=3)` in [discovery.py](src/polyquant/agents/discovery.py): for each mechanical cluster, pick the top-3 markets by liquidity. Deduplicate across clusters.
2. Introduce a new `MarketCluster.constraint_source = "cross_event_logical"` variant in [market_models.py](src/polyquant/data/market_models.py) (or wherever `MarketCluster` lives) that carries markets drawn from *multiple* source events. Its `cluster_id` must reference the source cluster IDs for traceability (e.g. `cross_event_{hash}`).
3. Build cross-event clusters after Phase 2c by semantic pre-clustering the representatives (reuse the existing MiniLM `all-MiniLM-L6-v2` path at threshold ~0.40) — only emit a cross-event cluster if ≥2 representatives from *different* source events land in the same semantic group.
4. In [map_maker.py:`_analyze_cluster`](src/polyquant/map_maker.py#L654), route `cross_event_logical` through the existing LogicArchitect/Validator path, NOT the mechanical bypass. The LLM does the reasoning; the Validator filters false positives.
5. Add a prompt variant in [logic_architect.py:244-387](src/polyquant/agents/logic_architect.py#L244) — a short addendum when `cluster.constraint_source == "cross_event_logical"`: *"These markets come from different events. Look only for inter-event SUBSET / MUTUALLY_EXCLUSIVE relationships. Partition constraints are already handled upstream — do NOT emit them."*
6. Ensure the cache hash at [map_maker.py:544-569](src/polyquant/map_maker.py#L544-L569) treats the cross-event cluster as a distinct hashable unit (include `constraint_source` in the hash preimage).
7. Add a `PHASE 4: CROSS-EVENT LOGICAL ANALYSIS` section to the map-maker report at [map_maker.py:463-496](src/polyquant/map_maker.py#L463-L496) so cross-event constraints are surfaced explicitly.
8. The Validator path already filters by confidence; no Validator changes needed. Emit only constraints with `confidence ≥ 0.7` to avoid noise.
9. **Tag cross-event clusters distinctly in the knowledge map** (flagged by user):
   - Persist `cluster_type: "cross_event_logical"` on the `ConstraintManifest` in [constraint_store.py](src/polyquant/data/constraint_store.py) — add it as a top-level field on the JSON schema (bump schema version). The current `source_cluster_id` is insufficient; a dedicated discriminator is needed so downstream consumers (Navigator, dashboard, report) can filter/group by type without parsing the cluster_id string.
   - Do the same for the existing mechanical types (`negrisk`, `native_partition`, `cross_market_partition`, `llm_analysis`) — today they are implicit in the cluster_id prefix. Promoting `cluster_type` to a first-class field makes the knowledge map queryable by source.
   - Update the web dashboard ([web/src/components/Dashboard.tsx](web/src/components/Dashboard.tsx)) to render a colored badge per cluster type: NegRisk (blue), Native Partition (green), Cross-Market Partition (teal), LLM Analysis (purple), **Cross-Event Logical (orange)**, Cross-Exchange Pair (red). This makes the new cluster class visually identifiable at a glance.
   - Group the map-maker report's summary statistics by `cluster_type` so the user can see counts per class in one view.

### Step 5 — Gap 5 (Limitless-first discovery + full-universe matching)

**5a. Widen the cross-exchange candidate pool (critical fix — flagged by user)**
1. Audit [exchange_matcher.py](src/polyquant/agents/exchange_matcher.py) `run_matching_pipeline()`: confirm whether the Polymarket candidate set is drawn from `all_polymarket_events` (full fetch) or from `clustered_polymarket_markets` (post-clustering). If the latter, **change it to the former** — matching must run against every fetched Polymarket event regardless of whether it ended up in a cluster.
2. For any cross-exchange pair found on a previously-unclustered Polymarket event, **spawn a new 2-market `cross_exchange_pair` cluster** so the manifest still emits the equivalence constraint and the Navigator can trade it. This ensures that `cluster membership` is not a gate for cross-exchange matching.
3. Add a counter to the map-maker report: *"Pairs rescued from unclustered Polymarket events: N"* — so any future regression is visible.

**5b. Limitless-first discovery (closes the pure-Limitless arb hole)**
1. Mirror [polymarket_client.py:get_active_events](src/polyquant/data/polymarket_client.py#L337) with a `get_active_events()` equivalent in [limitless_client.py](src/polyquant/data/limitless_client.py), returning the same event/market shape.
2. In [discovery.py](src/polyquant/agents/discovery.py), after Polymarket clustering + full-universe cross-exchange matching (step 5a), fetch Limitless events and identify any whose markets are *not yet* referenced by any cluster or cross-exchange pair. Cluster those as a second pass using the same Tier 1-4 strategy (NegRisk-equivalent detection on Limitless markets if Limitless exposes a negRisk flag; then LLM for residuals).
3. Add `LIMITLESS_DISCOVERY` config flag (default: true) so the user can disable it if rate-limited.
4. Update the map-maker report to show Limitless-originated clusters separately from Polymarket-originated ones.

## Verification

1. **Gap 2** — unit tests for `canonicalize_resolution_source()` covering known aliases and edge cases. Integration test: feed a deliberately-mismatched pair (different resolution sources) and confirm rejection even when `is_match=true`. Confirm cache schema bump re-verifies old entries.
2. **Gap 3** — run `python -m polyquant.main map --limit 20` and confirm at least one new 2-outcome `native_partition` cluster is produced. Confirm no regression in NegRisk cluster count.
3. **Gap 4** — set `excluded_tags: ["Sports"]` in config, rerun map-maker, confirm zero Sports-tagged events in output and report shows skip count. Confirm partition-detector runtime drops (tag prior working).
4. **Gap 1** — pick a known cross-event implication (e.g., "Trump wins 2024" from a NegRisk election event and "Republican wins 2024" from a separate NegRisk event) and confirm a SUBSET constraint with both token IDs appears in `.polyquant/constraints/_cluster_cross_event_*_meta.json`. Inspect the new `PHASE 4: CROSS-EVENT LOGICAL ANALYSIS` section of the map-maker report. Regression check: NegRisk Σ=1 constraint counts unchanged; Gap 1 should *only add* new clusters, not perturb existing ones.
5. **Gap 5a** — log the size of the Polymarket candidate pool handed to the matcher; confirm it equals `len(all_polymarket_events)`, not `len(clustered_polymarket_markets)`. Run against a live slice and confirm the new "Pairs rescued from unclustered Polymarket events" counter is non-zero at least occasionally.
6. **Gap 5b** — disable Polymarket fetching temporarily (or point at a narrow slice), enable Limitless-first discovery, confirm clusters are built from pure-Limitless events.
7. **End-to-end** — run `pytest tests/`, then `python -m polyquant.main map --limit 100 --force` and inspect the full map-maker report for all four phases (Discovery, Cross-Exchange, Reasoning, Cross-Event Logical). Run `python -m polyquant.main trade` in paper mode for at least 10 minutes to confirm the Navigator loads the new constraint classes without error.
8. **Regression baseline** — before starting, save the current manifest set as a golden copy. After each step, diff: the set of manifests should *only grow*; no constraint should disappear or change coefficients unless the change is explicitly expected.

## Critical Files

- [src/polyquant/agents/discovery.py](src/polyquant/agents/discovery.py) — Gaps 1, 3, 4
- [src/polyquant/map_maker.py](src/polyquant/map_maker.py) — Gap 1 routing, report output
- [src/polyquant/agents/logic_architect.py](src/polyquant/agents/logic_architect.py) — Gap 1 prompt variant
- [src/polyquant/agents/match_prefilter.py](src/polyquant/agents/match_prefilter.py) — Gap 2 canonicalization
- [src/polyquant/agents/exchange_matcher.py](src/polyquant/agents/exchange_matcher.py) — Gap 2 prompt + decision logic
- [src/polyquant/data/market_models.py](src/polyquant/data/market_models.py) — possibly new `MarketCluster` variant discriminator
- [src/polyquant/data/constraint_store.py](src/polyquant/data/constraint_store.py) — Gap 1 step 9, `cluster_type` field on manifest schema
- [web/src/components/Dashboard.tsx](web/src/components/Dashboard.tsx) — Gap 1 step 9, colored badge per cluster type

## Sources

- [Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets (arXiv 2508.03474)](https://arxiv.org/abs/2508.03474)
- [Polymarket docs — Markets & Events](https://docs.polymarket.com/concepts/markets-events)
- [Polymarket docs — Negative Risk Markets](https://docs.polymarket.com/advanced/neg-risk)
- [NegRisk Market Rebalancing: How $29M Was Extracted — Navnoor Bawa](https://medium.com/@navnoorbawa/negrisk-market-rebalancing-how-29m-was-extracted-from-multi-condition-prediction-markets-2f1f91644c5b)
- [Beyond Simple Arbitrage: 4 Polymarket Strategies Bots Profit From in 2026](https://medium.com/illumination/beyond-simple-arbitrage-4-polymarket-strategies-bots-actually-profit-from-in-2026-ddacc92c5b4f)
