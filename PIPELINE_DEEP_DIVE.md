# PolyQuant 2.0 — Complete Pipeline Deep Dive

> **Purpose**: This document is the definitive technical reference for every piece of the PolyQuant codebase. After reading it, you should understand what every file does, how data flows through the system, and the latency/CPU cost of each stage.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Entry Point & CLI](#2-entry-point--cli)
3. [Configuration System](#3-configuration-system)
4. [Data Models](#4-data-models)
5. [The Slow Brain — Map Maker Pipeline](#5-the-slow-brain--map-maker-pipeline)
   - 5.1 [Discovery Agent](#51-discovery-agent)
   - 5.2 [Logic Architect](#52-logic-architect)
   - 5.3 [Validator Agent](#53-validator-agent)
   - 5.4 [Correlation Agent](#54-correlation-agent)
   - 5.5 [Constraint Store & Manifests](#55-constraint-store--manifests)
6. [The Fast Brain — Navigator Pipeline](#6-the-fast-brain--navigator-pipeline)
   - 6.1 [Startup & Manifest Loading](#61-startup--manifest-loading)
   - 6.2 [WebSocket Client & Price Cache](#62-websocket-client--price-cache)
   - 6.3 [Microstructure Agent](#63-microstructure-agent)
   - 6.4 [Execution Guard](#64-execution-guard)
   - 6.5 [Arbitrage Detector (Frank-Wolfe)](#65-arbitrage-detector-frank-wolfe)
   - 6.6 [SCIP Solver (LMO Oracle)](#66-scip-solver-lmo-oracle)
   - 6.7 [Position Sizer (Kelly Criterion)](#67-position-sizer-kelly-criterion)
   - 6.8 [Kill Switch](#68-kill-switch)
   - 6.9 [Trade Executor](#69-trade-executor)
7. [Supporting Infrastructure](#7-supporting-infrastructure)
   - 7.1 [LLM Client (OpenRouter)](#71-llm-client-openrouter)
   - 7.2 [Redis Cache](#72-redis-cache)
   - 7.3 [Monitoring API Server](#73-monitoring-api-server)
   - 7.4 [Profiling & Watchdog](#74-profiling--watchdog)
8. [Complete Data Flow Diagrams](#8-complete-data-flow-diagrams)
9. [Latency & Performance Budget](#9-latency--performance-budget)
10. [File Index](#10-file-index)

---

## 1. High-Level Architecture

PolyQuant operates as a **two-brain system**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            PolyQuant 2.0                                    │
│                                                                              │
│  ┌─────────────────────────┐                ┌─────────────────────────┐     │
│  │      MAP MAKER          │                │       NAVIGATOR         │     │
│  │     "Slow Brain"        │─── manifests ──▶│      "Fast Brain"      │     │
│  │   (runs periodically)   │    (JSON files)│   (runs continuously)  │     │
│  │                         │                │                         │     │
│  │  • Discovery Agent      │                │  • ExecutionGuard       │     │
│  │  • Logic Architect      │                │  • PriceCache (WS)     │     │
│  │  • Validator Agent      │                │  • MicrostructureAgent  │     │
│  │  • Correlation Agent    │                │  • ArbitrageDetector    │     │
│  │  • LLM calls (slow)     │                │  • SCIP/FW Solver      │     │
│  │                         │                │  • PositionSizer        │     │
│  │  ⏱ 5-10 minutes total   │                │  • KillSwitch           │     │
│  │  🧠 CPU: Low (mostly I/O)│                │  • TradeExecutor        │     │
│  └─────────────────────────┘                │                         │     │
│                                              │  ⏱ <50ms per tick       │     │
│                                              │  🧠 CPU: Medium-High    │     │
│                                              └─────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Why two brains?**
- The Slow Brain uses expensive LLM calls to **think deeply** about market relationships. This happens offline — you can wait minutes.
- The Fast Brain uses pre-computed constraints to **react instantly** to price changes. Zero LLM calls. Pure math. Target: <50ms tick-to-trade.

---

## 2. Entry Point & CLI

**File**: [`main.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/main.py) (197 lines)

This is the single entry point for the entire system. It uses `argparse` to expose two modes:

```bash
python -m polyquant.main map --min-liquidity 1000 --limit 0  # Scan all events with $1k+ liquidity
python -m polyquant.main map --limit 20 --force              # Re-analyze top 20 liquid events
python -m polyquant.main trade                               # Start the real-time Navigator
```

### What happens on startup:

| Step | What | Code |
|------|-------|------|
| 1 | Parse CLI arguments (`mode`, `--limit`, `--min-liquidity`, `--force`) | Lines 126-170 |
| 2 | Print ASCII banner with current mode | Lines 173-179 |
| 3 | Branch to `run_map_maker()` or `run_navigator()` | Lines 181-188 |
| 4 | `run_map_maker()` creates a `MapMaker` context manager, calls `build_map()`, prints results | Lines 49-89 |
| 5 | `run_navigator()` creates a `Navigator`, registers SIGINT/SIGTERM handlers, calls `navigator.run()` | Lines 92-121 |

**Latency**: Startup itself is <100ms. The lazy imports (`from polyquant.map_maker import MapMaker`) happen inside the functions to keep initial import fast.

**CPU**: Negligible — just argument parsing and printing.

---

## 3. Configuration System

**File**: [`utils/config.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/utils/config.py) (227 lines)

All configuration is loaded from environment variables via **Pydantic Settings**. The config is a singleton — loaded once and cached with `@lru_cache`.

### Key Configuration Groups:

| Group | Variables | Purpose |
|-------|-----------|---------|
| **API Keys** | `GEMINI_API_KEY`, `ALCHEMY_API_KEY`, `POLYGON_PRIVATE_KEY` | All stored as `SecretStr` — never printed in logs |
| **Polymarket URLs** | `POLYMARKET_GAMMA_URL`, `POLYMARKET_CLOB_URL`, `POLYMARKET_DATA_URL`, `POLYMARKET_WS_URL` | 4 separate API endpoints |
| **Redis** | `REDIS_URL` | Default: `redis://localhost:6379/0` |
| **Trading** | `EXTRACTION_ALPHA` (0.9), `MAX_DRAWDOWN` (0.15), `ORDERBOOK_DEPTH_CAP` (0.5), `VWAP_SLIPPAGE_LIMIT` (0.05) | Risk parameters |
| **Frank-Wolfe** | `INITIAL_EPSILON` (0.1), `MIN_PROFIT_THRESHOLD` ($0.05), `FW_MAX_ITERATIONS` (150) | Solver tuning |
| **Execution** | `TRADING_MODE` (`paper` / `live`), `PRIVATE_RPC_URL` | Mode selection |

**How it works**: Pydantic reads `.env` → validates types → creates `PolyQuantConfig` instance → cached globally as `config`.

**CPU**: One-time 10-20ms at import time. Zero cost thereafter.

---

## 4. Data Models

**File**: [`data/market_models.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/data/market_models.py) (315 lines)

Every piece of data flowing through the system is a Pydantic model. Here's the complete hierarchy:

### Core Models

| Model | Fields | Purpose |
|-------|--------|---------|
| **`Market`** | `market_id`, `question`, `description`, `outcomes[]`, `volume`, `liquidity`, `negrisk`, `group_id`, `microstructure` | A prediction market from Polymarket |
| **`Outcome`** | `outcome_id`, `name`, `price` (0-1), `token_id` | Single outcome (e.g., "Yes" at 0.65) |
| **`OrderBook`** | `outcome_id`, `bids[]`, `asks[]`, `timestamp` | Level 2 order book snapshot |
| **`OrderLevel`** | `price` (0-1), `size` (shares) | Single price level in book |

### Signal Models

| Model | Fields | Purpose |
|-------|--------|---------|
| **`MicrostructureSignal`** | `imbalance` (-1 to +1), `spread`, `weighted_midpoint` | Real-time order book analysis |
| **`CorrelationSignal`** | `leader_id`, `laggard_id`, `correlation` (-1 to +1), `lead_time_seconds` | Statistical market relationship |

### Trade Models

| Model | Fields | Purpose |
|-------|--------|---------|
| **`MarketDependency`** | `source_market_id`, `target_market_id`, `relationship` (implies/excludes/correlates) | LLM-detected dependency |
| **`ProposedTrade`** | `market_id`, `outcome_id`, `side` (buy/sell), `size`, `limit_price`, `time_in_force` (IOC by default), `priority` | Optimizer output |
| **`ArbitrageOpportunity`** | `markets[]`, `trades[]`, `expected_profit`, `guaranteed_profit`, `confidence` | Detected arbitrage |

### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `OrderSide` | `BUY`, `SELL` | Trade direction |
| `TimeInForce` | `GTC`, `IOC` (default, safe), `FOK` | Order execution policy |

### Key Computed Properties on `OrderBook`:

- `best_bid` / `best_ask`: O(n) scan of levels
- `spread`: `best_ask - best_bid`
- `mid_price`: `(best_bid + best_ask) / 2`
- `get_vwap(side, size)`: Walks the book to calculate the actual fill price for a given size

**Memory**: Each `OrderBook` is ~500 bytes for a typical 10-level book. With 200 outcomes tracked, that's ~100KB in `PriceCache`.

---

## 5. The Slow Brain — Map Maker Pipeline

**File**: [`map_maker.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/map_maker.py) (~487 lines)

The Map Maker is an `async with` context manager that orchestrates the offline analysis pipeline. On `__aenter__`, it initializes all agents and clients. On `build_map()`, it runs the full pipeline:

```
build_map()
  ├── Step 1: DiscoveryAgent.scan_markets()        → MarketCluster[]
  ├── Step 2: For each cluster:
  │     ├── LogicArchitect.analyze_cluster()        → AnalysisResult
  │     └── ValidatorAgent.validate()               → ValidatedResult
  ├── Step 3: CorrelationAgent.analyze_pairs()      → CorrelationSignal[]
  └── Step 4: ConstraintStore.save_manifest()       → JSON files on disk
```

**Total time**: 5-10 minutes for 100 markets (dominated by LLM API calls).
**CPU**: Low — mostly waiting on HTTP responses from OpenRouter.

---

### 5.1 Discovery Agent

**File**: [`agents/discovery.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/agents/discovery.py) (~460 lines)

**Purpose**: Scan Polymarket for active markets and group them into clusters where arbitrage might exist.

**Pipeline (3 phases)**:

#### Phase 1: Fetch Events (No LLM, ~2-5 seconds)

```python
events = await self._polymarket.get_active_events(
    min_liquidity=min_liquidity,
    max_events=limit,
)
```

- Calls the **Gamma API** `/events` endpoint
- Events come pre-sorted by liquidity (highest first)
- Each event contains multiple markets (e.g., "2024 US Election" has "Trump wins", "Biden wins", etc.)
- Pagination handles large result sets

**CPU**: Negligible (HTTP I/O bound).
**Latency**: 1-5 seconds depending on Polymarket API response time.

#### Phase 2: Auto-Cluster NegRisk (No LLM, ~10ms)

NegRisk markets are Polymarket's multi-outcome events where all outcomes are mutually exclusive and exhaustive (sum to 1.0). These are **automatically clustered** without any LLM call by detecting the `negRiskMarketID` field in the event data:

```python
if neg_risk_id and len(valid_markets) > 1:
    clusters.append(MarketCluster(
        cluster_id=f"negrisk_{event_id}",
        topic=f"[AUTO] {event_title} (NegRisk, Sum={total_price:.2f})",
        ...
    ))
```

Also filters out **zombie markets** (prices at 0.00/1.00 indicating resolution):

```python
def _is_zombie_market(self, market: Market) -> bool:
    # Markets with extreme prices are likely resolved
```

**CPU**: Negligible — just dict operations.
**Latency**: <10ms.

#### Phase 3: LLM Clustering (Expensive, ~30-120 seconds)

Markets that aren't NegRisk are sent to the LLM for analysis. The system prompt is highly optimized for arbitrage detection — it asks the LLM to find 4 types of constraints:

| Constraint Type | Meaning | Price Rule |
|-----------------|---------|------------|
| `MUTUALLY_EXCLUSIVE` | Only one can be YES | Sum ≤ 1 |
| `EXHAUSTIVE` | All possibilities covered | Sum = 1 |
| `IMPLICATION` | A implies B | P(B) ≥ P(A) |
| `CONDITIONAL` | A's outcome affects B | Correlation link |

The prompt includes **actual prices and liquidity** for each market, enabling the LLM to spot price violations immediately.

```python
market_lines.append(
    f"ID: {m.market_id} | Question: {m.question} | YES Price: {yes_price:.2f} | Liquidity: ${m.liquidity:,.0f}"
)
```

**CPU**: Negligible (waiting on API).
**Latency**: 30-120 seconds depending on batch size and model response time.
**Cost**: Free via OpenRouter free models, but rate-limited.

#### Output: `list[MarketCluster]`

Each cluster contains:
- `cluster_id`: Unique ID
- `topic`: Human description (e.g., "2024 US Presidential Election")
- `markets`: List of `Market` objects
- `potential_dependencies`: String descriptions of constraints

---

### 5.2 Logic Architect

**File**: [`agents/logic_architect.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/agents/logic_architect.py) (659 lines)

**Purpose**: Take a cluster of related markets and extract **formal mathematical constraints** that can be fed to the optimizer.

**Input**: `MarketCluster` from Discovery Agent
**Output**: `AnalysisResult` containing `LogicalConstraint` objects

#### The Constraint Format

All constraints use the form: **A^T × z ≥ b**

Where:
- `z` is a binary vector of market outcomes (1 = resolved YES, 0 = NO)
- `coefficients` maps outcome_id → coefficient value
- `rhs` is the right-hand side bound

**Example**: "If Trump wins (M1=Yes), then a Republican wins (M2=Yes)"
→ `z[M2] - z[M1] ≥ 0` → coefficients: `{M2: 1, M1: -1}`, rhs: `0`

#### How It Works

1. **Format markets** for the LLM prompt (question, outcomes, prices, IDs)
2. **Call LLM** via OpenRouter with a detailed system prompt that explains constraint matrix format
3. **Parse response** into typed `LogicalConstraint` objects
4. **Validate constraints** against market prices (sanity checks):
   - If A implies B, then `price(A) <= price(B)` must hold
   - Reject constraints with obviously wrong relationships
5. **Check for edge cases** (resolution ambiguity, conditional dependencies)

**CPU**: Negligible (API bound).
**Latency**: 10-30 seconds per cluster (LLM response time).
**Total contribution to pipeline**: 2-8 minutes (runs per cluster, partially parallelizable with semaphore).

---

### 5.3 Validator Agent

**File**: [`agents/validator.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/agents/validator.py) (413 lines)

**Purpose**: Quality control over the Logic Architect's output. Uses a separate LLM call with a "verification" mindset to catch errors.

**Input**: `AnalysisResult` from Logic Architect
**Output**: `ValidatedResult` with `is_valid` flag and `ValidationIssue` list

#### Validation Passes

| Pass | What It Checks | Method |
|------|----------------|--------|
| 1. **Consistency** | Do constraints contradict each other? | LLM verification |
| 2. **Coverage** | Do constraints cover all relevant outcomes? | LLM verification |
| 3. **Edge Cases** | Resolution ambiguity, conditional dependencies? | LLM verification |
| 4. **Mathematical Sanity** | Are coefficient matrices well-formed? | Code check |
| 5. **Confidence Calibration** | Are confidence scores reasonable? | Code heuristics |

If `error_count > 0`, the analysis is rejected. Warnings are recorded but don't block.

**CPU**: Negligible (API bound).
**Latency**: 10-20 seconds per cluster.

---

### 5.4 Correlation Agent

**File**: [`agents/correlation.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/agents/correlation.py) (66 lines)

**Purpose**: Find **statistical leader-laggard relationships** between markets. Unlike the Logic Architect (which uses reasoning), this uses pure math on historical price data.

**Algorithm**: Pearson correlation on historical price timeseries.

```python
def _calculate_correlation(self, series_a, series_b) -> float:
    # Manual Pearson calculation (no numpy dependency required)
    num = sum((a - mean_a) * (b - mean_b) for a, b in zip(series_a, series_b))
    den = sqrt(sum((a-mean_a)**2) * sum((b-mean_b)**2))
    return num / den
```

**Thresholds**: `min_correlation = 0.8`, `lookback_hours = 24`, `min_data_points = 10`

**Output**: `CorrelationSignal` objects stored in the `ConstraintManifest.correlations` field.

**CPU**: Low — pure arithmetic, O(N²) pairs but N is typically small (10-30 markets per cluster).
**Latency**: <1 second for typical cluster sizes.

> **Note**: The correlation engine is currently a scaffold. The `analyze_pairs()` method returns an empty list. The `_calculate_correlation()` math is implemented and ready for integration with the historical price data from `PolymarketClient.get_history()`.

---

### 5.5 Constraint Store & Manifests

**File**: [`data/constraint_store.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/data/constraint_store.py) (~284 lines)

**Purpose**: Persist and load the Map Maker's output as JSON files. These manifests are the **bridge** between the Slow Brain and the Fast Brain.

#### The `ConstraintManifest` Model

```python
class ConstraintManifest(BaseModel):
    manifest_id: str
    cluster_id: str
    topic: str
    created_at: datetime
    markets: list[Market]
    dependencies: list[MarketDependency]
    validated_constraints: list[LogicalConstraint]  # The formal A^T z ≥ b constraints
    correlations: dict[str, list[CorrelationSignal] | None]  # Statistical signals
    version: int = 1
```

#### Storage

- **Location**: `.polyquant/manifests/` directory
- **Format**: One JSON file per cluster (e.g., `manifest_negrisk_abc123.json`)
- **Size**: 5-50 KB per manifest (depends on cluster complexity)
- **Immutability**: Once written, manifests are not modified — only replaced on re-scan

#### Key Methods

| Method | What | I/O |
|--------|------|-----|
| `save_manifest(manifest)` | Serialize to JSON, write to disk | Blocking file write (~1-5ms) |
| `load_manifest(cluster_id)` | Read JSON, parse into model | Blocking file read (~1-5ms) |
| `load_all_manifests()` | Load all manifests in directory | Sequential reads (~10-50ms total) |
| `get_latest_manifests()` | Load all, sorted by creation time | Same as above |

**CPU**: Negligible — just JSON serialization/deserialization.
**Disk I/O**: ~1-5ms per manifest. For 20 manifests, ~20-100ms total on startup.

---

## 6. The Fast Brain — Navigator Pipeline

**File**: [`navigator.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/navigator.py) (~730 lines)

The Navigator is the real-time trading engine. It runs continuously, processing price updates and executing trades when opportunities arise. **Zero LLM calls** — all intelligence is pre-computed by the Map Maker.

### Overall Flow (Per Tick)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         NAVIGATOR TICK                               │
│                                                                      │
│  1. await price_cache.wait_for_update()    ← Event-driven wake-up   │
│                                              (0ms CPU, blocks)       │
│  2. Check WebSocket health                 ← Stale data protection  │
│                                              (<1ms)                  │
│  3. KillSwitch.can_trade()                 ← Safety gate            │
│                                              (<1ms)                  │
│  4. price_cache.get_all()                  ← O(1) dict access       │
│                                              (<1ms)                  │
│  5. _detect_opportunities(books)           ← THE HOT PATH           │
│     ├── Group books by cluster             ← O(n) dict lookup       │
│     ├── MicrostructureAgent.analyze()      ← Imbalance signal       │
│     ├── ExecutionGuard.check_trade()       ← Constraint validation  │
│     └── ArbitrageDetector.detect()         ← Frank-Wolfe + SCIP     │
│                                              (10-100ms)              │
│  6. For each opportunity:                                            │
│     ├── PositionSizer.calculate_size()     ← Kelly Criterion        │
│     ├── KillSwitch checks                  ← Final safety          │
│     └── TradeExecutor.execute_atomic()     ← Submit to Polymarket   │
│                                              (5-30ms)                │
│  7. Record latency metrics                 ← Profiling              │
│                                              (<1ms)                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 6.1 Startup & Manifest Loading

When the Navigator starts (`__aenter__`), it:

1. **Loads all manifests** from disk via `ConstraintStore.load_all_manifests()` — ~20-100ms
2. **Builds the ExecutionGuard** by parsing constraints into an in-memory matrix — ~5ms
3. **Initializes PriceCache** with 5-second staleness threshold — instant
4. **Connects PolymarketClient** (HTTP sessions, WebSocket) — ~200ms
5. **Initializes MicrostructureAgent** — instant
6. **Initializes ArbitrageDetector** (wraps FWSolver + SCIPSolver) — ~10ms
7. **Initializes KillSwitch** (loads state from Redis) — ~5ms
8. **Starts FastAPI sidecar server** on port 8000 — ~50ms

**Total startup**: ~500ms-1 second.

---

### 6.2 WebSocket Client & Price Cache

**Files**: [`data/polymarket_client.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/data/polymarket_client.py) (lines 570-893) and [`data/price_cache.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/data/price_cache.py) (188 lines)

#### PolymarketWSClient

This is the real-time data feed. It:

1. **Connects** to `wss://ws-subscriptions-clob.polymarket.com/ws/market`
2. **Subscribes** to specific token IDs (outcome-level order book updates)
3. **Receives** Level 2 order book snapshots as JSON messages
4. **Parses** into `OrderBook` objects and feeds to `PriceCache`
5. **Reconnects** automatically on disconnect with exponential backoff (1s → 2s → 4s → ... → 60s max)
6. **Tracks health**: `is_connection_healthy(max_age_seconds)` and `get_connection_age()` for stale connection detection

**Message rate**: 10-100 messages/second depending on subscribed markets.
**CPU per message**: ~1-3ms (JSON parse + OrderBook construction + cache update + subscriber notification).

#### PriceCache

An in-memory dictionary that stores the latest `OrderBook` for each token:

```python
class PriceCache:
    _books: dict[str, OrderBook]          # token_id → latest OrderBook
    _last_update: dict[str, float]        # token_id → monotonic timestamp
    _update_event: asyncio.Event()        # Wake up Navigator on any update
```

**Key design decisions**:
- Uses `time.monotonic()` instead of `datetime.utcnow()` — saves ~1ms per tick
- `asyncio.Event()` enables **event-driven architecture** — Navigator blocks on `wait_for_update()` instead of polling
- Subscribers are notified in **parallel** via `asyncio.gather()` — saves 5-20ms vs sequential
- Data older than 5 seconds is considered **stale** and filtered out

**CPU**: <1ms per `get_all()` call (dict iteration + timestamp comparison).

---

### 6.3 Microstructure Agent

**File**: [`agents/microstructure.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/agents/microstructure.py) (53 lines)

**Purpose**: Extract real-time signals from the order book to predict short-term price direction.

**Input**: `OrderBook` (from PriceCache)
**Output**: `MicrostructureSignal`

#### Computed Metrics

| Metric | Formula | Range | Meaning |
|--------|---------|-------|---------|
| **Spread** | `best_ask - best_bid` | 0-1 | Wider = less liquid |
| **Imbalance** | `(BidVol - AskVol) / (BidVol + AskVol)` (top 3 levels) | -1 to +1 | +1 = strong buy pressure, -1 = strong sell |
| **Weighted Midpoint** | `(Bid × AskVol + Ask × BidVol) / (BidVol + AskVol)` | 0-1 | Fair price accounting for order flow |

**When imbalance exceeds a threshold** (e.g., >0.6), the Navigator logs a critical signal. This can be used to filter which clusters to analyze with the expensive solver.

**CPU**: <0.1ms per call — just 6 float operations.
**Memory**: Zero allocation — reuses the same `MicrostructureSignal` instance.

---

### 6.4 Execution Guard

**File**: [`navigator.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/navigator.py) (lines 120-300)

**Purpose**: The **last line of defense** before a trade is submitted. Pre-flight validation that must complete in <1ms.

#### The Constraint Matrix

On startup, the Navigator builds an inverted index from manifests:

```python
# _constraint_matrix: outcome_id → list of constraints affecting this outcome
{
    "outcome_abc": [
        {"constraint_id": "c1", "coefficient": 1.0, "rhs": 0.0},
        {"constraint_id": "c2", "coefficient": -1.0, "rhs": -1.0},
    ],
    ...
}
```

This enables **O(1) lookup** to find all constraints for any outcome.

#### Validation Checks (in order)

| # | Check | Time |
|---|-------|------|
| 1 | Price bounds: `0 < price < 1` | <0.01ms |
| 2 | Size positive | <0.01ms |
| 3 | Extreme price warning: `< 0.02` or `> 0.98` (likely resolved market) | <0.01ms |
| 4 | Valid side: `buy` or `sell` | <0.01ms |
| 5 | Constraint matrix lookup: does this outcome have constraints? | O(1) dict |
| 6 | Constraint sanity: coefficient/RHS plausibility | O(k) where k = constraints per outcome |

**Total time**: <0.5ms for typical outcomes with 2-3 constraints.

> **Design note**: The heavy constraint validation (LP feasibility) was already done by the ArbitrageDetector. ExecutionGuard is a lightweight sanity check, not a full solver run.

---

### 6.5 Arbitrage Detector (Frank-Wolfe)

**File**: [`solver/fw_solver.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/solver/fw_solver.py) (780 lines)

This is the **mathematical heart** of PolyQuant. It implements the Barrier Frank-Wolfe algorithm to find arbitrage-free prices, then compares them with actual market prices to detect profit opportunities.

#### Class Hierarchy

```
ArbitrageDetector (high-level interface)
  └── FWSolver (algorithm implementation)
        └── SCIPSolver (Linear Minimization Oracle)
```

#### Algorithm Overview

The algorithm finds the **closest arbitrage-free distribution** to current market prices by minimizing KL divergence subject to constraints:

```
minimize    D_KL(μ || θ)    = Σ μ_i · log(μ_i / θ_i)
subject to  A^T × μ ≥ b     (logical constraints from MapMaker)
            μ ∈ Δ            (probability simplex: Σμ_i = 1, μ_i ≥ 0)
```

Where `θ` = current market prices, `μ` = arbitrage-free prices.

**If `D_KL > 0`**, there's arbitrage. The difference `θ - μ` tells you which outcomes are mispriced.

#### Step-by-Step Execution

| Step | Function | What | Time |
|------|----------|------|------|
| 1 | `ArbitrageDetector.detect()` | Entry point, validates inputs | <1ms |
| 2 | `FWSolver.init_fw()` | **Algorithm 3: InitFW** — constructs valid starting vertices Z₀ and interior point u. Runs SCIP feasibility checks to find which outcome assignments satisfy constraints. | 10-50ms (cold), <1ms (Redis cached) |
| 3 | `FWSolver.barrier_fw()` | **Algorithm 2: Barrier Frank-Wolfe** — iteratively minimizes KL divergence. Each iteration: compute gradient → call LMO → update step. | 20-100ms (50-150 iterations) |
| 4 | Profit calculation | Compare optimal `μ*` with market `θ`. If `θ_i > μ*_i`, sell outcome i. If `θ_i < μ*_i`, buy outcome i. | <1ms |
| 5 | Trade construction | Build `ProposedTrade` objects with sizes from Kelly Criterion | <1ms |

**Total time**: 30-150ms per cluster (dominated by Frank-Wolfe iterations).

#### Vectorized Path

For clusters with >5 outcomes, the solver uses **numpy-vectorized** operations:

```python
def _vectorized_gradient(self, mu_vec, theta_vec):
    # ~50× faster than dict loop for 100+ outcomes
    return np.log(mu_vec / theta_vec) + 1.0
```

| Operation | Dict Loop | NumPy | Speedup |
|-----------|-----------|-------|---------|
| Gradient | 2ms | 0.04ms | 50× |
| KL Divergence | 1.5ms | 0.05ms | 30× |
| FW Gap | 1ms | 0.05ms | 20× |
| Step Update | 0.5ms | 0.03ms | 15× |

**CPU**: Medium-High during solving. Each FW iteration is ~0.5-2ms (vectorized) or ~5-10ms (dict-based).

---

### 6.6 SCIP Solver (LMO Oracle)

**File**: [`solver/scip_solver.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/solver/scip_solver.py) (613 lines)

**Purpose**: The Linear Minimization Oracle (LMO) for the Frank-Wolfe algorithm. Called 20-100 times per opportunity detection.

#### What SCIP Does

**SCIP** (Solving Constraint Integer Programs) is the fastest open-source solver for mixed-integer programming. In PolyQuant, it solves:

```
minimize    c^T × z        (gradient direction from Frank-Wolfe)
subject to  A^T × z ≥ b    (logical constraints)
            z ∈ {0, 1}^n   (binary outcomes)
```

This finds the **extreme point** of the constraint polytope in the gradient direction — the core operation of every Frank-Wolfe iteration.

#### Model Persistence (Performance Critical!)

The biggest optimization: **cache the SCIP model** so it's not rebuilt every iteration.

```python
constraint_key = self._get_constraint_hash(validated)

if constraint_key == self._cached_model_key:
    # FAST PATH: Reuse existing model, just update objective
    model = self._cached_model    # ~0ms
else:
    # SLOW PATH: Build new model from scratch
    model = Model("lmo")          # ~5-10ms
    # Add constraints, variables...
    self._cached_model = model
```

**Cache hit rate**: >95% for same cluster (constraints don't change between ticks).

#### Aggressive Tuning

```python
model.setParam("limits/time", 0.01)         # 10ms timeout per LMO call
model.setParam("limits/gap", 0.01)          # 1% optimality gap acceptable
model.setParam("presolving/maxrounds", 0)   # Skip presolve
model.setParam("separating/maxrounds", 1)   # Minimal cut generation
```

This trades 1-2% optimality for **3-5× speed improvement**.

**CPU per LMO call**: 2-10ms (cached), 10-50ms (cold).
**Impact**: Called 20-100× per opportunity → 40-200ms (cached) or 200-5000ms (not cached).

---

### 6.7 Position Sizer (Kelly Criterion)

**File**: [`risk/position_sizing.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/risk/position_sizing.py) (304 lines)

**Purpose**: Calculate optimal trade sizes using the **Modified Kelly Criterion**.

#### The Kelly Formula

```
f* = (p × b - q) / b

Where:
  f* = fraction of capital to bet
  p  = probability of winning
  q  = 1 - p (probability of losing)
  b  = odds (payout ratio)
```

PolyQuant uses **Half Kelly** (fraction = 0.5) for safety:

```python
kelly_fraction: float = 0.5  # Even experts recommend half-Kelly
```

#### Constraint Stack

The position sizer applies **4 limits** and takes the minimum:

| Limit | Default | Purpose |
|-------|---------|---------|
| Kelly Optimal (× fraction) | 50% Kelly | Mathematical optimum |
| Single Trade | 10% of capital | Diversification |
| Total Exposure | 50% of capital | Portfolio protection |
| Order Book Depth | 50% of book | Market impact prevention |

```python
recommended_size = min(kelly_size, single_trade_limit, remaining_exposure, book_limit)
```

**CPU**: <0.1ms per calculation — pure arithmetic.

---

### 6.8 Kill Switch

**File**: [`risk/kill_switch.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/risk/kill_switch.py) (471 lines)

**Purpose**: Emergency halt mechanism that stops all trading when risk thresholds are breached.

#### Trigger Conditions

| Condition | Threshold | What Happens |
|-----------|-----------|-------------|
| **Drawdown** | 15% from high water mark | All trading halts |
| **Solver Timeout** | >50% of solves timeout | Solver is misbehaving |
| **API Errors** | Configurable rate | Connection issues |
| **Latency** | Configurable ms threshold | System too slow |
| **Manual** | UI button or API call | Human override |

#### State Persistence

Kill switch state is saved to Redis so it survives restarts:

```python
await cache.set_kill_switch_state({
    "is_triggered": True,
    "high_water_mark": 10500.0,
    "current_capital": 8925.0,
    "trigger_count": 3,
})
```

**Reset requires explicit confirmation**: `kill_switch.reset(confirm=True)`.

**CPU**: <0.5ms per `can_trade()` check (metric comparison + Redis read).

---

### 6.9 Trade Executor

**File**: [`execution/executor.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/execution/executor.py) (320 lines)

**Purpose**: Execute multi-leg arbitrage trades **atomically** — all legs succeed or all are unwound.

#### Execution Strategy

1. **Sort trades by priority** (lower = illiquid, execute first for fail-fast)
2. **Group by priority** level
3. **Execute each priority group**:
   - Trades within a group execute **in parallel** via `asyncio.gather()`
   - Groups execute **sequentially** (dependent)
4. **If any leg fails**: unwind all previous fills by reversing the trades

```python
# Group trades by priority
sorted_trades = sorted(result.trades, key=lambda t: t.priority)
groups = groupby(sorted_trades, key=lambda t: t.priority)

for priority, group_trades in groups:
    fills = await self._execute_batch(list(group_trades))
    if len(fills) < expected:
        await self._unwind(all_previous_fills)
        return ExecutionResult(success=False, reason="partial_fill")
```

#### Paper vs Live Mode

```python
if self._mode == "paper":
    # Simulate fill at limit price
    fill = Fill(trade=trade, filled_size=trade.size, filled_price=trade.limit_price)
else:
    # Submit to Polymarket CLOB API
    result = await self._client.place_order(trade)
```

**CPU**: <1ms for paper mode. 10-30ms for live mode (dominated by API latency).

---

## 7. Supporting Infrastructure

### 7.1 LLM Client (OpenRouter)

**File**: [`utils/llm_client.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/utils/llm_client.py) (112 lines)

A thin wrapper around the OpenAI SDK pointed at OpenRouter's API:

```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.gemini_api_key.get_secret_value(),
)
```

Key function: `call_llm_json()` — sends a prompt, requests JSON response format, parses the result. Handles markdown code blocks in LLM output.

**Only used by Map Maker agents** — never in the Navigator hot path.

---

### 7.2 Redis Cache

**File**: [`utils/cache.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/utils/cache.py) (~390 lines)

Redis provides:
- `is_market_processed(market_id)` / `mark_market_processed(market_id)` — Skip already-analyzed markets
- `set_solver_result()` / `get_solver_result()` — Persist InitFW results across restarts
- `set_kill_switch_state()` / `get_kill_switch_state()` — Survive crash recovery
- `increment_manifest_version()` — Track manifest generation count
- Heartbeat methods for watchdog

**Connection**: `redis://localhost:6379/0` (configurable via `REDIS_URL`).

---

### 7.3 Monitoring API Server

**File**: [`api/server.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/api/server.py) (141 lines)

A **FastAPI sidecar** that runs alongside the Navigator on port 8000. Provides:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ws` | WebSocket | Real-time state streaming to UI |
| `/status` | GET | Current system state (JSON) |
| `/kill` | POST | Emergency kill switch trigger |

The `Monitor` singleton tracks:
- `status`: OFFLINE / RUNNING / HALTED
- `net_liquidation_value`: Current portfolio value
- `active_solvers`: How many clusters are being analyzed
- `global_latency_ms`: Current tick latency
- `kill_switch_active`: Whether trading is halted
- `clusters[]`: Discovered clusters from MapMaker
- `opportunities[]`: Detected arbitrage (last 10)
- `logs[]`: Last 100 log messages

**Broadcasting** uses `asyncio.gather()` with per-client 1-second timeout to prevent slow clients from blocking the pipeline.

---

### 7.4 Profiling & Watchdog

**File**: [`utils/profiling.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/utils/profiling.py) (~250 lines)

Provides `@async_timed` decorator and `timed_operation` context manager for measuring component latency.

**File**: [`utils/watchdog.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/utils/watchdog.py) (~100 lines)

Monitors process health via Redis heartbeats. If no heartbeat in configurable timeout, the watchdog triggers an alert.

---

## 8. Complete Data Flow Diagrams

### Map Maker (Offline)

```
Polymarket REST API
    │
    ▼
DiscoveryAgent.scan_markets()
    │   Fetches events from /events endpoint
    │   Auto-clusters NegRisk groups (no LLM)
    │   Sends remaining to LLM for clustering
    │
    ▼
list[MarketCluster]          ──── 5-30 clusters, 2-10 markets each
    │
    ├──▶ LogicArchitect.analyze_cluster()
    │       │   LLM extracts A^T z ≥ b constraints
    │       ▼
    │   AnalysisResult
    │       │
    │       ├──▶ ValidatorAgent.validate()
    │       │       │   LLM verifies constraint correctness
    │       │       ▼
    │       │   ValidatedResult
    │       │
    │       └──▶ CorrelationAgent.analyze_pairs()
    │               │   Pearson correlation on price history
    │               ▼
    │           CorrelationSignal[]
    │
    ▼
ConstraintStore.save_manifest()
    │
    ▼
.polyquant/manifests/*.json    ──── Constraints persisted to disk
```

### Navigator (Real-Time)

```
.polyquant/manifests/*.json
    │
    ▼
ConstraintStore.load_all_manifests() ─────▶ ExecutionGuard (in-memory matrix)
                                                    │
Polymarket WebSocket                                │
    │                                               │
    ▼                                               │
PolymarketWSClient._listen()                        │
    │   Receives L2 order book JSON                 │
    │   Parses into OrderBook objects                │
    ▼                                               │
PriceCache.update()                                 │
    │   Stores in dict, sets Event                  │
    ▼                                               │
Navigator: await price_cache.wait_for_update()      │
    │                                               │
    ▼                                               │
price_cache.get_all() ─────▶ Group by cluster ──────┤
    │                                               │
    ├──▶ MicrostructureAgent.analyze()              │
    │       Imbalance, spread, wmid signals         │
    │                                               │
    ├──▶ ArbitrageDetector.detect()                 │
    │       │                                       │
    │       ├── FWSolver.init_fw()                  │
    │       │     SCIP feasibility checks           │
    │       │                                       │
    │       ├── FWSolver.barrier_fw()               │
    │       │     50-150 iterations:                │
    │       │       gradient → LMO → step           │
    │       │                                       │
    │       └── Profit calculation                  │
    │           θ vs μ* comparison                  │
    │                                               │
    ▼                                               │
ArbitrageOpportunity                                │
    │                                               │
    ├──▶ ExecutionGuard.check_trade() ◀─────────────┘
    │       Price bounds, constraint sanity
    │
    ├──▶ PositionSizer.calculate_size()
    │       Kelly Criterion + limits
    │
    ├──▶ KillSwitch.can_trade()
    │       Drawdown, latency, error checks
    │
    └──▶ TradeExecutor.execute_atomic()
            Sort by priority
            Parallel execution within groups
            Unwind on failure
            │
            ▼
        Polymarket CLOB API (or paper simulation)
```

---

## 9. Latency & Performance Budget

### Target: <50ms End-to-End (Tick-to-Trade)

| Component | Time Budget | Actual (Estimated) | Notes |
|-----------|------------|-------------------|-------|
| WebSocket → PriceCache | 1-3ms | ~2ms | JSON parse + dict update |
| Event wake-up → get_all() | <1ms | ~0.5ms | `asyncio.Event` + dict iteration |
| MicrostructureAgent | <1ms | ~0.1ms | 6 float operations |
| ExecutionGuard | <1ms | ~0.5ms | Dict lookup + bounds check |
| ArbitrageDetector (warm) | 20-100ms | ~40ms | FW iterations + cached SCIP |
| PositionSizer | <1ms | ~0.1ms | Kelly arithmetic |
| KillSwitch check | <1ms | ~0.5ms | Metric comparison |
| TradeExecutor (paper) | <1ms | ~0.5ms | Simulated fill |
| TradeExecutor (live) | 10-30ms | ~20ms | CLOB API round trip |
| **Total (paper)** | **<50ms** | **~44ms** | |
| **Total (live)** | **<80ms** | **~64ms** | Network latency dominant |

### CPU Utilization

| Mode | CPU Usage | Notes |
|------|-----------|-------|
| Map Maker | 5-10% | Mostly idle, waiting on LLM API |
| Navigator (idle) | <1% | Blocked on `asyncio.Event` |
| Navigator (active) | 20-40% | FW solver + SCIP is compute-heavy |
| Navigator (peak) | 60-80% | Multiple clusters solving simultaneously |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| PriceCache (200 tokens) | ~100KB | OrderBook objects in dict |
| Manifests (20 clusters) | ~1MB | Loaded from disk on startup |
| ExecutionGuard matrix | ~50KB | Inverted index of constraints |
| SCIP model (cached) | ~5-20MB | Single model instance, reused |
| Python runtime | ~50MB | Base Python + asyncio + Pydantic |
| **Total** | **~70-80MB** | Fits comfortably on 2GB+ instance |

---

## 10. File Index

### Core Modules

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| [`main.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/main.py) | ~220 | 6.5KB | CLI entry point, `map`/`trade` mode routing |
| [`map_maker.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/map_maker.py) | ~500 | 18KB | Offline analysis orchestrator |
| [`navigator.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/navigator.py) | ~730 | 30KB | Real-time trading engine |

### Agents (`agents/`)

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| [`discovery.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/agents/discovery.py) | ~500 | 19KB | 3-Phase market scanning (event-based) |
| [`logic_architect.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/agents/logic_architect.py) | 659 | 25KB | Dependency → constraint matrix conversion |
| [`validator.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/agents/validator.py) | 413 | 15KB | Constraint quality control |
| [`correlation.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/agents/correlation.py) | 66 | 2.4KB | Statistical leader-laggard detection |
| [`microstructure.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/agents/microstructure.py) | 53 | 1.9KB | Order book imbalance analysis |

### Data Layer (`data/`)

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| [`polymarket_client.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/data/polymarket_client.py) | ~900 | 32KB | REST + WebSocket client for all 4 APIs |
| [`market_models.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/data/market_models.py) | 315 | 9.6KB | All Pydantic data models |
| [`constraint_store.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/data/constraint_store.py) | ~300 | 10KB | Manifest persistence (JSON) |
| [`price_cache.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/data/price_cache.py) | 188 | 6KB | In-memory order book cache |
| [`auth.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/data/auth.py) | ~180 | 5.9KB | EIP-712 local signing for Polygon |

### Solver (`solver/`)

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| [`fw_solver.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/solver/fw_solver.py) | 780 | 29KB | Barrier Frank-Wolfe + ArbitrageDetector |
| [`scip_solver.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/solver/scip_solver.py) | 613 | 23KB | SCIP integer programming oracle |

### Risk (`risk/`)

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| [`kill_switch.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/risk/kill_switch.py) | 471 | 16KB | Emergency trading halt |
| [`position_sizing.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/risk/position_sizing.py) | 304 | 10KB | Modified Kelly Criterion |

### Execution (`execution/`)

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| [`executor.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/execution/executor.py) | 320 | 11KB | Atomic multi-leg trade execution |

### API (`api/`)

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| [`server.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/api/server.py) | 141 | 5KB | FastAPI sidecar (WebSocket + REST) |

### Utilities (`utils/`)

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| [`config.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/utils/config.py) | 227 | 7.9KB | Pydantic settings from `.env` |
| [`cache.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/utils/cache.py) | ~390 | 12KB | Redis client wrapper |
| [`llm_client.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/utils/llm_client.py) | 112 | 3.3KB | OpenRouter/OpenAI SDK wrapper |
| [`logging_config.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/utils/logging_config.py) | ~190 | 6KB | Structured logging (structlog) |
| [`profiling.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/utils/profiling.py) | ~250 | 8KB | Latency measurement decorators |
| [`watchdog.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/utils/watchdog.py) | ~100 | 3.2KB | Process health monitoring |
| [`market_utils.py`](file:///d:/GithubLocal/PolyQuant/src/polyquant/utils/market_utils.py) | ~25 | 0.7KB | Helper to extract market ID from outcome ID |

---

### Total Codebase

| Metric | Value |
|--------|-------|
| **Python Files** | 26 |
| **Total Lines** | ~6,700 |
| **Total Size** | ~250 KB |
| **External Dependencies** | pydantic, httpx, websockets, pyscipopt, numpy, openai, redis, fastapi, uvicorn, structlog |

---

*Last updated: 2026-02-27*
