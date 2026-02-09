# PolyQuant Architecture Guide

## Overview

PolyQuant uses a **two-mode architecture** that separates offline analysis (slow, thorough) from real-time trading (fast, optimized):

```
┌─────────────────────────────────────────────────────────────┐
│                     PolyQuant 2.0                           │
│                                                             │
│  ┌──────────────────┐              ┌──────────────────┐   │
│  │   MAP MAKER      │              │    NAVIGATOR     │   │
│  │  (Slow Brain)    │──────────────▶│  (Fast Brain)   │   │
│  │                  │   Manifests   │                  │   │
│  └──────────────────┘              └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Mode 1: Map Maker (Offline - "Slow Brain")

**Purpose**: Analyze market structures and generate constraint manifests

**What it does**:
1. **Discovery**: Fetches all active markets from Polymarket
2. **Clustering**: Groups related markets (e.g., "Trump wins PA" + "GOP wins PA")
3. **Logic Analysis**: Uses LLMs (Gemini) to identify logical dependencies
4. **Validation**: Verifies constraints are mathematically sound
5. **Persistence**: Saves validated constraint manifests to disk

**When to run**:
- First time setup
- Periodically (e.g., hourly) to discover new markets
- After major market events that create new dependencies

**How to run**:
```bash
python -m polyquant.main map
```

**Output**:
- Constraint manifests saved to `./manifests/` directory
- JSON files containing market dependencies and constraints

**Latency**:
- Not critical (can take minutes)
- Uses LLMs for deep analysis

---

## Mode 2: Navigator (Real-Time - "Fast Brain")

**Purpose**: Execute trades in real-time with <50ms latency

**What it does**:
1. **Load Manifests**: Reads pre-computed constraints from disk
2. **WebSocket Connection**: Connects to Polymarket for real-time prices
3. **Opportunity Detection**: Runs Frank-Wolfe solver on price updates
4. **Execution**: Submits trades when profitable opportunities found
5. **Risk Management**: Monitors kill switch, position sizing, latency

**When to run**:
- After MapMaker has generated manifests
- Runs continuously (24/7)

**How to run**:
```bash
python -m polyquant.main trade
```

**Latency Requirements**:
- Tick-to-Decision: <10ms
- Decision-to-Execution: <30ms
- **Total Target: <50ms**

**Key Features**:
- No LLM calls (all logic pre-computed)
- In-memory constraint checking (ExecutionGuard)
- WebSocket for real-time prices
- Atomic trade execution

---

## Typical Workflow

### Initial Setup
```bash
# 1. Build the constraint map (first time)
python -m polyquant.main map

# 2. Start trading
python -m polyquant.main trade
```

### Production Operation
```bash
# Terminal 1: Run MapMaker hourly (cron job)
0 * * * * python -m polyquant.main map

# Terminal 2: Run Navigator continuously
python -m polyquant.main trade
```

---

## Component Architecture

### Map Maker Components
```
MapMaker
├── DiscoveryAgent (Gemini 2.0 Flash)
│   └── Scans Polymarket API for active markets
├── LogicArchitect (Gemini 2.0 Flash Thinking)
│   └── Analyzes logical dependencies between markets
├── ValidatorAgent (Gemini 2.0 Flash Thinking)
│   └── Validates constraint matrices
└── ConstraintStore
    └── Persists manifests to disk
```

### Navigator Components
```
Navigator
├── ConstraintStore
│   └── Loads pre-computed manifests
├── ExecutionGuard
│   └── Fast in-memory constraint validation (O(1))
├── PriceCache
│   └── Real-time price updates from WebSocket
├── ArbitrageDetector
│   └── Detects opportunities using Frank-Wolfe solver
├── PositionSizer
│   └── Kelly Criterion for position sizing
├── KillSwitch
│   └── Risk management (drawdown, latency limits)
└── Executor
    └── Atomic multi-leg trade execution
```

---

## File Structure

```
src/polyquant/
├── main.py                    # Entry point (map/trade modes)
├── map_maker.py               # Offline constraint generation
├── navigator.py               # Real-time trading engine
│
├── agents/                    # LLM-based agents (MapMaker only)
│   ├── discovery.py
│   ├── logic_architect.py
│   └── validator.py
│
├── solver/                    # Optimization engines
│   ├── fw_solver.py          # Frank-Wolfe + Bregman projection
│   └── scip_solver.py        # Integer programming oracle
│
├── risk/                      # Risk management
│   ├── kill_switch.py
│   └── position_sizing.py
│
├── execution/                 # Trade execution
│   └── executor.py
│
├── data/                      # Data models and clients
│   ├── polymarket_client.py
│   ├── price_cache.py
│   └── constraint_store.py
│
└── utils/                     # Shared utilities
    ├── config.py
    └── logging_config.py
```

---

## Data Flow

### Map Maker Flow (Offline)
```
Polymarket API
    ↓
DiscoveryAgent (Gemini) → Market Clusters
    ↓
LogicArchitect (Gemini) → Dependencies + Constraints
    ↓
ValidatorAgent (Gemini) → Validated Constraints
    ↓
ConstraintStore → manifests/*.json (saved to disk)
```

### Navigator Flow (Real-Time)
```
ConstraintStore → Load manifests → ExecutionGuard (in-memory)
                                           ↓
Polymarket WebSocket → PriceCache → ArbitrageDetector
                                           ↓
                                    Opportunity Found?
                                           ↓
                                    PositionSizer
                                           ↓
                                    KillSwitch Check
                                           ↓
                                    Executor → Trade Submitted
```

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| MapMaker Total Time | <10 min for 100 markets | TBD |
| Navigator Latency | <50ms end-to-end | TBD |
| Navigator Uptime | >99.5% | TBD |
| Constraint Load Time | <1 second | TBD |
| Solver Convergence | <150 iterations | Implemented |

---

## Migration from Legacy Orchestrator

**Old Way (Deprecated)**:
```python
from polyquant.legacy_orchestrator import PolyQuantOrchestrator

async with PolyQuantOrchestrator() as orch:
    await orch.run_pipeline()
```

**New Way**:
```python
from polyquant.main import run_map_maker, run_navigator

# Build map
await run_map_maker()

# Trade
await run_navigator()
```

**Why the change?**
- **Separation of Concerns**: Offline analysis vs. real-time trading
- **Performance**: Navigator doesn't call LLMs (pre-computed logic)
- **Reliability**: Manifests are immutable once generated
- **Scalability**: Can run multiple Navigators with same manifests

---

## Configuration

Key environment variables (`.env`):

```bash
# Polymarket API
POLYMARKET_API_KEY=your_key
POLYMARKET_API_SECRET=your_secret
POLYGON_PRIVATE_KEY=your_private_key

# Trading Mode
TRADING_MODE=paper  # or "live"

# Risk Management
MAX_DRAWDOWN=0.15  # 15%
MAX_POSITION_SIZE=0.25  # 25% of capital
VWAP_SLIPPAGE_LIMIT=0.05  # 5%

# Solver
FW_MAX_ITERATIONS=150
FW_ALPHA_EXTRACTION=0.9
FW_MIN_PROFIT=0.05

# LLM (for MapMaker)
GEMINI_API_KEY=your_gemini_key
```

---

## Next Steps

See [REFACTORING_PLAN.txt](./REFACTORING_PLAN.txt) for the full implementation roadmap, including:

- Phase 1: Critical bug fixes (WebSocket, Frank-Wolfe epsilon, execution)
- Phase 2: Architecture improvements (this document)
- Phase 3: Performance optimizations (vectorization, caching)
- Phase 4: Risk management enhancements
- Phase 5: Testing and validation

---

## Questions?

- MapMaker not finding markets? Check Polymarket API connectivity
- Navigator failing to load manifests? Run MapMaker first
- Trades not executing? Check `TRADING_MODE` is set correctly
- High latency? Profile WebSocket connection and solver performance
