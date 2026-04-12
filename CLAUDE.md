# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolyQuant 2.0 is an autonomous arbitrage extraction system for prediction markets (Polymarket on Polygon, Limitless on Base). It uses a hybrid Python + Rust architecture:

- **Python**: LLM-based market analysis, constraint generation, real-time arbitrage detection
- **Rust (oms-sidecar)**: Low-latency order execution with EIP-712 signing (<10ms overhead)
- **Web (React/Vite)**: Real-time monitoring dashboard

## Build & Run Commands

### Python
```bash
pip install -e ".[dev]"           # Install with dev dependencies
pytest tests/ -v                   # Run tests
pytest tests/test_solvers.py -v   # Run specific test file
ruff format src/ && ruff check src/ --fix  # Format and lint
mypy src/polyquant --strict       # Type checking
```

### Rust (oms-sidecar)
```bash
cd oms-sidecar
cargo build --release             # Build optimized binary
cargo run --release               # Run sidecar
cargo test                        # Run tests
cargo clippy && cargo fmt         # Lint and format
```

### Web UI
```bash
cd web
npm install && npm run dev        # Development server
```

### Running the System
```bash
# 1. Start Rust sidecar (Terminal 1)
cd oms-sidecar && cargo run --release

# 2. Generate constraint manifests (one-time or periodic)
python -m polyquant.main map --limit 500

# 3. Start real-time trading (Terminal 2)
python -m polyquant.main trade
```

## Architecture: Two-Mode System

### Map Maker ("Slow Brain" - Offline)
Generates constraint manifests periodically. Pipeline:
1. **DiscoveryAgent** (Gemini) → Fetch & cluster markets
2. **LogicArchitect** (DeepSeek) → Generate constraint matrices (A^T·z ≥ b)
3. **ValidatorAgent** (Gemini) → Verify mathematical soundness
4. **ConstraintStore** → Persist to `manifests/*.json`

### Navigator ("Fast Brain" - Real-Time)
Executes trades with <50ms latency. No LLM calls - uses pre-computed constraints.
1. Load manifests → WebSocket price feeds → Frank-Wolfe solver
2. Risk checks (kill switch, position limits) → Atomic execution via Rust sidecar

## Key Component Locations

```
src/polyquant/
├── main.py              # Entry point (map/trade modes)
├── map_maker.py         # Offline constraint generation
├── navigator.py         # Real-time trading engine
├── agents/              # LLM agents (Discovery, LogicArchitect, Validator, etc.)
├── solver/
│   ├── fw_solver.py     # Frank-Wolfe barrier method (Kroer et al. 2016)
│   └── scip_solver.py   # Integer programming oracle
├── execution/
│   ├── executor.py      # Multi-leg atomic execution with unwind
│   └── rust_client.py   # ZeroMQ IPC to Rust sidecar
├── risk/
│   ├── kill_switch.py   # Drawdown/latency triggers
│   └── position_sizing.py  # Kelly Criterion + Dutching
└── data/
    ├── polymarket_client.py  # REST + WebSocket
    ├── limitless_client.py   # REST + WebSocket
    └── constraint_store.py   # Manifest persistence

oms-sidecar/src/
├── main.rs              # ZMQ REQ/REP server
├── config.rs            # Shared state, signers
├── journal.rs           # Crash recovery
└── exchanges/
    ├── polymarket.rs    # Polygon EIP-712 signing
    └── limitless.rs     # Base network execution
```

## Python ↔ Rust IPC

Communication via ZeroMQ (tcp://127.0.0.1:5555):
- Python serializes `ProposedTrade` objects to JSON
- Rust sidecar handles EIP-712 signing and blockchain submission
- Returns `ExecutionResponse` with fill details

## Configuration

All config via `.env` (copy from `.env.example`). Critical variables:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | LLM for Discovery/Validator agents |
| `POLYGON_PRIVATE_KEY` | Wallet for Polymarket (Polygon) |
| `BASE_PRIVATE_KEY` | Wallet for Limitless (Base) |
| `TRADING_MODE` | `paper` (simulation) or `live` |
| `EXTRACTION_ALPHA` | Capture % of edge (default 0.9) |
| `MAX_DRAWDOWN` | Kill switch threshold (default 0.15) |
| `SIZING_STRATEGY` | `kelly` or `dutching` |

## Code Patterns

- **Async context managers**: All I/O uses `async with` for proper cleanup
- **Logging**: `structlog` with JSON output - use `from polyquant.utils import get_logger`
- **Config access**: `from polyquant.utils.config import config`
- **Type hints**: Full MyPy strict mode enforced

## Constraint Manifest Format (v1.1)

```json
{
  "cluster_id": "string",
  "market_ids": ["id1", "id2"],
  "market_exchanges": {"id1": "polymarket", "id2": "limitless"},
  "constraints": [{
    "coefficients": {"outcome1": 1.0, "outcome2": 1.0},
    "rhs": 1.0,
    "confidence": 0.95
  }]
}
```

## Dependencies

- **SCIP**: Required for integer programming - ensure `SCIP_HOME` env var is set
- **Redis**: State caching and IPC - must be running (`redis-cli ping`)
- **ZMQ**: Rust sidecar must be running before Navigator starts
