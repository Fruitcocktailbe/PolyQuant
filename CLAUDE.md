# PolyQuant 2.0

Autonomous arbitrage extraction on prediction markets (Polymarket + Limitless) via multi-agent LLM reasoning and mathematical optimization.

**Stack**: Python 3.11+ (core), Rust (OMS sidecar), React/Vite (dashboard)

## Quick Commands

```bash
# Run
python -m polyquant.main map [--limit 500] [--min-liquidity 1000] [--force]   # Build constraint manifests (offline)
python -m polyquant.main trade                                                  # Real-time trading (online)

# Test & Lint
pytest tests/
ruff check src/
black --check src/
mypy src/

# Rust OMS sidecar
cd oms-sidecar && cargo build --release

# Web dashboard
cd web && npm install && npm run dev   # http://localhost:5173
```

## Architecture: Two-Brain System

**Map Maker (Slow Brain)** — offline, LLM-driven, runs periodically:
```
Polymarket API → DiscoveryAgent → MarketCluster
                                      ↓
                              LogicArchitect → AnalysisResult
                                      ↓
                              ValidatorAgent → ValidatedResult
                                      ↓
                              ConstraintStore → .polyquant/constraints/{cluster_id}.json
```

**Navigator (Fast Brain)** — online, no LLMs, <50ms latency:
```
WebSocket Feed → PriceCache → BayesianUpdater → FWSolver → SCIPSolver → TradeExecutor → Rust OMS
                 (in-memory)   (phantom-arb       (optimize)  (sizing)    (atomic exec)   (<5ms ZMQ)
                                prevention)
```

Bridge between brains: JSON constraint manifests on disk (`.polyquant/constraints/`)

## Project Structure

```
src/polyquant/
├── main.py              # Entry point: "map" and "trade" modes
├── map_maker.py         # Orchestrates offline constraint pipeline
├── navigator.py         # Real-time trading loop
├── agents/
│   ├── discovery.py         # Market scanning, clustering, zombie filtering
│   ├── logic_architect.py   # LLM constraint extraction (DeepSeek/Gemini)
│   ├── validator.py         # Mathematical + liquidity validation (Gemini Flash)
│   ├── bayesian_updater.py  # Adjusts correlated prices to prevent phantom arb
│   ├── correlation.py       # Historical price correlation detection
│   ├── exchange_matcher.py  # Match Polymarket ↔ Limitless markets (TF-IDF + LLM)
│   └── microstructure.py    # Order book imbalance/spread analysis
├── solver/
│   ├── fw_solver.py             # Barrier Frank-Wolfe optimization (Algorithm 2 & 3)
│   ├── fw_solver_dutching_new.py # Dutching-specific solver variant
│   └── scip_solver.py          # SCIP integer programming (LMO, feasibility oracle)
├── data/
│   ├── market_models.py     # Pydantic models: Market, Outcome, OrderBook, ProposedTrade, etc.
│   ├── polymarket_client.py # Async client for Gamma/CLOB/Data/WebSocket APIs
│   ├── limitless_client.py  # Async client for Limitless REST/WebSocket APIs
│   ├── constraint_store.py  # JSON persistence for constraint manifests
│   ├── price_cache.py       # In-memory order book cache with staleness detection
│   ├── auth.py              # Authentication helpers
│   └── trade_store.py       # Trade history persistence
├── execution/
│   ├── executor.py      # Atomic multi-leg execution with unwind logic
│   └── rust_client.py   # ZMQ bridge to Rust OMS sidecar
├── risk/
│   ├── kill_switch.py      # Emergency halt (drawdown, latency, toxic flow)
│   └── position_sizing.py  # Modified Kelly + dutching sizing
├── utils/
│   ├── config.py          # Pydantic Settings (all config from .env)
│   ├── llm_client.py      # OpenRouter wrapper (free models)
│   ├── logging_config.py  # structlog setup (dev console / prod JSON)
│   ├── cache.py           # Redis integration
│   ├── profiling.py       # @timed_operation / @async_timed decorators
│   ├── market_utils.py    # Helper functions
│   └── watchdog.py        # Monitoring utility
└── api/                   # FastAPI monitoring server
```

Other top-level directories:
- `oms-sidecar/` — Rust OMS (Tokio, ZeroMQ, Alloy for Base network)
- `web/` — React/Vite dashboard (Recharts, Tailwind, Framer Motion)
- `tests/` — pytest suite (asyncio_mode="auto")
- `scripts/` — Utilities (benchmark signing, verify credentials, test latency)
- `.polyquant/` — Runtime cache (constraint manifests, reports)

## Coding Conventions

- **Type hints**: Strict mypy. Use `str | None` not `Optional[str]`. Generic types: `list[str]`, `dict[str, Any]`
- **Models**: Pydantic v2 BaseModel with Field constraints (`ge=`, `le=`, `description=`)
- **Logging**: structlog — `from polyquant.utils.logging_config import get_logger; logger = get_logger(__name__)`
- **Async**: async/await throughout all I/O. Use `httpx.AsyncClient` not `requests`
- **Formatting**: ruff (line-length 100, py311) + black. Rules: E, F, I, N, W, UP
- **Naming**: PascalCase classes, snake_case functions/variables, UPPER_SNAKE constants, _leading_underscore private
- **Docstrings**: Google-style for public classes/methods
- **Config**: All settings via environment variables loaded through Pydantic Settings (`utils/config.py`). Secrets use `SecretStr`. Singleton via `@lru_cache(maxsize=1)`

## IDE False Positives (Ignore These)

- **ALL** Pydantic BaseModel field errors ("Unexpected keyword argument in `object.__init__`") — false positives
- Type inference errors on dict/list/float operations (`+=`, `-` operators) — false positives
- Import resolution errors for `polyquant.*` modules — false positives
- These never affect runtime

## Key Patterns

### Solver Constraints
- Frank-Wolfe solver expects **A^T * z >= b** format
- `LogicalConstraint` with `<=` must be converted to `>=` by **negating both coefficients AND rhs**
- NegRisk markets: outcome prices should sum to 1.0; deviations signal arbitrage
- Tolerances: `TIGHT_TOLERANCE = 0.02` (2%), `LOOSE_TOLERANCE = 0.05` (5%)

### Execution Safety
- Net profit = gross - (taker_fee + gas + VWAP_slippage); abort if <= 0
- Order lifecycle: submit → poll 5s for confirmation → cancel if not confirmed (prevents ghost orders)
- Unwind on partial failure: 3 retries with widening spread (3% → 6% → 9%) → kill switch
- Trades sorted by priority (illiquid legs first), parallel within priority groups

### HTTP Resilience
- CLOB API: 5s timeout, Gamma API: 10s timeout
- 3 retries with exponential backoff
- 429 rate limit detection with Retry-After header
- Permanent errors (400, 401, 403, 404): no retry

### Bayesian Price Adjustment
- Prevents phantom arbitrage from price lag between correlated markets
- Max adjustment: 20% of original price
- Confidence-weighted pass-through (weak dependencies = 50%)
- Only adjusts if delta > $0.005

## External APIs

| API | Base URL | Purpose |
|-----|----------|---------|
| Polymarket Gamma | `gamma-api.polymarket.com` | Market discovery, metadata |
| Polymarket CLOB | `clob.polymarket.com` | Order books, prices, trading |
| Polymarket Data | `data-api.polymarket.com` | Positions, activity |
| Polymarket WS | `ws-subscriptions-clob.polymarket.com` | Real-time price feeds |
| Limitless REST | `api.limitless.exchange` | Markets, order books, trading |
| Limitless WS | `stream.limitless.exchange` | Real-time order books |
| OpenRouter | `openrouter.ai/api/v1` | LLM routing (Gemini, DeepSeek) |
| Redis | `localhost:6379` | Caching, kill switch state, solver results |

## Safety Mechanisms

**Kill Switch** (`risk/kill_switch.py`) triggers on:
- Drawdown > 15% (`MAX_DRAWDOWN`)
- >50% solver timeouts in 5 minutes
- Average latency > 3x target (90ms when target is 30ms)
- >3 fills in <500ms (toxic flow / faster competitors)
- Persists to Redis across restarts

**Other safeguards**:
- Dead man's switch: halt if WebSocket data > 200ms old
- Order expiration: 6s TTL (~3 Polygon blocks)
- VWAP slippage limit: abort if > $0.05
- Position limits: 5% single trade, 25% total exposure, 50% of order book depth
- Atomic execution: all-or-nothing multi-leg trades

## Environment Setup

1. Copy `.env.example` → `.env`
2. Required keys: `GEMINI_API_KEY`, `POLYGON_PRIVATE_KEY`, `ALCHEMY_API_KEY`
3. Optional: `BASE_PRIVATE_KEY`, `LIMITLESS_API_KEY` (for Limitless exchange)
4. Set `TRADING_MODE=paper` for testing, `live` for production
5. Redis must be running for caching (`REDIS_URL=redis://localhost:6379/0`)

## Testing

```bash
pytest tests/                          # Full suite
pytest tests/test_solvers.py           # Solver tests
pytest tests/test_dutching.py          # Dutching strategy
pytest tests/test_integration_check.py # Integration tests
```

Test config: `asyncio_mode = "auto"` in pyproject.toml — no need for `@pytest.mark.asyncio`.
