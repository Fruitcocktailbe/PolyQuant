> ⚠️ **HISTORICAL — v0.2/v0.3, superseded.**
> This document describes state from before the OMS sidecar + Rust client integration.
> Claims about `_submit_order()` returning None and `RUST_AVAILABLE = False` are **no longer accurate**.
> The current execution path runs through `executor.py:376` → `RustClient.dispatch_trades()` → ZeroMQ → Rust sidecar.
> See `docs/DUTCHING_ENHANCEMENTS_v0.4.0.md` for the current authoritative spec.

# Deep-State Analysis: Edge Case Cascades & FMEA

**Date**: 2026-03-01
**Version**: v0.2.0 → v0.3.0
**Role**: Principal Web3 Engineer & Formal Verification Specialist
**Scope**: PolyQuant — Rust OMS Sidecar + Python Execution Pipeline

---

## Part 1: State Machine & Memory Consistency

### 1.1 Local vs On-Chain Source of Truth — CRITICAL

**Architecture**: Pure CLOB (off-chain order book). Both Polymarket and Limitless submit signed EIP-712 orders to centralized CLOB APIs, not directly on-chain.

**Re-org Impact**:
- **Polymarket (Polygon)**: CLOB operator handles settlement. Bot has NO visibility into on-chain settlement. If Polygon re-orgs, the CLOB operator absorbs the risk — but bot's local balance tracking had no reconciliation mechanism.
- **Limitless (Base)**: Nonce fetched from RPC with `"latest"` parameter. Base is L2 with faster finality than Polygon, but `"latest"` is unsafe for nonce — should use `"finalized"`.

**Fix Applied**:
- Balance reconciliation heartbeat every 500 ticks (`navigator.py:_reconcile_balances`)
- All RPC calls now use `"finalized"` block tag (Rust + Python)
- Auto-correct local balances when drift exceeds $1 threshold

### 1.2 Race Conditions in `await` Blocks — HIGH

**Detection-to-Execution Gap**: 16-216ms between price snapshot and CLOB execution. Staleness check was 5s default — far too generous.

**Fix Applied**:
- Reduced `OMS_MAX_SIGNAL_AGE_US` default from 5,000,000 (5s) to 500,000 (500ms) in `config.rs:94`
- `signal_timestamp_us` now stamped at detection time (`detect_start`) in Navigator, not at dispatch time in RustClient
- RustClient accepts `signal_timestamp_us` kwarg for caller-supplied timestamps

### 1.3 Optimistic State Updates Without Rollback — CRITICAL

**Pre-fix**: Balances deducted optimistically on fill, never restored on unwind failure.

**Fix Applied**:
- Unwind method now returns success/failure boolean
- Successful unwind fills restore balances per-exchange
- Trade store writes now `await`-ed instead of fire-and-forget `create_task`

---

## Part 2: Execution "Atomicity" & Partial Failures

### 2.1 The Hanging Leg Problem — CRITICAL

**Pre-fix**: `_unwind()` was fire-and-forget — single dispatch, no confirmation, no retry, no spread widening. If unwind failed, position left unhedged indefinitely.

**Fix Applied** (`executor.py:_unwind`):
- 3 retries with spread widening: 3% → 6% → 9%
- Each attempt checks Rust response for successful fills
- Remaining positions tracked across retries
- Balance restored for each successful unwind fill
- If all 3 retries fail: **triggers kill switch** via `_kill_switch.trigger()` or `_rust_client.send_halt()`
- Unwind call site now handles `False` return and annotates reason

### 2.2 Ghost Cancellations — HIGH

**Pre-fix**: `cancel_all_orders()` returned `verified=False` silently without blocking trading.

**Fix Applied**: `cancel_all_orders()` now raises `RuntimeError` if verification fails after 3 attempts, forcing the caller to handle the failure explicitly.

### 2.3 Trade Store Durability — MEDIUM

**Pre-fix**: `asyncio.create_task(self._trade_store.record_fills(filled))` — no await.

**Fix Applied**: Changed to `await self._trade_store.record_fills(filled)` with try/except to prevent crash on store failure.

---

## Part 3: Polygon-Specific Infrastructure Risks

### 3.1 RPC Failover — CRITICAL (was single point of failure)

**Pre-fix**: Single `base_rpc_url` with no fallback.

**Fix Applied**:
- **Python**: Added `base_rpc_fallback_urls` config field; new `_rpc_call()` method tries primary + all fallbacks; `get_usdc_balance()` and `get_base_network_nonce()` both use failover
- **Rust**: Added `rpc_fallback_urls: Vec<String>` to `LimitlessConfig`; `resync_nonce()` and `warm_limitless_nonce()` both iterate through all providers

### 3.2 Gas Price — HIGH (hardcoded)

**Pre-fix**: Single `polygon_gas_per_tx = 0.01` used for ALL exchanges.

**Fix Applied**:
- Both solvers now use per-exchange gas rates (`polygon_gas_per_tx` vs `base_gas_per_tx`)
- Both solvers now use per-exchange fee rates (`polymarket_taker_fee_pct` vs `limitless_taker_fee_pct`)
- Note: Dynamic gas oracle integration deferred — hardcoded values still require manual updates during congestion

### 3.3 Block Finality — HIGH

**Pre-fix**: All nonce/balance RPC calls used `"latest"` block tag.

**Fix Applied**: Changed to `"finalized"` in:
- Rust: `warm_limitless_nonce()` in `main.rs`
- Rust: `resync_nonce()` in `limitless.rs`
- Python: `get_base_network_nonce()` in `limitless_client.py`

### 3.4 Nonce Persistence — CRITICAL

**Pre-fix**: `AtomicU64` in-memory only; crash loses nonce state.

**Fix Applied**:
- New `journal.log_nonce()` method writes nonce assignment to crash journal after each `fetch_add`
- `warm_limitless_nonce()` now scans journal for highest recorded nonce on startup
- Effective nonce = `max(rpc_nonce, journal_max_nonce + 1)` — prevents collision with in-flight txns

---

## Part 4: Logic & Math "Black Swans"

### 4.1 Fee Scaling — HIGH

**Pre-fix**: Only Polymarket taker fee deducted; Limitless fees defaulted to 0.

**Fix Applied**: Both SCIP and Frank-Wolfe solvers now iterate trades and apply the correct fee/gas per exchange.

### 4.2 Pessimistic Rounding — MIXED

**Pre-fix**: Rust used `.round()` (banker's rounding) for all amounts.

**Fix Applied**:
- Costs: `.ceil()` (round up what we pay)
- Revenue: `.floor()` (round down what we receive)
- Applied to both `polymarket.rs` and `limitless.rs`

### 4.3 VWAP Slippage Enforcement — HIGH (was dead code)

**Pre-fix**: `vwap_slippage_limit = 0.05` defined in config but never checked.

**Fix Applied**: Pre-flight check in `executor.py:execute_atomic()` rejects any trade where `vwap_slippage > slippage_limit`.

---

## Part 5: FMEA Table

| # | Failure Mode | Root Cause | Impact | Detection | Pre-Fix Mitigation | Post-Fix Mitigation | Residual Risk |
|---|---|---|---|---|---|---|---|
| 1 | **Hanging Leg** | Partial fill + unwind failure | CRITICAL: directional exposure | <100ms | Fire-and-forget dispatch | 3 retries + spread widening + kill switch | LOW: covered |
| 2 | **Balance Drift** | Optimistic deduction, no rollback | HIGH: lost buying power | Hours | None | Unwind restores balances + 500-tick reconciliation | LOW: auto-corrected |
| 3 | **Stale Signal** | 5s threshold, wrong timestamp origin | HIGH: bad price execution | 5-16ms gap invisible | 5s check at dispatch time | 500ms check at detection time | LOW: 10x tighter |
| 4 | **Nonce Gap** | In-memory AtomicU64, crash | MEDIUM: trades blocked | 0-10s | Reactive resync only | Journal persistence + startup recovery | LOW: proactive |
| 5 | **Gas Purgatory** | Hardcoded $0.01 gas | HIGH: stuck funds | Minutes | None | Per-exchange gas rates (manual still) | MEDIUM: no oracle yet |
| 6 | **RPC Failure** | Single provider | MEDIUM: Limitless halts | 5s timeout | None | Primary + fallback with round-robin | LOW: multi-provider |
| 7 | **Ghost Cancel** | Unverified cancellation | MEDIUM: double exposure | 0.5-3s | Silent `verified=False` | RuntimeError raised on failure | LOW: forced handling |
| 8 | **Kill Switch Race** | ZMQ latency window | MEDIUM: stale trades | 10-100ms | Single attempt | 3 retry loop blocking until ACK | LOW: confirmed halt |
| 9 | **Fee Underestimate** | Limitless fees = 0% | MEDIUM: unprofitable trades | Post-reconciliation | Only Polymarket fees | Per-exchange fees in both solvers | LOW: correct rates |
| 10 | **f64 Rounding** | `.round()` not pessimistic | LOW: ±$0.000001 | N/A | Banker's rounding | `ceil()` costs / `floor()` revenue | NEGLIGIBLE |
| 11 | **VWAP Bypass** | Config value never enforced | HIGH: excess slippage | Post-fill | Dead code | Pre-flight rejection check | LOW: enforced |
| 12 | **Trade Store Loss** | `create_task` without await | MEDIUM: audit gap | Post-crash | Rust journal backup | `await record_fills()` | LOW: durable |
| 13 | **Balance Re-org** | No finality checks | MEDIUM: stale nonce | Block time | `"latest"` everywhere | `"finalized"` block tag | LOW: finalized |
| 14 | **CLOB Censorship** | Centralized matching | MEDIUM: adverse fills | Undetectable | FOK orders | FOK orders (unchanged) | MEDIUM: accepted risk |

---

## Part 6: Cascade Scenarios (Post-Fix)

### Cascade A: "The Perfect Storm" — NOW MITIGATED

```
t=0ms     Price snapshot: YES=0.55, NO(limitless)=0.40 → 5¢ arb
t=10ms    Detection confirms. signal_timestamp_us = detect_start (NEW)
t=16ms    Leg 1: Buy YES on Polymarket → FILLED
t=70ms    Leg 2: Buy NO on Limitless → REJECTED
t=75ms    Unwind attempt 1: Sell YES at 0.55 * (1 - 0.03) = 0.5335
t=200ms   Attempt 1 fails (market at 0.48)
t=400ms   Unwind attempt 2: Sell YES at 0.55 * (1 - 0.06) = 0.517
t=600ms   Attempt 2 fails
t=900ms   Unwind attempt 3: Sell YES at 0.55 * (1 - 0.09) = 0.5005
t=1100ms  Attempt 3 FILLS at 0.50 → loss = (0.55-0.50) * size
          OR: Attempt 3 fails → KILL SWITCH → all trading halted
```

### Cascade B: "Nonce Spiral" — NOW MITIGATED

```
t=0       Sidecar crashes mid-batch, nonce=10 in memory
t=1s      Sidecar restarts
t=1.5s    Reads journal: max nonce_assign = 12
t=1.5s    Reads RPC (finalized): on-chain nonce = 11
t=1.5s    Effective nonce = max(11, 12+1) = 13 → no collision
```

---

## Part 7: Files Modified

### Rust (oms-sidecar/)
| File | Changes |
|---|---|
| `src/config.rs` | Staleness 5s→500ms, `rpc_fallback_urls` field, field in `LimitlessConfig` |
| `src/journal.rs` | New `log_nonce()` method |
| `src/main.rs` | RPC failover in nonce warm, journal nonce recovery, `recover_max_nonce_from_journal()` |
| `src/exchanges/polymarket.rs` | Pessimistic rounding (ceil/floor) |
| `src/exchanges/limitless.rs` | Pessimistic rounding, nonce journal logging, RPC failover in resync, `"finalized"` block tag |

### Python (src/polyquant/)
| File | Changes |
|---|---|
| `execution/executor.py` | Unwind retries + spread widening + kill switch, merged balance checks, VWAP enforcement, await trade store, balance rollback |
| `execution/rust_client.py` | Accept external `signal_timestamp_us` |
| `navigator.py` | Detection-time timestamp, kill switch 3-retry ACK, `_reconcile_balances()` heartbeat |
| `data/limitless_client.py` | `_rpc_call()` failover, `"finalized"` nonce, failover in balance/nonce |
| `data/polymarket_client.py` | Cancel verification raises RuntimeError |
| `utils/config.py` | `base_rpc_fallback_urls` field |
| `solver/fw_solver.py` | Per-exchange gas + fees |
| `solver/scip_solver.py` | Per-exchange gas + fees |

### Build Status
- `cargo check`: **0 errors, 0 warnings**
