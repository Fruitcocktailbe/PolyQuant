# OMS Sidecar Audit & Fix Report

**Date**: 2026-03-01
**Version**: v0.1.0 → v0.2.0
**Scope**: Rust OMS Sidecar + Python execution pipeline

---

## Part 1: Audit Findings

### Codebase Summary (Pre-Fix)

| Metric | Value |
|---|---|
| Rust source files | 5 (`main.rs`, `lib.rs`, `models.rs`, `polymarket.rs`, `limitless.rs`) |
| Rust LoC | ~320 |
| Dependencies | 13 crates |
| IPC mechanism | ZeroMQ REP/REQ over TCP (localhost:5555) |
| Serialization | JSON (`serde_json`) |
| Signing library | `alloy` 0.7.1 |
| Async runtime | `tokio` 1.40 (full) |
| Cargo edition | `"2024"` (INVALID - does not exist in stable Rust) |

### 1. Rust Memory Safety & Concurrency

**State Management**: The sidecar was fully stateless (no `Arc`, `Mutex`, or `Atomic` types). Each trade was processed in isolation via `tokio::task::spawn`. No deadlock risk, but also no local order tracking for crash recovery.

**Serialization**: JSON everywhere on the hot path (Python→Rust, Rust→CLOB, Rust→Python). No `bincode` or `protobuf`. Kept as-is per user decision (latency dominated by HTTP round-trip).

**Panic Handling**:
- `msg.get(0).unwrap()` in main.rs — crashes if ZMQ delivers empty multipart message
- `serde_json::to_string(&response).unwrap()` — crashes on serialization failure
- No `catch_unwind` anywhere
- FOK order semantics mitigate orphaned orders on crash

### 2. Hot Path Execution

**Per-Trade Initialization** (CRITICAL):
- `env::var()` x5 called per trade (~5-20us each)
- `PrivateKeySigner::from_str()` per trade (~50-100us)
- `Client::builder().build()` per trade (~100-500us, loses connection pooling)
- `eip712_domain!{}` reconstructed per trade (~10-50us)
- HMAC secret base64-decoded per trade

**Limitless Nonce RPC** (CRITICAL):
- `eth_getTransactionCount` HTTP call per trade (+50-200ms)
- Race condition: concurrent trades read same nonce
- Semantically questionable (on-chain tx nonce vs CLOB order nonce)

**Amount Calculation** (HIGH):
- `(price * size * 1e6) as u64` truncates instead of rounding
- Example: `0.55 * 100.0 * 1e6 = 54999999.99999999` → truncated to `54999999` (should be `55000000`)

**Salt Predictability** (MEDIUM):
- `SystemTime::nanos as u64` — deterministic, not random

### 3. IPC & State Synchronicity

**Stale Signal**: No timestamp field on `ProposedTrade`. If Python→Rust delivery delayed 10-100ms, sidecar executes at stale prices.

**Backpressure**: ZMQ REP/REQ provides natural backpressure (request-response). No unbounded queuing risk. Head-of-line blocking on slow trades.

### 4. Polygon/Polymarket Specifics

**MEV Protection**: Not applicable — sidecar submits to CLOB APIs, not on-chain. Settlement is the CLOB operator's responsibility.

**Nonce Management**: Polymarket correctly uses nonce=0 (offchain CLOB). Limitless incorrectly fetched on-chain nonce per trade.

### 5. Python-Side Issues

**Placeholder Bug** (CRITICAL): `executor.py:_execute_batch()` created placeholder `Fill` objects with `order_id="rust_oms_placeholder"`, ignoring actual Rust response data (filled sizes, prices, order IDs).

**No Kill Switch Integration**: Kill switch only set `_is_running=False` in Python. Rust sidecar was unaware and could continue executing trades.

---

## Part 2: Execution Bottleneck Report (Pre-Fix)

| # | Component | Logic/Code Issue | Latency Penalty (Est.) | Severity |
|---|---|---|---|---|
| 1 | Config Loading | `env::var()` x5 per trade | +5-20us | CRITICAL |
| 2 | Signer Init | `PrivateKeySigner::from_str()` per trade | +50-100us | CRITICAL |
| 3 | HTTP Client | `Client::builder().build()` per trade | +100-500us | CRITICAL |
| 4 | Nonce RPC | Limitless `eth_getTransactionCount` per trade | +50-200ms | CRITICAL |
| 5 | Signal Staleness | No timestamp validation | Unbounded | CRITICAL |
| 6 | Amount Calc | f64 truncation | +/-1-100 microUSDC | CRITICAL |
| 7 | EIP-712 Domain | Recomputed per trade | +10-50us | HIGH |
| 8 | Panic Safety | `.unwrap()` on ZMQ frame | Process crash | HIGH |
| 9 | HTTP Response | No status code check | Silent failures | HIGH |
| 10 | Error Reporting | Failed trades silently dropped | N/A (correctness) | HIGH |
| 11 | Salt | Deterministic nanosecond timestamp | N/A (security) | HIGH |
| 12 | Python Fills | Placeholder objects ignore Rust data | N/A (correctness) | HIGH |
| 13 | TCP_NODELAY | Not set on reqwest client | +0-40ms (Nagle) | MEDIUM |
| 14 | HMAC Serialize | Double serialization of body | +50-100us | MEDIUM |
| 15 | Tokio Runtime | Multi-threaded for I/O-bound work | Scheduling overhead | MEDIUM |

### Estimated Latency Budget (Per Trade, Pre-Fix vs Post-Fix)

| Phase | Pre-Fix (Est.) | Post-Fix (Est.) |
|---|---|---|
| ZMQ recv + JSON parse | 100-200us | 100-200us (kept JSON) |
| Env var + signer + client init | 200-600us | 0us (cached at startup) |
| EIP-712 domain + sign | 50-150us | 10-50us (cached domain) |
| Nonce RPC (Limitless only) | 50-200ms | 0ms (AtomicU64) |
| HMAC computation (Polymarket) | 50-100us | 5-20us (pre-decoded, single serialize) |
| HTTP POST to CLOB | 50-200ms | 50-200ms (irreducible) |
| Response parse + ZMQ reply | 50-100us | 50-100us |
| **Total** | **~100-400ms** | **~50-200ms** |

---

## Part 3: Implementation Plan (Completed)

### Files Modified/Created

| File | Action | Key Changes |
|------|--------|-------------|
| `oms-sidecar/Cargo.toml` | Edit | Fixed edition `"2024"`→`"2021"`, added `chrono` + `uuid` |
| `oms-sidecar/src/models.rs` | Rewrite | `signal_timestamp_us` on ProposedTrade, `error` on Fill, `errors` on ExecutionResponse, `IncomingMessage` enum (trade batch or control command), `ControlCommand`, `JournalEntry` |
| `oms-sidecar/src/journal.rs` | **Create** | Append-only JSONL crash journal (`CrashJournal` with `Mutex<BufWriter<File>>`). Methods: `log_pre_submit()`, `log_post_submit()`, `log_error()`. Flushes after each entry. Never crashes the trade path. |
| `oms-sidecar/src/config.rs` | **Create** | `SharedState` struct initialized once at startup: `PolymarketConfig` (signer, cached EIP-712 domain, pre-decoded HMAC secret), `LimitlessConfig` (signer, `AtomicU64` nonce, domain cache), shared `reqwest::Client` with `tcp_nodelay(true)`, `AtomicBool` halt flag |
| `oms-sidecar/src/lib.rs` | Edit | Added `pub mod config;` and `pub mod journal;` |
| `oms-sidecar/src/main.rs` | Rewrite | `#[tokio::main(worker_threads = 2)]`, `SharedState` init at startup, Limitless nonce warmed via async RPC before ZMQ loop, `IncomingMessage` enum parsing (trades or halt/status/reset commands), staleness rejection, panic-safe ZMQ handling, fills + errors in response |
| `oms-sidecar/src/exchanges/polymarket.rs` | Rewrite | Uses `SharedState` (no per-trade init), `.round() as u64` amounts, `Uuid::new_v4()` salt, cached EIP-712 domain, single serialization for HMAC + HTTP body, HTTP status code check, journal entries before/after submission |
| `oms-sidecar/src/exchanges/limitless.rs` | Rewrite | Uses `SharedState`, `AtomicU64` nonce with `fetch_add` (no RPC per trade), domain cache (`Mutex<HashMap<Address, Eip712Domain>>`), nonce resync on error, same fixes as polymarket |
| `src/polyquant/execution/rust_client.py` | Rewrite | `signal_timestamp_us` on every trade payload, `send_halt(reason)`, `send_reset()`, `send_status()` methods |
| `src/polyquant/execution/executor.py` | Edit | Fixed placeholder bug: now parses actual `fills[]` and `errors[]` from Rust response with real `order_id`, `filled_size`, `filled_price`. Handles `"halted"`, `"rejected"`, `"partial"` status values. |
| `src/polyquant/navigator.py` | Edit | `_on_kill_switch_trigger()` now calls `self._rust_client.send_halt()` with trigger reason |

### Fixes by Severity

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | CRITICAL | Per-trade env::var/signer/client init | `config.rs` SharedState, init once at startup |
| 2 | CRITICAL | Limitless nonce RPC per trade (+50-200ms) | `AtomicU64` with `fetch_add`, warmed once from RPC |
| 3 | CRITICAL | No signal staleness check | `signal_timestamp_us` field + rejection in `main.rs` |
| 4 | CRITICAL | f64 truncation in amounts | `.round() as u64` |
| 5 | HIGH | Panic on `.unwrap()` (ZMQ frame) | `match` with error response |
| 6 | HIGH | EIP-712 domain recomputed per trade | Static cache (Polymarket), `HashMap` cache (Limitless) |
| 7 | HIGH | No HTTP status code checking | Check before parsing, return `Fill` with error |
| 8 | HIGH | Failed trades silently dropped | `errors` field in `ExecutionResponse` |
| 9 | HIGH | Salt predictability | `uuid::Uuid::new_v4()` |
| 10 | HIGH | Python ignores actual Rust fills | Parse `fills`/`errors` from response |
| 11 | MEDIUM | No TCP_NODELAY | `Client::builder().tcp_nodelay(true)` |
| 12 | MEDIUM | Double HMAC serialization | Serialize once, use `.body()` |
| 13 | MEDIUM | Tokio over-provisioned | `worker_threads = 2` |
| 14 | NEW | No crash journal | `journal.rs` append-only JSONL log |
| 15 | NEW | No kill switch to Rust integration | Halt command via ZMQ + `AtomicBool` |

---

## Part 4: Architecture (Post-Fix)

### Rust Sidecar v0.2.0 Module Structure

```
oms-sidecar/src/
  main.rs          ZMQ loop, startup init, halt/staleness/batch dispatch
  lib.rs           Module declarations
  models.rs        ProposedTrade, Fill, ExecutionResponse, IncomingMessage, JournalEntry
  config.rs        SharedState (signers, HTTP client, domains, halt flag, journal)
  journal.rs       CrashJournal (append-only JSONL)
  exchanges/
    mod.rs         Module declarations
    polymarket.rs  Polymarket CLOB execution (EIP-712 + HMAC-SHA256 L2 auth)
    limitless.rs   Limitless execution (EIP-712, AtomicU64 nonce, domain cache)
```

### Control Flow

```
Python Navigator
  |
  |-- detect_opportunity()
  |-- execute_atomic() via TradeExecutor
  |     |
  |     |-- Pre-flight: balance checks, in-flight capital limit
  |     |-- Priority grouping (illiquid legs first)
  |     |
  |     |-- dispatch_trades() via RustClient
  |     |     |
  |     |     |-- Add signal_timestamp_us (microseconds)
  |     |     |-- JSON serialize → ZMQ REQ send
  |     |     |
  |     |     v
  |     |   RUST SIDECAR (tcp://127.0.0.1:5555)
  |     |     |
  |     |     |-- Parse IncomingMessage (TradeBatch or Control)
  |     |     |-- Check halt flag (AtomicBool)
  |     |     |-- Staleness check (reject if age > max_signal_age_us)
  |     |     |-- Spawn tokio::task per trade
  |     |     |     |-- Journal: log_pre_submit()
  |     |     |     |-- EIP-712 sign (cached domain + signer)
  |     |     |     |-- HTTP POST to CLOB API (shared client, TCP_NODELAY)
  |     |     |     |-- Check HTTP status code
  |     |     |     |-- Journal: log_post_submit() or log_error()
  |     |     |     |-- Return Fill with error field
  |     |     |
  |     |     |-- Collect fills + errors
  |     |     |-- ZMQ REP send {status, fills[], errors[]}
  |     |     |
  |     |     v
  |     |-- Parse actual fills from Rust response
  |     |-- If partial failure: unwind all previous fills
  |     |-- Record fills to SQLite (fire-and-forget)
  |     |
  |     v
  |-- ExecutionResult
  |
  |-- KillSwitch trigger
        |-- send_halt() → Rust sidecar sets AtomicBool
        |-- All subsequent trades rejected with status "halted"
```

### Crash Journal Format

File: `oms_crash_journal.jsonl` (append-only, one JSON object per line)

```json
{"timestamp":"2026-03-01T12:00:00Z","event":"pre_submit","exchange":"polymarket","outcome_id":"12345","side":"Buy","size":"100","limit_price":"0.55"}
{"timestamp":"2026-03-01T12:00:00Z","event":"post_submit","exchange":"polymarket","outcome_id":"12345","side":"Buy","size":"100","limit_price":"0.55","order_id":"abc123","filled_size":"100","filled_price":"0.55","http_status":200}
{"timestamp":"2026-03-01T12:00:01Z","event":"error","exchange":"limitless","outcome_id":"67890","error":"http_429: rate limited","http_status":429}
```

### ZMQ Control Commands

| Command | Request | Response | Effect |
|---------|---------|----------|--------|
| Halt | `{"command":"halt","reason":"kill_switch: drawdown"}` | `{"status":"halted","message":"..."}` | Sets AtomicBool, rejects all trades |
| Status | `{"command":"status"}` | `{"status":"running"}` or `{"status":"halted","halt_reason":"..."}` | Read-only |
| Reset | `{"command":"reset"}` | `{"status":"running","message":"halt cleared"}` | Clears halt flag |

---

## Part 5: Verification Checklist

1. `cargo check` in `oms-sidecar/` — compiles with zero warnings
2. Start sidecar, send trade batch with `signal_timestamp_us` from Python — verify fills contain actual order_id/filled_size
3. Send `{"command": "halt", "reason": "test"}` — verify subsequent trades rejected with status `"halted"`
4. Send `{"command": "reset"}` — verify trades work again
5. Check `oms_crash_journal.jsonl` for `pre_submit` + `post_submit` entries
6. Execute two Limitless trades concurrently — verify journal shows different nonce values (no race)
7. Kill switch trigger in Python — verify Rust sidecar receives halt and refuses trades
