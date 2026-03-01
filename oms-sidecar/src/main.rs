use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use dotenvy::dotenv;
use serde_json::json;
use tokio::task;
use tracing::{error, info, warn};
use zeromq::{RepSocket, Socket, SocketRecv, SocketSend};

use oms_sidecar::config::{self, SharedState};
use oms_sidecar::exchanges::{limitless, polymarket};
use oms_sidecar::models::{ControlCommand, ExecutionResponse, Fill, IncomingMessage, ProposedTrade};

#[tokio::main(worker_threads = 2)]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("Starting PolyQuant OMS Sidecar v0.2...");

    match dotenv() {
        Ok(_) => info!("Loaded .env file"),
        Err(e) => warn!("No .env file: {:?}", e),
    }

    // Initialize shared state ONCE (signers, HTTP client, cached domains, journal)
    let state = config::init_shared_state()?;
    info!("Shared state initialized");

    // Warm Limitless nonce from RPC before entering the trading loop
    if state.limitless.is_some() {
        match warm_limitless_nonce(&state).await {
            Ok(n) => info!("Limitless nonce warmed: {}", n),
            Err(e) => error!("Failed to warm Limitless nonce: {:?}", e),
        }
    }

    let mut socket = RepSocket::new();
    socket.bind("tcp://127.0.0.1:5555").await?;
    info!("Listening on tcp://127.0.0.1:5555");

    loop {
        match socket.recv().await {
            Ok(msg) => {
                let payload_bytes = match msg.get(0) {
                    Some(frame) => frame.to_vec(),
                    None => {
                        error!("Received empty ZMQ message");
                        let err = json!({"status": "error", "message": "empty message"});
                        let _ = socket.send(err.to_string().into()).await;
                        continue;
                    }
                };

                let payload_str = String::from_utf8_lossy(&payload_bytes);

                let response_json = match serde_json::from_str::<IncomingMessage>(&payload_str) {
                    Ok(IncomingMessage::TradeBatch(trades)) => {
                        handle_trade_batch(trades, &state).await
                    }
                    Ok(IncomingMessage::Control(cmd)) => handle_control_command(cmd, &state),
                    Err(e) => {
                        error!("Failed to parse payload: {:?}", e);
                        json!({"status": "error", "message": e.to_string()}).to_string()
                    }
                };

                if let Err(e) = socket.send(response_json.into()).await {
                    error!("Failed to send ZMQ response: {:?}", e);
                }
            }
            Err(e) => {
                error!("ZMQ recv error: {:?}", e);
            }
        }
    }
}

async fn handle_trade_batch(trades: Vec<ProposedTrade>, state: &Arc<SharedState>) -> String {
    if state.is_halted() {
        let reason = state.halt_reason_str();
        warn!("Rejecting batch: sidecar is halted ({})", reason);
        let resp = ExecutionResponse {
            status: "halted".into(),
            fills: vec![],
            errors: vec![],
        };
        return serde_json::to_string(&resp).unwrap_or_default();
    }

    // Idempotency dedup: check if we've already processed this request_id
    let request_id = trades
        .first()
        .and_then(|t| t.request_id.clone());
    if let Some(ref rid) = request_id {
        if let Some(cached) = state.dedup_check(rid) {
            info!("Dedup hit for request_id={}, returning cached response", rid);
            return cached;
        }
    }

    info!("Received batch of {} trades", trades.len());

    // Staleness check
    let now_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    for trade in &trades {
        if let Some(ts) = trade.signal_timestamp_us {
            let age = now_us.saturating_sub(ts);
            if age > state.max_signal_age_us {
                warn!(
                    "Stale signal rejected: age={}us max={}us outcome={}",
                    age, state.max_signal_age_us, trade.outcome_id
                );
                let resp = ExecutionResponse {
                    status: "rejected".into(),
                    fills: vec![],
                    errors: vec![Fill {
                        trade: trade.clone(),
                        filled_size: "0".into(),
                        filled_price: "0".into(),
                        order_id: String::new(),
                        error: format!(
                            "stale_signal: age={}us exceeds max={}us",
                            age, state.max_signal_age_us
                        ),
                    }],
                };
                return serde_json::to_string(&resp).unwrap_or_default();
            }
        }
    }

    let (fills, errors) = execute_batch(trades, state).await;

    let status = if errors.is_empty() {
        "success"
    } else if fills.is_empty() {
        "error"
    } else {
        "partial"
    };

    let resp = ExecutionResponse {
        status: status.into(),
        fills,
        errors,
    };

    let response_json = serde_json::to_string(&resp).unwrap_or_else(|e| {
        json!({"status": "error", "message": format!("serialization failed: {}", e)}).to_string()
    });

    // Cache response for idempotency dedup
    if let Some(rid) = request_id {
        state.dedup_insert(rid, response_json.clone());
    }

    response_json
}

fn handle_control_command(cmd: ControlCommand, state: &Arc<SharedState>) -> String {
    match cmd.command.as_str() {
        "halt" => {
            info!("HALT command received: {}", cmd.reason);
            state.set_halt(&cmd.reason);
            state.journal.log_error(
                "system",
                "ALL",
                &format!("HALT: {}", cmd.reason),
                None,
                None,
            );
            json!({"status": "halted", "message": cmd.reason}).to_string()
        }
        "status" => {
            let halted = state.is_halted();
            json!({
                "status": if halted { "halted" } else { "running" },
                "halt_reason": state.halt_reason_str(),
            })
            .to_string()
        }
        "reset" => {
            info!("RESET command received — clearing halt");
            state.clear_halt();
            json!({"status": "running", "message": "halt cleared"}).to_string()
        }
        other => {
            warn!("Unknown command: {}", other);
            json!({"status": "error", "message": format!("unknown command: {}", other)})
                .to_string()
        }
    }
}

async fn execute_batch(
    trades: Vec<ProposedTrade>,
    state: &Arc<SharedState>,
) -> (Vec<Fill>, Vec<Fill>) {
    let mut tasks = Vec::new();

    for trade in trades {
        let st = Arc::clone(state);
        let mut halt_rx = state.halt_rx.clone();
        let t = task::spawn(async move {
            tokio::select! {
                biased;
                _ = halt_rx.changed() => {
                    warn!("Halt received mid-batch, aborting trade for {}", trade.outcome_id);
                    Ok(Fill {
                        trade: trade.clone(),
                        filled_size: "0".into(),
                        filled_price: "0".into(),
                        order_id: String::new(),
                        error: "halted_mid_batch".into(),
                    })
                }
                result = async {
                    match trade.exchange.as_str() {
                        "polymarket" => polymarket::execute(&trade, &st).await,
                        "limitless" => limitless::execute(&trade, &st).await,
                        other => {
                            error!("Unsupported exchange: {}", other);
                            Ok(Fill {
                                trade: trade.clone(),
                                filled_size: "0".into(),
                                filled_price: "0".into(),
                                order_id: String::new(),
                                error: format!("unsupported_exchange: {}", other),
                            })
                        }
                    }
                } => result,
            }
        });
        tasks.push(t);
    }

    let mut fills = Vec::new();
    let mut errors = Vec::new();

    for t in tasks {
        match t.await {
            Ok(Ok(fill)) => {
                if fill.error.is_empty() {
                    fills.push(fill);
                } else {
                    errors.push(fill);
                }
            }
            Ok(Err(e)) => {
                error!("Trade execution error: {:?}", e);
            }
            Err(e) => {
                error!("Tokio task panicked: {:?}", e);
            }
        }
    }

    (fills, errors)
}

async fn warm_limitless_nonce(state: &Arc<SharedState>) -> Result<u64> {
    let limitless = state
        .limitless
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Limitless not configured"))?;

    // Fetch on-chain nonce (use "finalized" to avoid re-org races) with failover
    let nonce_req = json!({
        "jsonrpc": "2.0",
        "method": "eth_getTransactionCount",
        "params": [limitless.maker_address.to_string(), "finalized"],
        "id": 1
    });

    let mut urls = vec![limitless.rpc_url.clone()];
    urls.extend(limitless.rpc_fallback_urls.iter().cloned());

    let mut rpc_nonce: u64 = 0;
    let mut rpc_ok = false;
    for url in &urls {
        match state.http_client.post(url).json(&nonce_req).send().await {
            Ok(res) => {
                if let Ok(data) = res.json::<serde_json::Value>().await {
                    if let Some(hex) = data["result"].as_str() {
                        rpc_nonce = u64::from_str_radix(hex.trim_start_matches("0x"), 16)?;
                        rpc_ok = true;
                        break;
                    }
                }
            }
            Err(e) => {
                warn!("RPC failover (nonce warm): {} failed: {:?}", url, e);
                continue;
            }
        }
    }
    if !rpc_ok {
        return Err(anyhow::anyhow!("All RPC providers failed for nonce warm"));
    }

    // Also scan crash journal for highest nonce_assign to handle in-flight txns
    let journal_nonce = recover_max_nonce_from_journal("oms_crash_journal.jsonl");
    // Use the higher of RPC and journal+1 (journal records used nonces, next is +1)
    let effective_nonce = if journal_nonce > 0 {
        std::cmp::max(rpc_nonce, journal_nonce + 1)
    } else {
        rpc_nonce
    };

    if effective_nonce != rpc_nonce {
        info!(
            "Nonce recovery: RPC={} journal_max={} using={}",
            rpc_nonce, journal_nonce, effective_nonce
        );
    }

    limitless.nonce.store(effective_nonce, Ordering::SeqCst);
    Ok(effective_nonce)
}

/// Scan the JSONL crash journal for the highest nonce assigned to limitless.
fn recover_max_nonce_from_journal(path: &str) -> u64 {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return 0,
    };

    let mut max_nonce: u64 = 0;
    for line in BufReader::new(file).lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        // Quick filter before full JSON parse
        if !line.contains("nonce_assign") || !line.contains("limitless") {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<serde_json::Value>(&line) {
            if entry["event"].as_str() == Some("nonce_assign")
                && entry["exchange"].as_str() == Some("limitless")
            {
                // Nonce value is stored in the "size" field
                if let Some(n) = entry["size"].as_str().and_then(|s| s.parse::<u64>().ok()) {
                    if n > max_nonce {
                        max_nonce = n;
                    }
                }
            }
        }
    }
    max_nonce
}
