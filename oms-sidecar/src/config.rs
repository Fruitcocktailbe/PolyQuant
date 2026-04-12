use std::collections::HashMap;
use std::env;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use alloy::primitives::Address;
use alloy::signers::local::PrivateKeySigner;
use alloy::sol_types::{eip712_domain, Eip712Domain};
use anyhow::Result;
use base64::{engine::general_purpose::STANDARD, Engine};
use reqwest::Client;
use tokio::sync::watch;
use tracing::info;

use crate::journal::CrashJournal;

pub struct PolymarketConfig {
    pub api_url: String,
    pub api_key: String,
    pub api_passphrase: String,
    pub signer: PrivateKeySigner,
    pub maker_address: Address,
    pub eip712_domain: Eip712Domain,
    /// Pre-decoded HMAC secret (base64-decoded once at startup).
    pub hmac_secret_bytes: Vec<u8>,
}

pub struct LimitlessConfig {
    pub api_url: String,
    pub api_key: String,
    pub signer: PrivateKeySigner,
    pub maker_address: Address,
    pub rpc_url: String,
    /// Fallback RPC URLs (tried in order if primary fails).
    pub rpc_fallback_urls: Vec<String>,
    /// Thread-safe nonce counter. Fetched once from RPC, then incremented atomically.
    pub nonce: AtomicU64,
    /// Cache of EIP-712 domains keyed by verifying contract address.
    pub domain_cache: Mutex<HashMap<Address, Eip712Domain>>,
    /// P0-1: cached slug → exchange-contract address. Avoids the per-trade
    /// `GET /markets/{slug}` HTTP round trip (50–200 ms each).
    pub market_contract_cache: Mutex<HashMap<String, Address>>,
    /// P0-6: allow-list of legitimate Limitless CTF exchange contracts. If
    /// non-empty, market lookups whose `exchangeContract` is not in this set
    /// are rejected. Empty = no allow-listing (legacy behavior).
    pub exchange_contract_allowlist: Vec<Address>,
    /// Serialize concurrent nonce resync attempts to prevent race conditions.
    pub nonce_resync_lock: tokio::sync::Mutex<()>,
    /// P0-3: consecutive failure counter. Reset on any successful submit.
    /// When this exceeds `consecutive_failure_halt_threshold`, the sidecar
    /// halts itself so an operator can investigate before nonce drift becomes
    /// catastrophic.
    pub consecutive_failures: AtomicU64,
}

/// Cached response entry for idempotency deduplication.
pub struct DedupEntry {
    pub response: String,
    pub created: Instant,
}

pub struct SharedState {
    pub http_client: Client,
    pub polymarket: Option<PolymarketConfig>,
    pub limitless: Option<LimitlessConfig>,
    pub halt: AtomicBool,
    pub halt_reason: Mutex<String>,
    pub journal: CrashJournal,
    /// Max staleness for signals in microseconds (default: 500ms).
    pub max_signal_age_us: u64,
    /// Idempotency dedup cache: request_id → (response JSON, timestamp).
    /// Entries older than 60s are evicted on next check.
    pub dedup_cache: Mutex<HashMap<String, DedupEntry>>,
    /// Watch channel sender for halt signal propagation to spawned tasks.
    pub halt_tx: watch::Sender<bool>,
    /// Watch channel receiver (clone for each spawned task).
    pub halt_rx: watch::Receiver<bool>,
    /// P0-3: number of consecutive Limitless failures that trips an automatic
    /// halt. Default 10. Set via `OMS_CONSECUTIVE_FAILURE_HALT`.
    pub consecutive_failure_halt_threshold: u64,
}

impl SharedState {
    pub fn is_halted(&self) -> bool {
        self.halt.load(Ordering::Acquire)
    }

    pub fn set_halt(&self, reason: &str) {
        self.halt.store(true, Ordering::Release);
        if let Ok(mut r) = self.halt_reason.lock() {
            *r = reason.to_string();
        }
        // Notify all spawned tasks via watch channel
        let _ = self.halt_tx.send(true);
    }

    pub fn clear_halt(&self) {
        self.halt.store(false, Ordering::Release);
        if let Ok(mut r) = self.halt_reason.lock() {
            r.clear();
        }
        let _ = self.halt_tx.send(false);
    }

    pub fn halt_reason_str(&self) -> String {
        self.halt_reason
            .lock()
            .map(|r| r.clone())
            .unwrap_or_default()
    }

    /// Check dedup cache for a request_id. Returns cached response if found and not expired.
    pub fn dedup_check(&self, request_id: &str) -> Option<String> {
        let cache = self.dedup_cache.lock().ok()?;
        if let Some(entry) = cache.get(request_id) {
            if entry.created.elapsed() < Duration::from_secs(60) {
                return Some(entry.response.clone());
            }
        }
        None
    }

    /// Insert a response into the dedup cache. Also evicts entries older than 60s.
    pub fn dedup_insert(&self, request_id: String, response: String) {
        if let Ok(mut cache) = self.dedup_cache.lock() {
            // Evict stale entries (older than 60s)
            cache.retain(|_, v| v.created.elapsed() < Duration::from_secs(60));
            cache.insert(request_id, DedupEntry {
                response,
                created: Instant::now(),
            });
        }
    }
}

/// Initialize all shared state once at startup. Returns `Arc<SharedState>`.
pub fn init_shared_state() -> Result<Arc<SharedState>> {
    let http_client = Client::builder()
        .timeout(Duration::from_secs(5))
        .tcp_nodelay(true)
        .pool_max_idle_per_host(4)
        .build()?;

    let polymarket = init_polymarket()?;
    let limitless = init_limitless()?;
    let journal = CrashJournal::open("oms_crash_journal.jsonl")?;

    let max_signal_age_us: u64 = env::var("OMS_MAX_SIGNAL_AGE_US")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500_000); // 500ms default (was 5s — far too generous for HFT)

    let consecutive_failure_halt_threshold: u64 = env::var("OMS_CONSECUTIVE_FAILURE_HALT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let (halt_tx, halt_rx) = watch::channel(false);

    Ok(Arc::new(SharedState {
        http_client,
        polymarket,
        limitless,
        halt: AtomicBool::new(false),
        halt_reason: Mutex::new(String::new()),
        journal,
        max_signal_age_us,
        dedup_cache: Mutex::new(HashMap::new()),
        halt_tx,
        halt_rx,
        consecutive_failure_halt_threshold,
    }))
}

fn init_polymarket() -> Result<Option<PolymarketConfig>> {
    let private_key = env::var("POLYGON_PRIVATE_KEY").unwrap_or_default();
    let api_secret = env::var("POLYMARKET_SECRET").unwrap_or_default();

    if private_key.is_empty() || api_secret.is_empty() {
        info!("Polymarket not configured (missing POLYGON_PRIVATE_KEY or POLYMARKET_SECRET)");
        return Ok(None);
    }

    let signer = PrivateKeySigner::from_str(&private_key)?;
    let maker_address = signer.address();

    let exchange_contract_str = env::var("POLYMARKET_EXCHANGE_CONTRACT")
        .unwrap_or_else(|_| "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E".to_string());
    let exchange_contract = Address::from_str(&exchange_contract_str)?;

    let domain = eip712_domain! {
        name: "Polymarket CTF Exchange",
        version: "1",
        chain_id: 137,
        verifying_contract: exchange_contract,
    };

    let hmac_secret_bytes = STANDARD.decode(&api_secret)?;

    info!(
        "Polymarket configured: maker={}",
        maker_address
    );

    Ok(Some(PolymarketConfig {
        api_url: env::var("POLYMARKET_API_URL")
            .unwrap_or_else(|_| "https://clob.polymarket.com".to_string()),
        api_key: env::var("POLYMARKET_API_KEY").unwrap_or_default(),
        api_passphrase: env::var("POLYMARKET_PASSPHRASE").unwrap_or_default(),
        signer,
        maker_address,
        eip712_domain: domain,
        hmac_secret_bytes,
    }))
}

fn init_limitless() -> Result<Option<LimitlessConfig>> {
    let private_key = env::var("BASE_PRIVATE_KEY").unwrap_or_default();

    if private_key.is_empty() {
        info!("Limitless not configured (missing BASE_PRIVATE_KEY)");
        return Ok(None);
    }

    let signer = PrivateKeySigner::from_str(&private_key)?;
    let maker_address = signer.address();

    info!("Limitless configured: maker={}", maker_address);

    let rpc_fallback_urls: Vec<String> = env::var("BASE_RPC_FALLBACK_URLS")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // P0-6: parse the comma-separated allow-list. An empty value disables
    // allow-listing entirely (legacy behavior). Bad addresses fail startup
    // loudly rather than silently dropping the protection.
    let exchange_contract_allowlist: Vec<Address> =
        match env::var("LIMITLESS_ALLOWED_EXCHANGE_CONTRACTS") {
            Ok(s) if !s.trim().is_empty() => s
                .split(',')
                .map(|c| c.trim())
                .filter(|c| !c.is_empty())
                .map(Address::from_str)
                .collect::<Result<Vec<_>, _>>()?,
            _ => Vec::new(),
        };
    if exchange_contract_allowlist.is_empty() {
        info!(
            "Limitless exchange-contract allow-list is empty — accepting any \
             exchangeContract returned by the market API. Set \
             LIMITLESS_ALLOWED_EXCHANGE_CONTRACTS for production."
        );
    } else {
        info!(
            "Limitless exchange-contract allow-list: {} entries",
            exchange_contract_allowlist.len()
        );
    }

    Ok(Some(LimitlessConfig {
        api_url: env::var("LIMITLESS_API_URL")
            .unwrap_or_else(|_| "https://api.limitless.exchange/v1".to_string()),
        api_key: env::var("LIMITLESS_API_KEY").unwrap_or_default(),
        signer,
        maker_address,
        rpc_url: env::var("BASE_RPC_URL")
            .unwrap_or_else(|_| "https://mainnet.base.org".to_string()),
        rpc_fallback_urls,
        nonce: AtomicU64::new(0),
        domain_cache: Mutex::new(HashMap::new()),
        market_contract_cache: Mutex::new(HashMap::new()),
        exchange_contract_allowlist,
        nonce_resync_lock: tokio::sync::Mutex::new(()),
        consecutive_failures: AtomicU64::new(0),
    }))
}
