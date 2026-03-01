use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedTrade {
    pub outcome_id: String,
    pub side: OrderSide,
    pub size: String,
    pub limit_price: String,
    pub exchange: String,
    pub reason: String,
    #[serde(default)]
    pub signal_timestamp_us: Option<u64>,
    /// Idempotency key: if present, Rust dedup cache rejects duplicate batch IDs.
    #[serde(default)]
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub trade: ProposedTrade,
    pub filled_size: String,
    pub filled_price: String,
    pub order_id: String,
    #[serde(default)]
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResponse {
    pub status: String,
    pub fills: Vec<Fill>,
    #[serde(default)]
    pub errors: Vec<Fill>,
}

/// Incoming message: either a trade batch (JSON array) or a control command (JSON object).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum IncomingMessage {
    TradeBatch(Vec<ProposedTrade>),
    Control(ControlCommand),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlCommand {
    pub command: String,
    #[serde(default)]
    pub reason: String,
}

/// Journal entry written to the append-only JSONL crash log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    pub timestamp: DateTime<Utc>,
    pub event: String,
    pub exchange: String,
    pub outcome_id: String,
    pub side: String,
    pub size: String,
    pub limit_price: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filled_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filled_price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_body: Option<String>,
}
