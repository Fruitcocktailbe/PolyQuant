use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::Mutex;

use anyhow::Result;
use chrono::Utc;
use tokio::sync::mpsc;
use tracing::error;

use crate::models::JournalEntry;

/// Append-only JSONL crash-recovery journal.
///
/// Safety-critical writes (`log_pre_submit`) are synchronous: they flush to
/// disk before returning so that intent is durable before the HTTP call.
///
/// Non-critical writes (`log_post_submit`, `log_nonce`, `log_error`) are
/// sent through an async mpsc channel to a background writer, unblocking the
/// Tokio executor threads.
pub struct CrashJournal {
    /// Synchronous writer for safety-critical entries (pre_submit).
    sync_writer: Mutex<BufWriter<File>>,
    /// Async channel sender for non-critical entries.
    async_tx: mpsc::UnboundedSender<JournalEntry>,
}

impl CrashJournal {
    pub fn open(path: &str) -> Result<Self> {
        // Sync writer: used only for pre_submit (safety-critical, must fsync before HTTP)
        let sync_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let sync_writer = Mutex::new(BufWriter::new(sync_file));

        // Async writer: background thread for post_submit, nonce, error entries
        let async_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let (async_tx, async_rx) = mpsc::unbounded_channel::<JournalEntry>();
        Self::spawn_async_writer(async_file, async_rx);

        Ok(CrashJournal {
            sync_writer,
            async_tx,
        })
    }

    /// Spawn a background thread (via spawn_blocking) that drains the async
    /// channel and writes entries to disk. Flushes after each batch.
    fn spawn_async_writer(file: File, mut rx: mpsc::UnboundedReceiver<JournalEntry>) {
        std::thread::spawn(move || {
            let mut writer = BufWriter::new(file);
            // Block on channel recv in a dedicated OS thread (not Tokio worker)
            while let Some(entry) = rx.blocking_recv() {
                if let Ok(line) = serde_json::to_string(&entry) {
                    let _ = writeln!(writer, "{}", line);
                }
                // Drain any buffered entries before flushing
                while let Ok(entry) = rx.try_recv() {
                    if let Ok(line) = serde_json::to_string(&entry) {
                        let _ = writeln!(writer, "{}", line);
                    }
                }
                let _ = writer.flush();
            }
        });
    }

    /// Synchronous write + flush. Used ONLY for safety-critical pre_submit entries.
    fn write_entry_sync(&self, entry: &JournalEntry) -> Result<()> {
        let line = serde_json::to_string(entry)?;
        let mut writer = self
            .sync_writer
            .lock()
            .map_err(|e| anyhow::anyhow!("journal lock poisoned: {}", e))?;
        writeln!(writer, "{}", line)?;
        writer.flush()?;
        Ok(())
    }

    /// Async write via mpsc channel. Non-blocking for the caller.
    fn write_entry_async(&self, entry: JournalEntry) {
        if let Err(e) = self.async_tx.send(entry) {
            error!("journal async send failed: {:?}", e);
        }
    }

    /// Log intent to submit an order (before HTTP call).
    /// SYNCHRONOUS: flushes to disk before returning (safety-critical).
    pub fn log_pre_submit(
        &self,
        exchange: &str,
        outcome_id: &str,
        side: &str,
        size: &str,
        limit_price: &str,
    ) {
        let entry = JournalEntry {
            timestamp: Utc::now(),
            event: "pre_submit".into(),
            exchange: exchange.into(),
            outcome_id: outcome_id.into(),
            side: side.into(),
            size: size.into(),
            limit_price: limit_price.into(),
            order_id: None,
            filled_size: None,
            filled_price: None,
            error: None,
            http_status: None,
            response_body: None,
        };
        if let Err(e) = self.write_entry_sync(&entry) {
            error!("journal sync write failed: {:?}", e);
        }
    }

    /// Log order submission result (after HTTP response).
    /// ASYNC: sent to background writer (non-blocking).
    pub fn log_post_submit(
        &self,
        exchange: &str,
        outcome_id: &str,
        side: &str,
        size: &str,
        limit_price: &str,
        order_id: &str,
        filled_size: &str,
        filled_price: &str,
        http_status: Option<u16>,
        response_body: Option<&str>,
    ) {
        let entry = JournalEntry {
            timestamp: Utc::now(),
            event: "post_submit".into(),
            exchange: exchange.into(),
            outcome_id: outcome_id.into(),
            side: side.into(),
            size: size.into(),
            limit_price: limit_price.into(),
            order_id: Some(order_id.into()),
            filled_size: Some(filled_size.into()),
            filled_price: Some(filled_price.into()),
            error: None,
            http_status,
            response_body: response_body.map(|s| s.to_string()),
        };
        self.write_entry_async(entry);
    }

    /// Log nonce assignment for crash recovery.
    /// ASYNC: sent to background writer (non-blocking).
    pub fn log_nonce(&self, exchange: &str, nonce: u64, outcome_id: &str) {
        let entry = JournalEntry {
            timestamp: Utc::now(),
            event: "nonce_assign".into(),
            exchange: exchange.into(),
            outcome_id: outcome_id.into(),
            side: String::new(),
            size: nonce.to_string(),
            limit_price: String::new(),
            order_id: None,
            filled_size: None,
            filled_price: None,
            error: None,
            http_status: None,
            response_body: None,
        };
        self.write_entry_async(entry);
    }

    /// Log an error event.
    /// ASYNC: sent to background writer (non-blocking).
    pub fn log_error(
        &self,
        exchange: &str,
        outcome_id: &str,
        error_msg: &str,
        http_status: Option<u16>,
        response_body: Option<&str>,
    ) {
        let entry = JournalEntry {
            timestamp: Utc::now(),
            event: "error".into(),
            exchange: exchange.into(),
            outcome_id: outcome_id.into(),
            side: String::new(),
            size: String::new(),
            limit_price: String::new(),
            order_id: None,
            filled_size: None,
            filled_price: None,
            error: Some(error_msg.into()),
            http_status,
            response_body: response_body.map(|s| s.to_string()),
        };
        self.write_entry_async(entry);
    }
}
