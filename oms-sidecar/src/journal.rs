use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use chrono::Utc;
use tokio::sync::mpsc;
use tracing::{error, warn};

use crate::models::JournalEntry;

const JOURNAL_CHANNEL_CAPACITY: usize = 1024;
const ASYNC_BATCH_LIMIT: usize = 64;

/// Append-only JSONL crash-recovery journal.
///
/// All writes go through a single shared `BufWriter<File>` guarded by a
/// `Mutex`, so sync and async writers cannot interleave at byte boundaries.
///
/// Safety-critical writes (`log_pre_submit`) are synchronous: they take the
/// lock, write, and flush before returning so intent is durable before the
/// HTTP call.
///
/// Non-critical writes (`log_post_submit`, `log_nonce`, `log_error`) are
/// sent through a *bounded* mpsc channel to a background writer thread. If
/// the channel fills (disk stall), entries are dropped and counted; the
/// caller can inspect `dropped_count()` to trigger an operator halt.
pub struct CrashJournal {
    writer: Arc<Mutex<BufWriter<File>>>,
    async_tx: mpsc::Sender<JournalEntry>,
    dropped_count: Arc<AtomicU64>,
}

impl CrashJournal {
    pub fn open(path: &str) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let writer = Arc::new(Mutex::new(BufWriter::new(file)));

        let (async_tx, async_rx) = mpsc::channel::<JournalEntry>(JOURNAL_CHANNEL_CAPACITY);
        let dropped_count = Arc::new(AtomicU64::new(0));

        Self::spawn_async_writer(Arc::clone(&writer), async_rx);

        Ok(CrashJournal {
            writer,
            async_tx,
            dropped_count,
        })
    }

    /// Background OS thread that drains the async channel and writes through
    /// the shared `BufWriter`. Batches up to `ASYNC_BATCH_LIMIT` entries per
    /// flush to amortize lock + flush cost.
    fn spawn_async_writer(
        writer: Arc<Mutex<BufWriter<File>>>,
        mut rx: mpsc::Receiver<JournalEntry>,
    ) {
        std::thread::spawn(move || {
            while let Some(first) = rx.blocking_recv() {
                let mut batch: Vec<JournalEntry> = Vec::with_capacity(ASYNC_BATCH_LIMIT);
                batch.push(first);
                while batch.len() < ASYNC_BATCH_LIMIT {
                    match rx.try_recv() {
                        Ok(entry) => batch.push(entry),
                        Err(_) => break,
                    }
                }

                match writer.lock() {
                    Ok(mut w) => {
                        for entry in &batch {
                            match serde_json::to_string(entry) {
                                Ok(line) => {
                                    if let Err(e) = writeln!(w, "{}", line) {
                                        error!("journal async writeln failed: {:?}", e);
                                    }
                                }
                                Err(e) => error!("journal async serialize failed: {:?}", e),
                            }
                        }
                        if let Err(e) = w.flush() {
                            error!("journal async flush failed: {:?}", e);
                        }
                    }
                    Err(e) => {
                        error!("journal writer lock poisoned in async thread: {:?}", e);
                        break;
                    }
                }
            }
        });
    }

    fn write_entry_sync(&self, entry: &JournalEntry) -> Result<()> {
        let line = serde_json::to_string(entry)?;
        let mut w = self
            .writer
            .lock()
            .map_err(|e| anyhow::anyhow!("journal lock poisoned: {}", e))?;
        writeln!(w, "{}", line)?;
        w.flush()?;
        Ok(())
    }

    fn write_entry_async(&self, entry: JournalEntry) {
        match self.async_tx.try_send(entry) {
            Ok(_) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                let prev = self.dropped_count.fetch_add(1, Ordering::Relaxed);
                if prev % 100 == 0 {
                    warn!(
                        "journal async channel full; dropped {} entries (cumulative)",
                        prev + 1
                    );
                }
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                error!("journal async channel closed");
            }
        }
    }

    /// Number of async entries dropped due to channel backpressure.
    /// Used by the main loop to trigger a halt if it grows too fast.
    pub fn dropped_count(&self) -> u64 {
        self.dropped_count.load(Ordering::Relaxed)
    }

    /// Force-flush the underlying writer. Called by the graceful shutdown path.
    pub fn flush(&self) {
        if let Ok(mut w) = self.writer.lock() {
            let _ = w.flush();
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
            nonce: None,
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
            nonce: None,
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
            size: String::new(),
            limit_price: String::new(),
            order_id: None,
            filled_size: None,
            filled_price: None,
            error: None,
            http_status: None,
            response_body: None,
            nonce: Some(nonce),
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
            nonce: None,
        };
        self.write_entry_async(entry);
    }
}
