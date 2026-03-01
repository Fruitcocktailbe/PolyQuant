use std::str::FromStr;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use alloy::hex;
use alloy::primitives::{Address, U256};
use alloy::signers::Signer;
use alloy::sol;
use alloy::sol_types::{eip712_domain, SolStruct};
use anyhow::{anyhow, Result};
use serde_json::Value;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::config::SharedState;
use crate::models::{Fill, OrderSide, ProposedTrade};

sol! {
    #[derive(Debug, Default)]
    struct Order {
        uint256 salt;
        address maker;
        address signer;
        address taker;
        uint256 tokenId;
        uint256 makerAmount;
        uint256 takerAmount;
        uint256 expiration;
        uint256 nonce;
        uint256 feeRateBps;
        uint8 side;
        uint8 signatureType;
    }
}

pub async fn execute(trade: &ProposedTrade, state: &Arc<SharedState>) -> Result<Fill> {
    info!("Preparing Limitless execution for {}", trade.outcome_id);

    let limitless = state
        .limitless
        .as_ref()
        .ok_or_else(|| anyhow!("Limitless not configured"))?;

    let side_str = format!("{:?}", trade.side);

    // Extract market slug from reason field
    let slug = if trade.reason.contains("slug:") {
        let parts: Vec<&str> = trade.reason.split("slug:").collect();
        if parts.len() > 1 {
            parts[1].trim_end_matches(')').to_string()
        } else {
            trade.outcome_id.clone()
        }
    } else {
        trade.outcome_id.clone()
    };

    // Fetch market verifying contract from Limitless API (varies per market)
    let market_res = state
        .http_client
        .get(format!("{}/markets/{}", limitless.api_url, slug))
        .header("X-API-Key", &limitless.api_key)
        .send()
        .await?;

    let market_http_status = market_res.status().as_u16();
    if market_http_status >= 400 {
        let body = market_res.text().await.unwrap_or_default();
        let err_msg = format!("market_fetch_http_{}: {}", market_http_status, &body[..body.len().min(200)]);
        state.journal.log_error("limitless", &trade.outcome_id, &err_msg, Some(market_http_status), Some(&body));
        return Ok(Fill {
            trade: trade.clone(),
            filled_size: "0".into(),
            filled_price: "0".into(),
            order_id: String::new(),
            error: err_msg,
        });
    }

    let market_data: Value = market_res.json().await?;
    let verifying_contract_str = market_data["exchangeContract"]
        .as_str()
        .ok_or_else(|| anyhow!("Missing exchangeContract in market data for slug={}", slug))?;
    let verifying_contract = Address::from_str(verifying_contract_str)?;

    // Atomic nonce: fetch_add guarantees unique nonce per concurrent trade
    let nonce = limitless.nonce.fetch_add(1, Ordering::SeqCst);
    // Persist nonce to journal for crash recovery
    state.journal.log_nonce("limitless", nonce, &trade.outcome_id);

    // Parse price/size
    let is_buy = matches!(trade.side, OrderSide::Buy);
    let price: f64 = trade.limit_price.parse()?;
    let size: f64 = trade.size.parse()?;

    // Pessimistic rounding: ceil() for what we pay, floor() for what we receive
    let (maker_amount, taker_amount, side_uint) = if is_buy {
        // BUY: we pay makerAmount (USDC) → ceil; we receive takerAmount (tokens) → floor
        let m = (price * size * 1e6).ceil() as u64;
        let t = (size * 1e6).floor() as u64;
        (m, t, 0u8)
    } else {
        // SELL: we pay makerAmount (tokens) → ceil; we receive takerAmount (USDC) → floor
        let m = (size * 1e6).ceil() as u64;
        let t = (price * size * 1e6).floor() as u64;
        (m, t, 1u8)
    };

    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let expiration = now + 60;
    let salt = Uuid::new_v4().as_u128() as u64; // FIX: random salt

    let order = Order {
        salt: U256::from(salt),
        maker: limitless.maker_address,
        signer: limitless.maker_address,
        taker: Address::ZERO,
        tokenId: U256::from_str_radix(
            trade.outcome_id.trim_start_matches("0x"),
            if trade.outcome_id.starts_with("0x") { 16 } else { 10 },
        )
        .unwrap_or(U256::ZERO),
        makerAmount: U256::from(maker_amount),
        takerAmount: U256::from(taker_amount),
        expiration: U256::from(expiration),
        nonce: U256::from(nonce),
        feeRateBps: U256::ZERO,
        side: side_uint,
        signatureType: 0,
    };

    // EIP-712 sign with cached domain (insert on miss)
    let domain = {
        let cache = limitless.domain_cache.lock().map_err(|e| anyhow!("domain cache lock: {}", e))?;
        cache.get(&verifying_contract).cloned()
    };

    let domain = domain.unwrap_or_else(|| {
        let d = eip712_domain! {
            name: "Limitless CTF Exchange",
            version: "1",
            chain_id: 8453,
            verifying_contract: verifying_contract,
        };
        if let Ok(mut cache) = limitless.domain_cache.lock() {
            cache.insert(verifying_contract, d.clone());
        }
        d
    });

    let hash = order.eip712_signing_hash(&domain);
    let signature = limitless.signer.sign_hash(&hash).await?;
    let sig_hex = format!("0x{}", hex::encode(signature.as_bytes()));

    // Build request body
    let submit_payload = serde_json::json!({
        "order": {
            "maker": order.maker.to_string(),
            "signer": order.signer.to_string(),
            "taker": order.taker.to_string(),
            "tokenId": order.tokenId.to_string(),
            "makerAmount": order.makerAmount.to_string(),
            "takerAmount": order.takerAmount.to_string(),
            "side": order.side,
            "salt": order.salt.to_string(),
            "expiration": order.expiration.to_string(),
            "nonce": order.nonce.to_string(),
            "feeRateBps": order.feeRateBps.to_string(),
            "signatureType": order.signatureType,
            "orderType": "FOK"
        },
        "signature": sig_hex
    });

    // Journal: intent to submit
    state.journal.log_pre_submit(
        "limitless",
        &trade.outcome_id,
        &side_str,
        &trade.size,
        &trade.limit_price,
    );

    // Halt check: abort before committing to HTTP POST
    if state.is_halted() {
        return Err(anyhow!("halted_before_submit"));
    }

    // Submit using shared HTTP client
    let submit_res = state
        .http_client
        .post(format!("{}/orders", limitless.api_url))
        .header("X-API-Key", &limitless.api_key)
        .json(&submit_payload)
        .send()
        .await?;

    let http_status = submit_res.status().as_u16();
    let res_text = submit_res.text().await?;
    debug!("Limitless response [{}]: {}", http_status, res_text);

    // Check HTTP status
    if http_status >= 400 {
        let err_msg = format!("http_{}: {}", http_status, &res_text[..res_text.len().min(200)]);
        state.journal.log_error(
            "limitless",
            &trade.outcome_id,
            &err_msg,
            Some(http_status),
            Some(&res_text),
        );

        // On nonce-related errors, try to resync nonce
        if res_text.contains("nonce") || res_text.contains("Nonce") {
            warn!("Nonce error detected, attempting resync");
            if let Err(e) = resync_nonce(state).await {
                warn!("Nonce resync failed: {:?}", e);
            }
        }

        return Ok(Fill {
            trade: trade.clone(),
            filled_size: "0".into(),
            filled_price: "0".into(),
            order_id: String::new(),
            error: err_msg,
        });
    }

    // Parse response
    let res_json: Value = serde_json::from_str(&res_text).unwrap_or(serde_json::json!({}));
    let order_id = res_json["id"].as_str().unwrap_or("").to_string();
    let actual_filled_size = res_json["filledAmount"]
        .as_f64()
        .or_else(|| res_json["filledSize"].as_f64())
        .unwrap_or(0.0);

    let final_filled_size = if actual_filled_size == 0.0
        && ["filled", "closed", "matched"]
            .contains(&res_json["status"].as_str().unwrap_or(""))
    {
        size
    } else {
        actual_filled_size
    };

    let status_str = res_json["status"].as_str().unwrap_or("");

    // Journal: result
    state.journal.log_post_submit(
        "limitless",
        &trade.outcome_id,
        &side_str,
        &trade.size,
        &trade.limit_price,
        &order_id,
        &final_filled_size.to_string(),
        &price.to_string(),
        Some(http_status),
        Some(&res_text),
    );

    let error = if final_filled_size > 0.0 {
        String::new()
    } else if !order_id.is_empty() {
        format!("order_status: {}", status_str)
    } else {
        "no_order_id_returned".into()
    };

    info!(
        "Limitless order: id={} status={} filled={}",
        order_id, status_str, final_filled_size
    );

    Ok(Fill {
        trade: trade.clone(),
        filled_size: final_filled_size.to_string(),
        filled_price: price.to_string(),
        order_id,
        error,
    })
}

/// Resync nonce from RPC after a nonce error. Tries primary + fallback RPCs.
async fn resync_nonce(state: &Arc<SharedState>) -> Result<()> {
    let limitless = state
        .limitless
        .as_ref()
        .ok_or_else(|| anyhow!("Limitless not configured"))?;

    let nonce_req = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "eth_getTransactionCount",
        "params": [limitless.maker_address.to_string(), "finalized"],
        "id": 1
    });

    // Try primary + all fallback URLs
    let mut urls = vec![limitless.rpc_url.clone()];
    urls.extend(limitless.rpc_fallback_urls.iter().cloned());

    for url in &urls {
        match state
            .http_client
            .post(url)
            .json(&nonce_req)
            .send()
            .await
        {
            Ok(res) => {
                if let Ok(data) = res.json::<Value>().await {
                    if let Some(hex) = data["result"].as_str() {
                        let nonce = u64::from_str_radix(hex.trim_start_matches("0x"), 16)?;
                        limitless.nonce.store(nonce, Ordering::SeqCst);
                        info!("Limitless nonce resynced to {} via {}", nonce, url);
                        return Ok(());
                    }
                }
            }
            Err(e) => {
                warn!("RPC failover: {} failed: {:?}", url, e);
                continue;
            }
        }
    }

    Err(anyhow!("All RPC providers failed for nonce resync"))
}
