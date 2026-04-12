use std::str::FromStr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use alloy::hex;
use alloy::primitives::{Address, U256};
use alloy::signers::Signer;
use alloy::sol;
use alloy::sol_types::SolStruct;
use anyhow::{anyhow, Result};
use hmac::{Hmac, Mac};
use reqwest::header::{HeaderMap, HeaderValue};
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use serde_json::Value;
use sha2::Sha256;
use tracing::{debug, info};
use uuid::Uuid;

use base64::{engine::general_purpose::STANDARD, Engine};

use crate::config::SharedState;
use crate::models::{Fill, OrderSide, ProposedTrade};
use crate::util::truncate_safely;

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
    info!("Preparing Polymarket execution for {}", trade.outcome_id);

    let poly = state
        .polymarket
        .as_ref()
        .ok_or_else(|| anyhow!("Polymarket not configured"))?;

    let side_str = trade.side.as_str();

    // Parse price/size with exact decimal arithmetic (no f64 precision loss)
    let price = Decimal::from_str(&trade.limit_price)
        .map_err(|e| anyhow!("invalid limit_price '{}': {}", trade.limit_price, e))?;
    let size = Decimal::from_str(&trade.size)
        .map_err(|e| anyhow!("invalid size '{}': {}", trade.size, e))?;
    let scale = Decimal::from(1_000_000u64);

    // Pessimistic rounding: ceil() for what we pay, floor() for what we receive
    let is_buy = matches!(trade.side, OrderSide::Buy);
    let (maker_amount, taker_amount, side_uint) = if is_buy {
        // BUY: we pay makerAmount (USDC) → ceil; we receive takerAmount (tokens) → floor
        let m = (price * size * scale).ceil().to_u64()
            .ok_or_else(|| anyhow!("maker_amount overflow for price={} size={}", price, size))?;
        let t = (size * scale).floor().to_u64()
            .ok_or_else(|| anyhow!("taker_amount overflow for size={}", size))?;
        (m, t, 0u8)
    } else {
        // SELL: we pay makerAmount (tokens) → ceil; we receive takerAmount (USDC) → floor
        let m = (size * scale).ceil().to_u64()
            .ok_or_else(|| anyhow!("maker_amount overflow for size={}", size))?;
        let t = (price * size * scale).floor().to_u64()
            .ok_or_else(|| anyhow!("taker_amount overflow for price={} size={}", price, size))?;
        (m, t, 1u8)
    };

    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let expiration = now.checked_add(60).ok_or_else(|| anyhow!("timestamp overflow"))?;
    let salt = Uuid::new_v4().as_u128(); // Full 128-bit entropy (was truncated to u64)

    // P1-7: parse tokenId explicitly so we surface a real error instead of
    // signing an order with tokenId=0 (which the exchange rejects with a
    // confusing message).
    let token_id = U256::from_str_radix(
        trade.outcome_id.trim_start_matches("0x"),
        if trade.outcome_id.starts_with("0x") { 16 } else { 10 },
    )
    .map_err(|e| anyhow!("invalid outcome_id '{}': {}", trade.outcome_id, e))?;

    let order = Order {
        salt: U256::from(salt),
        maker: poly.maker_address,
        signer: poly.maker_address,
        taker: Address::ZERO,
        tokenId: token_id,
        makerAmount: U256::from(maker_amount),
        takerAmount: U256::from(taker_amount),
        expiration: U256::from(expiration),
        nonce: U256::ZERO, // Polymarket CLOB: always 0
        feeRateBps: U256::ZERO,
        side: side_uint,
        signatureType: 0,
    };

    // EIP-712 sign using CACHED domain
    let hash = order.eip712_signing_hash(&poly.eip712_domain);
    let signature = poly.signer.sign_hash(&hash).await?;
    let sig_hex = format!("0x{}", hex::encode(signature.as_bytes()));

    // Build request body (serialize ONCE for both HMAC and HTTP body)
    let submit_payload = serde_json::json!({
        "order": {
            "salt": order.salt.to_string(),
            "maker": order.maker.to_string(),
            "signer": order.signer.to_string(),
            "taker": order.taker.to_string(),
            "tokenId": order.tokenId.to_string(),
            "makerAmount": order.makerAmount.to_string(),
            "takerAmount": order.takerAmount.to_string(),
            "expiration": order.expiration.to_string(),
            "nonce": order.nonce.to_string(),
            "feeRateBps": order.feeRateBps.to_string(),
            "side": order.side,
            "signatureType": order.signatureType,
        },
        "owner": poly.maker_address.to_string(),
        "signature": sig_hex,
        "orderType": "FOK"
    });

    // Single serialization for HMAC + HTTP body
    let body_str = serde_json::to_string(&submit_payload)?;

    // HMAC-SHA256 L2 auth (using pre-decoded secret)
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs()
        .to_string();
    let path = "/order";
    let message = format!("{}POST{}{}", timestamp, path, body_str);

    let mut mac = Hmac::<Sha256>::new_from_slice(&poly.hmac_secret_bytes)
        .map_err(|e| anyhow!("HMAC error: {}", e))?;
    mac.update(message.as_bytes());
    let l2_signature = STANDARD.encode(mac.finalize().into_bytes());

    let mut headers = HeaderMap::new();
    headers.insert("POLY_API_KEY", HeaderValue::from_str(&poly.api_key)?);
    headers.insert("POLY_SIGNATURE", HeaderValue::from_str(&l2_signature)?);
    headers.insert("POLY_TIMESTAMP", HeaderValue::from_str(&timestamp)?);
    headers.insert(
        "POLY_PASSPHRASE",
        HeaderValue::from_str(&poly.api_passphrase)?,
    );
    headers.insert("Content-Type", HeaderValue::from_static("application/json"));

    // Journal: intent to submit
    state.journal.log_pre_submit(
        "polymarket",
        &trade.outcome_id,
        side_str,
        &trade.size,
        &trade.limit_price,
    );

    // Halt check: abort before committing to HTTP POST
    if state.is_halted() {
        return Err(anyhow!("halted_before_submit"));
    }

    // Submit using shared HTTP client and pre-serialized body
    let submit_res = state
        .http_client
        .post(format!("{}{}", poly.api_url, path))
        .headers(headers)
        .body(body_str)
        .send()
        .await?;

    let http_status = submit_res.status().as_u16();
    let res_text = submit_res.text().await?;
    debug!("Polymarket response [{}]: {}", http_status, res_text);

    // Check HTTP status
    if http_status >= 400 {
        let err_msg = format!("http_{}: {}", http_status, truncate_safely(&res_text, 200));
        state.journal.log_error(
            "polymarket",
            &trade.outcome_id,
            &err_msg,
            Some(http_status),
            Some(&res_text),
        );
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
    let order_id = res_json["orderID"].as_str().unwrap_or("").to_string();
    let status = res_json["status"].as_str().unwrap_or("");

    // P1-6 (Polymarket): trust the response. The CLOB returns takingAmount /
    // makingAmount in 1e6 USDC units; recover the filled size by dividing by
    // the scale, and the actual fill price by dividing maker against taker.
    let taking_amount = res_json["takingAmount"]
        .as_str()
        .and_then(|s| Decimal::from_str(s).ok());
    let making_amount = res_json["makingAmount"]
        .as_str()
        .and_then(|s| Decimal::from_str(s).ok());
    let scale_dec = Decimal::from(1_000_000u64);

    let (final_filled_size, final_filled_price) = match (taking_amount, making_amount, is_buy) {
        // BUY: makingAmount = USDC paid, takingAmount = tokens received.
        // size  = takingAmount / 1e6
        // price = makingAmount / takingAmount
        (Some(t), Some(m), true) if !t.is_zero() => {
            let filled_size = (t / scale_dec).to_string();
            let filled_price = (m / t).to_string();
            (filled_size, filled_price)
        }
        // SELL: makingAmount = tokens sent, takingAmount = USDC received.
        // size  = makingAmount / 1e6
        // price = takingAmount / makingAmount
        (Some(t), Some(m), false) if !m.is_zero() => {
            let filled_size = (m / scale_dec).to_string();
            let filled_price = (t / m).to_string();
            (filled_size, filled_price)
        }
        // Fall back to status-based heuristic only if amounts are missing.
        _ => {
            if status == "matched" {
                (size.to_string(), price.to_string())
            } else {
                ("0".to_string(), price.to_string())
            }
        }
    };

    // Journal: result
    state.journal.log_post_submit(
        "polymarket",
        &trade.outcome_id,
        side_str,
        &trade.size,
        &trade.limit_price,
        &order_id,
        &final_filled_size,
        &final_filled_price,
        Some(http_status),
        Some(&res_text),
    );

    let filled_dec = Decimal::from_str(&final_filled_size).unwrap_or(Decimal::ZERO);
    let error = if !filled_dec.is_zero() {
        String::new()
    } else if !order_id.is_empty() {
        format!("zero_fill_status: {}", status)
    } else {
        "no_order_id_returned".into()
    };

    info!(
        "Polymarket order: id={} status={} filled={} price={}",
        order_id, status, final_filled_size, final_filled_price
    );

    Ok(Fill {
        trade: trade.clone(),
        filled_size: final_filled_size,
        filled_price: final_filled_price,
        order_id,
        error,
    })
}
