use pyo3::prelude::*;
use alloy_sol_types::{sol, SolStruct, Eip712Domain};
use alloy_primitives::{Address, U256};
use alloy_signer::SignerSync;
use alloy_signer_local::PrivateKeySigner;
use std::str::FromStr;

// Define Polymarket Order Struct (EIP-712)
sol! {
    #[derive(Debug)]
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

#[pyclass]
struct LocalSigner {
    signer: PrivateKeySigner,
    domain: Eip712Domain,
}

#[pymethods]
impl LocalSigner {
    #[new]
    fn new(private_key: &str, chain_id: u64, verifying_contract: &str) -> PyResult<Self> {
        let signer = PrivateKeySigner::from_str(private_key)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid private key: {}", e)))?;
        
        let verifying_contract = Address::from_str(verifying_contract)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid contract address: {}", e)))?;
            
        // Construct EIP-712 Domain Separator
        let domain = Eip712Domain::new(
            Some("Polymarket CTF Exchange".into()), 
            Some("1".into()), 
            Some(U256::from(chain_id)),
            Some(verifying_contract),
            None,
        );

        Ok(LocalSigner { signer, domain })
    }

    #[allow(clippy::too_many_arguments)]
    fn sign_order(
        &self,
        salt: u64,
        maker: &str,
        signer: &str,
        taker: &str,
        token_id: &str,
        maker_amount: &str,
        taker_amount: &str,
        expiration: u64,
        nonce: u64,
        fee_rate_bps: u64,
        side: u8,
        signature_type: u8,
    ) -> PyResult<String> {
        // Parse inputs
        let maker = Address::from_str(maker)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid maker: {}", e)))?;
        let signer_addr = Address::from_str(signer)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid signer: {}", e)))?;
        let taker = Address::from_str(taker)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid taker: {}", e)))?;
        let token_id = U256::from_str(token_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid token_id: {}", e)))?;
        let maker_amount = U256::from_str(maker_amount)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid maker_amount: {}", e)))?;
        let taker_amount = U256::from_str(taker_amount)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid taker_amount: {}", e)))?;

        // Create Order struct
        let order = Order {
            salt: U256::from(salt),
            maker,
            signer: signer_addr,
            taker,
            tokenId: token_id,
            makerAmount: maker_amount,
            takerAmount: taker_amount,
            expiration: U256::from(expiration),
            nonce: U256::from(nonce),
            feeRateBps: U256::from(fee_rate_bps),
            side,
            signatureType: signature_type,
        };

        // Calculate EIP-712 Hash
        let hash = order.eip712_signing_hash(&self.domain);

        // Sign the hash
        let signature = self.signer.sign_hash_sync(&hash)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Signing failed: {}", e)))?;

        // Return hex signature
        Ok(format!("0x{}", hex::encode(signature.as_bytes())))
    }
}

#[pymodule]
fn execution_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LocalSigner>()?;
    Ok(())
}
