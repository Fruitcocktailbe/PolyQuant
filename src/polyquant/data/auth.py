import os
import time
from typing import Any, Dict

from eth_account import Account
from eth_account.messages import encode_typed_data
# from polyquant.utils import config, get_logger # Avoid circular imports if possible
import logging

logger = logging.getLogger(__name__)

# Try to import Rust accelerator
# try:
#     from polyquant.execution_core import LocalSigner as RustSigner
#     RUST_AVAILABLE = True
# except Exception as e:
#     # logger.debug(f"Rust accelerator not available: {e}")
#     RUST_AVAILABLE = False
# import logging # Already imported
from pydantic import BaseModel

class ApiCredentials(BaseModel):
    """
    Credentials for Polymarket API access.
    """
    api_key: str
    api_secret: str
    passphrase: str


class PolymarketAuth:
    """
    Helper for generating Polymarket API authentication headers.
    """
    def __init__(self, credentials: ApiCredentials):
        self.credentials = credentials
        
    def get_headers(self, timestamp: int, method: str, path: str, body: str = "") -> Dict[str, str]:
        """
        Generate L2 headers for CLOB API.
        """
        # TODO: Implement actual signing if needed here, 
        # but typically this is done via clob-client or LocalSigner
        return {}
    
RUST_AVAILABLE = False

class LocalSigner:
    """
    Handles local EIP-712 signing of Polymarket orders.
    Uses Rust implementation if available, otherwise falls back to eth_account.
    """
    def __init__(self, private_key: str, chain_id: int = 137, verifying_contract: str = None):
        self._pk = private_key
        self._chain_id = chain_id
        # Default to Polymarket CTF Exchange on Polygon if not provided
        self._verifying_contract = verifying_contract or "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
        
        # Initialize Rust signer if available
        self._rust_signer = None
        # if RUST_AVAILABLE:
        #     try:
        #         self._rust_signer = RustSigner(
        #             self._pk, 
        #             self._chain_id, 
        #             self._verifying_contract
        #         )
        #         logger.info("🚀 Rust execution_core accelerator loaded for signing")
        #     except Exception as e:
        #         logger.error(f"Failed to init Rust signer: {e}")
        
        # Initialize Python signer (eth_account)
        if not self._rust_signer:
            self._account = Account.from_key(self._pk)
            logger.info("Using Python eth_account for signing (slower but reliable)")

    def sign_order(self, order_dict: Dict[str, Any]) -> str:
        """
        Sign an order using the fastest available method.
        """
        # 1. Rust Fast Path
        if self._rust_signer:
            try:
                # TODO: Map dict to exact args expected by Rust
                # This requires precise mapping based on lib.rs signature
                return self._rust_signer.sign_order(
                    int(order_dict.get("salt", 0)),
                    order_dict.get("maker"),
                    order_dict.get("signer"),
                    order_dict.get("taker"),
                    str(order_dict.get("tokenId")),
                    str(order_dict.get("makerAmount")),
                    str(order_dict.get("takerAmount")),
                    int(order_dict.get("expiration", 0)),
                    int(order_dict.get("nonce", 0)),
                    int(order_dict.get("feeRateBps", 0)),
                    int(order_dict.get("side", 0)),
                    int(order_dict.get("signatureType", 0)),
                )
            except Exception as e:
                logger.error(f"Rust signing failed, falling back to Python: {e}")
        
        # 2. Python Fallback (eth_account)
        # Construct EIP-712 payload manually
        # Note: This is a simplified version. Real implementation needs full EIP-712 struct definition.
        # For now, we assume usage via py-clob-client which handles this internally.
        # This method is exposed for benchmarking/custom bypass.
        
        # Define types (Simplified for visual benchmark, actual logic is complex)
        types = {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "Order": [
                {"name": "salt", "type": "uint256"},
                {"name": "maker", "type": "address"},
                {"name": "signer", "type": "address"},
                {"name": "taker", "type": "address"},
                {"name": "tokenId", "type": "uint256"},
                {"name": "makerAmount", "type": "uint256"},
                {"name": "takerAmount", "type": "uint256"},
                {"name": "expiration", "type": "uint256"},
                {"name": "nonce", "type": "uint256"},
                {"name": "feeRateBps", "type": "uint256"},
                {"name": "side", "type": "uint8"},
                {"name": "signatureType", "type": "uint8"},
            ]
        }
        
        domain = {
            "name": "Polymarket CTF Exchange",
            "version": "1",
            "chainId": self._chain_id,
            "verifyingContract": self._verifying_contract,
        }
        
        # Message is the order_dict
        
        data = {
            "types": types,
            "domain": domain,
            "primaryType": "Order",
            "message": order_dict,
        }
        
        signed = self._account.sign_message(encode_typed_data(full_message=data))
        return signed.signature.hex()
