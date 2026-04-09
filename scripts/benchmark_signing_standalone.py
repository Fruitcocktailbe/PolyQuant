import time
import secrets
import sys
import os
from eth_account import Account
from eth_account.messages import encode_typed_data

class LocalSigner:
    """
    Handles local EIP-712 signing of Polymarket orders.
    Uses Python eth_account implementation (inlined for benchmark).
    """
    def __init__(self, private_key: str, chain_id: int = 137, verifying_contract: str = None):
        self._pk = private_key
        self._chain_id = chain_id
        # Default to Polymarket CTF Exchange on Polygon if not provided
        self._verifying_contract = verifying_contract or "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
        self._account = Account.from_key(self._pk)
        print("Using Python eth_account for signing")

    def sign_order(self, order_dict):
        # Python EIP-712 implementation (simplified types for benchmark)
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
        
        data = {
            "types": types,
            "domain": domain,
            "primaryType": "Order",
            "message": order_dict,
        }
        
        signed = self._account.sign_message(encode_typed_data(full_message=data))
        return signed.signature.hex()

def benchmark():
    print(f"\n🚀 Benchmarking Order Signing Speed (Standalone)")
    print("=" * 40)
    
    # 1. Setup
    pk = "0x" + secrets.token_hex(32)
    try:
        signer = LocalSigner(pk)
    except Exception as e:
        print(f"Error init signer: {e}")
        return

    order = {
        "salt": 123456,
        "maker": "0x" + secrets.token_hex(20),
        "signer": "0x" + secrets.token_hex(20),
        "taker": "0x0000000000000000000000000000000000000000",
        "tokenId": 123,
        "makerAmount": 1000000,
        "takerAmount": 2000000,
        "expiration": 0,
        "nonce": 1,
        "feeRateBps": 0,
        "side": 0,
        "signatureType": 0,
    }
    
    # 2. Warmup
    print("Warming up JIT...")
    signer.sign_order(order)
    
    # 3. Local Signing loops
    print("Running 100 iterations...")
    start = time.perf_counter()
    count = 100
    for _ in range(count):
        signer.sign_order(order)
    duration = time.perf_counter() - start
    
    local_avg = (duration / count) * 1000
    print(f"\n✅ Local Signing (Python): {local_avg:.2f} ms per order")
    
    # 4. Simulate API Relayer
    relayer_latency_ms = 200.0
    print(f"🐌 Standard API (Relayer): ~{relayer_latency_ms:.2f} ms per order")
    
    speedup = relayer_latency_ms / local_avg
    print(f"\n🎉 Speedup Factor: {speedup:.1f}x FASTER")
    print("=" * 40)

if __name__ == "__main__":
    benchmark()
