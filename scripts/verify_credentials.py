"""
Credential Verification Script for PolyQuant

This script verifies that your Polymarket credentials are correctly configured.
Run this after setting up your .env file to ensure authentication will work.

USAGE:
------
    python scripts/verify_credentials.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polyquant.utils import config, get_logger
from polyquant.data.auth import PolymarketAuth, verify_credentials

logger = get_logger(__name__)


def check_env_vars():
    """Check that required environment variables are set."""
    print("\n=== Environment Check ===")
    
    # Gemini API Key
    gemini_key = config.gemini_api_key.get_secret_value()
    if gemini_key:
        print(f"✅ GEMINI_API_KEY: Set ({len(gemini_key)} chars)")
    else:
        print("❌ GEMINI_API_KEY: Not set")
    
    # Alchemy API Key
    alchemy_key = config.alchemy_api_key.get_secret_value()
    if alchemy_key:
        print(f"✅ ALCHEMY_API_KEY: Set ({len(alchemy_key)} chars)")
    else:
        print("⚠️  ALCHEMY_API_KEY: Not set (optional but recommended)")
    
    # Polygon Private Key
    polygon_key = config.polygon_private_key.get_secret_value()
    if polygon_key:
        print(f"✅ POLYGON_PRIVATE_KEY: Set ({len(polygon_key)} chars)")
        if not polygon_key.startswith("0x"):
            print("   ⚠️  Warning: Private key should start with '0x'")
    else:
        print("❌ POLYGON_PRIVATE_KEY: Not set (REQUIRED for trading)")
        return False
    
    return True


def check_wallet():
    """Check wallet derivation and address."""
    print("\n=== Wallet Check ===")
    
    try:
        auth = PolymarketAuth()
        print(f"✅ Wallet Address: {auth.address}")
        return True
    except ValueError as e:
        print(f"❌ Wallet Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False


def check_api_credentials():
    """Try to derive Polymarket API credentials."""
    print("\n=== API Credential Check ===")
    
    try:
        auth = PolymarketAuth()
        creds = auth.derive_api_credentials()
        
        print(f"✅ API Key: {creds.api_key[:12]}...")
        print(f"✅ API Secret: {creds.api_secret[:8]}...")
        print(f"✅ API Passphrase: {'*' * len(creds.api_passphrase)}")
        
        return True
    except RuntimeError as e:
        print(f"❌ API Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False


def main():
    """Run all credential checks."""
    print("=" * 50)
    print("PolyQuant Credential Verification")
    print("=" * 50)
    
    # Step 1: Environment variables
    env_ok = check_env_vars()
    
    if not env_ok:
        print("\n⛔ Cannot proceed without POLYGON_PRIVATE_KEY.")
        print("   Add it to your .env file and try again.")
        sys.exit(1)
    
    # Step 2: Wallet derivation
    wallet_ok = check_wallet()
    
    if not wallet_ok:
        print("\n⛔ Wallet derivation failed.")
        print("   Check that your private key is valid.")
        sys.exit(1)
    
    # Step 3: API credentials
    api_ok = check_api_credentials()
    
    print("\n" + "=" * 50)
    if env_ok and wallet_ok and api_ok:
        print("✅ All checks passed! Ready for trading.")
        print("   Run: python -m polyquant.navigator")
    else:
        print("❌ Some checks failed. Review errors above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
