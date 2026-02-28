# PolyQuant Deployment Requirements

To deploy PolyQuant for live trading, you must provide the following credentials and system configurations.

## 1. Identity & Funding (Polygon PoS)
**Why:** To sign EIP-712 authentication messages and execute trades on-chain.
- [ ] **Private Key**: The private key of your trading wallet.
    - *Variable*: `POLYGON_PRIVATE_KEY`
    - *Format*: `0x...` (64 hex characters)
    - *Action*: Add to `.env`. Ensure this wallet has MATIC for gas and USDC.e (bridged USDC) for trading collateral.

## 2. Infrastructure Provider
**Why:** Reliability. The public Polygon RPC is too slow/unreliable for event listening.
- [ ] **Alchemy/Infura Key**:
    - *Variable*: `ALCHEMY_API_KEY` (or full `WEB3_PROVIDER_URI`)
    - *Action*: Sign up at [alchemy.com](https://www.alchemy.com/), create a Polygon/Matic app, and copy the API Key.

## 3. Polymarket API Access
**Why:** To access the order book (CLOB) and execute trades.
- [ ] **API Keys (`api_key`, `secret`, `passphrase`)**:
    - *Action*: **Do not creates these manually.** The system will derive them using your Private Key via the `derive_api_key()` function we will implement in Phase 1.
    - *Requirement*: Use a dedicated trading wallet, not your cold storage.

## 4. AI Models
**Why:** To power the Discovery, Logic, and Validator agents.
- [ ] **Google Gemini API Key**:
    - *Variable*: `GEMINI_API_KEY`
    - *Status*: **Existing**. (Check `.env` to confirm).

## 5. System Dependencies
**Why:** The combinatorial solver needs a high-performance backend.
- [ ] **SCIP Optimization Suite**:
    - *Action*: Download and install the SCIP Opt Suite for your OS from [scipopt.org](https://scipopt.org/index.php#download).
    - *Verify*: Ensure the installation path is in your system `PATH` so `pyscipopt` can find the libraries.

## 6. Environment Checklist
Copy the following into your `.env` file:

```bash
# --- Identity ---
POLYGON_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE

# --- Infrastructure ---
ALCHEMY_API_KEY=YOUR_ALCHEMY_KEY_HERE
# or
WEB3_PROVIDER_URI=https://polygon-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_KEY

# --- AI ---
GEMINI_API_KEY=YOUR_GEMINI_KEY

# --- Trading Config ---
TRADING_MODE=live
```
