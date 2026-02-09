# Requirements to Run PolyQuant 2.0

To make the application fully functional, you need to provide the following credentials and configuration.

## 1. API Keys (For `.env` file)
Create a `.env` file in the root directory if it doesn't exist, and add these keys:

```bash
# Polymarket (Required for Execution)
# You need a dedicated wallet for the bot (Polygon chain).
# The Proxy Wallet is a smart contract wallet (Gnosis Safe) created by Polymarket.
POLYMARKET_API_KEY="<your-api-key>"         # From https://polymarket.com/settings/api
POLYMARKET_SECRET="<your-api-secret>"       # From same page
POLYMARKET_PASSPHRASE="<your-passphrase>"   # From same page
PRIVATE_KEY="<your-wallet-private-key>"     # Private key of the EOA (Externally Owned Account) controlling the proxy

# AI Models (Required for Logic & Validation)
DEEPSEEK_API_KEY="<your-deepseek-key>"      # For Logic Architect (Reasoning)
GEMINI_API_KEY="<your-google-ai-key>"       # For Validator (Flash Thinking)

# Infrastructure (Required for Speed)
# The "RPC" (Remote Procedure Call) is your gateway to the blockchain.
# Public/Free nodes are too slow (throttled) for arbitrage.
# You need a dedicated, private connection to see the "mempool" (pending trades) faster.
POLYGON_RPC_URL="<your-alchemy-url>"        # e.g., https://polygon-mainnet.g.alchemy.com/v2/...
```

## 2. Environment Setup
### SCIP Solver 
The optimization engine uses SCIP. You cannot just install the python package; you need the system libraries.
1.  **Download SCIPOptSuite** for Windows: [https://scipopt.org/index.php#download](https://scipopt.org/index.php#download)
2.  Install it.
3.  Add the installation `bin` folder to your System PATH.
4.  Set standard environment variable: `SCIPOPTDIR="C:\Program Files\SCIPOptSuite 9.0.0"` (or wherever you installed it).

### Redis (Optional but Recommended)
For caching market states between runs.
1.  Install Redis for Windows or run via Docker: `docker run -p 6379:6379 redis`

## 3. What's Next?
Once you have these keys:
1.  **Test Connection**: Run a simple script to fetch your balance and open orders.
2.  **Dry Run**: Set `TRADING_ENABLED=False` in `config.py` (or similar) to let the bot find opportunities without executing.
3.  **Fund Wallet**: Send USDC.e (Bridged USDC) and MATIC (for gas) to your Polymarket Proxy Wallet address.
