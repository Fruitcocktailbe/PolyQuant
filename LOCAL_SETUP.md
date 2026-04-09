
# Local Setup Guide for PolyQuant 2.0

Follow these steps to get PolyQuant running on your local machine for paper trading.

## 1. System Dependencies
Since you've already installed SCIP and Redis, ensure they are correctly configured:

### SCIP Optimization Suite
*   **Verification**: Run `python -c "import pyscipopt; print('SCIP successfully imported. Tech version:', pyscipopt.Model().getTechVersion())"`
*   **Success**: The fact that your previous command didn't fail on `import pyscipopt` means SCIP is correctly installed and linked!
*   **Issue?**: If `import pyscipopt` fails, ensure `SCIP_HOME` points to your installation.

### Redis
*   **Verification**: Run `.\redis\redis-cli.exe ping`. You should get `PONG`.
*   **Startup**: If Redis isn't running, start it with `.\redis\redis-server.exe .\redis\redis.windows.conf`.

---

## 2. Python Environment Setup

```powershell
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 3. Environment Configuration (`.env`)
Create/Update your `.env` file with the following keys. 

> [!IMPORTANT]
> Since we moved to OpenRouter, you only need **one key** for all AI agents.

```env
# AI Agents (OpenRouter)
GEMINI_API_KEY=your_openrouter_api_key_here

# Polymarket API (Optional for discovery, but recommended)
POLYGON_PRIVATE_KEY=your_wallet_private_key
ALCHEMY_API_KEY=your_alchemy_rpc_key

# Solver Settings
FW_MIN_PROFIT=0.01  # Target $0.01 profit per cluster
MAX_CONCURRENT_LLM=3
```

---

## 4. Running PolyQuant

PolyQuant operates in two phases: **Map Making** and **Trading**.

### 1. Build the Market Map
The "Map Maker" analyzes Polymarket to find related markets and build logical constraints.

```powershell
# Basic run (scans for high-liquidity markets)
python -m polyquant.main map

# Advanced: Lower liquidity threshold to find more markets (e.g., $100)
python -m polyquant.main map --min-liquidity 100

# Advanced: Re-scan markets even if they were analyzed before
python -m polyquant.main map --force
```
*   **Output**: JSON manifests stored in `.polyquant/manifests/`.
*   **Note**: This takes 5-10 minutes depending on cluster size.

### Phase 2: The Navigator (Execution/Trading)
The Navigator loads the manifests and starts a low-latency loop waiting for order book updates via WebSocket.

```powershell
# Start real-time arbitrage detector in Paper Mode
python -m polyquant.main trade --mode paper
```
*   **UI**: Open `http://localhost:8000` in your browser to see the live dashboard, latency, and detected opportunities.

---

## 5. Troubleshooting
*   **Import Errors**: If you see `ModuleNotFoundError`, ensure you are in the `venv` and the `src` directory is in your PYTHONPATH: `$env:PYTHONPATH = "src"`.
*   **Redis Connection**: If the app crashes on start, check if Redis is running on port 6379.
*   **Empty Dashboard**: If the Navigator starts but shows no markets, ensure you have ran the `map` command first to generate manifests.
