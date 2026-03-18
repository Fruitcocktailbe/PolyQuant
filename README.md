# PolyQuant 2.0 - Liquidity Vacuum

**Real-Time Momentum Scanning System for Polymarket**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

PolyQuant 2.0 has been upgraded from a cross-exchange arbitrage engine into an ultra-fast **momentum execution scanner** natively built for Polymarket. Using a strategy known as the "Liquidity Vacuum", it scans for underpriced markets (<$0.10 YES price), monitors real-time WebSockets, and executes entries when massive volume spikes and momentum swings occur simultaneously.

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Tier 1     │ -> │  Tier 2     │ -> │  Tier 3     │ -> │  OMS Sidecar│
│ Discovery   │    │ (Warmup)    │    │ (Websocket  │    │   (Rust)    │
│ (Gamma API) │    │ (CLOB Hist) │    │  Monitor)   │    │(Execution)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

- **Python (Intelligence)**: 
  - Iterates through the API to find illiquid/cheap markets.
  - Generates 200m EMAs.
  - Connects to the Polymarket WebSocket and filters conditions in real time.
- **Rust (Execution)**: A dedicated Order Management System (OMS) sidecar that handles signing, submission, and multi-exchange connectivity with <10ms overhead.
- **Web UI (Monitoring)**: A React/Vite dashboard for real-time monitoring and manual overrides (KillSwitch).

## 📁 Project Structure

```
polyquant/
├── src/
│   ├── api/                 # Monitoring API (FastAPI)
│   ├── data/                # Data layer (Websocket/Gamma Clients)
│   ├── execution/           # Trade execution logic
│   ├── risk/                # Risk (KillSwitch/PositionSizer)
│   └── utils/               # Config/Logging
├── oms-sidecar/             # Rust Execution Engine
├── web/                     # React/Vite Dashboard
├── .env                     # Centralized configuration
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Rust 1.75+

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/polyquant.git
cd polyquant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
pip install -e .

# Build Rust OMS sidecar
cd oms-sidecar
cargo build --release

# Install Web dependencies
cd ../web
npm install
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your Polymarket + Polygon Keys
```

### Running the System

1. **Start the OMS Sidecar**:
   ```bash
   cd oms-sidecar
   cargo run --release
   ```

2. **Run the Scanner**:
   ```bash
   # In a new terminal
   python src/polyquant/main.py
   ```

3. **Launch the Dashboard**:
   ```bash
   cd web
   npm run dev
   ```

## 🔒 Risk Management

- **Modified Kelly Criterion**: Position sizing capped based on risk models.
- **Kill Switch**: Automatic halt if drawdown exceeds threshold.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

This software is for educational purposes only. Automated trading on prediction markets may have legal implications depending on jurisdiction. Use at your own risk.
