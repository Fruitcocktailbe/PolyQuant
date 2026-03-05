# PolyQuant 2.0

**Autonomous Arbitrage Extraction via Multi-Agent Logical Reasoning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

PolyQuant 2.0 is a modular agent swarm that autonomously extracts arbitrage opportunities from Polymarket prediction markets. The system translates human language market descriptions into mathematical constraints and executes optimal trades with <30ms latency.

## 🏗️ Architecture

PolyQuant 2.0 uses a **hybrid architecture** combining Python's high-level reasoning with Rust's low-level execution speed.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Discovery  │ -> │  Map Maker  │ -> │  Validator  │ -> │ Navigator   │ -> │  OMS Sidecar│
│   Agent     │    │(Correlation)│    │ (Reasoning) │    │ (Fast Brain)│    │   (Rust)    │
│(Gemini 2.0) │    │(Logic Arch) │    │(Thinking)   │    │  (Python)   │    │(Execution)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       ^                                                        |                  |
       └─────────────────────────── manifests ──────────────────┘                  v
                                                                            Polymarket/Limitless
```

- **Python (Intelligence)**: Handles market discovery, LLM-based reasoning, and real-time arbitrage detection.
- **Rust (Execution)**: A dedicated Order Management System (OMS) sidecar that handles signing, submission, and multi-exchange connectivity with <10ms overhead.
- **Web UI (Monitoring)**: A React/Vite dashboard for real-time monitoring and manual overrides (KillSwitch).

## 📁 Project Structure

```
polyquant/
├── src/
│   ├── agents/              # AI Agent implementations
│   ├── solver/              # Optimization engine (SCIP/FW)
│   ├── api/                 # Monitoring API (FastAPI)
│   ├── data/                # Data layer
│   ├── risk/                # Risk (KillSwitch/PositionSizer)
│   └── utils/               # Config/Logging
├── oms-sidecar/             # [NEW] Rust Execution Engine
│   ├── src/                 # Multi-exchange execution logic
│   └── Cargo.toml           # Optimized binary build
├── web/                     # [NEW] React/Vite Dashboard
│   └── src/components/      # PipelineMonitor, KillSwitch, LogTerminal
├── manifests/               # Pre-computed market logic files
├── .env                     # Centralized configuration
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Rust 1.75+
- SCIP Optimization Suite

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
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# - GEMINI_API_KEY
# - DEEPSEEK_API_KEY
# - ALCHEMY_API_KEY
```

### Running the System

1. **Start the OMS Sidecar**:
   ```bash
   cd oms-sidecar
   cargo run --release
   ```

2. **Run the Agent Swarm**:
   ```bash
   # In a new terminal
   python -m polyquant.main trade
   ```

3. **Launch the Dashboard**:
   ```bash
   cd web
   npm run dev
   ```
# Or run individual agents
python -m polyquant.agents.discovery
```

## ⚙️ Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MIN_LIQUIDITY` | Minimum market liquidity (USD) | 1000.0 |
| `LLM_TEMPERATURE` | Global agent reasoning temp | 0.0 |
| `SCIP_GAP` | Solver optimality precision | 0.001 |
| `MIN_PROFIT_THRESHOLD` | Execute if profit > X | 0.05 |
| `MAX_DRAWDOWN` | Emergency kill switch | 0.15 |

## 📊 Success Metrics

| Metric | Target |
|--------|--------|
| Extraction Efficiency | $500+ avg profit/trade |
| Logical Accuracy | >81% on dependent pairs |
| Latency | <30ms decision-to-mempool |

## 🔒 Risk Management

- **Modified Kelly Criterion**: Position sizing capped at 50% of order book depth
- **Kill Switch**: Automatic halt if drawdown exceeds 15%
- **VWAP Guardrail**: Abort if slippage exceeds $0.05 profit margin
- **Solver Timeout**: Halt if 5-minute rolling average timeout exceeded

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

This software is for educational purposes only. Automated trading on prediction markets may have legal implications depending on jurisdiction. Use at your own risk.
