# PolyQuant 2.0

**Autonomous Arbitrage Extraction via Multi-Agent Logical Reasoning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

PolyQuant 2.0 is a modular agent swarm that autonomously extracts arbitrage opportunities from Polymarket prediction markets. The system translates human language market descriptions into mathematical constraints and executes optimal trades with <30ms latency.

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Discovery  │ -> │  Reasoning  │ -> │ Verification│ -> │Optimization │ -> │  Execution  │
│   Agent     │    │  (Logic     │    │  (Validator)│    │   (SCIP)    │    │  (Rust HFT) │
│(Gemini 2.0) │    │  Architect) │    │(Gemini     │    │             │    │             │
│             │    │             │    │ Thinking)   │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 📁 Project Structure

```
polyquant/
├── src/
│   ├── agents/              # AI Agent implementations
│   │   ├── __init__.py
│   │   ├── discovery.py     # Phase 1: Market scanner (Gemini 2.0 Flash)
│   │   ├── logic_architect.py # Phase 2: Dependency detection (DeepSeek-R1)
│   │   └── validator.py     # Phase 3: Constraint verification (Gemini Thinking)
│   ├── solver/              # Optimization engine
│   │   ├── __init__.py
│   │   ├── bregman.py       # Bregman projection algorithm
│   │   └── scip_solver.py   # SCIP integer programming
│   ├── executor/            # Trade execution (Rust)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs
│   │       ├── websocket.rs
│   │       └── order_manager.rs
│   ├── data/                # Data layer
│   │   ├── __init__.py
│   │   ├── polymarket_client.py
│   │   └── market_models.py
│   ├── risk/                # Risk management
│   │   ├── __init__.py
│   │   ├── position_sizing.py
│   │   └── kill_switch.py
│   └── utils/               # Shared utilities
│       ├── __init__.py
│       ├── config.py
│       └── logging_config.py
├── tests/                   # Test suite
├── docs/                    # Documentation
├── .env.example             # Environment variables template
├── pyproject.toml           # Python dependencies
├── requirements.txt         # Pip requirements
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

# Build Rust executor
cd src/executor
cargo build --release
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

### Running

```bash
# Start the agent swarm
python -m polyquant.main

# Or run individual agents
python -m polyquant.agents.discovery
```

## ⚙️ Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `EXTRACTION_ALPHA` | Target extraction efficiency | 0.9 (90%) |
| `MAX_DRAWDOWN` | Kill switch threshold | 0.15 (15%) |
| `ORDERBOOK_DEPTH_CAP` | Position size limit | 0.5 (50%) |
| `LATENCY_TARGET_MS` | Max decision-to-mempool | 30 |

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
