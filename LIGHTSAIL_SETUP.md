# AWS Lightsail Deployment Guide: PolyQuant 2.0

This guide explains how to set up PolyQuant from scratch on an AWS Lightsail instance.

## 1. Instance Recommendation
- **Blueprint**: Ubuntu 22.04 LTS
- **Plan**: $12 USD/month (2 GB RAM, 2 vCPUs) or higher. 
  - *Note: SCIP and Rust builds enjoy more CPU for initial setup.*

## 2. System Prerequisites

Run these commands on your fresh instance:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git redis-server python3-pip python3-venv pkg-config libssl-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install SCIP Optimization Suite (Pre-compiled for Ubuntu recommended)
sudo apt install -y scip
```

## 3. Project Setup

```bash
git clone https://github.com/yourusername/PolyQuant.git
cd PolyQuant

# Python Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env
cp .env.example .env
nano .env  # Add your API keys. POLYGON_PRIVATE_KEY can be empty for Paper Trading.

# Generate Market Map (REQUIRED before trading)
# This scans for arbitrage clusters and local constraints.
python -m polyquant.map_maker
```

## 4. Running the Services

You will need 3 terminal sessions (or use `screen`/`tmux`):

### Session 1: The Execution Engine (Rust)
The OMS Sidecar must be running for orders to be submitted.
```bash
cd oms-sidecar
cargo run --release
```

### Session 2: The Agent Swarm (Python)
The "Fast Brain" Navigator.
```bash
# In root directory
source venv/bin/activate
python -m polyquant.main trade
```

### Session 3: The Web Dashboard
```bash
cd web
npm install
# Fix Vite permissions
chmod +x node_modules/.bin/vite
npm run dev -- --host
```

## 5. Network & Firewall

In the Lightsail Console, open the following **Inbound Ports**:
- **8000**: Python API (Internal/Monitoring)
- **5173**: React Dashboard (Vite)

> [!TIP]
> **Security Alternative**: Instead of opening ports publicly, use an SSH tunnel from your local machine:
> `ssh -L 5173:localhost:5173 -L 8000:localhost:8000 ubuntu@your-lightsail-ip`

## 6. Maintenance Commands

- **Stop All**: `CTRL+C` in all sessions.
- **Restart Redis**: `sudo systemctl restart redis-server`
- **Clear Scan Cache**: `redis-cli FLUSHALL` (Use this if you want to force a full rescan of all markets)
- **View Logs**: Check `logs/` directory in the root or the Web Dashboard terminal.

> [!NOTE]
> `sentence-transformers` is used for cross-exchange matching. The first time the Map Maker runs, it will download a small model (~80MB). This is normal.
