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

# Create a Swap File (Recommended for 2GB RAM instances)
# This helps prevent out-of-memory errors during memory-intensive operations like Map Maker.
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 3. Project Setup

```bash
git clone https://github.com/yourusername/PolyQuant.git
cd PolyQuant

# Python Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Create .env
cp .env.example .env
nano .env  # Add your API keys (Polymarket, Limitless, OpenRouter, etc.)
# Note: If trading in 'paper' mode without a POLYGON_PRIVATE_KEY, your 
# starting paper-trading UI balance will default to $10,000.00.

# Generate Market Map (REQUIRED before trading)
# This scans for arbitrage clusters, local constraints, and cross-exchange mappings.
python -m polyquant.map_maker


# LOW-RAM TIP: If Map Maker hangs, add: ENABLE_SEMANTIC_MATCHING=false

```

## 4. Running the Services (with tmux)

Use `tmux` so your services **keep running** after you disconnect from SSH.

```bash
# Start a named tmux session
tmux new -s polyquant

# Activate venv (do this once per session)
cd ~/PolyQuant
source venv/bin/activate
```

### Window 0: Redis + Map Maker
```bash
redis-server --daemonize yes
python -m polyquant.map_maker   # Run once to generate constraints
```

### Window 1: The Execution Engine (Rust)
Press `Ctrl+B`, then `C` to create a new window.
```bash
cd ~/PolyQuant/oms-sidecar
cargo run --release
```

### Window 2: The Agent Swarm (Python)
Press `Ctrl+B`, then `C` to create a new window.
```bash
cd ~/PolyQuant
source venv/bin/activate
python -m polyquant.main trade
```

### Window 3: The Web Dashboard (optional)
Press `Ctrl+B`, then `C` to create a new window.
```bash
cd ~/PolyQuant/web
npm install
chmod +x node_modules/.bin/vite
npm run dev -- --host
```

Once everything is running, press `Ctrl+B`, then `D` to **detach**. You can now safely close SSH — everything keeps running on the server.

## 5. Tmux Basics

Tmux is a terminal multiplexer that keeps your programs alive on the server even when you disconnect.

| Action | Command / Shortcut |
|---|---|
| **Create a session** | `tmux new -s polyquant` |
| **Detach** (leave but keep running) | `Ctrl+B`, then `D` |
| **Reattach** (reconnect later) | `tmux attach` or `tmux attach -t polyquant` |
| **New window** (like a new tab) | `Ctrl+B`, then `C` |
| **Next / Previous window** | `Ctrl+B`, then `N` / `P` |
| **List windows** | `Ctrl+B`, then `W` |
| **Kill current window** | Type `exit` or `Ctrl+D` |
| **List all sessions** | `tmux ls` |

> [!TIP]
> **Daily workflow**: SSH in → `tmux attach` → check your windows → `Ctrl+B, D` to detach → close SSH. Done!

## 6. Network & Firewall

In the Lightsail Console, open the following **Inbound Ports**:
- **8000**: Python API (Internal/Monitoring)
- **5173**: React Dashboard (Vite)

> [!TIP]
> **Security Alternative**: Instead of opening ports publicly, use an SSH tunnel from your local machine:
> `ssh -L 5173:localhost:5173 -L 8000:localhost:8000 ubuntu@your-lightsail-ip`

## 7. Maintenance Commands

- **Stop All**: `CTRL+C` in all sessions.
- **Restart Redis**: `sudo systemctl restart redis-server`
- **Clear Scan Cache**: `redis-cli FLUSHALL` (Use this if you want to force a full rescan of all markets)
- **View Logs**: Check `logs/` directory in the root or the Web Dashboard terminal.

> [!NOTE]
> `sentence-transformers` is used for cross-exchange matching. The first time the Map Maker runs, it will download a small model (~80MB). This is normal.

## 8. Troubleshooting: "Connection Refused"

If you can't connect to the dashboard (localhost:5173):

1. **Check if Vite is running**: Ensure Session 3 says `Local: http://localhost:5173/` and `Network: http://<internal-ip>:5173/`.
2. **SSH Tunneling (Recommended)**: 
   > [!IMPORTANT]
   > Run this command from a terminal on **YOUR OWN COMPUTER** (not the Lightsail server), replacing the path to your private key:
   
   `ssh -i /path/to/your-key.pem -L 5173:localhost:5173 -L 8000:localhost:8000 ubuntu@your-lightsail-ip`
   
   Then open `http://localhost:5173` in your browser.
3. **Public IP Access**: If not using a tunnel, open `http://<your-lightsail-ip>:5173`. Ensure ports 5173 and 8000 are open in the Lightsail Firewall settings.
