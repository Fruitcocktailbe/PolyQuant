# Polymarket Liquidity Vacuum Scanner

Real-time market scanner for Polymarket that detects trading opportunities using a "Liquidity Vacuum" strategy. It polls the Gamma and CLOB APIs, applies a three-tier filter funnel, and logs strong signals.

## Strategy

The scanner uses a three-tier funnel to efficiently find opportunities without hammering the API:

1. **Tier 1 (Light)** — Fetches all active markets, keeps only those with YES price < $0.10
2. **Tier 2 (Medium)** — Checks volume spikes (>2.5× average) and price momentum (>2% in 90s)
3. **Tier 3 (Heavy)** — Analyzes order book imbalance (>70% one-sided) and EMA(200) deviation (>12%)

## Quick Start

```bash
pip install -r requirements.txt
python scanner.py
```

Signals are logged to both console and `scanner_alerts.txt`.

## Configuration

All parameters are defined at the top of `scanner.py`:

| Parameter | Default | Description |
|---|---|---|
| `MAX_YES_PRICE` | 0.10 | Tier-1 price ceiling |
| `VOLUME_SPIKE_MULT` | 2.5 | Volume spike multiplier |
| `MOMENTUM_THRESHOLD` | 0.02 | 90-second momentum threshold |
| `ORDERBOOK_IMBALANCE` | 0.70 | Order book imbalance ratio |
| `EMA_DEVIATION` | 0.12 | EMA deviation trigger |
| `SCAN_INTERVAL` | 60s | Time between scan cycles |

## Alert Format

```
[TIMESTAMP] - [MARKET_SLUG] - [PRICE] - [VOL_MULT] - [EMA_DEV] - [SIGNAL_TYPE]
```

## Disclaimer

This tool is for informational and educational purposes only. It does not execute trades. Use at your own risk.
