# gemini-conversation-selection
Exported on: 18-3-2026, 21:25:37

---

**Gemini:**
Yes, it is absolutely possible to transition this into a **Live Scanner**. Instead of iterating through a local CSV or database, the script will act as a "Watchdog" that polls the Polymarket API at regular intervals (e.g., every 30–60 seconds) to find opportunities across all active markets.

## The Adjusted "Live Scanner" Prompt

Use this prompt to generate the code for a real-time monitor.

---

**Prompt:**

> "Act as a Quantitative Developer. I want to build a **Real-Time Market Scanner** for Polymarket based on a 'Liquidity Vacuum' strategy.
> 
> **1. Core Logic & Filters:**
> 
> * **Market Discovery:** Use the Gamma API (`https://gamma-api.polymarket.com/markets`) to fetch all active, non-closed markets.
> * **Initial Filter:** Only process markets where the 'Yes' price is < $0.10.
> * **Dynamic Metrics:**
> 
> * **Volume Spike:** Fetch the 10-day average volume for each filtered market. Compare against the current 1-minute volume. Trigger if `current_vol > 2.5 * avg_vol`.
> * **Momentum:** Calculate the price change over the last 90 seconds. Trigger if `change > 2%`.
> * **Imbalance:** Use the `/book` endpoint to calculate if > 70% of orders are on one side (Buy/Sell).
> 
> **2. Execution logic:**
> 
> * **Expected Price:** `expected_price = EMA(200) * (1 + sentiment_offset)`.
> * **Signal:** If a market meets all criteria and `current_price` deviates > 12% from `expected_price`, log a **'STRONG SIGNAL'**.
> 
> **3. Pipeline Architecture:**
> 
> * **Market Cache:** Since fetching 200 periods of history for *every* market is heavy, implement a `MarketTracker` class that stores the last 200 minutes of OHLCV data for active markets in memory (using a `deque`).
> * **Worker Loop:** Run a main loop every 60 seconds.
> * **Rate Limit Management:** Respect Polymarket's 2026 limits (e.g., 300 req/10s for Gamma, 1,000 req/10s for CLOB prices). Implement a cooldown if a `429 Too Many Requests` error is received.
> 
> **4. Output:**
> 
> * Maintain a live log file `scanner_alerts.txt`.
> * Format: `[TIMESTAMP] - [MARKET_SLUG] - [PRICE] - [VOL_MULT] - [EMA_DEV] - [SIGNAL_TYPE]`
> 
> Please provide the code in a clean, object-oriented Python structure using `asyncio` for concurrent API calls."
> 
> 

---

## How to Build a Solid "Live" Pipeline

When moving from backtesting to live scanning, the "Pipeline" shifts from a linear process to a **Circular Loop**. Here is the recommended architecture for your 2026 bot:

#### 1. The Three-Tier Filter (Efficiency)

To avoid hitting API limits, don't calculate the EMA for every market. Use a "Funnel":

* **Tier 1 (Light):** Get all active markets from Gamma API. Filter for `price < 0.10`.
* **Tier 2 (Medium):** For those \~50–100 markets, check current volume and price movement.
* **Tier 3 (Heavy):** Only for markets that pass Tier 2, fetch the historical 200 candles to calculate the EMA and check the Order Book for the 70% imbalance.

#### 2. The "Warm-Up" Phase

When you first start the script, it will have a "blank" memory. Your code needs a `warm_up()` function that:

1. Downloads the last 200 minutes of data for all candidate markets.
2. Populates the `EMA` values so the bot can trade immediately upon start.

#### 3. Handling the "90-Second" Rule

Standard API candles are usually 1-minute or 5-minute. To accurately track a **90-second** window:

* Store 1-minute candles.
* Keep a "Real-Time Price" variable that updates every 10 seconds.
* Compare `current_price` (now) vs. `price` (90 seconds ago) by interpolating between the current and previous 1-minute candles.

## API Rate Limit Strategy

| Endpoint | Use Case | Limit (Approx) | Strategy |
| --- | --- | --- | --- |
| `/markets` | Discovering new opportunities | 300 req / 10s | Call once every 5 minutes. |
| `/prices-history` | Calculating EMA(200) | 1,000 req / 10s | Call only once per market at startup or when "flagged." |
| `/price` | Real-time monitoring | 1,500 req / 10s | Use for the final "Entry" check. |
| **WebSockets** | Price/Orderbook updates | No limit | **Highly Recommended:** Subscribe to the `market` channel for high-liquidity markets to get instant price changes without polling. |

Exporteren naar SpreadsheetsWould you like me to show you how to structure the **Market Cache** using Python's `collections.deque` to keep the last 200 price points efficiently?

---

