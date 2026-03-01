# PolyQuant: Optimal Production Setup

To run PolyQuant 2.0 at its absolute peak performance (lowest latency, highest reasoning accuracy, best execution rate), you must optimize its connection to the outside world. This document outlines the ultimate, state-of-the-art configuration for LLMs, RPCs, and networking.

---

## 1. LLM Model Selection (The "Brains")

PolyQuant divides cognitive labor across three agents. Using the same model for everything is sub-optimal. Here is the best-in-class model configuration for each specific task based on the current AI landscape:

### Phase 1: Discovery Agent (The Scanner)
* **Goal**: Process hundreds of market descriptions, group them by topic, and output structured JSON, *fast*.
* **Optimal Model**: **Google Gemini 2.0 Flash** (or Anthropic Claude 3.5 Haiku)
* **Why**: Gemini 2.0 Flash is unparalleled in processing large contexts instantly and returning perfectly formatted JSON at a fraction of the cost. You want this agent to chew through the entire Polymarket active event list in seconds.
* **Provider**: Google AI Studio (lowest latency) or OpenRouter.

### Phase 2: Logic Architect (The Mathematician)
* **Goal**: Perform deep logical deduction to translate ambiguous market conditions into strict mathematical constraints ($A^T \cdot z \ge b$).
* **Optimal Model**: **OpenAI `o3-mini` (High Effort)** or **DeepSeek-R1**
* **Why**: This task requires genuine reasoning, not just pattern matching. `o3-mini` (set to high reasoning effort) excels at STEM, logic puzzles, and rigorous mathematical formulation without hallucinating impossible constraints. DeepSeek-R1 is an excellent open-weight alternative for mathematical density.
* **Provider**: OpenAI direct API or OpenRouter.

### Phase 3: Validator Agent (The Auditor)
* **Goal**: Double-check the Architect's math, catch edge cases, and output a final pass/fail confidence score.
* **Optimal Model**: **Anthropic Claude 3.7 Sonnet** (or GPT-4o)
* **Why**: Claude 3.7 Sonnet is currently the industry leader in coding, logic verification, and instruction following. It is phenomenal at spotting subtle logical contradictions that step-by-step reasoning models might have glazed over during generation.
* **Provider**: Anthropic direct API.

---

## 2. RPC Endpoints (The "Nerves")

When the Fast Brain (Navigator) detects an arbitrage, the limiting factor switches from "how smart is the AI" to "how fast can the Rust sidecar hit the mempool."

### Polygon (Polymarket Execution)
Polymarket's CLOB (Central Limit Order Book) requires blisteringly fast order submission and EIP-712 signature verification.
* **Good**: Alchemy or QuickNode Dedicated Node.
* **Optimal**: **bloXroute (BDN)** or a **bare-metal co-located Validator Node** (running Reth or Erigon) in the exact AWS region Polymarket hosts its CLOB matchers.
* **Why**: In competitive arbitrage, a 10ms difference means your order gets filled while the competition gets a "Price Expired" error. bloXroute propagates transactions directly to block builders.

### Base (Limitless Execution)
Limitless operates fully on-chain on the Base network (an Optimistic Rollup).
* **Optimal**: **Alchemy Dedicated Rollup Node** or a local `op-node` setup.
* **Why**: Base has very fast block times (2 seconds) but priority fees matter heavily. A dedicated node ensures your nonce fetches and transaction submissions aren't queued behind public RPC traffic.

### ⚠️ How to Ensure RPCs are Actually in Europe
If your server is in London but your Polygon RPC silently routes your transactions to Virginia, you lose the latency advantage. To guarantee European routing:
1. **Use Region-Specific Endpoints**: Many tier-1 providers (like QuickNode and bloXroute) allow you to specify the geographic region of your endpoint (e.g., `europe-west3.quiknode.pro`). Avoid "Global" endpoints that rely on Anycast GeoDNS, as they can sometimes misroute.
2. **Ping Test from your Server**: SSH into your Lightsail instance and `ping` your RPC URL. If the response time is `~2-15ms`, the node is local. If it hits `~70-90ms`, you are likely crossing the Atlantic.
3. **Run Your Own Node**: The ultimate guarantee. Run a local `Reth` or `Erigon` node on the same network or even the same bare-metal machine. This drops RPC round-trip times from `~10ms` over the internet to `<1ms` over `localhost`.

---

## 3. Network Architecture (The "Body")

To minimize latency across the board, the physical location of your PolyQuant server matters.

* **Server Hosting**: **AWS EC2 / Lightsail in `eu-west-2` (London) or `eu-west-1` (Ireland)**. 
  * *Reasoning*: If Polymarket's matching engine (CLOB) is hosted in London, co-locating your server in AWS London (`eu-west-2`) or Ireland (`eu-west-1`) minimizes the speed-of-light delay to the order book. This will give you a significant latency advantage over competitors running from the US. However, ensure your chosen RPC provider also has low-latency nodes in that specific European region to prevent the transaction submission from bottlenecking the round trip.
* **Internal Routing**: Keep the Python Map Maker, the Python Navigator, and the Rust OMS Sidecar on the **same physical machine**. 
  * *Reasoning*: The ZeroMQ IPC (Inter-Process Communication) running over `127.0.0.1` takes <0.1ms. If you split the UI/OMS and the Python logic across the internet, you introduce 20-50ms of network lag per tick.

---

## Summary: The "God Mode" `.env` Setup

If you have unlimited resources, your config should look like this:

```env
# LLM Routing (Use direct APIs for lowest latency, bypass aggregator hops)
DISCOVERY_LLM_PROVIDER="gemini"
DISCOVERY_MODEL="gemini-2.0-flash"

ARCHITECT_LLM_PROVIDER="openai"
ARCHITECT_MODEL="o3-mini"

VALIDATOR_LLM_PROVIDER="anthropic"
VALIDATOR_MODEL="claude-3.7-sonnet"

# Network (Dedicated, private RPCs)
POLYGON_RPC_URL="https://polygon-mainnet.g.alchemy.com/v2/<DEDICATED_KEY>"
BASE_RPC_URL="https://base-mainnet.g.alchemy.com/v2/<DEDICATED_KEY>"

# Execution Mode
TRADING_MODE="live"
```
