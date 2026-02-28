"""
Logic Architect Agent for PolyQuant 2.0 - Phase 2

The Logic Architect is the reasoning engine of the PolyQuant system. It
analyzes market clusters from the Discovery Agent and identifies formal
logical dependencies between markets.

RESPONSIBILITIES:
-----------------
1. Analyze market descriptions for logical relationships
2. Extract IF/THEN rules (e.g., "IF Trump wins PA THEN >70% Trump wins election")
3. Identify edge cases in resolution criteria
4. Output structured constraint matrices for the optimizer

WHY GEMINI 2.0 FLASH THINKING?
---------------------------
Gemini 2.0 Flash Thinking was chosen for this task because:
1. Strong reasoning capabilities with visible "thinking" process
2. Excellent at logical deduction and constraint extraction
3. Consolidates all AI to a single provider (Google)
4. Cost-effective with generous rate limits

HOW IT WORKS:
-------------
1. Receive market clusters from Discovery Agent
2. For each cluster, analyze all market descriptions
3. Use Chain-of-Thought prompting to identify logical rules
4. Formalize rules into constraint format: A^T × z ≥ b
5. Return MarketDependency objects for the Validator

CONSTRAINT FORMAT:
-----------------
The output constraint matrix represents logical implications:

    A^T × z ≥ b

Where:
- z is a binary vector (0/1) of market outcomes
- A is the constraint coefficient matrix
- b is the right-hand side vector

Example: "If market M1=Yes THEN market M2=Yes" becomes:
    z[M2] - z[M1] ≥ 0  (i.e., M2 ≥ M1)

USAGE:
------
    architect = LogicArchitect()
    
    async with architect:
        # Analyze a cluster of related markets
        dependencies = await architect.analyze_cluster(cluster)
        
        for dep in dependencies:
            print(f"If {dep.source_outcome} then {dep.target_outcome}")
            print(f"Confidence: {dep.confidence}")
"""

import json
from datetime import datetime
from typing import Any

from polyquant.utils.llm_client import call_llm_json
from pydantic import BaseModel, Field

from polyquant.agents.discovery import MarketCluster
from polyquant.data import Market, MarketDependency
from polyquant.utils import config, get_logger

logger = get_logger(__name__)


class LogicalConstraint(BaseModel):
    """
    A formal logical constraint between market outcomes.
    
    Represents a rule of the form: A^T × z ≥ b
    where z is the binary outcome vector.
    
    Attributes:
        constraint_id: Unique identifier
        description: Human-readable description of the rule
        coefficients: Dict mapping outcome_id -> coefficient in A matrix
        rhs: Right-hand side value (b)
        confidence: How confident the agent is (0-1)
        source_markets: Markets involved in this constraint
        reasoning: The logical reasoning that led to this constraint
    """
    constraint_id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()))
    description: str
    coefficients: dict[str, float] = Field(default_factory=dict)
    rhs: float = 0.0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_markets: list[str] = Field(default_factory=list)
    reasoning: str = ""

    # Phase 6: Execution metadata
    liquidity_score: str | None = None         # "high", "medium", "low"
    min_tradeable_size: float | None = None    # USD, based on order book depth
    urgency: str | None = None                 # "high", "medium", "low"
    arbitrage_type: str | None = None          # "buy_all", "sell_all", "price_violation"
    expected_profit_pct: float | None = None   # Expected profit percentage


class AnalysisResult(BaseModel):
    """
    Complete analysis result from the Logic Architect.
    
    Contains both the high-level dependencies and the formal
    constraint matrices ready for the optimizer.
    
    Attributes:
        cluster_id: Which cluster was analyzed
        dependencies: List of identified dependencies
        constraints: Formal constraints in matrix form
        edge_cases:  List of potential edge cases or exceptions
        analyzed_at: Timestamp of analysis
    """
    cluster_id: str
    dependencies: list[MarketDependency] = Field(default_factory=list)
    constraints: list[LogicalConstraint] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class LogicArchitect:
    """
    Phase 2: Logical Dependency Detection and Formalization
    
    The Logic Architect uses Gemini 2.0 Flash Thinking to analyze market descriptions
    and extract formal logical constraints.
    
    Architecture:
    - Uses Google Generative AI SDK with Chain-of-Thought prompting
    - Multi-step reasoning for complex dependencies
    - Outputs both human-readable and machine-readable formats
    
    Example:
        architect = LogicArchitect()
        
        async with architect:
            result = await architect.analyze_cluster(cluster)
            
            for dep in result.dependencies:
                print(f"Rule: {dep.source_outcome} -> {dep.target_outcome}")
            
            for constraint in result.constraints:
                print(f"Constraint: {constraint.coefficients} >= {constraint.rhs}")
    """
    
    # System prompt explaining the task to DeepSeek-R1
    # Uses Chain-of-Thought structure for better reasoning
    # System prompt explaining the task to DeepSeek-R1 / Gemini 2.0 Flash Thinking
    # Uses Chain-of-Thought structure for better reasoning
    ANALYSIS_PROMPT = """You are the **Logic Architect** for a high-frequency arbitrage system trading on Polymarket.

YOUR GOAL:
Identify **TRADEABLE Logical Constraints** between prediction market outcomes that create arbitrage opportunities.
Focus on constraints with sufficient liquidity to execute profitably.

═══════════════════════════════════════════════════════════════
SECTION 1: POLYMARKET MECHANICS (READ THIS FIRST)
═══════════════════════════════════════════════════════════════

**NegRisk Markets:**
- Special market type where outcomes are MUTUALLY EXCLUSIVE + EXHAUSTIVE
- Prices MUST sum to exactly 1.0 (e.g., all presidential candidates)
- If price sum < 0.98: **BUY ARBITRAGE opportunity** (underpriced)
- If price sum > 1.02: **SELL ARBITRAGE opportunity** (overpriced)
- You will see "TYPE: NegRisk Group <ID>" in market metadata
- NegRisk markets are the HIGHEST CONFIDENCE constraints (confidence: 0.95+)

**Conditional Markets:**
- Markets that depend on a parent market resolving first
- Example: "Will X win IF Y happens?" depends on Y
- Check for "CONDITIONAL PARENT" in metadata
- If A is conditional on B, then: P(A) <= P(B) MUST hold

**Resolution Mechanics:**
- Markets resolve based on OFFICIAL sources (not projections)
- Check "RESOLUTION CRITERIA" in descriptions for edge cases
- Time zones matter: "11:59 PM ET" vs "11:59 PM UTC" are different
- "By end of 2024" means different things for different markets

═══════════════════════════════════════════════════════════════
SECTION 2: LIQUIDITY FILTERING (CRITICAL)
═══════════════════════════════════════════════════════════════

You will receive ORDER BOOK data for each outcome:
- **Bid/Ask**: Top of the order book (best prices)
- **Spread**: Ask - Bid (smaller = more liquid)
- **Depth**: $ available to trade

**LIQUIDITY THRESHOLDS:**
- **IGNORE** outcomes with "NO ORDER BOOK" or "ILLIQUID" status
- **IGNORE** outcomes with Spread > 0.10 (too wide, can't profit)
- **REQUIRE** at least $500 depth on BOTH sides for constraints
- **PRIORITIZE** constraints where all outcomes have >$2000 depth

**Output Liquidity Scores:**
For each constraint, include:
```json
"liquidity_score": "high",  // "high" (>$2K), "medium" ($500-$2K), "low" (<$500)
"min_tradeable_size": 1000.00  // USD, based on min depth across all legs
```

═══════════════════════════════════════════════════════════════
SECTION 3: RELATIONSHIP TAXONOMY (Enhanced)
═══════════════════════════════════════════════════════════════

**A. MUTUALLY_EXCLUSIVE (Disjoint)**
Two outcomes cannot BOTH be TRUE.
Constraint: `z[A] + z[B] <= 1`
Price Hint: Sum <= 1.0 (e.g., $0.60 + $0.35 = $0.95)
Liquidity Check: Both outcomes must be tradeable

**B. PARTITION (Exhaustive)**
Outcomes are mutually exclusive AND cover all possibilities.
Constraint: `sum(z[i]) == 1`
Price Hint: Sum ~1.0 (tolerance: 0.98-1.02 for FAIR, outside = ARBITRAGE)

**CRITICAL - NegRisk Deviation Detection:**
For NegRisk markets, ALWAYS flag deviations from 1.0:
- If sum = 0.95 → "UNDERPRICED_PARTITION, arbitrage_type: buy_all_outcomes, deviation: -5.0%"
- If sum = 1.04 → "OVERPRICED_PARTITION, arbitrage_type: sell_all_outcomes, deviation: +4.0%"

**C. SUBSET (Implication)**
If A happens, B MUST happen.
Constraint: `z[A] <= z[B]` or `z[B] - z[A] >= 0`
Price Hint: P(A) <= P(B) MUST hold (with 2% tolerance for noise)

**VIOLATION CHECK:**
If P(A) > P(B) + 0.02, this is INVALID or an arbitrage signal.

Liquidity Check: Sizes must match (can't buy $10k of A if only $2k of B available)

**D. NUMERICAL_DEVIATION (NEW)**
Price deviations from theoretical relationships.
Example: "Trump wins PA" at 0.60, "Trump wins election" at 0.55
Theoretical: P(election) >= P(PA), so this is a 5% violation

Output:
```json
{
    "relationship": "NUMERICAL_DEVIATION",
    "expected_relationship": "P(election) >= P(PA)",
    "actual_prices": {"election": 0.55, "PA": 0.60},
    "price_deviation_pct": 8.3,  // (0.60-0.55)/0.60
    "arbitrage_signal": "buy election, sell PA"
}
```

**E. CAUSAL_GROUP**
Complex multi-market dependencies.
Example: "Dems win Senate" + "Dems win House" → higher P("Dems win both chambers")
Use when simple SUBSET/PARTITION doesn't capture the logic.

═══════════════════════════════════════════════════════════════
SECTION 4: TIME-TO-RESOLUTION URGENCY
═══════════════════════════════════════════════════════════════

Markets closing soon have higher urgency:
- **<24 hours**: HIGH urgency (immediate arbitrage)
- **1-7 days**: MEDIUM urgency
- **>7 days**: LOW urgency (prices may converge naturally)

Include in output:
```json
"urgency": "high",  // or "medium" or "low"
"closes_in_hours": 12.5
```

═══════════════════════════════════════════════════════════════
SECTION 5: REAL-WORLD DATA HINTS
═══════════════════════════════════════════════════════════════

You will receive market data including **PRICE**, **VOLUME**, and **ORDER BOOKS**.
- **Use Price Checks**: If you think A implies B, but Price(A) > Price(B), **YOU ARE WRONG**. Reject it.
- **Use Volume**: High volume markets are "truth anchors". Trust them more.
- **Use Liquidity**: Only create constraints for tradeable outcomes (see SECTION 2)
- **Ignore Small Gaps**: Prices are noisy. $0.99 + $0.02 = $1.01 might still be a Partition.

═══════════════════════════════════════════════════════════════
SECTION 6: OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Output JSON immediately following your `<thinking>` block.

```json
{
    "dependencies": [
        {
            "source_market_id": "...",
            "source_outcome": "Yes",
            "target_market_id": "...",
            "target_outcome": "Yes",
            "relationship": "SUBSET",
            "confidence": 0.95,
            "reasoning": "If Trump wins PA, he must win the election (PA is subset)",
            "liquidity_score": "high",
            "min_tradeable_size": 2000.0,
            "urgency": "high",
            "price_deviation_pct": 5.2
        }
    ],
    "constraints": [
        {
            "constraint_id": "c1",
            "description": "NegRisk partition (UNDERPRICED): prices sum to 0.95",
            "coefficients": {"outcome1": 1.0, "outcome2": 1.0, "outcome3": 1.0},
            "operator": ">=",
            "rhs": 0.95,
            "confidence": 0.98,
            "arbitrage_type": "buy_all_outcomes",
            "expected_profit_pct": 5.0,
            "liquidity_score": "high"
        }
    ],
    "edge_cases": [
        "Market 'By end of 2024' may resolve on Dec 31 11:59 PM ET or UTC",
        "Check official source: AP vs NYT for election calls"
    ]
}
```

═══════════════════════════════════════════════════════════════
SECTION 7: REASONING PROCESS (Chain-of-Thought)
═══════════════════════════════════════════════════════════════

Before outputting JSON, you MUST use a `<thinking>` block:

1. **Identify NegRisk Groups**: Find all "TYPE: NegRisk" markets
2. **Calculate Price Sums**: For each group, sum all outcome prices
3. **Detect Deviations**: Flag any sums outside 0.98-1.02 range
4. **Check Liquidity**: Verify min $500 depth on all sides (reject ILLIQUID outcomes)
5. **Find Implications**: Look for A→B relationships, validate P(A) <= P(B)
6. **Check Time Urgency**: Flag markets closing <24h
7. **Verify Execution**: Ensure tradeable sizes match across constraint legs

═══════════════════════════════════════════════════════════════
CRITICAL RULES:
═══════════════════════════════════════════════════════════════
✓ ALWAYS check liquidity before creating constraints
✓ ALWAYS quantify deviations numerically (percentage)
✓ ALWAYS flag NegRisk price sum deviations
✓ REJECT constraints with any ILLIQUID outcome or spread > 0.10
✓ PRIORITIZE constraints with urgency="high"
✓ Include execution guidance (which direction to trade)

"""

    def __init__(self, polymarket_client=None):
        """
        Initialize the Logic Architect.

        Args:
            polymarket_client: Optional PolymarketClient for order book fetching (Phase 3)
        """
        self._llm_available = False
        self._polymarket_client = polymarket_client
        self._order_book_cache = {}
        self._order_book_timestamps = {}

        logger.info("LogicArchitect initialized", has_client=polymarket_client is not None)
    
    async def __aenter__(self) -> "LogicArchitect":
        """Async context manager - check LLM availability."""
        from polyquant.utils.llm_client import get_llm_client
        self._llm_available = get_llm_client() is not None
        if not self._llm_available:
            logger.warning("LLM not available - running in No-LLM mode")
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager - cleanup."""
        # Gemini SDK doesn't require explicit cleanup
        pass
    
    async def analyze_cluster(self, cluster: MarketCluster) -> AnalysisResult:
        """
        Analyze a cluster of markets for logical dependencies.
        
        This is the main entry point for the Logic Architect. It takes
        a cluster from the Discovery Agent and returns a full analysis.
        
        Algorithm:
        1. Format market information for the prompt
        2. Call DeepSeek-R1 with Chain-of-Thought prompting
        3. Parse the structured response
        4. Convert to typed MarketDependency and LogicalConstraint objects
        
        Args:
            cluster: MarketCluster from the Discovery Agent
            
        Returns:
            AnalysisResult with dependencies, constraints, and edge cases
        """
        if not self._llm_available:
            logger.info("LLM not available - using fallback heuristics")
            return self._apply_fallback_heuristics(cluster)
        
        logger.info(
            "Analyzing cluster",
            cluster_id=cluster.cluster_id,
            market_count=len(cluster.markets),
            topic=cluster.topic,
        )

        # Phase 3: Pre-fetch all order books in parallel for liquidity filtering
        if self._polymarket_client:
            order_books = await self._fetch_all_order_books(cluster.markets)
            self._order_book_cache.update(order_books)
            logger.debug("Order books cached", count=len(order_books))

        # Step 1: Format market data for the prompt (now includes order book data)
        market_descriptions = self._format_markets(cluster.markets)
        
        # Step 2: Call Gemini (or use fallback if unavailable)
        if not self._llm_available:
            logger.info("LLM not available - using fallback heuristics")
            return self._apply_fallback_heuristics(cluster)


        try:
            response = await self._call_gemini(market_descriptions)
        except Exception as e:
            logger.error("Gemini API call failed - using fallback heuristics", error=str(e))
            return self._apply_fallback_heuristics(cluster)
        
        # Step 3: Parse the response
        result = self._parse_response(response, cluster)

        # Step 4: Validate the result
        result = self._validate_constraints(result, cluster)

        logger.info(
            "Analysis complete",
            cluster_id=cluster.cluster_id,
            dependencies_found=len(result.dependencies),
            constraints_found=len(result.constraints),
            edge_cases=len(result.edge_cases),
        )

        return result
    
    async def analyze_pair(
        self,
        market1: Market,
        market2: Market,
    ) -> list[MarketDependency]:
        """
        Analyze a specific pair of markets for dependencies.
        
        Use this when you already have two markets you suspect are related.
        More focused than analyze_cluster().
        
        Args:
            market1: First market
            market2: Second market
            
        Returns:
            List of dependencies between the two markets
        """
        # Create a mini-cluster and analyze
        cluster = MarketCluster(
            markets=[market1, market2],
            topic=f"{market1.question[:30]} vs {market2.question[:30]}",
        )
        
        result = await self.analyze_cluster(cluster)
        return result.dependencies
    
    def _format_markets(self, markets: list[Market]) -> str:
        """
        Format market data with comprehensive arbitrage information.

        Phase 3 Enhancement: Includes order books, full resolution criteria (500+ chars),
        temporal data, and cluster-level hints for better LLM reasoning.

        Includes:
        - Order book data (bid/ask/spread/depth) for liquidity filtering
        - Full resolution criteria (500+ chars vs 300)
        - Temporal data (end_date, hours remaining, urgency)
        - Market type and conditional parent
        - Pre-calculated cluster hints (price sums, deviations)
        """
        from datetime import datetime

        formatted = []

        # Pre-calculate cluster hints (price sums, deviations, urgency)
        cluster_hints = self._calculate_cluster_hints(markets)

        for market in markets:
            # === HEADER ===
            header = f"MARKET ID: {market.market_id}"

            # NegRisk + Type metadata
            if market.negrisk:
                header += f" (TYPE: NegRisk Group {market.group_id or 'Unknown'})"
            if hasattr(market, 'market_type') and market.market_type:
                header += f" [Type: {market.market_type}]"

            # === TEMPORAL DATA ===
            time_info = f"VOLUME: ${market.volume:,.0f}"
            if market.end_date:
                now = datetime.utcnow()
                hours_remaining = (market.end_date - now).total_seconds() / 3600
                if hours_remaining > 0:
                    urgency = "HIGH" if hours_remaining < 24 else "MEDIUM" if hours_remaining < 168 else "LOW"
                    time_info += f"\nCLOSES: {market.end_date.isoformat()} ({hours_remaining:.1f}h remaining, URGENCY: {urgency})"

            # === RESOLUTION CRITERIA (Enhanced - 500+ chars) ===
            resolution = self._extract_resolution_criteria(market.description)

            # === CONDITIONAL PARENT (if applicable) ===
            conditional_info = ""
            if hasattr(market, 'conditional_parent_id') and market.conditional_parent_id:
                conditional_info = f"\nCONDITIONAL PARENT: {market.conditional_parent_id}"

            # === OUTCOMES WITH ORDER BOOK DATA ===
            outcomes_list = []
            for o in market.outcomes:
                ob = self._get_order_book_cached(o.token_id or o.outcome_id)

                if ob:
                    # Extract order book data
                    bid = getattr(ob, 'best_bid', None) or 0.0
                    ask = getattr(ob, 'best_ask', None) or 0.0
                    spread = getattr(ob, 'spread', None) or (ask - bid if ask and bid else 0.0)

                    # Calculate depth
                    if hasattr(ob, 'total_bid_depth'):
                        bid_depth = ob.total_bid_depth()
                    elif hasattr(ob, 'bids'):
                        bid_depth = sum(level.size for level in ob.bids) if ob.bids else 0.0
                    else:
                        bid_depth = 0.0

                    if hasattr(ob, 'total_ask_depth'):
                        ask_depth = ob.total_ask_depth()
                    elif hasattr(ob, 'asks'):
                        ask_depth = sum(level.size for level in ob.asks) if ob.asks else 0.0
                    else:
                        ask_depth = 0.0

                    # Liquidity status
                    status = "LIQUID" if spread < 0.05 and min(bid_depth, ask_depth) > 500 else "ILLIQUID"

                    outcomes_list.append(
                        f"  - {o.name} (Token: {o.token_id or o.outcome_id}):\n"
                        f"    Mid: ${o.price:.3f} | Bid: ${bid:.3f} | Ask: ${ask:.3f} | Spread: {spread:.3f}\n"
                        f"    Depth: ${bid_depth:,.0f} bid / ${ask_depth:,.0f} ask | Status: {status}"
                    )
                else:
                    # No order book data available
                    outcomes_list.append(
                        f"  - {o.name} (Token: {o.token_id or o.outcome_id}): ${o.price:.3f} [NO ORDER BOOK]"
                    )

            outcomes_str = "\n".join(outcomes_list)

            # === ASSEMBLE MARKET BLOCK ===
            market_block = (
                f"{header}\n"
                f"QUESTION: {market.question}\n"
                f"{time_info}\n"
                f"RESOLUTION CRITERIA:\n{resolution}\n"
                f"{conditional_info}"
                f"OUTCOMES:\n{outcomes_str}\n"
            )
            formatted.append(market_block)

        # === CLUSTER SUMMARY ===
        summary = self._format_cluster_hints(cluster_hints)

        separator = "=" * 60
        return (
            f"{separator}\n"
            f"CLUSTER SUMMARY:\n{summary}\n"
            f"{separator}\n\n" +
            f"\n{separator}\n".join(formatted) +
            f"\n{separator}"
        )

    def _calculate_cluster_hints(self, markets: list[Market]) -> dict[str, Any]:
        """
        Pre-calculate cluster-level hints to reduce LLM cognitive load.

        Returns:
            Dict with:
            - NegRisk group price sums + deviations
            - High urgency market list (<24h)
            - Price violation flags
        """
        from datetime import datetime, timedelta

        hints = {
            "total_markets": len(markets),
            "negrisk_groups": {},
            "high_urgency_markets": [],
        }

        # Group NegRisk markets
        negrisk_by_group = {}
        for m in markets:
            if m.negrisk and m.group_id:
                if m.group_id not in negrisk_by_group:
                    negrisk_by_group[m.group_id] = []
                negrisk_by_group[m.group_id].append(m)

        # Calculate price sums for NegRisk groups
        for group_id, group_markets in negrisk_by_group.items():
            total = sum(
                m.outcomes[0].price for m in group_markets
                if m.outcomes
            )
            deviation = total - 1.0
            hints["negrisk_groups"][group_id] = {
                "market_count": len(group_markets),
                "price_sum": round(total, 4),
                "deviation_pct": round(deviation * 100, 2),
                "status": "UNDERPRICED" if total < 0.98 else "OVERPRICED" if total > 1.02 else "FAIR",
            }

        # Detect high urgency markets
        now = datetime.utcnow()
        for m in markets:
            if m.end_date:
                hours_remaining = (m.end_date - now).total_seconds() / 3600
                if hours_remaining < 24 and hours_remaining > 0:
                    hints["high_urgency_markets"].append({
                        "market_id": m.market_id,
                        "question": m.question[:60],
                        "hours_remaining": round(hours_remaining, 1),
                    })

        return hints

    def _format_cluster_hints(self, hints: dict[str, Any]) -> str:
        """Format cluster hints for the prompt."""
        lines = [f"Total Markets: {hints['total_markets']}"]

        # NegRisk groups
        if hints["negrisk_groups"]:
            lines.append("\nNegRisk Groups Detected:")
            for group_id, data in hints["negrisk_groups"].items():
                lines.append(
                    f"  - Group {group_id}: {data['market_count']} markets, "
                    f"Sum={data['price_sum']:.4f} ({data['status']}), "
                    f"Deviation={data['deviation_pct']:+.2f}%"
                )

        # High urgency
        if hints["high_urgency_markets"]:
            lines.append("\nHigh Urgency Markets (closing <24h):")
            for m in hints["high_urgency_markets"][:5]:  # Limit to 5
                lines.append(f"  - {m['question']} ({m['hours_remaining']:.1f}h)")

        return "\n".join(lines)

    def _extract_resolution_criteria(self, description: str) -> str:
        """
        Extract resolution criteria (up to 500+ chars).

        Strategy: Find keywords ("resolves", "official source"),
        extract from sentence start to 600 chars at sentence boundary.
        """
        if not description or len(description) <= 500:
            return description or "N/A"

        keywords = ["resolves", "resolution", "criteria", "official", "source", "determine"]
        lower = description.lower()

        # Find first keyword
        best_idx = -1
        for keyword in keywords:
            idx = lower.find(keyword)
            if idx != -1:
                if best_idx == -1 or idx < best_idx:
                    best_idx = idx

        if best_idx != -1:
            # Start from sentence beginning
            start = description.rfind('.', 0, best_idx) + 1
            start = max(0, start)

            # End at sentence boundary after 600 chars
            end = min(start + 600, len(description))
            period_idx = description.find('.', end) + 1
            end = period_idx if period_idx > 0 else len(description)

            return description[start:end].strip()

        # Fallback: first 500 chars at sentence boundary
        end = min(500, len(description))
        period_idx = description.find('.', end) + 1
        return description[:period_idx if period_idx > 0 else end].strip() + "..."

    def _get_order_book_cached(self, token_id: str):
        """
        Fetch order book with 5-minute caching.

        Strategy:
        - Check in-memory cache (TTL: 5 min)
        - Return None if unavailable (graceful degradation)
        - Production: Pre-fetch step handles this
        """
        from datetime import datetime

        if not hasattr(self, '_order_book_cache'):
            self._order_book_cache = {}
            self._order_book_timestamps = {}

        now = datetime.utcnow()
        TTL = 300  # 5 minutes

        # Check cache
        if token_id in self._order_book_cache:
            age = (now - self._order_book_timestamps.get(token_id, now)).total_seconds()
            if age < TTL:
                return self._order_book_cache[token_id]

        # Cache miss - return None for now
        # Production: Pre-fetch step handles this
        return None

    async def _fetch_all_order_books(self, markets: list[Market]) -> dict[str, Any]:
        """
        Pre-fetch all order books in parallel (call at start of analyze_cluster).
        """
        import asyncio

        if not hasattr(self, '_polymarket_client') or not self._polymarket_client:
            return {}

        tasks = []
        token_ids = []
        for m in markets:
            for o in m.outcomes:
                if o.token_id:
                    tasks.append(self._polymarket_client.get_order_book(o.token_id))
                    token_ids.append(o.token_id)

        if not tasks:
            return {}

        # Fetch in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        order_books = {}
        for token_id, result in zip(token_ids, results):
            if not isinstance(result, Exception) and result is not None:
                order_books[token_id] = result

        logger.info(
            "Pre-fetched order books",
            requested=len(token_ids),
            successful=len(order_books),
        )

        return order_books

    async def _call_gemini(self, market_descriptions: str) -> dict[str, Any]:
        """
        Call the LLM via OpenRouter for market analysis.
        """
        import asyncio
        
        logger.debug("Calling LLM for constraint analysis")
        
        result = await asyncio.to_thread(
            call_llm_json,
            prompt=f"Analyze these markets for logical dependencies:\n\n{market_descriptions}",
            system_prompt=self.ANALYSIS_PROMPT,
            temperature=config.llm_temperature,
        )
        
        if not result:
            raise ValueError("LLM returned empty response")
        
        return result
    
    def _parse_response(
        self,
        response: dict[str, Any],
        cluster: MarketCluster,
    ) -> AnalysisResult:
        """
        Parse DeepSeek response into typed objects.
        """
        dependencies = []
        for dep_data in response.get("dependencies", []):
            try:
                dep = MarketDependency(
                    source_market_id=dep_data.get("source_market_id", ""),
                    source_outcome=dep_data.get("source_outcome", ""),
                    target_market_id=dep_data.get("target_market_id", ""),
                    target_outcome=dep_data.get("target_outcome", ""),
                    relationship=dep_data.get("relationship", "implies"),
                    confidence=float(dep_data.get("confidence", 0.5)),
                    reasoning=dep_data.get("reasoning", ""),
                    # Phase 6: Execution metadata from enhanced LLM output
                    liquidity_score=dep_data.get("liquidity_score"),
                    min_tradeable_size=float(dep_data["min_tradeable_size"]) if dep_data.get("min_tradeable_size") else None,
                    urgency=dep_data.get("urgency"),
                    price_deviation_pct=float(dep_data["price_deviation_pct"]) if dep_data.get("price_deviation_pct") else None,
                )
                dependencies.append(dep)
            except Exception as e:
                logger.warning("Failed to parse dependency", error=str(e))
        
        constraints = []
        for cons_data in response.get("constraints", []):
            try:
                # Handle sense (default to >= if not present)
                sense = cons_data.get("sense", ">=")
                coeffs = cons_data.get("coefficients", {})
                rhs = float(cons_data.get("rhs", 0))

                if sense == "<=":
                    # Convert to >= by negating coefficients and RHS
                    coeffs = {k: -float(v) for k, v in coeffs.items()}
                    rhs = -rhs
                
                # Note: '=' constraints could be handled as two inequalities,
                # but for now we treat them as >= for simplicity or error.
                # Ideally the prompt produces <= or >=.
                
                cons = LogicalConstraint(
                    description=cons_data.get("description", ""),
                    coefficients=coeffs,
                    rhs=rhs,
                    confidence=float(cons_data.get("confidence", 0.99)),
                    reasoning=cons_data.get("reasoning", ""),
                    source_markets=[m.market_id for m in cluster.markets],
                    # Phase 6: Execution metadata from enhanced LLM output
                    liquidity_score=cons_data.get("liquidity_score"),
                    min_tradeable_size=float(cons_data["min_tradeable_size"]) if cons_data.get("min_tradeable_size") else None,
                    urgency=cons_data.get("urgency"),
                    arbitrage_type=cons_data.get("arbitrage_type"),
                    expected_profit_pct=float(cons_data["expected_profit_pct"]) if cons_data.get("expected_profit_pct") else None,
                )
                constraints.append(cons)
            except Exception as e:
                logger.warning("Failed to parse constraint", error=str(e), data=cons_data)
        
        # Parse Primitives (New Logic Scout feature)
        # We can store them in edge_cases or a new field, but let's just log them for now
        primitives = response.get("primitives", [])
        if primitives:
            logger.info("Identified primitives", primitives=primitives)
        
        edge_cases = response.get("edge_cases", [])
        
        return AnalysisResult(
            cluster_id=cluster.cluster_id,
            dependencies=dependencies,
            constraints=constraints,
            edge_cases=edge_cases,
        )

    def _validate_constraints(
        self,
        result: AnalysisResult,
        cluster: MarketCluster,
    ) -> AnalysisResult:
        """
        Validate constraints against market data.

        This performs sanity checks on the LLM's output:
        1. Price consistency: If A implies B, price(A) <= price(B)
        2. Mutual exclusion: If A and B are mutually exclusive, price(A) + price(B) <= 1
        3. Partition: If outcomes form a partition, sum of prices ~= 1

        Args:
            result: Analysis result from LLM
            cluster: Original market cluster with price data

        Returns:
            Filtered AnalysisResult with only valid constraints
        """
        logger.debug("Validating constraints", cluster_id=cluster.cluster_id)

        # Build price lookup
        price_map: dict[str, float] = {}
        for market in cluster.markets:
            for outcome in market.outcomes:
                price_map[outcome.outcome_id] = outcome.price

        # Filter dependencies based on price consistency
        valid_dependencies = []
        for dep in result.dependencies:
            is_valid, reason = self._check_dependency_validity(dep, price_map)
            if is_valid:
                valid_dependencies.append(dep)
            else:
                logger.warning(
                    "Rejected dependency - price inconsistency",
                    dependency=f"{dep.source_outcome} -> {dep.target_outcome}",
                    reason=reason,
                )

        # Filter constraints (basic sanity checks)
        valid_constraints = []
        for constraint in result.constraints:
            if self._check_constraint_sanity(constraint, price_map):
                valid_constraints.append(constraint)
            else:
                logger.warning(
                    "Rejected constraint - failed sanity check",
                    constraint_id=constraint.constraint_id,
                )

        # Return filtered result
        return AnalysisResult(
            cluster_id=result.cluster_id,
            dependencies=valid_dependencies,
            constraints=valid_constraints,
            edge_cases=result.edge_cases,
            analyzed_at=result.analyzed_at,
        )

    def _check_dependency_validity(
        self,
        dep: MarketDependency,
        price_map: dict[str, float],
    ) -> tuple[bool, str]:
        """
        Check if a dependency is consistent with prices.

        ENHANCED (Week 5): Now validates IMPLICATION relationships against prices.

        For SUBSET relationships (A implies B):
        - price(A) should be <= price(B)
        - If price(A) > price(B), this is a violation (arbitrage opportunity OR error)

        Args:
            dep: Dependency to check
            price_map: Mapping of outcome_id -> price

        Returns:
            (is_valid, reason)
        """
        # We need to map dependency to actual outcome prices
        # Current limitation: Dependencies use market_id, but price_map uses outcome_id
        # For now, we implement basic checks that can be enhanced later

        if dep.relationship in ["SUBSET", "implies", "IMPLICATION"]:
            # A implies B: price(A) <= price(B) should hold
            # We need to find prices for source and target outcomes

            source_price = None
            target_price = None

            # Try to find prices (this requires outcome_id matching)
            # For now, we'll do a heuristic check if outcome names match
            for outcome_id, price in price_map.items():
                # Check if outcome_id contains market_id (simple heuristic)
                if dep.source_market_id in outcome_id:
                    source_price = price
                if dep.target_market_id in outcome_id:
                    target_price = price

            # If we found both prices, validate
            if source_price is not None and target_price is not None:
                tolerance = 0.05  # 5% tolerance for price noise

                if source_price > (target_price + tolerance):
                    # VIOLATION: Price(A) > Price(B) but A implies B
                    logger.warning(
                        "IMPLICATION VIOLATION detected",
                        source_market=dep.source_market_id,
                        target_market=dep.target_market_id,
                        source_price=f"{source_price:.4f}",
                        target_price=f"{target_price:.4f}",
                        violation=f"P(A)={source_price:.4f} > P(B)={target_price:.4f}",
                        reasoning="If A implies B, then P(A) <= P(B) must hold",
                    )
                    return False, f"price_violation: P(A)={source_price:.4f} > P(B)={target_price:.4f}"
                else:
                    logger.debug(
                        "IMPLICATION validated",
                        source_price=f"{source_price:.4f}",
                        target_price=f"{target_price:.4f}",
                        relationship=dep.relationship,
                    )

        # Accept if no violation detected
        return True, "accepted"

    def _check_constraint_sanity(
        self,
        constraint: LogicalConstraint,
        price_map: dict[str, float],
    ) -> bool:
        """
        Enhanced sanity check for constraints.

        ENHANCED (Week 5): Now validates partition constraints against price sums.

        Args:
            constraint: Constraint to check
            price_map: Mapping of outcome_id -> price

        Returns:
            True if constraint passes sanity checks
        """
        # Check 1: Coefficients are reasonable (not extreme values)
        for coeff in constraint.coefficients.values():
            if abs(coeff) > 1000:  # Unreasonably large
                logger.warning(
                    "Constraint rejected: extreme coefficient",
                    constraint_id=constraint.constraint_id,
                    coeff=coeff,
                )
                return False

        # Check 2: RHS is reasonable
        if abs(constraint.rhs) > 1000:
            logger.warning(
                "Constraint rejected: extreme RHS",
                constraint_id=constraint.constraint_id,
                rhs=constraint.rhs,
            )
            return False

        # Check 3: At least one non-zero coefficient
        if all(abs(c) < 1e-9 for c in constraint.coefficients.values()):
            logger.warning(
                "Constraint rejected: all coefficients near zero",
                constraint_id=constraint.constraint_id,
            )
            return False

        # Check 4: PARTITION constraint price consistency
        # If constraint looks like sum(z) = 1 or sum(z) >= X, validate against prices
        all_positive_unit_coeffs = all(
            abs(c - 1.0) < 0.01 for c in constraint.coefficients.values()
        )

        if all_positive_unit_coeffs:
            # This is a partition-like constraint (sum of outcomes)
            # Calculate actual price sum for these outcomes
            actual_sum = 0.0
            found_count = 0

            for outcome_id in constraint.coefficients.keys():
                if outcome_id in price_map:
                    actual_sum += price_map[outcome_id]
                    found_count += 1

            if found_count == len(constraint.coefficients):
                # We have all prices
                tolerance = 0.1  # 10% tolerance

                # Check if RHS is consistent with actual price sum
                if abs(constraint.rhs - actual_sum) > tolerance:
                    logger.warning(
                        "Partition constraint price mismatch",
                        constraint_id=constraint.constraint_id,
                        rhs=f"{constraint.rhs:.4f}",
                        actual_price_sum=f"{actual_sum:.4f}",
                        deviation=f"{abs(constraint.rhs - actual_sum):.4f}",
                        reasoning="RHS deviates significantly from actual price sum",
                    )
                    # Don't reject - this might be intentional for arbitrage detection
                    # But log the warning

        return True

    def _apply_fallback_heuristics(
        self,
        cluster: MarketCluster,
    ) -> AnalysisResult:
        """
        Apply heuristic constraint detection when LLM fails.

        ENHANCED (Week 5): Dynamic constraint formulation based on actual price sums.
        This captures arbitrage opportunities from price deviations.

        This provides a safety net for common cases:
        1. NegRisk markets: Dynamic partition constraints
        2. Price-based detection: Prices summing to ~1.0
        3. Price deviation detection: Underpriced/overpriced markets

        Args:
            cluster: Market cluster to analyze

        Returns:
            AnalysisResult with heuristically-derived constraints
        """
        logger.info("Applying fallback heuristics with deviation detection", cluster_id=cluster.cluster_id)

        constraints = []
        dependencies = []

        # Define tolerance thresholds
        TIGHT_TOLERANCE = 0.02  # 2% - considered "fair price"
        LOOSE_TOLERANCE = 0.05  # 5% - still valid but with deviation

        # Heuristic 1: NegRisk markets form partitions (with dynamic constraints)
        for market in cluster.markets:
            if market.negrisk and len(market.outcomes) > 1:
                # Calculate actual price sum
                price_sum = sum(o.price for o in market.outcomes)

                # Dynamic constraint based on deviation
                coeffs = {o.outcome_id: 1.0 for o in market.outcomes}

                if price_sum < (1.0 - TIGHT_TOLERANCE):
                    # UNDERPRICED: Use inequality constraint
                    # sum(z) >= price_sum (allows buying up to fair value)
                    rhs = price_sum
                    constraint_type = "UNDERPRICED_PARTITION"
                    reasoning = (
                        f"Automatic NegRisk partition (UNDERPRICED). "
                        f"Prices sum to {price_sum:.4f} < 1.0. "
                        f"Constraint: sum(z) >= {rhs:.4f} allows buy arbitrage."
                    )

                elif price_sum > (1.0 + TIGHT_TOLERANCE):
                    # OVERPRICED: Convert to >= by negating
                    # Original: sum(z) <= price_sum
                    # Converted: -sum(z) >= -price_sum
                    coeffs = {o.outcome_id: -1.0 for o in market.outcomes}
                    rhs = -price_sum
                    constraint_type = "OVERPRICED_PARTITION"
                    reasoning = (
                        f"Automatic NegRisk partition (OVERPRICED). "
                        f"Prices sum to {price_sum:.4f} > 1.0. "
                        f"Constraint: sum(z) <= {price_sum:.4f} allows sell arbitrage."
                    )

                else:
                    # FAIR PRICE: Use equality (represented as standard constraint)
                    rhs = 1.0
                    constraint_type = "FAIR_PARTITION"
                    reasoning = (
                        f"Automatic NegRisk partition (FAIR). "
                        f"Prices sum to {price_sum:.4f} ≈ 1.0. "
                        f"Constraint: sum(z) = 1.0 (standard partition)."
                    )

                constraints.append(LogicalConstraint(
                    description=f"{constraint_type} for market {market.market_id}",
                    coefficients=coeffs,
                    rhs=rhs,
                    confidence=0.95,  # High confidence for NegRisk
                    reasoning=reasoning,
                    source_markets=[market.market_id],
                ))

                logger.info(
                    "Created dynamic NegRisk constraint",
                    market_id=market.market_id,
                    price_sum=f"{price_sum:.4f}",
                    constraint_type=constraint_type,
                    rhs=f"{rhs:.4f}",
                )

        # Heuristic 2: General markets with price-based partitions (with dynamic constraints)
        for market in cluster.markets:
            if market.negrisk:
                continue  # Already handled above

            if len(market.outcomes) < 2:
                continue  # Single outcome can't form partition

            price_sum = sum(o.price for o in market.outcomes)

            # Only create constraint if prices suggest partition structure
            if (1.0 - LOOSE_TOLERANCE) <= price_sum <= (1.0 + LOOSE_TOLERANCE):
                coeffs = {o.outcome_id: 1.0 for o in market.outcomes}

                # Determine constraint type based on deviation
                if price_sum < (1.0 - TIGHT_TOLERANCE):
                    rhs = price_sum
                    constraint_type = "UNDERPRICED_IMPLIED_PARTITION"
                    reasoning = (
                        f"Price-based partition (UNDERPRICED). "
                        f"Prices sum to {price_sum:.4f}. "
                        f"Constraint: sum(z) >= {rhs:.4f}."
                    )

                elif price_sum > (1.0 + TIGHT_TOLERANCE):
                    # Convert to <= by negating
                    coeffs = {o.outcome_id: -1.0 for o in market.outcomes}
                    rhs = -price_sum
                    constraint_type = "OVERPRICED_IMPLIED_PARTITION"
                    reasoning = (
                        f"Price-based partition (OVERPRICED). "
                        f"Prices sum to {price_sum:.4f}. "
                        f"Constraint: sum(z) <= {price_sum:.4f}."
                    )

                else:
                    rhs = 1.0
                    constraint_type = "FAIR_IMPLIED_PARTITION"
                    reasoning = (
                        f"Price-based partition (FAIR). "
                        f"Prices sum to {price_sum:.4f} ≈ 1.0."
                    )

                constraints.append(LogicalConstraint(
                    description=f"{constraint_type} for market {market.market_id}",
                    coefficients=coeffs,
                    rhs=rhs,
                    confidence=0.7,  # Medium confidence for non-NegRisk
                    reasoning=reasoning,
                    source_markets=[market.market_id],
                ))

                logger.info(
                    "Created dynamic implied partition constraint",
                    market_id=market.market_id,
                    price_sum=f"{price_sum:.4f}",
                    constraint_type=constraint_type,
                )

        return AnalysisResult(
            cluster_id=cluster.cluster_id,
            dependencies=dependencies,
            constraints=constraints,
            edge_cases=["Generated using enhanced fallback heuristics with deviation detection"],
        )


# Convenience function
async def analyze_markets(cluster: MarketCluster) -> AnalysisResult:
    """
    Convenience function to analyze a market cluster.
    
    Example:
        result = await analyze_markets(cluster)
        print(f"Found {len(result.dependencies)} dependencies")
    """
    async with LogicArchitect() as architect:
        return await architect.analyze_cluster(cluster)
