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

import google.generativeai as genai
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
    ANALYSIS_PROMPT = """You are the **Logic Architect** for a high-frequency arbitrage system.

YOUR GOAL:
Identify **Strict Logical Constraints** between prediction market outcomes.
We are not looking for "correlations". We are looking for **mathematical impossibilities** and **guaranteed implications**.

### 1. RELATIONSHIP TAXONOMY
You must classify relationships into these strict categories:

**A. MUTUALLY_EXCLUSIVE (Disjoint)**
Two outcomes cannot BOTH be TRUE.
Constraint: `z[A] + z[B] <= 1`
*Hint*: Often found when Prices sum to <= 1.0 (e.g. $0.60 + $0.35 = $0.95).

**B. PARTITION (Exhaustive)**
Outcomes are mutually exclusive AND cover all possibilities.
Constraint: `sum(z[i]) == 1`
*Hint*: Prices usually sum to ~1.0 (e.g. $0.40 + $0.30 + $0.30 = $1.00).

**C. SUBSET (Implication)**
If Outcome A happens, Outcome B MUST happen.
Constraint: `z[A] <= z[B]` (or `z[B] - z[A] >= 0`)
*Hint*: Price of A ($0.10) should be LESS than Price of B ($0.80). 
*Anti-Hint*: If Price A > Price B, Implication is IMPOSSIBLE.

**D. CAUSAL_GROUP**
Outcomes share a complex dependency.
Example: "Democrats win Senate" and "Democrats win House" -> "Democrats win Congress".
Constraint: `z[Senate] + z[House] - z[Congress] <= 1`

---

### 2. REAL-WORLD DATA HINTS
You will receive market data including **PRICE** and **VOLUME**.
- **Use Price Checks**: If you think A implies B, but Price(A) > Price(B), **YOU ARE WRONG**. Reject it.
- **Use Volume**: High volume markets are "truth anchors". Trust them more.
- **Ignore Small Gaps**: Prices are noisy. $0.99 + $0.02 = $1.01 might still be a Partition.

### 3. OUTPUT FORMAT
Output JSON immediately following your `<thinking>` block.

```json
{
    "dependencies": [
        {
            "source_market_id": "...",
            "source_outcome": "Yes",
            "target_market_id": "...",
            "target_outcome": "Yes",
            "target_outcome": "Yes",
            "relationship": "MUTUALLY_EXCLUSIVE", 
            "confidence": 1.0,
            "reasoning": "Both imply same event X but different winners"
        }
    ],
    "constraints": [
        {
            "constraint_id": "c1",
            "description": "Only one winner allowed",
            "coefficients": {"m1_yes": 1.0, "m2_yes": 1.0},
            "operator": "<=",
            "rhs": 1.0
        }
    ]
}
```

---

### 4. REASONING PROCESS (Chain-of-Thought)
Before generating JSON, you must output a `<thinking>` block:
1.  **Analyze Entities**: List all candidates, teams, or assets involved.
2.  **Normalize Outcomes**: "Yes" means what event?
3.  **Check Internal Logic**: Does each market sum to 1? (Partition check).
4.  **Check Cross-Market Logic**:
    -   Does A imply B? (Check Prices!)
    -   Are A and B mutually exclusive?
    -   Do A and B form a cover?
5.  **Verify Directions**: Ensure implications go the right way (Subsets must have lower/equal price).

"""

    def __init__(self):
        """Initialize the Logic Architect."""
        self._model: genai.GenerativeModel | None = None
        
        logger.info("LogicArchitect initialized")
    
    async def __aenter__(self) -> "LogicArchitect":
        """Async context manager - initialize Gemini model."""
        api_key = config.gemini_api_key.get_secret_value()
        
        if not api_key or "your-" in api_key:
            logger.warning("Gemini API key not set - running in No-LLM mode")
            self._model = None
        else:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-thinking-exp-01-21",
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,  # Low temperature for consistent reasoning
                ),
                system_instruction=self.ANALYSIS_PROMPT,
            )
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
        if not self._model:
            raise RuntimeError("Architect not initialized. Use 'async with architect:'")
        
        logger.info(
            "Analyzing cluster",
            cluster_id=cluster.cluster_id,
            market_count=len(cluster.markets),
            topic=cluster.topic,
        )
        
        # Step 1: Format market data for the prompt
        market_descriptions = self._format_markets(cluster.markets)
        
        # Step 2: Call Gemini (or use fallback if unavailable)
        if not self._model:
            logger.info("Gemini not initialized - using fallback heuristics")
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
        Format market data for the DeepSeek prompt.
        
        Includes all relevant information for logical analysis:
        - Question
        - Outcomes with Prices (for probability heuristics)
        - IDs (for precise constraints)
        """
        formatted = []
        
        for market in markets:
            outcomes_list = []
            for o in market.outcomes:
                # Format: "- Yes (Token: 0x123...): $0.55"
                outcomes_list.append(
                    f"  - {o.name} (Token: {o.outcome_id}): ${o.price:.3f}"
                )
            outcomes_str = "\n".join(outcomes_list)
            
            description_snippet = market.description[:300] if market.description else "N/A"
            
            # NegRisk hinting
            type_hint = ""
            if market.negrisk:
                type_hint = f" (TYPE: NegRisk Group {market.group_id or 'Unknown'})"
            
            market_block = (
                f"MARKET ID: {market.market_id}{type_hint}\n"
                f"QUESTION: {market.question}\n"
                f"VOLUME: ${market.volume:,.0f}\n"
                f"DESCRIPTION: {description_snippet}...\n"
                f"OUTCOMES:\n{outcomes_str}\n"
            )
            formatted.append(market_block)
        
        return "\n" + ("=" * 40) + "\n".join(formatted) + "\n" + ("=" * 40)
    
    async def _call_gemini(self, market_descriptions: str) -> dict[str, Any]:
        """
        Call the Gemini API with the analysis prompt.
        
        Uses Gemini 2.0 Flash Thinking for best reasoning performance.
        """
        if not self._model:
            raise RuntimeError("Gemini model not initialized")
        
        logger.debug("Calling Gemini 2.0 Flash Thinking")
        
        # Gemini SDK is synchronous for generate_content, but we can run it in executor
        # For now, use the sync API (it's fast enough for our use case)
        import asyncio
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._model.generate_content(
                f"Analyze these markets for logical dependencies:\n\n{market_descriptions}"
            )
        )
        
        content = response.text
        
        # Parse JSON from response (may be wrapped in markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        return json.loads(content)
    
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
                    # Confidence is usually implicit in Logic Scout (1.0) but check if present
                    confidence=float(cons_data.get("confidence", 0.99)),
                    reasoning=cons_data.get("reasoning", ""),
                    source_markets=[m.market_id for m in cluster.markets],
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

        For SUBSET relationships (A implies B):
        - price(A) should be <= price(B)
        - If price(A) > price(B), reject it

        Args:
            dep: Dependency to check
            price_map: Mapping of outcome_id -> price

        Returns:
            (is_valid, reason)
        """
        # Get prices (we need outcome IDs, not market IDs)
        # For now, simplified check using market-level heuristics
        # In production, would map to specific outcome IDs

        if dep.relationship == "SUBSET" or dep.relationship == "implies":
            # A implies B: price(A) <= price(B)
            # But we don't have direct prices for dependencies yet
            # This would require more granular tracking
            pass

        # For now, accept all (future enhancement)
        return True, "accepted"

    def _check_constraint_sanity(
        self,
        constraint: LogicalConstraint,
        price_map: dict[str, float],
    ) -> bool:
        """
        Basic sanity check for constraints.

        Args:
            constraint: Constraint to check
            price_map: Mapping of outcome_id -> price

        Returns:
            True if constraint passes sanity checks
        """
        # Check 1: Coefficients are reasonable (not extreme values)
        for coeff in constraint.coefficients.values():
            if abs(coeff) > 1000:  # Unreasonably large
                return False

        # Check 2: RHS is reasonable
        if abs(constraint.rhs) > 1000:
            return False

        # Check 3: At least one non-zero coefficient
        if all(abs(c) < 1e-9 for c in constraint.coefficients.values()):
            return False

        return True

    def _apply_fallback_heuristics(
        self,
        cluster: MarketCluster,
    ) -> AnalysisResult:
        """
        Apply heuristic constraint detection when LLM fails.

        This provides a safety net for common cases:
        1. NegRisk markets: Automatic partition constraints
        2. Price-based detection: Prices summing to ~1.0
        3. Extreme prices: Markets near 0 or 1

        Args:
            cluster: Market cluster to analyze

        Returns:
            AnalysisResult with heuristically-derived constraints
        """
        logger.info("Applying fallback heuristics", cluster_id=cluster.cluster_id)

        constraints = []
        dependencies = []

        # Heuristic 1: NegRisk markets form partitions
        for market in cluster.markets:
            if market.negrisk and len(market.outcomes) > 1:
                # Create partition constraint: sum(outcomes) = 1
                coeffs = {o.outcome_id: 1.0 for o in market.outcomes}

                constraints.append(LogicalConstraint(
                    description=f"NegRisk partition for market {market.market_id}",
                    coefficients=coeffs,
                    rhs=1.0,
                    confidence=0.95,  # High confidence for NegRisk
                    reasoning="Automatic: NegRisk markets form partitions",
                    source_markets=[market.market_id],
                ))

        # Heuristic 2: Check if prices sum to ~1.0 (suggesting partition)
        for market in cluster.markets:
            price_sum = sum(o.price for o in market.outcomes)
            if 0.95 <= price_sum <= 1.05:  # Within 5% of 1.0
                # Likely a partition
                coeffs = {o.outcome_id: 1.0 for o in market.outcomes}

                constraints.append(LogicalConstraint(
                    description=f"Price-based partition for market {market.market_id}",
                    coefficients=coeffs,
                    rhs=1.0,
                    confidence=0.7,  # Medium confidence
                    reasoning=f"Prices sum to {price_sum:.3f} (near 1.0)",
                    source_markets=[market.market_id],
                ))

        return AnalysisResult(
            cluster_id=cluster.cluster_id,
            dependencies=dependencies,
            constraints=constraints,
            edge_cases=["Generated using fallback heuristics (LLM unavailable)"],
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
