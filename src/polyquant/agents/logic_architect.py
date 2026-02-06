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

WHY DeepSeek-R1?
----------------
DeepSeek-R1 was chosen for this task because:
1. 79.8% Pass@1 on AIME 2024 - excellent mathematical reasoning
2. Reinforcement learning training optimizes for logical deduction
3. Cost-effective ($2.19/$8.79 per 1M tokens) vs OpenAI o1
4. Strong performance on multi-step proofs without external tools

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

import httpx
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
    
    The Logic Architect uses DeepSeek-R1 to analyze market descriptions
    and extract formal logical constraints.
    
    Architecture:
    - Uses DeepSeek API with Chain-of-Thought prompting
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
    ANALYSIS_PROMPT = """You are a logical analysis agent for prediction market arbitrage.

Your task is to find LOGICAL DEPENDENCIES between prediction markets.

A logical dependency exists when:
- The outcome of one market IMPLIES something about another market
- Both markets share underlying events that constrain possible outcomes
- Resolution criteria create mathematical relationships

EXAMPLES OF DEPENDENCIES:
1. "Trump wins Pennsylvania" -> increases probability of "Trump wins election"
2. "Democrats win Senate" + "Democrats win House" -> implies "Democrats control Congress"
3. "Bitcoin > $100k by March" incompatible with "Bitcoin < $80k by March"

ANALYSIS STEPS:
1. Read each market's question and resolution criteria carefully
2. Identify shared entities, events, or conditions
3. Determine if outcomes are:
   - Mutually exclusive (both cannot be true)
   - Implicative (one implies the other)
   - Correlated (share common factors)
4. Quantify the logical relationship strength

OUTPUT FORMAT (JSON):
{
    "dependencies": [
        {
            "source_market_id": "id1",
            "source_outcome": "Yes",
            "target_market_id": "id2",
            "target_outcome": "Yes",
            "relationship": "implies",
            "confidence": 0.85,
            "reasoning": "If Trump wins PA (a key swing state with 19 electoral votes)..."
        }
    ],
    "constraints": [
        {
            "description": "PA win implies increased election probability",
            "coefficients": {"outcome_id_election_yes": 1, "outcome_id_pa_yes": -0.7},
            "rhs": 0,
            "reasoning": "Historical data shows PA winner wins election >80% of time"
        }
    ],
    "edge_cases": [
        "Market M1 resolves on popular vote, M2 on electoral college - not perfectly correlated"
    ]
}

Be thorough but conservative - only report high-confidence dependencies."""

    # DeepSeek API endpoint
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    def __init__(self):
        """Initialize the Logic Architect."""
        self._client: httpx.AsyncClient | None = None
        
        logger.info("LogicArchitect initialized")
    
    async def __aenter__(self) -> "LogicArchitect":
        """Async context manager - initialize HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=120.0,  # DeepSeek-R1 reasoning can take time
            headers={
                "Authorization": f"Bearer {config.deepseek_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager - cleanup."""
        if self._client:
            await self._client.aclose()
    
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
        if not self._client:
            raise RuntimeError("Architect not initialized. Use 'async with architect:'")
        
        logger.info(
            "Analyzing cluster",
            cluster_id=cluster.cluster_id,
            market_count=len(cluster.markets),
            topic=cluster.topic,
        )
        
        # Step 1: Format market data for the prompt
        market_descriptions = self._format_markets(cluster.markets)
        
        # Step 2: Call DeepSeek-R1
        try:
            response = await self._call_deepseek(market_descriptions)
        except Exception as e:
            logger.error("DeepSeek API call failed", error=str(e))
            return AnalysisResult(cluster_id=cluster.cluster_id)
        
        # Step 3: Parse the response
        result = self._parse_response(response, cluster)
        
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
        
        Includes all relevant information for logical analysis.
        """
        formatted = []
        
        for market in markets:
            outcomes_str = ", ".join(
                f"{o.name} (id={o.outcome_id}, price={o.price:.2f})"
                for o in market.outcomes
            )
            
            formatted.append(
                f"MARKET: {market.market_id}\n"
                f"Question: {market.question}\n"
                f"Description: {market.description[:500] if market.description else 'N/A'}\n"
                f"Outcomes: {outcomes_str}\n"
                f"Volume: ${market.volume:,.0f}\n"
            )
        
        return "\n---\n".join(formatted)
    
    async def _call_deepseek(self, market_descriptions: str) -> dict[str, Any]:
        """
        Call the DeepSeek API with the analysis prompt.
        
        Uses the R1 model for best reasoning performance.
        """
        if not self._client:
            raise RuntimeError("Client not initialized")
        
        logger.debug("Calling DeepSeek-R1")
        
        response = await self._client.post(
            self.DEEPSEEK_API_URL,
            json={
                "model": "deepseek-reasoner",  # DeepSeek-R1
                "messages": [
                    {"role": "system", "content": self.ANALYSIS_PROMPT},
                    {
                        "role": "user",
                        "content": f"Analyze these markets for logical dependencies:\n\n{market_descriptions}",
                    },
                ],
                "temperature": 0.1,  # Low temperature for consistent reasoning
                "max_tokens": 4000,
            },
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
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
                )
                dependencies.append(dep)
            except Exception as e:
                logger.warning("Failed to parse dependency", error=str(e))
        
        constraints = []
        for cons_data in response.get("constraints", []):
            try:
                cons = LogicalConstraint(
                    description=cons_data.get("description", ""),
                    coefficients=cons_data.get("coefficients", {}),
                    rhs=float(cons_data.get("rhs", 0)),
                    confidence=float(cons_data.get("confidence", 0.5)),
                    reasoning=cons_data.get("reasoning", ""),
                    source_markets=[m.market_id for m in cluster.markets],
                )
                constraints.append(cons)
            except Exception as e:
                logger.warning("Failed to parse constraint", error=str(e))
        
        edge_cases = response.get("edge_cases", [])
        
        return AnalysisResult(
            cluster_id=cluster.cluster_id,
            dependencies=dependencies,
            constraints=constraints,
            edge_cases=edge_cases,
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
