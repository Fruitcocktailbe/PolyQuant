"""
Validator Agent for PolyQuant 2.0 - Phase 3

The Validator Agent is the quality control phase of the pipeline. It takes
the dependencies identified by the Logic Architect and verifies them for
correctness, completeness, and edge cases.

RESPONSIBILITIES:
-----------------
1. Verify logical constraints are mathematically consistent
2. Check for edge cases in market resolution criteria
3. Cross-reference with actual market descriptions
4. Flag any inconsistencies or potential issues
5. Output validated constraints for the optimizer

WHY GEMINI 2.0 FLASH THINKING?
------------------------------
Gemini 2.0 Flash Thinking was chosen for validation because:
1. Strong reasoning capabilities similar to o1
2. Cost-effective compared to OpenAI's reasoning models
3. Good at catching edge cases and logical inconsistencies
4. Native JSON output support

VALIDATION PROCESS:
-------------------
1. Constraint Consistency: Are all constraints mathematically satisfiable?
2. Coverage Check: Do constraints cover all relevant outcomes?
3. Edge Case Detection: Are there resolution edge cases we're missing?
4. Confidence Calibration: Are confidence scores well-calibrated?

USAGE:
------
    validator = ValidatorAgent()
    
    async with validator:
        validated = await validator.validate(analysis_result)
        
        if validated.is_valid:
            # Send to optimizer
            pass
        else:
            for issue in validated.issues:
                print(f"Issue: {issue}")
"""

import asyncio
import json
from datetime import datetime
from typing import Any

from polyquant.utils.llm_client import call_llm_json
from pydantic import BaseModel, Field

from polyquant.agents.logic_architect import AnalysisResult, LogicalConstraint
from polyquant.data import MarketDependency
from polyquant.utils import config, get_logger

logger = get_logger(__name__)


def _is_partition_shape(constraint: LogicalConstraint) -> bool:
    """A constraint is treated as partition-shaped when every coefficient is
    +1.0 (within float noise) and the RHS is close to 1.0. This covers the
    structural family the architect is most likely to hallucinate members
    into — NegRisk sums, native partitions, cross-market partitions."""
    if not constraint.coefficients:
        return False
    if abs(constraint.rhs - 1.0) > 0.05:
        return False
    return all(abs(c - 1.0) < 1e-6 for c in constraint.coefficients.values())


def _expected_partition_tokens(cluster: Any) -> set[str]:
    """Derive the authoritative token set for a cluster's partition shape.

    For binary polar markets (YES/NO labelled), a partition sum ranges over
    YES legs only — mirroring `build_partition_constraint`. For non-polar
    multi-outcome markets (native partitions), all outcome tokens participate.
    Returns an empty set when the cluster isn't amenable to this check
    (e.g. LLM-only cross-event clusters where the architect is the authority).
    """
    if cluster is None:
        return set()

    from polyquant.utils.market_utils import (
        get_yes_outcome,
        has_binary_polarity,
    )

    source = getattr(cluster, "constraint_source", "")
    markets = getattr(cluster, "markets", []) or []
    expected: set[str] = set()

    if source == "native_partition" and len(markets) == 1:
        for outcome in markets[0].outcomes:
            token = outcome.token_id or outcome.outcome_id
            if token:
                expected.add(token)
        return expected

    # NegRisk / cross_market_partition / cross_event clusters: sum-to-1 over
    # the YES leg of each polar market in the cluster.
    for m in markets:
        if not has_binary_polarity(m):
            continue
        yes = get_yes_outcome(m)
        if yes is None:
            continue
        token = yes.token_id or yes.outcome_id
        if token:
            expected.add(token)
    return expected


def _deterministic_partition_check(
    constraint: LogicalConstraint,
    expected_tokens: set[str],
) -> str | None:
    """If `constraint` is partition-shaped, assert its coefficient set matches
    `expected_tokens` exactly. Returns a failure reason string, or None when
    the check passes or does not apply.

    Catches two hallucination modes:
    - LLM invents a token id that isn't in the cluster (phantom member).
    - LLM omits a market from a NegRisk group, leaving the sum short.
    """
    if not _is_partition_shape(constraint):
        return None
    actual_tokens = set(constraint.coefficients.keys())
    phantom = actual_tokens - expected_tokens
    missing = expected_tokens - actual_tokens
    if phantom:
        return f"phantom_tokens={sorted(phantom)[:3]}"
    if missing:
        # Some architect outputs legitimately drop low-volume NegRisk legs. We
        # only flag when more than one is missing — a single drop is within
        # the cleanup behaviour we expect.
        if len(missing) > 1:
            return f"missing_tokens={sorted(missing)[:3]}"
    return None


class ValidationIssue(BaseModel):
    """
    Represents an issue found during validation.
    
    Attributes:
        severity: 'error' (blocks execution), 'warning' (proceed with caution)
        category: Type of issue (consistency, edge_case, coverage, etc.)
        description: Human-readable description
        affected_constraints: Which constraints are affected
        suggested_fix: How to address the issue (if known)
    """
    severity: str = Field(default="warning")
    category: str = Field(default="general")
    description: str
    affected_constraints: list[str] = Field(default_factory=list)
    suggested_fix: str = ""


class ValidatedResult(BaseModel):
    """
    The output of the validation process.
    
    Contains the original analysis with validation status and any
    issues that were found.
    
    Attributes:
        original: The original analysis from Logic Architect
        is_valid: Whether the analysis passed validation
        issues: List of issues found (if any)
        validated_dependencies: Dependencies that passed validation
        validated_constraints: Constraints that passed validation
        adjusted_confidences: Updated confidence scores after validation
        validation_notes: General notes from the validator
        validated_at: When validation was performed
    """
    original: AnalysisResult
    is_valid: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)
    validated_dependencies: list[MarketDependency] = Field(default_factory=list)
    validated_constraints: list[LogicalConstraint] = Field(default_factory=list)
    adjusted_confidences: dict[str, float] = Field(default_factory=dict)
    market_exchanges: dict[str, str] = Field(default_factory=dict)
    validation_notes: str = ""
    validated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def error_count(self) -> int:
        """Number of blocking errors."""
        return len([i for i in self.issues if i.severity == "error"])
    
    @property
    def warning_count(self) -> int:
        """Number of non-blocking warnings."""
        return len([i for i in self.issues if i.severity == "warning"])


class ValidatorAgent:
    """
    Phase 3: Constraint Validation and Edge Case Detection
    
    The Validator Agent uses Gemini 2.0 Flash Thinking to verify the
    logical constraints from the Logic Architect are correct and complete.
    
    Architecture:
    - Uses Gemini Flash Thinking for rigorous logical verification
    - Multi-pass validation for different aspects
    - Confidence calibration based on validation results
    
    Example:
        validator = ValidatorAgent()
        
        async with validator:
            validated = await validator.validate(analysis_result)
            
            if validated.is_valid:
                print("All constraints validated!")
            else:
                for issue in validated.issues:
                    print(f"[{issue.severity}] {issue.description}")
    """
    
    # Validation system prompt for Gemini
    VALIDATION_PROMPT = """You are a rigorous logical validator for prediction market arbitrage constraints.

Your task is to verify that logical constraints between markets are:
1. CONSISTENT: The constraints don't contradict each other
2. COMPLETE: Important relationships aren't missing
3. CORRECT: The logic accurately reflects the market descriptions
4. EDGE-CASE-FREE: Resolution criteria edge cases are handled

VALIDATION CHECKLIST:
□ Do the constraint coefficients make mathematical sense?
□ Are mutual exclusivity relationships correct?
□ Are implication directions correct (A->B vs B->A)?
□ Are there edge cases in market resolution that could break constraints?
□ Are confidence scores calibrated appropriately?

NOTE: Do NOT filter on snapshot-time market conditions (current spread, depth,
liquidity, time-to-resolution). Those are evaluated by the Navigator at trade
time. Your job is structural validation: is the constraint mathematically and
logically correct? A constraint with currently-thin liquidity is still a valid
constraint — it may become tradeable when prices move.

PARTIAL FILTERING IS REQUIRED. When you find a problem, do NOT mark the entire
analysis invalid. Instead, flag the specific bad constraints by their
constraint_id in the issue's `affected_constraints` array, and the downstream
pipeline will filter them out individually. Good constraints in the same
cluster must still survive. Only flag a constraint with severity="error" when
it is mathematically inconsistent or logically broken — use severity="warning"
for items that are merely suboptimal or worth a human glance.

COMMON EDGE CASES TO CHECK:
- "Win by X points" vs "Win outright" - different resolutions
- Date/time boundaries and timezone issues
- "Official" results vs "projected" results
- Tiebreaker scenarios
- Market manipulation or unusual resolutions

OUTPUT FORMAT (JSON):
{
    "is_valid": true/false,
    "issues": [
        {
            "severity": "error" or "warning",
            "category": "consistency/completeness/correctness/edge_case/liquidity",
            "description": "What the issue is",
            "affected_constraints": ["constraint_id_1"],
            "suggested_fix": "How to fix it"
        }
    ],
    "adjusted_confidences": {
        "constraint_id": 0.7
    },
    "execution_warnings": [
        "Constraint c1: Spread cost (0.03) may eat into profit (0.05)"
    ],
    "validation_notes": "General observations about the constraints"
}

Be thorough and conservative. Flag anything that could cause issues."""

    def __init__(self):
        """Initialize the Validator Agent."""
        self._llm_available = False
        
        logger.info("ValidatorAgent initialized")
    
    async def __aenter__(self) -> "ValidatorAgent":
        """Async context manager - check LLM availability."""
        from polyquant.utils.llm_client import get_llm_client
        self._llm_available = get_llm_client() is not None
        if not self._llm_available:
            logger.error(
                "LLM not available — Validator is running in No-LLM mode. "
                "MapMaker will refuse to persist any cluster while in this state."
            )
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager - cleanup."""
        pass
    
    async def validate(
        self,
        analysis: AnalysisResult,
        *,
        cluster: Any = None,
    ) -> ValidatedResult:
        """
        Validate an analysis result from the Logic Architect.

        This is the main entry point. It performs multiple validation
        passes and aggregates the results.

        Validation Passes:
        1. Deterministic partition-shape check (when the originating cluster
           is provided — catches hallucinated membership without an LLM call)
        2. Mathematical consistency check (LLM)
        3. Edge case detection (LLM)
        4. Confidence calibration

        Args:
            analysis: AnalysisResult from the Logic Architect
            cluster: Optional MarketCluster the analysis came from. When
                supplied, enables the deterministic partition-shape check
                that rejects constraints whose token set doesn't match the
                cluster's actual outcomes. Kept optional for back-compat
                with callers that only have the AnalysisResult.

        Returns:
            ValidatedResult with validation status and any issues
        """
        if not self._llm_available:
            logger.info("LLM not available, skipping validation")
            return ValidatedResult(
                original=analysis,
                is_valid=True,  # Assume valid in No-LLM mode
                validated_dependencies=analysis.dependencies,
                validated_constraints=analysis.constraints,
                validation_notes="No-LLM mode: Validation skipped."
            )

        logger.info(
            "Validating analysis",
            cluster_id=analysis.cluster_id,
            dependency_count=len(analysis.dependencies),
            constraint_count=len(analysis.constraints),
        )

        # Deterministic partition-shape pass. Runs before the LLM call so we
        # can short-circuit without burning quota when the architect clearly
        # hallucinated a partition member set.
        expected_tokens = _expected_partition_tokens(cluster) if cluster else set()
        deterministic_issues: list[ValidationIssue] = []
        if expected_tokens:
            for cons in analysis.constraints:
                reason = _deterministic_partition_check(cons, expected_tokens)
                if reason:
                    deterministic_issues.append(
                        ValidationIssue(
                            severity="error",
                            category="consistency",
                            description=(
                                "Partition-shape constraint failed "
                                f"deterministic token-set check ({reason})."
                            ),
                            affected_constraints=[cons.constraint_id],
                            suggested_fix=(
                                "Re-run architect; coefficient keys must "
                                "match the cluster's outcome tokens exactly."
                            ),
                        )
                    )

        # Format the analysis for the prompt
        analysis_text = self._format_analysis(analysis)

        try:
            # Call Gemini for validation
            response = await self._call_gemini(analysis_text)
            result = self._parse_response(response, analysis)
        except Exception as e:
            logger.error("Validation failed", error=str(e))
            # Return a result indicating validation couldn't complete
            result = ValidatedResult(
                original=analysis,
                is_valid=False,
                issues=[
                    ValidationIssue(
                        severity="error",
                        category="validation_failure",
                        description=f"Validation process failed: {str(e)}",
                    )
                ],
            )

        # Merge deterministic findings in and drop any constraints they
        # flagged — they override the LLM verdict because they're structural
        # facts, not model output.
        if deterministic_issues:
            blocked_ids = {
                cid
                for issue in deterministic_issues
                for cid in issue.affected_constraints
            }
            result.issues = list(result.issues) + deterministic_issues
            result.validated_constraints = [
                c for c in result.validated_constraints
                if c.constraint_id not in blocked_ids
            ]
            result.is_valid = len(result.validated_constraints) > 0

        logger.info(
            "Validation complete",
            is_valid=result.is_valid,
            error_count=result.error_count,
            warning_count=result.warning_count,
            deterministic_rejections=len(deterministic_issues),
        )

        return result
    
    async def validate_single_constraint(
        self,
        constraint: LogicalConstraint,
    ) -> tuple[bool, list[ValidationIssue]]:
        """
        Validate a single constraint in isolation.

        Useful for quick checks during development or debugging.

        Args:
            constraint: The constraint to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        # Create a minimal analysis with just this constraint
        analysis = AnalysisResult(
            cluster_id="single_constraint",
            constraints=[constraint],
        )
        
        result = await self.validate(analysis)
        return result.is_valid, result.issues
    
    def _format_analysis(self, analysis: AnalysisResult) -> str:
        """
        Format the analysis for the validation prompt.
        """
        sections = [
            f"CLUSTER ID: {analysis.cluster_id}",
            f"ANALYZED AT: {analysis.analyzed_at.isoformat()}",
            "",
            "DEPENDENCIES:",
        ]
        
        for i, dep in enumerate(analysis.dependencies):
            sections.append(
                f"  {i+1}. {dep.source_market_id}/{dep.source_outcome} "
                f"--[{dep.relationship}]--> "
                f"{dep.target_market_id}/{dep.target_outcome} "
                f"(confidence: {dep.confidence:.2f})"
            )
        
        sections.append("")
        sections.append("CONSTRAINTS:")
        
        for i, cons in enumerate(analysis.constraints):
            sections.append(
                f"  {i+1}. [{cons.constraint_id}] {cons.description}\n"
                f"      Coefficients: {cons.coefficients}\n"
                f"      RHS: {cons.rhs}\n"
                f"      Confidence: {cons.confidence:.2f}\n"
                f"      Reasoning: {cons.reasoning}"
            )
        
        if analysis.edge_cases:
            sections.append("")
            sections.append("KNOWN EDGE CASES:")
            for case in analysis.edge_cases:
                sections.append(f"  - {case}")
        
        return "\n".join(sections)
    
    async def _call_gemini(self, analysis_text: str) -> dict[str, Any]:
        """
        Call LLM via OpenRouter for validation.
        """
        logger.debug("Calling LLM for constraint validation")

        result = await asyncio.to_thread(
            call_llm_json,
            prompt=f"ANALYSIS TO VALIDATE:\n\n{analysis_text}",
            system_prompt=self.VALIDATION_PROMPT,
            temperature=0.1,
            model=config.llm_model_validator,
        )
        
        if not result:
            raise ValueError("LLM returned empty response")
        
        return result
    
    def _parse_response(
        self,
        response: dict[str, Any],
        original: AnalysisResult,
    ) -> ValidatedResult:
        """
        Parse the Gemini response into a ValidatedResult.
        """
        issues = []
        for issue_data in response.get("issues", []):
            issues.append(
                ValidationIssue(
                    severity=issue_data.get("severity", "warning"),
                    category=issue_data.get("category", "general"),
                    description=issue_data.get("description", ""),
                    affected_constraints=issue_data.get("affected_constraints", []),
                    suggested_fix=issue_data.get("suggested_fix", ""),
                )
            )
        
        # Filter constraints with severity="error" individually. A cluster
        # is valid as long as at least one constraint survives — partial
        # filtering replaces the old all-or-nothing verdict.
        error_constraint_ids: set[str] = set()
        for issue in issues:
            if issue.severity == "error":
                error_constraint_ids.update(issue.affected_constraints)

        validated_constraints = [
            cons for cons in original.constraints
            if cons.constraint_id not in error_constraint_ids
        ]

        validated_dependencies = list(original.dependencies)

        is_valid = len(validated_constraints) > 0

        return ValidatedResult(
            original=original,
            is_valid=is_valid,
            issues=issues,
            validated_dependencies=validated_dependencies,
            validated_constraints=validated_constraints,
            adjusted_confidences=response.get("adjusted_confidences", {}),
            validation_notes=response.get("validation_notes", ""),
        )


# Convenience function
async def validate_analysis(analysis: AnalysisResult) -> ValidatedResult:
    """
    Convenience function to validate an analysis result.
    
    Example:
        validated = await validate_analysis(analysis)
        if validated.is_valid:
            proceed_to_optimizer(validated)
    """
    async with ValidatorAgent() as validator:
        return await validator.validate(analysis)
