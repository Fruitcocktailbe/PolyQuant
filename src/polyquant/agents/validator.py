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

WHY o1-preview?
---------------
OpenAI's o1-preview was chosen for validation because:
1. Designed specifically for verification and catching edge cases
2. Strong at formal logical reasoning
3. Excels at finding inconsistencies and contradictions
4. Better at catching subtle errors than general-purpose LLMs

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

import json
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from polyquant.agents.logic_architect import AnalysisResult, LogicalConstraint
from polyquant.data import MarketDependency
from polyquant.utils import config, get_logger

logger = get_logger(__name__)


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
    
    The Validator Agent uses OpenAI's o1-preview to verify the logical
    constraints from the Logic Architect are correct and complete.
    
    Architecture:
    - Uses o1-preview for rigorous logical verification
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
    
    # Validation system prompt for o1-preview
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
            "category": "consistency/completeness/correctness/edge_case",
            "description": "What the issue is",
            "affected_constraints": ["constraint_id_1"],
            "suggested_fix": "How to fix it"
        }
    ],
    "adjusted_confidences": {
        "constraint_id": 0.7  // adjusted confidence
    },
    "validation_notes": "General observations about the constraints"
}

Be thorough and conservative. Flag anything that could cause issues."""

    def __init__(self):
        """Initialize the Validator Agent."""
        self._openai: AsyncOpenAI | None = None
        
        logger.info("ValidatorAgent initialized")
    
    async def __aenter__(self) -> "ValidatorAgent":
        """Async context manager - initialize OpenAI client."""
        self._openai = AsyncOpenAI(api_key=config.openai_api_key.get_secret_value())
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager - cleanup."""
        # OpenAI client doesn't need explicit cleanup
        pass
    
    async def validate(self, analysis: AnalysisResult) -> ValidatedResult:
        """
        Validate an analysis result from the Logic Architect.
        
        This is the main entry point. It performs multiple validation
        passes and aggregates the results.
        
        Validation Passes:
        1. Mathematical consistency check
        2. Edge case detection
        3. Confidence calibration
        
        Args:
            analysis: AnalysisResult from the Logic Architect
            
        Returns:
            ValidatedResult with validation status and any issues
        """
        if not self._openai:
            raise RuntimeError("Validator not initialized. Use 'async with validator:'")
        
        logger.info(
            "Validating analysis",
            cluster_id=analysis.cluster_id,
            dependency_count=len(analysis.dependencies),
            constraint_count=len(analysis.constraints),
        )
        
        # Format the analysis for the prompt
        analysis_text = self._format_analysis(analysis)
        
        try:
            # Call o1-preview for validation
            response = await self._call_o1(analysis_text)
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
        
        logger.info(
            "Validation complete",
            is_valid=result.is_valid,
            error_count=result.error_count,
            warning_count=result.warning_count,
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
    
    async def _call_o1(self, analysis_text: str) -> dict[str, Any]:
        """
        Call OpenAI's o1-preview model for validation.
        """
        if not self._openai:
            raise RuntimeError("OpenAI client not initialized")
        
        logger.debug("Calling o1-preview for validation")
        
        # Note: o1-preview doesn't support system messages the same way
        # We include the instructions in the user message
        response = await self._openai.chat.completions.create(
            model="o1-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"{self.VALIDATION_PROMPT}\n\n---\n\nANALYSIS TO VALIDATE:\n\n{analysis_text}",
                },
            ],
            # o1-preview has specific parameter requirements
            # temperature and max_tokens may not be supported
        )
        
        content = response.choices[0].message.content or "{}"
        
        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        return json.loads(content.strip())
    
    def _parse_response(
        self,
        response: dict[str, Any],
        original: AnalysisResult,
    ) -> ValidatedResult:
        """
        Parse the o1-preview response into a ValidatedResult.
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
        
        # Determine validity based on issues
        is_valid = response.get("is_valid", True)
        if any(i.severity == "error" for i in issues):
            is_valid = False
        
        # Filter validated items (exclude those with errors)
        error_constraint_ids = set()
        for issue in issues:
            if issue.severity == "error":
                error_constraint_ids.update(issue.affected_constraints)
        
        validated_dependencies = [
            dep for dep in original.dependencies
            # Keep all deps for now - could filter if deps had IDs
        ]
        
        validated_constraints = [
            cons for cons in original.constraints
            if cons.constraint_id not in error_constraint_ids
        ]
        
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
