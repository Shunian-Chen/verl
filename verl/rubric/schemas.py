# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Agentic Rubrics data structures.

This module defines the core data structures for executable rubrics:
- VerificationItem: A single verification check (v_j in V)
- EvidencePlan: A plan for collecting evidence (phi_j in Phi)
- AggregationRule: How to combine scores (Agg)
- ExecutableRubric: The complete program rho = <V, Phi, Agg>
- EvidenceRecord: Collected evidence with provenance tracking

Reference: framework.tex Section 2.4 (Definition of an Executable Rubric)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class VerificationCategory(str, Enum):
    """Category of verification item."""

    CORRECTNESS = "correctness"
    STYLE = "style"
    SAFETY = "safety"
    COMPLETENESS = "completeness"
    EFFICIENCY = "efficiency"


class ScoringPrimitive(str, Enum):
    """Built-in scoring primitives for verification."""

    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    REGEX = "regex"
    CODE_EXECUTION = "code_execution"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    LLM_JUDGE = "llm_judge"
    NUMERIC_COMPARISON = "numeric_comparison"


class EvidenceStatus(str, Enum):
    """Status of evidence collection."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


class AggregationMethod(str, Enum):
    """Method for aggregating scores."""

    WEIGHTED_SUM = "weighted_sum"
    MIN = "min"
    PRODUCT = "product"
    MEAN = "mean"


class VerificationItem(BaseModel):
    """
    A single verification item in the rubric checklist.

    Represents v_j in the verification set V.

    Attributes:
        id: Unique identifier for this verification item.
        description: Human-readable description of what is being verified.
        category: Type of verification (correctness, style, safety, etc.).
        weight: Weight for aggregation (default 1.0).
        is_gate: If True, failing this check results in total score of 0.
        scoring_primitive: The built-in primitive to use for scoring.
        scoring_config: Additional configuration for the scoring primitive.
    """

    id: str = Field(..., description="Unique identifier for this verification item")
    description: str = Field(..., description="What this item verifies")
    category: VerificationCategory = Field(
        default=VerificationCategory.CORRECTNESS,
        description="Category of verification",
    )
    weight: float = Field(default=1.0, ge=0.0, description="Weight for aggregation")
    is_gate: bool = Field(
        default=False,
        description="If True, failing this check results in total score of 0",
    )
    scoring_primitive: ScoringPrimitive = Field(
        default=ScoringPrimitive.EXACT_MATCH,
        description="The scoring primitive to use",
    )
    scoring_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration for scoring",
    )


class EvidencePlan(BaseModel):
    """
    A plan for collecting evidence for a verification item.

    Represents phi_j in the evidence plan set Phi.

    Attributes:
        check_id: The verification item ID this evidence is for.
        tool_name: Name of the tool to call for evidence collection.
        tool_arguments: Static arguments to pass to the tool.
        argument_extractor: Jinja2 template for extracting dynamic arguments
            from policy output (y) or trace (tau).
        timeout_ms: Maximum time to wait for tool execution.
        depends_on: List of check_ids that must be collected first.
    """

    check_id: str = Field(..., description="ID of the verification item this is for")
    tool_name: str = Field(..., description="Name of the tool to call")
    tool_arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Static arguments to pass to the tool",
    )
    argument_extractor: str | None = Field(
        default=None,
        description="Jinja2 template for extracting dynamic arguments from y/tau",
    )
    timeout_ms: int = Field(
        default=30000,
        ge=0,
        description="Timeout for tool execution in milliseconds",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Check IDs that must complete before this one",
    )


class AggregationRule(BaseModel):
    """
    Rule for aggregating individual check scores into final reward.

    Represents the Agg component of the rubric.

    Attributes:
        method: Aggregation method (weighted_sum, min, product, mean).
        normalize: Whether to normalize the final score to [0, 1].
        gate_threshold: Threshold below which a gate check fails (default 0.5).
    """

    method: AggregationMethod = Field(
        default=AggregationMethod.WEIGHTED_SUM,
        description="Method for combining scores",
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize final score to [0, 1]",
    )
    gate_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold below which a gate check fails",
    )


class EvidenceRecord(BaseModel):
    """
    A record of collected evidence with full provenance tracking.

    Represents e_j in the evidence set E.

    Attributes:
        check_id: The verification item ID this evidence is for.
        tool_name: Name of the tool that was called.
        tool_input: The actual arguments passed to the tool.
        tool_output: The raw output from the tool.
        success: Whether the tool execution succeeded.
        status: Status of the evidence collection (ok, error, timeout).
        error_message: Error message if execution failed.
        start_time_ms: Timestamp when execution started.
        end_time_ms: Timestamp when execution ended.
        latency_ms: Actual execution latency.
    """

    check_id: str = Field(..., description="ID of the verification item")
    tool_name: str = Field(..., description="Name of the tool called")
    tool_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Actual arguments passed to the tool",
    )
    tool_output: Any = Field(default=None, description="Raw output from the tool")
    success: bool = Field(default=False, description="Whether execution succeeded")
    status: EvidenceStatus = Field(
        default=EvidenceStatus.OK,
        description="Status of evidence collection",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if execution failed",
    )
    start_time_ms: float = Field(
        default=0.0,
        description="Timestamp when execution started",
    )
    end_time_ms: float = Field(
        default=0.0,
        description="Timestamp when execution ended",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Actual execution latency in milliseconds",
    )


class ExecutableRubric(BaseModel):
    """
    A complete executable rubric program.

    Represents rho = <V, Phi, Agg> as defined in framework.tex Section 2.4.

    The rubric is compiled from the task input during the offline phase
    and executed during the online phase to compute rewards.

    Attributes:
        id: Unique identifier for this rubric.
        task_intent: The extracted intent from the task input.
        verification_checklist: List of verification items (V).
        evidence_plans: List of evidence collection plans (Phi).
        aggregation: Rule for combining scores (Agg).
        estimated_latency_ms: Estimated total execution latency.
        metadata: Additional metadata about the rubric.
    """

    id: str = Field(..., description="Unique identifier for this rubric")
    task_intent: str = Field(
        ...,
        description="Extracted task intent from the input",
    )
    verification_checklist: list[VerificationItem] = Field(
        default_factory=list,
        description="List of verification items (V)",
    )
    evidence_plans: list[EvidencePlan] = Field(
        default_factory=list,
        description="List of evidence collection plans (Phi)",
    )
    aggregation: AggregationRule = Field(
        default_factory=AggregationRule,
        description="Rule for aggregating scores (Agg)",
    )
    estimated_latency_ms: int = Field(
        default=0,
        ge=0,
        description="Estimated execution latency in milliseconds",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the rubric",
    )

    def get_latency_bucket(self) -> int:
        """
        Get the latency bucket for this rubric.

        Buckets:
            0: < 1s (simple checks, regex)
            1: 1-5s (code execution)
            2: 5-15s (web search)
            3: 15-30s (complex simulation)
            4: > 30s (multi-tool chains)

        Returns:
            Bucket ID (0-4).
        """
        latency_s = self.estimated_latency_ms / 1000.0
        if latency_s < 1:
            return 0
        elif latency_s < 5:
            return 1
        elif latency_s < 15:
            return 2
        elif latency_s < 30:
            return 3
        else:
            return 4

    def get_check_by_id(self, check_id: str) -> VerificationItem | None:
        """Get a verification item by its ID."""
        for item in self.verification_checklist:
            if item.id == check_id:
                return item
        return None

    def get_evidence_plan_by_check_id(self, check_id: str) -> EvidencePlan | None:
        """Get the evidence plan for a specific check."""
        for plan in self.evidence_plans:
            if plan.check_id == check_id:
                return plan
        return None

    def get_gate_checks(self) -> list[VerificationItem]:
        """Get all gate checks that must pass for non-zero score."""
        return [item for item in self.verification_checklist if item.is_gate]


class RubricExecutionResult(BaseModel):
    """
    Result of executing a rubric.

    Attributes:
        reward_score: Final reward score in [0, 1].
        rubric_id: ID of the executed rubric.
        check_scores: Individual scores for each check.
        evidence_records: All collected evidence with provenance.
        gate_passed: Whether all gate checks passed.
        failed_gates: List of failed gate check IDs.
        total_latency_ms: Total execution latency.
    """

    reward_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final reward score",
    )
    rubric_id: str = Field(..., description="ID of the executed rubric")
    check_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Individual scores for each check",
    )
    evidence_records: list[EvidenceRecord] = Field(
        default_factory=list,
        description="All collected evidence",
    )
    gate_passed: bool = Field(
        default=True,
        description="Whether all gate checks passed",
    )
    failed_gates: list[str] = Field(
        default_factory=list,
        description="List of failed gate check IDs",
    )
    total_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total execution latency",
    )

    def to_reward_extra_info(self) -> dict[str, Any]:
        """Convert to the format expected by verl's reward system."""
        return {
            "rubric_id": self.rubric_id,
            "check_scores": self.check_scores,
            "evidence_count": len(self.evidence_records),
            "evidence_success_rate": (
                sum(1 for e in self.evidence_records if e.success)
                / max(len(self.evidence_records), 1)
            ),
            "gate_passed": self.gate_passed,
            "failed_gates": self.failed_gates,
            "total_latency_ms": self.total_latency_ms,
            "evidence_records": [e.model_dump() for e in self.evidence_records],
        }
