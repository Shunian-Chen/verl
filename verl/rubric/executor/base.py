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
Base class for rubric executors.

A rubric executor implements the online rubric execution pipeline:
1. Evidence Collection: Execute tool calls from evidence plans
2. Scoring: Score each verification item based on evidence
3. Aggregation: Combine scores according to aggregation rules

The executor has access to T_e^rub = T_e^pol ∪ T_e^ver (all tools).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from verl.rubric.schemas import (
    EvidenceRecord,
    ExecutableRubric,
    RubricExecutionResult,
)


class RubricExecutorConfig(BaseModel):
    """Configuration for rubric executors."""

    max_parallel_evidence: int = Field(
        default=5,
        ge=1,
        description="Maximum number of parallel evidence collection tasks",
    )
    default_timeout_ms: int = Field(
        default=30000,
        ge=0,
        description="Default timeout for tool execution in milliseconds",
    )
    fail_fast_on_gate: bool = Field(
        default=True,
        description="Stop execution early if a gate check fails",
    )
    collect_all_evidence: bool = Field(
        default=True,
        description="Collect all evidence even if some checks fail",
    )


@runtime_checkable
class ToolExecutor(Protocol):
    """Protocol for tool execution."""

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute a tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            timeout_ms: Optional timeout in milliseconds.

        Returns:
            Tool execution result.
        """
        ...


class BaseRubricExecutor(ABC):
    """
    Abstract base class for rubric executors.

    A rubric executor takes a rubric and evaluates a policy output
    by collecting evidence and scoring verification items.

    The execution pipeline is:
    1. Evidence Collection: For each evidence plan, execute the
       specified tool and record the result with provenance.
    2. Scoring: For each verification item, apply the scoring
       primitive to the collected evidence.
    3. Aggregation: Combine individual scores using the
       aggregation rule, handling gate checks.

    The executor has access to:
    - Task input (x)
    - Policy output (y)
    - Policy trace (τ) - tool calls made by the policy
    - Rubric (ρ) - the executable rubric
    - Tools (T_e^rub) - all available tools for evidence collection
    """

    def __init__(self, config: RubricExecutorConfig | None = None):
        """
        Initialize the rubric executor.

        Args:
            config: Configuration for the executor.
        """
        self.config = config or RubricExecutorConfig()

    @abstractmethod
    async def execute(
        self,
        task_input: str,
        policy_output: str,
        policy_trace: list[dict[str, Any]],
        rubric: ExecutableRubric,
        tool_executor: ToolExecutor | None = None,
    ) -> RubricExecutionResult:
        """
        Execute a rubric and compute the reward.

        Args:
            task_input: The original task description (x).
            policy_output: The policy's response (y).
            policy_trace: The policy's tool call trace (τ).
            rubric: The executable rubric (ρ).
            tool_executor: Optional tool executor for evidence collection.

        Returns:
            RubricExecutionResult containing the reward and details.
        """
        pass

    @abstractmethod
    async def collect_evidence(
        self,
        task_input: str,
        policy_output: str,
        policy_trace: list[dict[str, Any]],
        rubric: ExecutableRubric,
        tool_executor: ToolExecutor | None = None,
    ) -> list[EvidenceRecord]:
        """
        Collect evidence for all evidence plans in the rubric.

        Args:
            task_input: The original task description.
            policy_output: The policy's response.
            policy_trace: The policy's tool call trace.
            rubric: The executable rubric.
            tool_executor: Optional tool executor.

        Returns:
            List of EvidenceRecord with collected evidence.
        """
        pass

    @abstractmethod
    async def score_check(
        self,
        check_id: str,
        rubric: ExecutableRubric,
        evidence: list[EvidenceRecord],
        policy_output: str,
    ) -> float:
        """
        Score a single verification item.

        Args:
            check_id: ID of the verification item to score.
            rubric: The executable rubric.
            evidence: Collected evidence records.
            policy_output: The policy's response.

        Returns:
            Score in [0, 1] for the verification item.
        """
        pass

    @abstractmethod
    def aggregate_scores(
        self,
        rubric: ExecutableRubric,
        check_scores: dict[str, float],
    ) -> tuple[float, bool, list[str]]:
        """
        Aggregate individual check scores into final reward.

        Args:
            rubric: The executable rubric.
            check_scores: Mapping of check_id to score.

        Returns:
            Tuple of (final_score, gate_passed, failed_gate_ids).
        """
        pass
