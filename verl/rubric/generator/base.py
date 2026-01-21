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
Base class for rubric generators.

A rubric generator implements the two-stage rubric synthesis pipeline:
1. Stage 1 (Intent Extraction): x -> d (direction)
2. Stage 2 (Tool Compilation): (x, d, T_e^rub) -> rho (executable rubric)

The generator only sees tool schemas, never policy outputs or trajectories,
ensuring objective evaluation criteria.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from verl.rubric.schemas import ExecutableRubric


class ToolSchema(BaseModel):
    """Schema describing a tool available for rubric execution."""

    name: str = Field(..., description="Unique name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for tool parameters",
    )
    returns: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for tool return value",
    )


class IntentDirection(BaseModel):
    """
    Output of Stage 1: Intent Extraction.

    Captures what needs to be verified without specifying how.
    """

    task_intent: str = Field(
        ...,
        description="High-level description of what the task is asking for",
    )
    verification_dimensions: list[str] = Field(
        default_factory=list,
        description="Dimensions to verify (correctness, style, safety, etc.)",
    )
    key_requirements: list[str] = Field(
        default_factory=list,
        description="Key requirements extracted from the task",
    )
    expected_output_type: str = Field(
        default="text",
        description="Expected type of output (text, code, json, etc.)",
    )


class RubricGeneratorConfig(BaseModel):
    """Configuration for rubric generators."""

    model_name: str = Field(
        default="gpt-4o",
        description="LLM model to use for generation",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for generation",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens for generation",
    )
    retry_attempts: int = Field(
        default=3,
        ge=1,
        description="Number of retry attempts on failure",
    )
    timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Timeout for API calls in seconds",
    )


class BaseRubricGenerator(ABC):
    """
    Abstract base class for rubric generators.

    A rubric generator synthesizes executable rubrics from task inputs
    during the offline preprocessing phase. It implements a two-stage
    pipeline:

    Stage 1 (Intent Extraction):
        Input: task_input, specification?, verification_data?
        Output: IntentDirection (what to verify)

    Stage 2 (Tool Compilation):
        Input: task_input, direction, tool_schemas, verification_data?
        Output: ExecutableRubric (how to verify)

    The generator only sees:
    - Task input (x)
    - Optional specification
    - Optional verification_data (unit test I/O pairs)
    - Tool schemas (T_e^rub = T_e^pol ∪ T_e^ver)

    It never sees policy outputs (y) or trajectories (τ), ensuring
    the rubric is objective and not biased by specific outputs.
    """

    def __init__(self, config: RubricGeneratorConfig | None = None):
        """
        Initialize the rubric generator.

        Args:
            config: Configuration for the generator.
        """
        self.config = config or RubricGeneratorConfig()

    @abstractmethod
    async def extract_intent(
        self,
        task_input: str,
        specification: str | None = None,
        verification_data: list[dict[str, Any]] | None = None,
    ) -> IntentDirection:
        """
        Stage 1: Extract the verification intent from the task.

        This stage identifies what needs to be verified without
        specifying how to verify it.

        Args:
            task_input: The task description/prompt (x).
            specification: Optional formal specification.
            verification_data: Optional unit test I/O pairs.

        Returns:
            IntentDirection describing what to verify.
        """
        pass

    @abstractmethod
    async def compile_rubric(
        self,
        task_input: str,
        direction: IntentDirection,
        tool_schemas: list[ToolSchema],
        verification_data: list[dict[str, Any]] | None = None,
    ) -> ExecutableRubric:
        """
        Stage 2: Compile the intent into an executable rubric.

        This stage converts the high-level intent into concrete
        verification items with tool-based evidence collection plans.

        Args:
            task_input: The task description/prompt (x).
            direction: Output from Stage 1.
            tool_schemas: Available tools (T_e^rub).
            verification_data: Optional unit test I/O pairs.

        Returns:
            ExecutableRubric ready for execution.
        """
        pass

    async def generate(
        self,
        task_input: str,
        tool_schemas: list[ToolSchema],
        specification: str | None = None,
        verification_data: list[dict[str, Any]] | None = None,
    ) -> ExecutableRubric:
        """
        Generate a complete executable rubric.

        This is the main entry point that runs both stages.

        Args:
            task_input: The task description/prompt (x).
            tool_schemas: Available tools (T_e^rub).
            specification: Optional formal specification.
            verification_data: Optional unit test I/O pairs.

        Returns:
            ExecutableRubric ready for execution.
        """
        # Stage 1: Extract intent
        direction = await self.extract_intent(
            task_input=task_input,
            specification=specification,
            verification_data=verification_data,
        )

        # Stage 2: Compile to executable rubric
        rubric = await self.compile_rubric(
            task_input=task_input,
            direction=direction,
            tool_schemas=tool_schemas,
            verification_data=verification_data,
        )

        return rubric

    def estimate_latency(self, rubric: ExecutableRubric) -> int:
        """
        Estimate the total execution latency for a rubric.

        This is used for latency-based batching during training.

        Args:
            rubric: The rubric to estimate latency for.

        Returns:
            Estimated latency in milliseconds.
        """
        total_latency = 0
        for plan in rubric.evidence_plans:
            total_latency += plan.timeout_ms
        return total_latency
