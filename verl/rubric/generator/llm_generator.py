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
LLM-based rubric generator implementation.

Uses GPT-4o or similar models to synthesize executable rubrics
through a two-stage pipeline.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from verl.rubric.schemas import (
    AggregationMethod,
    AggregationRule,
    EvidencePlan,
    ExecutableRubric,
    ScoringPrimitive,
    VerificationCategory,
    VerificationItem,
)

from .base import (
    BaseRubricGenerator,
    IntentDirection,
    RubricGeneratorConfig,
    ToolSchema,
)

logger = logging.getLogger(__name__)


# Prompt templates for the two-stage pipeline
STAGE1_INTENT_EXTRACTION_PROMPT = """You are analyzing a task to determine what needs to be verified.

## Task Input
{task_input}

{specification_section}
{verification_data_section}

## Instructions
Extract the verification intent from this task. Identify:
1. What is the high-level goal of the task?
2. What dimensions should be verified (correctness, style, safety, etc.)?
3. What are the key requirements that must be met?
4. What type of output is expected (code, text, JSON, etc.)?

Respond with a JSON object:
```json
{{
    "task_intent": "Brief description of what the task is asking for",
    "verification_dimensions": ["dimension1", "dimension2", ...],
    "key_requirements": ["requirement1", "requirement2", ...],
    "expected_output_type": "code|text|json|other"
}}
```"""

STAGE2_TOOL_COMPILATION_PROMPT = """You are compiling a verification rubric with tool-based evidence collection.

## Task Input
{task_input}

## Verification Intent
- Task Intent: {task_intent}
- Dimensions to verify: {verification_dimensions}
- Key Requirements: {key_requirements}
- Expected Output Type: {expected_output_type}

{verification_data_section}

## Available Tools
{tool_schemas_json}

## Instructions
Create an executable rubric that verifies the task requirements using the available tools.

For each verification item:
1. Create a unique ID (e.g., "check_correctness_1")
2. Describe what is being verified
3. Choose an appropriate category (correctness, style, safety, completeness, efficiency)
4. Set weight (default 1.0) and is_gate (True if critical)
5. Choose a scoring primitive (exact_match, contains, regex, code_execution, semantic_similarity, llm_judge, numeric_comparison)

For each evidence plan:
1. Link it to a verification item via check_id
2. Specify which tool to call
3. Provide static tool_arguments
4. Optionally provide an argument_extractor (Jinja2 template to extract args from policy output)
   - Use {{{{ policy_output }}}} for the raw policy output
   - Use {{{{ policy_trace }}}} for the tool call trace
5. Set an appropriate timeout_ms

Common patterns:
- For code tasks: Use code execution tools with test cases
- For factual tasks: Use lookup or search tools
- For format tasks: Use regex or contains primitives

Respond with a JSON object:
```json
{{
    "verification_checklist": [
        {{
            "id": "check_id",
            "description": "What this verifies",
            "category": "correctness|style|safety|completeness|efficiency",
            "weight": 1.0,
            "is_gate": false,
            "scoring_primitive": "exact_match|contains|regex|code_execution|semantic_similarity|llm_judge|numeric_comparison",
            "scoring_config": {{}}
        }}
    ],
    "evidence_plans": [
        {{
            "check_id": "check_id",
            "tool_name": "tool_name",
            "tool_arguments": {{}},
            "argument_extractor": "optional jinja2 template",
            "timeout_ms": 30000
        }}
    ],
    "aggregation": {{
        "method": "weighted_sum|min|product|mean",
        "normalize": true,
        "gate_threshold": 0.5
    }}
}}
```"""


class LLMRubricGenerator(BaseRubricGenerator):
    """
    LLM-based rubric generator using OpenAI-compatible APIs.

    Implements the two-stage rubric synthesis pipeline:
    1. Intent Extraction: Understands what needs to be verified
    2. Tool Compilation: Creates executable verification plans

    Example usage:
        ```python
        config = RubricGeneratorConfig(model_name="gpt-4o")
        generator = LLMRubricGenerator(config)

        tool_schemas = [
            ToolSchema(
                name="sandbox_fusion",
                description="Execute code in a sandbox",
                parameters={"type": "object", "properties": {"code": {"type": "string"}}},
                returns={"type": "object", "properties": {"output": {"type": "string"}}}
            )
        ]

        rubric = await generator.generate(
            task_input="Write a function to compute fibonacci numbers",
            tool_schemas=tool_schemas,
            verification_data=[{"input": "5", "output": "5"}]
        )
        ```
    """

    def __init__(
        self,
        config: RubricGeneratorConfig | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        """
        Initialize the LLM rubric generator.

        Args:
            config: Generator configuration.
            api_key: OpenAI API key (or set OPENAI_API_KEY env var).
            api_base: Optional custom API base URL.
        """
        super().__init__(config)
        self.api_key = api_key
        self.api_base = api_base
        self._client = None

    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package required for LLMRubricGenerator. "
                    "Install with: pip install openai"
                )

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["base_url"] = self.api_base

            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def _call_llm(self, prompt: str) -> str:
        """Make an LLM API call and return the response content."""
        client = self._get_client()

        for attempt in range(self.config.retry_attempts):
            try:
                response = await client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.config.retry_attempts}): {e}"
                )
                if attempt == self.config.retry_attempts - 1:
                    raise

        raise RuntimeError("All LLM call attempts failed")

    def _extract_json_from_response(self, response: str) -> dict:
        """Extract JSON object from LLM response."""
        # Try to find JSON block in markdown code fence
        import re

        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to parse the entire response as JSON
            json_str = response.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}\nResponse: {response}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")

    def _format_verification_data_section(
        self,
        verification_data: list[dict[str, Any]] | None,
    ) -> str:
        """Format verification data for prompt inclusion."""
        if not verification_data:
            return ""

        lines = ["## Verification Data (Test Cases)"]
        for i, case in enumerate(verification_data[:10]):  # Limit to 10 cases
            lines.append(f"### Test Case {i + 1}")
            if "input" in case:
                lines.append(f"Input: {case['input']}")
            if "output" in case:
                lines.append(f"Expected Output: {case['output']}")
            if "description" in case:
                lines.append(f"Description: {case['description']}")
            lines.append("")

        return "\n".join(lines)

    async def extract_intent(
        self,
        task_input: str,
        specification: str | None = None,
        verification_data: list[dict[str, Any]] | None = None,
    ) -> IntentDirection:
        """
        Stage 1: Extract verification intent from the task.

        Args:
            task_input: The task description/prompt.
            specification: Optional formal specification.
            verification_data: Optional unit test I/O pairs.

        Returns:
            IntentDirection describing what to verify.
        """
        spec_section = ""
        if specification:
            spec_section = f"## Specification\n{specification}\n"

        verification_section = self._format_verification_data_section(verification_data)

        prompt = STAGE1_INTENT_EXTRACTION_PROMPT.format(
            task_input=task_input,
            specification_section=spec_section,
            verification_data_section=verification_section,
        )

        response = await self._call_llm(prompt)
        data = self._extract_json_from_response(response)

        return IntentDirection(
            task_intent=data.get("task_intent", ""),
            verification_dimensions=data.get("verification_dimensions", []),
            key_requirements=data.get("key_requirements", []),
            expected_output_type=data.get("expected_output_type", "text"),
        )

    async def compile_rubric(
        self,
        task_input: str,
        direction: IntentDirection,
        tool_schemas: list[ToolSchema],
        verification_data: list[dict[str, Any]] | None = None,
    ) -> ExecutableRubric:
        """
        Stage 2: Compile intent into executable rubric.

        Args:
            task_input: The task description/prompt.
            direction: Output from Stage 1.
            tool_schemas: Available tools.
            verification_data: Optional unit test I/O pairs.

        Returns:
            ExecutableRubric ready for execution.
        """
        # Format tool schemas for the prompt
        tool_schemas_json = json.dumps(
            [schema.model_dump() for schema in tool_schemas],
            indent=2,
        )

        verification_section = self._format_verification_data_section(verification_data)

        prompt = STAGE2_TOOL_COMPILATION_PROMPT.format(
            task_input=task_input,
            task_intent=direction.task_intent,
            verification_dimensions=", ".join(direction.verification_dimensions),
            key_requirements="\n".join(f"- {req}" for req in direction.key_requirements),
            expected_output_type=direction.expected_output_type,
            verification_data_section=verification_section,
            tool_schemas_json=tool_schemas_json,
        )

        response = await self._call_llm(prompt)
        data = self._extract_json_from_response(response)

        # Parse verification checklist
        verification_items = []
        for item_data in data.get("verification_checklist", []):
            try:
                item = VerificationItem(
                    id=item_data.get("id", f"check_{uuid.uuid4().hex[:8]}"),
                    description=item_data.get("description", ""),
                    category=VerificationCategory(
                        item_data.get("category", "correctness")
                    ),
                    weight=float(item_data.get("weight", 1.0)),
                    is_gate=bool(item_data.get("is_gate", False)),
                    scoring_primitive=ScoringPrimitive(
                        item_data.get("scoring_primitive", "exact_match")
                    ),
                    scoring_config=item_data.get("scoring_config", {}),
                )
                verification_items.append(item)
            except Exception as e:
                logger.warning(f"Failed to parse verification item: {e}")

        # Parse evidence plans
        evidence_plans = []
        for plan_data in data.get("evidence_plans", []):
            try:
                plan = EvidencePlan(
                    check_id=plan_data.get("check_id", ""),
                    tool_name=plan_data.get("tool_name", ""),
                    tool_arguments=plan_data.get("tool_arguments", {}),
                    argument_extractor=plan_data.get("argument_extractor"),
                    timeout_ms=int(plan_data.get("timeout_ms", 30000)),
                    depends_on=plan_data.get("depends_on", []),
                )
                evidence_plans.append(plan)
            except Exception as e:
                logger.warning(f"Failed to parse evidence plan: {e}")

        # Parse aggregation rule
        agg_data = data.get("aggregation", {})
        aggregation = AggregationRule(
            method=AggregationMethod(agg_data.get("method", "weighted_sum")),
            normalize=bool(agg_data.get("normalize", True)),
            gate_threshold=float(agg_data.get("gate_threshold", 0.5)),
        )

        # Calculate estimated latency
        estimated_latency = sum(plan.timeout_ms for plan in evidence_plans)

        rubric = ExecutableRubric(
            id=f"rubric_{uuid.uuid4().hex[:12]}",
            task_intent=direction.task_intent,
            verification_checklist=verification_items,
            evidence_plans=evidence_plans,
            aggregation=aggregation,
            estimated_latency_ms=estimated_latency,
            metadata={
                "generator": "LLMRubricGenerator",
                "model": self.config.model_name,
                "verification_dimensions": direction.verification_dimensions,
            },
        )

        return rubric


class MockRubricGenerator(BaseRubricGenerator):
    """
    Mock rubric generator for testing.

    Generates simple rubrics without calling an LLM.
    """

    async def extract_intent(
        self,
        task_input: str,
        specification: str | None = None,
        verification_data: list[dict[str, Any]] | None = None,
    ) -> IntentDirection:
        """Generate a mock intent direction."""
        return IntentDirection(
            task_intent=f"Complete the task: {task_input[:100]}",
            verification_dimensions=["correctness"],
            key_requirements=["Task completion"],
            expected_output_type="text",
        )

    async def compile_rubric(
        self,
        task_input: str,
        direction: IntentDirection,
        tool_schemas: list[ToolSchema],
        verification_data: list[dict[str, Any]] | None = None,
    ) -> ExecutableRubric:
        """Generate a mock rubric."""
        check_id = f"check_{uuid.uuid4().hex[:8]}"

        verification_items = [
            VerificationItem(
                id=check_id,
                description="Verify task completion",
                category=VerificationCategory.CORRECTNESS,
                weight=1.0,
                is_gate=False,
                scoring_primitive=ScoringPrimitive.CONTAINS,
                scoring_config={},
            )
        ]

        evidence_plans = []
        if tool_schemas:
            evidence_plans.append(
                EvidencePlan(
                    check_id=check_id,
                    tool_name=tool_schemas[0].name,
                    tool_arguments={},
                    timeout_ms=5000,
                )
            )

        return ExecutableRubric(
            id=f"mock_rubric_{uuid.uuid4().hex[:8]}",
            task_intent=direction.task_intent,
            verification_checklist=verification_items,
            evidence_plans=evidence_plans,
            aggregation=AggregationRule(),
            estimated_latency_ms=5000,
            metadata={"generator": "MockRubricGenerator"},
        )
