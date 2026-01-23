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
Default rubric executor implementation.

Executes rubrics by collecting evidence, scoring checks, and aggregating results.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from verl.rubric.schemas import (
    AggregationMethod,
    AggregationRule,
    EvidenceRecord,
    EvidenceStatus,
    ExecutableRubric,
    RubricExecutionResult,
    VerificationItem,
)

from .base import BaseRubricExecutor, RubricExecutorConfig, ToolExecutor
from .primitives import get_scoring_primitive

logger = logging.getLogger(__name__)


class DefaultRubricExecutor(BaseRubricExecutor):
    """
    Default implementation of the rubric executor.

    This executor:
    1. Collects evidence in parallel (respecting dependencies)
    2. Scores each verification item using the appropriate primitive
    3. Aggregates scores with gate checking

    Example usage:
        ```python
        executor = DefaultRubricExecutor()

        result = await executor.execute(
            task_input="Write a fibonacci function",
            policy_output="def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
            policy_trace=[],
            rubric=rubric,
            tool_executor=my_tool_executor,
        )

        print(f"Reward: {result.reward_score}")
        print(f"Gate passed: {result.gate_passed}")
        ```
    """

    def __init__(self, config: RubricExecutorConfig | None = None):
        """Initialize the default executor."""
        super().__init__(config)
        self._jinja_env = None

    def _get_jinja_env(self):
        """Get or create Jinja2 environment for argument extraction."""
        if self._jinja_env is None:
            try:
                from jinja2 import Environment, BaseLoader

                self._jinja_env = Environment(loader=BaseLoader())
            except ImportError:
                logger.warning("jinja2 not installed, argument extraction disabled")
                self._jinja_env = False
        return self._jinja_env

    def _extract_arguments(
        self,
        template: str,
        policy_output: str,
        policy_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Extract dynamic arguments using Jinja2 template.

        Args:
            template: Jinja2 template string.
            policy_output: The policy's response.
            policy_trace: The policy's tool call trace.

        Returns:
            Extracted arguments dictionary.
        """
        jinja_env = self._get_jinja_env()
        if not jinja_env:
            return {}

        try:
            tmpl = jinja_env.from_string(template)
            result = tmpl.render(
                policy_output=policy_output,
                policy_trace=policy_trace,
            )

            # Try to parse as JSON
            import json

            return json.loads(result)
        except Exception as e:
            logger.warning(f"Failed to extract arguments from template: {e}")
            return {}

    async def _collect_single_evidence(
        self,
        plan_idx: int,
        rubric: ExecutableRubric,
        task_input: str,
        policy_output: str,
        policy_trace: list[dict[str, Any]],
        tool_executor: ToolExecutor | None,
    ) -> EvidenceRecord:
        """Collect evidence for a single evidence plan."""
        plan = rubric.evidence_plans[plan_idx]
        start_time = time.time() * 1000

        # Build arguments
        arguments = dict(plan.tool_arguments)

        # Extract dynamic arguments if template provided
        if plan.argument_extractor:
            extracted = self._extract_arguments(
                plan.argument_extractor,
                policy_output,
                policy_trace,
            )
            arguments.update(extracted)

        # Execute tool
        tool_output = None
        success = False
        status = EvidenceStatus.OK
        error_message = None

        if tool_executor:
            try:
                result = await asyncio.wait_for(
                    tool_executor.execute(
                        tool_name=plan.tool_name,
                        arguments=arguments,
                        timeout_ms=plan.timeout_ms,
                    ),
                    timeout=plan.timeout_ms / 1000.0,
                )
                tool_output = result
                success = True
            except asyncio.TimeoutError:
                status = EvidenceStatus.TIMEOUT
                error_message = f"Tool execution timed out after {plan.timeout_ms}ms"
            except Exception as e:
                status = EvidenceStatus.ERROR
                error_message = str(e)
        else:
            # No tool executor - create a placeholder record
            status = EvidenceStatus.ERROR
            error_message = "No tool executor provided"

        end_time = time.time() * 1000

        return EvidenceRecord(
            check_id=plan.check_id,
            tool_name=plan.tool_name,
            tool_input=arguments,
            tool_output=tool_output,
            success=success,
            status=status,
            error_message=error_message,
            start_time_ms=start_time,
            end_time_ms=end_time,
            latency_ms=end_time - start_time,
        )

    async def collect_evidence(
        self,
        task_input: str,
        policy_output: str,
        policy_trace: list[dict[str, Any]],
        rubric: ExecutableRubric,
        tool_executor: ToolExecutor | None = None,
    ) -> list[EvidenceRecord]:
        """
        Collect evidence for all evidence plans.

        Evidence collection is parallelized where possible, respecting
        dependencies between plans.
        """
        if not rubric.evidence_plans:
            return []

        # Build dependency graph
        plan_indices = {
            plan.check_id: i for i, plan in enumerate(rubric.evidence_plans)
        }
        completed: set[str] = set()
        evidence_records: list[EvidenceRecord] = []

        # Process in waves based on dependencies
        remaining = set(range(len(rubric.evidence_plans)))

        while remaining:
            # Find plans that can be executed (all dependencies satisfied)
            ready = []
            for idx in remaining:
                plan = rubric.evidence_plans[idx]
                if all(dep in completed for dep in plan.depends_on):
                    ready.append(idx)

            if not ready:
                # Circular dependency or missing dependency
                logger.warning("Unable to resolve evidence plan dependencies")
                break

            # Execute ready plans in parallel (up to limit)
            semaphore = asyncio.Semaphore(self.config.max_parallel_evidence)

            async def limited_collect(idx: int) -> EvidenceRecord:
                async with semaphore:
                    return await self._collect_single_evidence(
                        idx,
                        rubric,
                        task_input,
                        policy_output,
                        policy_trace,
                        tool_executor,
                    )

            batch_results = await asyncio.gather(
                *[limited_collect(idx) for idx in ready],
                return_exceptions=True,
            )

            # Process results
            for idx, result in zip(ready, batch_results):
                if isinstance(result, Exception):
                    # Create error record
                    plan = rubric.evidence_plans[idx]
                    result = EvidenceRecord(
                        check_id=plan.check_id,
                        tool_name=plan.tool_name,
                        tool_input={},
                        tool_output=None,
                        success=False,
                        status=EvidenceStatus.ERROR,
                        error_message=str(result),
                        start_time_ms=time.time() * 1000,
                        end_time_ms=time.time() * 1000,
                        latency_ms=0,
                    )

                evidence_records.append(result)
                completed.add(rubric.evidence_plans[idx].check_id)
                remaining.remove(idx)

        return evidence_records

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
            check_id: ID of the check to score.
            rubric: The executable rubric.
            evidence: All collected evidence.
            policy_output: The policy's response.

        Returns:
            Score in [0, 1].
        """
        verification_item = rubric.get_check_by_id(check_id)
        if not verification_item:
            logger.warning(f"Unknown check ID: {check_id}")
            return 0.0

        # Get the scoring primitive
        try:
            primitive = get_scoring_primitive(verification_item.scoring_primitive)
        except ValueError:
            logger.warning(
                f"Unknown scoring primitive: {verification_item.scoring_primitive}"
            )
            return 0.0

        # Filter evidence for this check
        check_evidence = [e for e in evidence if e.check_id == check_id]

        # Score using the primitive
        try:
            score = primitive.score(verification_item, check_evidence, policy_output)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except Exception as e:
            logger.warning(f"Scoring failed for check {check_id}: {e}")
            return 0.0

    def _aggregate_with_gates(
        self,
        aggregation: AggregationRule,
        check_scores: dict[str, tuple[VerificationItem, float]],
    ) -> float:
        """
        Aggregate scores with gate handling.

        This is a helper method that takes check_scores as a dict mapping
        check_id to (VerificationItem, score) tuples.

        Args:
            aggregation: The aggregation rule to apply.
            check_scores: Dict mapping check_id to (item, score) tuples.

        Returns:
            Final aggregated score.
        """
        # Check gates first
        for check_id, (item, score) in check_scores.items():
            if item.is_gate and score < aggregation.gate_threshold:
                return 0.0

        # Collect scores and weights
        scores = []
        weights = []
        for check_id, (item, score) in check_scores.items():
            scores.append(score)
            weights.append(item.weight)

        if not scores:
            return 0.0

        # Apply aggregation method
        if aggregation.method == AggregationMethod.WEIGHTED_SUM:
            total_weight = sum(weights)
            if total_weight > 0:
                final_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            else:
                final_score = 0.0

        elif aggregation.method == AggregationMethod.MIN:
            final_score = min(scores)

        elif aggregation.method == AggregationMethod.PRODUCT:
            final_score = 1.0
            for s in scores:
                final_score *= s

        elif aggregation.method == AggregationMethod.MEAN:
            final_score = sum(scores) / len(scores)

        else:
            total_weight = sum(weights)
            if total_weight > 0:
                final_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            else:
                final_score = 0.0

        # Normalize if requested
        if aggregation.normalize:
            final_score = max(0.0, min(1.0, final_score))

        return final_score

    def aggregate_scores(
        self,
        rubric: ExecutableRubric,
        check_scores: dict[str, float],
    ) -> tuple[float, bool, list[str]]:
        """
        Aggregate individual check scores into final reward.

        Returns:
            Tuple of (final_score, gate_passed, failed_gate_ids).
        """
        agg = rubric.aggregation

        # Check gates first
        gate_passed = True
        failed_gates = []

        for item in rubric.verification_checklist:
            if item.is_gate:
                score = check_scores.get(item.id, 0.0)
                if score < agg.gate_threshold:
                    gate_passed = False
                    failed_gates.append(item.id)

        # If any gate failed, return 0
        if not gate_passed:
            return 0.0, False, failed_gates

        # Aggregate non-gate scores
        if not check_scores:
            return 0.0, True, []

        scores = []
        weights = []

        for item in rubric.verification_checklist:
            score = check_scores.get(item.id, 0.0)
            scores.append(score)
            weights.append(item.weight)

        if not scores:
            return 0.0, True, []

        # Apply aggregation method
        if agg.method == AggregationMethod.WEIGHTED_SUM:
            total_weight = sum(weights)
            if total_weight > 0:
                final_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            else:
                final_score = 0.0

        elif agg.method == AggregationMethod.MIN:
            final_score = min(scores)

        elif agg.method == AggregationMethod.PRODUCT:
            final_score = 1.0
            for s in scores:
                final_score *= s

        elif agg.method == AggregationMethod.MEAN:
            final_score = sum(scores) / len(scores)

        else:
            # Default to weighted sum
            total_weight = sum(weights)
            if total_weight > 0:
                final_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            else:
                final_score = 0.0

        # Normalize if requested
        if agg.normalize:
            final_score = max(0.0, min(1.0, final_score))

        return final_score, True, []

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

        This is the main entry point for rubric execution.
        """
        start_time = time.time() * 1000

        # Step 1: Collect evidence
        evidence_records = await self.collect_evidence(
            task_input=task_input,
            policy_output=policy_output,
            policy_trace=policy_trace,
            rubric=rubric,
            tool_executor=tool_executor,
        )

        # Step 2: Score each check
        check_scores: dict[str, float] = {}

        for item in rubric.verification_checklist:
            score = await self.score_check(
                check_id=item.id,
                rubric=rubric,
                evidence=evidence_records,
                policy_output=policy_output,
            )
            check_scores[item.id] = score

            # Early exit on gate failure if configured
            if (
                self.config.fail_fast_on_gate
                and item.is_gate
                and score < rubric.aggregation.gate_threshold
            ):
                # Return early with zero score
                end_time = time.time() * 1000
                return RubricExecutionResult(
                    reward_score=0.0,
                    rubric_id=rubric.id,
                    check_scores=check_scores,
                    evidence_records=evidence_records,
                    gate_passed=False,
                    failed_gates=[item.id],
                    total_latency_ms=end_time - start_time,
                )

        # Step 3: Aggregate scores
        final_score, gate_passed, failed_gates = self.aggregate_scores(
            rubric=rubric,
            check_scores=check_scores,
        )

        end_time = time.time() * 1000

        return RubricExecutionResult(
            reward_score=final_score,
            rubric_id=rubric.id,
            check_scores=check_scores,
            evidence_records=evidence_records,
            gate_passed=gate_passed,
            failed_gates=failed_gates,
            total_latency_ms=end_time - start_time,
        )


class MockToolExecutor:
    """
    Mock tool executor for testing.

    Returns configurable mock results for tool executions.
    """

    def __init__(self, mock_results: dict[str, Any] | None = None):
        """
        Initialize the mock executor.

        Args:
            mock_results: Mapping of tool_name to mock result.
        """
        self.mock_results = mock_results or {}
        self.call_history: list[dict[str, Any]] = []

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Execute a mock tool call."""
        self.call_history.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "timeout_ms": timeout_ms,
            }
        )

        if tool_name in self.mock_results:
            result = self.mock_results[tool_name]
            if callable(result):
                return result(arguments)
            return result

        # Default mock result
        return {"success": True, "output": "mock_output"}
