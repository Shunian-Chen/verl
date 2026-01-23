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
Rubric-based reward manager for verl.

Integrates executable rubrics into verl's reward computation pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.experimental.reward_loop.reward_manager.registry import register
from verl.rubric.executor.base import BaseRubricExecutor, ToolExecutor
from verl.rubric.executor.default_executor import DefaultRubricExecutor
from verl.rubric.schemas import ExecutableRubric

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("rubric")
class RubricRewardManager(RewardManagerBase):
    """
    Rubric-based reward manager.

    Computes rewards by executing executable rubrics against policy outputs.
    Rubrics are stored in the data's extra_info field and contain:
    - Verification checklist (what to verify)
    - Evidence plans (how to collect evidence)
    - Aggregation rules (how to combine scores)

    Usage:
        Configure in YAML:
        ```yaml
        reward_model:
          reward_manager: "rubric"
        ```

        Rubrics should be stored in DataProto:
        ```python
        data.non_tensor_batch["extra_info"]["rubric"] = rubric.model_dump()
        ```

    The manager extracts:
    - Rubric from extra_info["rubric"]
    - Policy output from decoded response tokens
    - Policy trace from tool_extra_fields["tool_calls"]
    - Task input from raw_prompt (if available)

    Returns:
        dict with:
        - "reward_score": float in [0, 1]
        - "reward_extra_info": dict containing:
            - "rubric_id": str
            - "check_scores": dict[str, float]
            - "evidence_count": int
            - "evidence_success_rate": float
            - "evidence": list[dict] (serialized evidence records)
    """

    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        executor: BaseRubricExecutor | None = None,
        tool_executor: ToolExecutor | None = None,
    ):
        """
        Initialize the rubric reward manager.

        Args:
            config: YAML configuration.
            tokenizer: Tokenizer for decoding responses.
            executor: Optional custom rubric executor.
            tool_executor: Optional tool executor for evidence collection.
        """
        super().__init__(config, tokenizer)
        self.executor = executor or DefaultRubricExecutor()
        self.tool_executor = tool_executor

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        """
        Compute reward for a single data item using rubric execution.

        Args:
            data: DataProto containing the policy output and rubric.

        Returns:
            dict with reward_score and reward_extra_info.
        """
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        # Extract response text
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(
            data_item.batch["attention_mask"][-response_length:].sum()
        )
        valid_response_ids = response_ids[:valid_response_length]

        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )

        # Extract extra info and rubric
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        if isinstance(extra_info, dict):
            rubric_data = extra_info.get("rubric")
        else:
            # Handle numpy array wrapping
            extra_info = extra_info.item() if hasattr(extra_info, "item") else extra_info
            rubric_data = extra_info.get("rubric") if extra_info else None

        if rubric_data is None:
            logger.warning("No rubric found in extra_info, returning zero reward")
            return {
                "reward_score": 0.0,
                "reward_extra_info": {
                    "error": "No rubric found",
                    "evidence": [],
                },
            }

        # Parse rubric
        try:
            if isinstance(rubric_data, ExecutableRubric):
                rubric = rubric_data
            else:
                rubric = ExecutableRubric.model_validate(rubric_data)
        except Exception as e:
            logger.error(f"Failed to parse rubric: {e}")
            return {
                "reward_score": 0.0,
                "reward_extra_info": {
                    "error": f"Failed to parse rubric: {e}",
                    "evidence": [],
                },
            }

        # Extract task input
        task_input = data_item.non_tensor_batch.get("raw_prompt", "")
        if hasattr(task_input, "item"):
            task_input = task_input.item()

        # Extract policy trace (tool calls made by the policy)
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", {})
        if hasattr(tool_extra_fields, "item"):
            tool_extra_fields = tool_extra_fields.item()

        policy_trace = []
        if tool_extra_fields:
            tool_calls = tool_extra_fields.get("tool_calls", [])
            if isinstance(tool_calls, list):
                policy_trace = tool_calls

        # Execute rubric
        try:
            result = await self.executor.execute(
                task_input=str(task_input),
                policy_output=response_str,
                policy_trace=policy_trace,
                rubric=rubric,
                tool_executor=self.tool_executor,
            )

            return {
                "reward_score": result.reward_score,
                "reward_extra_info": {
                    "rubric_id": result.rubric_id,
                    "check_scores": result.check_scores,
                    "evidence_count": len(result.evidence_records),
                    "evidence_success_rate": (
                        sum(1 for e in result.evidence_records if e.success)
                        / max(len(result.evidence_records), 1)
                    ),
                    "gate_passed": result.gate_passed,
                    "failed_gates": result.failed_gates,
                    "total_latency_ms": result.total_latency_ms,
                    "evidence": [e.model_dump() for e in result.evidence_records],
                },
            }
        except Exception as e:
            logger.error(f"Rubric execution failed: {e}")
            return {
                "reward_score": 0.0,
                "reward_extra_info": {
                    "error": f"Rubric execution failed: {e}",
                    "rubric_id": rubric.id,
                    "evidence": [],
                },
            }
