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

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.reward_loop.reward_manager import RubricRewardManager
from verl.protocol import DataProto
from verl.rubric.schemas import (
    AggregationRule,
    EvidenceRecord,
    EvidenceStatus,
    ExecutableRubric,
    RubricExecutionResult,
    VerificationItem,
)


class DummyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "ok"


class DummyExecutor:
    """Mock executor that returns configurable results."""

    def __init__(self, reward_score: float = 0.5):
        self.reward_score = reward_score
        self.call_count = 0

    async def execute(self, task_input, policy_output, policy_trace, rubric, tool_executor=None):
        self.call_count += 1
        return RubricExecutionResult(
            reward_score=self.reward_score,
            rubric_id=rubric.id,
            check_scores={"check_1": 1.0},
            evidence_records=[
                EvidenceRecord(
                    check_id="check_1",
                    tool_name="mock_tool",
                    tool_input={},
                    tool_output={"result": "success"},
                    success=True,
                    status=EvidenceStatus.OK,
                    start_time_ms=0,
                    end_time_ms=100,
                    latency_ms=100,
                )
            ],
            gate_passed=True,
            failed_gates=[],
            total_latency_ms=100,
        )


def create_test_data(rubric_dict: dict) -> DataProto:
    """Create test DataProto with the given rubric."""
    responses = torch.tensor([[1, 2]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1]], dtype=torch.long)
    batch = TensorDict(
        {
            "responses": responses,
            "attention_mask": attention_mask,
        },
        batch_size=1,
    )

    non_tensor_batch = {
        "data_source": np.array(["unit_test"], dtype=object),
        "reward_model": np.array([{"ground_truth": "gt"}], dtype=object),
        "extra_info": np.array([{"rubric": rubric_dict}], dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


@pytest.mark.asyncio
async def test_rubric_reward_manager_returns_evidence():
    """Test that the reward manager returns evidence in the result."""
    config = OmegaConf.create({})
    manager = RubricRewardManager(config=config, tokenizer=DummyTokenizer())
    manager.executor = DummyExecutor()

    rubric_dict = {
        "id": "rubric-1",
        "task_intent": "dummy",
        "verification_checklist": [],
        "evidence_plans": [],
        "aggregation": {"method": "weighted_sum", "normalize": True},
        "estimated_latency_ms": 0,
    }
    data = create_test_data(rubric_dict)

    result = await manager.run_single(data)

    assert "reward_score" in result
    assert "reward_extra_info" in result
    assert "evidence" in result["reward_extra_info"]


@pytest.mark.asyncio
async def test_rubric_reward_manager_handles_missing_rubric():
    """Test that the manager handles missing rubric gracefully."""
    config = OmegaConf.create({})
    manager = RubricRewardManager(config=config, tokenizer=DummyTokenizer())

    responses = torch.tensor([[1, 2]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1]], dtype=torch.long)
    batch = TensorDict(
        {
            "responses": responses,
            "attention_mask": attention_mask,
        },
        batch_size=1,
    )
    non_tensor_batch = {
        "data_source": np.array(["unit_test"], dtype=object),
        "reward_model": np.array([{"ground_truth": "gt"}], dtype=object),
        "extra_info": np.array([{}], dtype=object),  # No rubric
    }
    data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    result = await manager.run_single(data)

    assert result["reward_score"] == 0.0
    assert "error" in result["reward_extra_info"]


@pytest.mark.asyncio
async def test_rubric_reward_manager_uses_executor_result():
    """Test that the manager uses the executor's reward score."""
    config = OmegaConf.create({})
    manager = RubricRewardManager(config=config, tokenizer=DummyTokenizer())
    manager.executor = DummyExecutor(reward_score=0.75)

    rubric_dict = {
        "id": "rubric-2",
        "task_intent": "test task",
        "verification_checklist": [],
        "evidence_plans": [],
        "aggregation": {"method": "weighted_sum", "normalize": True},
        "estimated_latency_ms": 0,
    }
    data = create_test_data(rubric_dict)

    result = await manager.run_single(data)

    assert result["reward_score"] == 0.75
    assert result["reward_extra_info"]["rubric_id"] == "rubric-2"


@pytest.mark.asyncio
async def test_rubric_reward_manager_includes_check_scores():
    """Test that check scores are included in extra info."""
    config = OmegaConf.create({})
    manager = RubricRewardManager(config=config, tokenizer=DummyTokenizer())
    manager.executor = DummyExecutor()

    rubric_dict = {
        "id": "rubric-3",
        "task_intent": "test",
        "verification_checklist": [],
        "evidence_plans": [],
        "aggregation": {"method": "weighted_sum", "normalize": True},
        "estimated_latency_ms": 0,
    }
    data = create_test_data(rubric_dict)

    result = await manager.run_single(data)

    assert "check_scores" in result["reward_extra_info"]
    assert result["reward_extra_info"]["check_scores"]["check_1"] == 1.0


@pytest.mark.asyncio
async def test_rubric_reward_manager_includes_latency():
    """Test that total_latency_ms is recorded in extra info."""
    config = OmegaConf.create({})
    manager = RubricRewardManager(config=config, tokenizer=DummyTokenizer())
    manager.executor = DummyExecutor(reward_score=0.9)

    rubric_dict = {
        "id": "rubric-latency",
        "task_intent": "test latency",
        "verification_checklist": [],
        "evidence_plans": [],
        "aggregation": {"method": "weighted_sum", "normalize": True},
        "estimated_latency_ms": 0,
    }
    data = create_test_data(rubric_dict)

    result = await manager.run_single(data)

    assert "total_latency_ms" in result["reward_extra_info"]
    assert result["reward_extra_info"]["total_latency_ms"] == 100


@pytest.mark.asyncio
async def test_rubric_reward_manager_evidence_success_rate():
    """Test that evidence success rate is computed correctly."""
    config = OmegaConf.create({})
    manager = RubricRewardManager(config=config, tokenizer=DummyTokenizer())
    manager.executor = DummyExecutor()

    rubric_dict = {
        "id": "rubric-rate",
        "task_intent": "test rate",
        "verification_checklist": [],
        "evidence_plans": [],
        "aggregation": {"method": "weighted_sum", "normalize": True},
        "estimated_latency_ms": 0,
    }
    data = create_test_data(rubric_dict)

    result = await manager.run_single(data)

    # DummyExecutor returns 1 successful evidence record
    assert result["reward_extra_info"]["evidence_success_rate"] == 1.0
