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

"""Tests for the worker-level RubricRewardManager.__call__ adapter."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.rubric.schemas import (
    EvidenceRecord,
    EvidenceStatus,
    RubricExecutionResult,
)


class DummyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "dummy response"


class DummyExecutor:
    """Mock executor that returns a configurable reward score."""

    def __init__(self, reward_score: float = 0.5):
        self.reward_score = reward_score
        self.call_count = 0

    async def execute(self, task_input, policy_output, policy_trace, rubric, tool_executor=None):
        self.call_count += 1
        return RubricExecutionResult(
            reward_score=self.reward_score,
            rubric_id=rubric.id,
            check_scores={"check_1": self.reward_score},
            evidence_records=[
                EvidenceRecord(
                    check_id="check_1",
                    tool_name="mock_tool",
                    tool_input={},
                    tool_output={"result": "ok"},
                    success=True,
                    status=EvidenceStatus.OK,
                    start_time_ms=0,
                    end_time_ms=50,
                    latency_ms=50,
                )
            ],
            gate_passed=True,
            failed_gates=[],
            total_latency_ms=50,
        )


def _make_batch(n: int, rubric_dict: dict) -> DataProto:
    """Create a DataProto batch with *n* items sharing the same rubric."""
    seq_len = 4
    responses = torch.ones((n, seq_len), dtype=torch.long)
    attention_mask = torch.ones((n, seq_len), dtype=torch.long)
    batch = TensorDict(
        {"responses": responses, "attention_mask": attention_mask},
        batch_size=n,
    )
    non_tensor_batch = {
        "data_source": np.array(["unit_test"] * n, dtype=object),
        "reward_model": np.array([{"ground_truth": "gt"}] * n, dtype=object),
        "extra_info": np.array([{"rubric": rubric_dict}] * n, dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


RUBRIC_DICT = {
    "id": "rubric-call-test",
    "task_intent": "test",
    "verification_checklist": [],
    "evidence_plans": [],
    "aggregation": {"method": "weighted_sum", "normalize": True},
    "estimated_latency_ms": 0,
}


def test_call_returns_tensor():
    """__call__ with return_dict=False returns a reward tensor."""
    from verl.workers.reward_manager.rubric_manager import RubricRewardManager

    config = OmegaConf.create({})
    manager = RubricRewardManager(config=config, tokenizer=DummyTokenizer())
    manager.executor = DummyExecutor(reward_score=0.8)

    data = _make_batch(2, RUBRIC_DICT)
    result = manager(data, return_dict=False)

    assert isinstance(result, torch.Tensor)
    assert result.shape == data.batch["responses"].shape
    # Reward placed at last valid token position (index 3 since all mask=1 and seq_len=4)
    assert result[0, 3].item() == pytest.approx(0.8)
    assert result[1, 3].item() == pytest.approx(0.8)


def test_call_returns_dict():
    """__call__ with return_dict=True returns a dict with reward_tensor and reward_extra_info."""
    from verl.workers.reward_manager.rubric_manager import RubricRewardManager

    config = OmegaConf.create({})
    manager = RubricRewardManager(config=config, tokenizer=DummyTokenizer())
    manager.executor = DummyExecutor(reward_score=0.6)

    data = _make_batch(3, RUBRIC_DICT)
    result = manager(data, return_dict=True)

    assert isinstance(result, dict)
    assert "reward_tensor" in result
    assert "reward_extra_info" in result
    assert result["reward_tensor"].shape == data.batch["responses"].shape
    assert result["reward_tensor"][0, 3].item() == pytest.approx(0.6)
    # Extra info lists should have 3 entries (one per item)
    assert len(result["reward_extra_info"]["rubric_id"]) == 3


def test_call_shortcircuits_on_rm_scores():
    """__call__ returns pre-computed rm_scores when available."""
    from verl.workers.reward_manager.rubric_manager import RubricRewardManager

    config = OmegaConf.create({})
    manager = RubricRewardManager(config=config, tokenizer=DummyTokenizer())
    executor = DummyExecutor()
    manager.executor = executor

    seq_len = 4
    n = 2
    responses = torch.ones((n, seq_len), dtype=torch.long)
    attention_mask = torch.ones((n, seq_len), dtype=torch.long)
    rm_scores = torch.tensor([[0.0, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.7]])
    batch = TensorDict(
        {"responses": responses, "attention_mask": attention_mask, "rm_scores": rm_scores},
        batch_size=n,
    )
    non_tensor_batch = {
        "data_source": np.array(["test"] * n, dtype=object),
        "reward_model": np.array([{"ground_truth": "gt"}] * n, dtype=object),
        "extra_info": np.array([{"rubric": RUBRIC_DICT}] * n, dtype=object),
    }
    data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    result = manager(data, return_dict=False)

    # Should return the pre-computed scores, NOT call the executor.
    assert torch.equal(result, rm_scores)
    assert executor.call_count == 0


def test_call_invokes_executor_per_item():
    """__call__ invokes run_single once per data item."""
    from verl.workers.reward_manager.rubric_manager import RubricRewardManager

    config = OmegaConf.create({})
    manager = RubricRewardManager(config=config, tokenizer=DummyTokenizer())
    executor = DummyExecutor(reward_score=1.0)
    manager.executor = executor

    data = _make_batch(5, RUBRIC_DICT)
    manager(data, return_dict=False)

    assert executor.call_count == 5
