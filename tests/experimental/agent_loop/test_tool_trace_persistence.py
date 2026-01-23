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

from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.protocol import DataProto


class DummyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "ok"


@pytest.mark.asyncio
async def test_tool_trace_merged_into_extra_info():
    def compute_score(*, data_source, solution_str, ground_truth, extra_info, **kwargs):
        assert "tool_trace" in extra_info, "tool_trace must be present in extra_info"
        return {"score": 1.0}

    tokenizer = DummyTokenizer()
    config = OmegaConf.create({})
    reward_manager = NaiveRewardManager(config=config, tokenizer=tokenizer, compute_score=compute_score)

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
        "extra_info": np.array([{}], dtype=object),
        "tool_extra_fields": np.array([{"tool_trace": [{"tool_name": "t1"}]}], dtype=object),
    }
    data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    result = await reward_manager.run_single(data)
    assert result["reward_score"] == 1.0
