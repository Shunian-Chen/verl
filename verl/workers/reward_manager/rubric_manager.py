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
Compatibility registration for rubric reward manager.

This module bridges the worker reward-manager registry to the rubric
reward manager implementation under experimental reward loop.
It provides a ``__call__`` adapter so that the standard training loop
(``compute_reward`` in ``verl/trainer/ppo/reward.py``) can invoke the
rubric manager identically to other ``AbstractRewardManager`` implementations.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.rubric import (
    RubricRewardManager as _ExperimentalRubricRewardManager,
)

from .registry import register

logger = logging.getLogger(__name__)


@register("rubric")
class RubricRewardManager(_ExperimentalRubricRewardManager):
    """Worker-registry entry point for rubric reward management.

    Inherits the async ``run_single`` implementation from the experimental
    reward-loop manager and adds a synchronous ``__call__`` adapter that
    matches the ``AbstractRewardManager.__call__`` interface used by the
    standard PPO training loop.
    """

    # ------------------------------------------------------------------
    # Standard training-loop adapter
    # ------------------------------------------------------------------
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute rewards for a full batch via the standard training loop.

        Iterates over every item in *data*, delegates to the async
        ``run_single`` method (running it synchronously on the event loop
        inherited from ``RewardManagerBase``), and assembles the results
        into either a plain reward tensor or a dict that also carries
        ``reward_extra_info``.
        """
        # Short-circuit: if pre-computed rm_scores exist, return them directly.
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i : i + 1]  # run_single expects len == 1

            result = self.loop.run_until_complete(self.run_single(data_item))

            # Determine the position of the last valid response token.
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = int(
                data_item.batch["attention_mask"][-response_length:].sum()
            )

            reward_tensor[i, valid_response_length - 1] = result["reward_score"]

            for k, v in result.get("reward_extra_info", {}).items():
                reward_extra_info[k].append(v)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor

    # Re-export helper from AbstractRewardManager so it is available on
    # this subclass even though the MRO does not include AbstractRewardManager.
    @staticmethod
    def _extract_reward_from_rm_scores(
        data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any] | None:
        """Return pre-computed rm_scores when available."""
        if "rm_scores" not in data.batch.keys():
            return None

        if return_dict:
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
            return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        return data.batch["rm_scores"]
