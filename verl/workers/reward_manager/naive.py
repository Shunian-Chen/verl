# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        enable_length_reward: int = 0,
        length_reward_max_len: int | None = None,
        reward_max_length: int | None = None,
        format_answer_product: bool = False,
        strong_tag: bool = False,
    ) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            enable_length_reward: Whether to add a length-based reward term (disabled by default). 0=off, 1=on, 2=half length reward.
            length_reward_max_len: Legacy name for the reward-length denominator. Kept for backward compatibility.
            reward_max_length: Dedicated reward-length denominator. Falls back to ``length_reward_max_len`` or upstream defaults.
            format_answer_product: Whether to combine format & answer reward multiplicatively instead of additively.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.enable_length_reward = int(enable_length_reward)

        try:
            normalized_reward_max_length = (
                int(reward_max_length) if reward_max_length is not None else None
            )
        except (TypeError, ValueError):
            normalized_reward_max_length = None

        try:
            legacy_length_cap = int(length_reward_max_len) if length_reward_max_len is not None else None
        except (TypeError, ValueError):
            legacy_length_cap = None

        if legacy_length_cap is None and normalized_reward_max_length is not None:
            legacy_length_cap = normalized_reward_max_length
        elif normalized_reward_max_length is None and legacy_length_cap is not None:
            normalized_reward_max_length = legacy_length_cap

        self.length_reward_max_len = legacy_length_cap
        self.reward_max_length = normalized_reward_max_length
        self.format_answer_product = bool(format_answer_product)
        self.strong_tag = bool(strong_tag)

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            # 将长度奖励相关信息通过 extra_info 传入底层 scorer，避免在管理器重复加成
            if self.enable_length_reward > 0 and isinstance(data_source, str) and data_source == "iceberg":
                extra_info = dict(extra_info or {})
                extra_info["enable_length_reward"] = self.enable_length_reward
                reward_cap = (
                    self.reward_max_length
                    if self.reward_max_length is not None
                    else self.length_reward_max_len
                )
                if reward_cap is not None:
                    extra_info["reward_max_length"] = reward_cap
                    # 兼容旧字段，仍然写入 legacy key
                    extra_info["length_reward_max_len"] = reward_cap
                extra_info["response_valid_length"] = int(
                    valid_response_length.item() if hasattr(valid_response_length, "item") else int(valid_response_length)
                )

            if self.format_answer_product and isinstance(data_source, str) and data_source == "iceberg":
                extra_info = dict(extra_info or {})
                extra_info["format_answer_product"] = True

            # 补充 strong_tag 开关到 extra_info（仅 iceberg 使用）
            if isinstance(extra_info, dict) and isinstance(data_source, str) and data_source == "iceberg":
                extra_info = dict(extra_info)
                extra_info["strong_tag"] = self.strong_tag

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # 仅记录可数值聚合的字段，避免将 dict/list 注入验证指标
                for key, value in score.items():
                    if key in {"format", "tags"}:
                        continue
                    if isinstance(value, (int, float, bool)):
                        reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
