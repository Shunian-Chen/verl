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
"""

from __future__ import annotations

from verl.experimental.reward_loop.reward_manager.rubric import (
    RubricRewardManager as _ExperimentalRubricRewardManager,
)

from .registry import register


@register("rubric")
class RubricRewardManager(_ExperimentalRubricRewardManager):
    """Worker-registry entry point for rubric reward management."""

