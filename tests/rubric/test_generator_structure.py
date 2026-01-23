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

import pytest

from verl.rubric.generator.llm_generator import LLMRubricGenerator
from verl.rubric.schemas import (
    AggregationRule,
    EvidencePlan,
    ExecutableRubric,
    VerificationItem,
)


@pytest.mark.asyncio
async def test_llm_generator_returns_executable_rubric(monkeypatch):
    def fake_init(self):
        return None

    async def fake_extract(self, task_input, specification):
        assert task_input == "task input"
        assert specification is None
        return {"task_intent": "intent"}

    async def fake_compile(self, task_input, direction, tools):
        assert tools == [{"name": "sandbox_fusion"}]
        return ExecutableRubric(
            id="rubric-1",
            task_intent=direction["task_intent"],
            verification_checklist=[
                VerificationItem(
                    id="check-1",
                    description="pass checks",
                    category="correctness",
                )
            ],
            evidence_plans=[
                EvidencePlan(
                    check_id="check-1",
                    tool_name="sandbox_fusion",
                    tool_arguments={"code": "{{ policy_output }}"},
                    argument_extractor="{{ policy_output }}",
                )
            ],
            aggregation=AggregationRule(method="weighted_sum", normalize=True),
        )

    def fake_estimate(self, rubric):
        return 123

    monkeypatch.setattr(LLMRubricGenerator, "_init_client", fake_init)
    monkeypatch.setattr(LLMRubricGenerator, "_extract_intent", fake_extract)
    monkeypatch.setattr(LLMRubricGenerator, "_compile_to_executable", fake_compile)
    monkeypatch.setattr(LLMRubricGenerator, "_estimate_latency", fake_estimate)

    generator = LLMRubricGenerator(model="dummy", api_key=None)
    rubric = await generator.generate("task input", None, [{"name": "sandbox_fusion"}])

    assert isinstance(rubric, ExecutableRubric)
    assert rubric.task_intent == "intent"
    assert rubric.estimated_latency_ms == 123
