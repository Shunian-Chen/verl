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
from pydantic import ValidationError

from verl.rubric.schemas import (
    AggregationRule,
    EvidencePlan,
    ExecutableRubric,
    VerificationItem,
)


def test_verification_item_defaults():
    item = VerificationItem(
        id="check-1",
        description="basic correctness",
        category="correctness",
    )

    assert item.weight == 1.0
    assert item.is_gate is False


def test_evidence_plan_defaults():
    plan = EvidencePlan(
        check_id="check-1",
        tool_name="sandbox_fusion",
        tool_arguments={"code": "print(1)"},
        argument_extractor="{{ policy_output }}",
    )

    assert plan.timeout_ms == 30000


def test_aggregation_rule_validates_method():
    AggregationRule(method="weighted_sum", normalize=True)

    with pytest.raises(ValidationError):
        AggregationRule(method="average", normalize=True)


def test_executable_rubric_round_trip():
    rubric = ExecutableRubric(
        id="rubric-1",
        task_intent="verify correctness",
        verification_checklist=[
            VerificationItem(
                id="check-1",
                description="passes unit tests",
                category="correctness",
                weight=2.0,
                is_gate=True,
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

    payload = rubric.model_dump()
    assert payload["id"] == "rubric-1"
    assert payload["aggregation"]["method"] == "weighted_sum"
    assert payload["estimated_latency_ms"] == 0
    assert payload["verification_checklist"][0]["id"] == "check-1"
    assert payload["evidence_plans"][0]["tool_name"] == "sandbox_fusion"


def test_executable_rubric_requires_task_intent():
    with pytest.raises(ValidationError):
        ExecutableRubric(
            id="rubric-1",
            verification_checklist=[],
            evidence_plans=[],
            aggregation=AggregationRule(method="min", normalize=True),
        )
