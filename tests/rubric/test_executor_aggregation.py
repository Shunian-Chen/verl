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

from verl.rubric.executor.default_executor import DefaultRubricExecutor
from verl.rubric.schemas import AggregationRule, VerificationItem


def _check(check_id, score, weight=1.0, is_gate=False):
    item = VerificationItem(
        id=check_id,
        description="score check",
        category="correctness",
        weight=weight,
        is_gate=is_gate,
    )
    return item, score


def test_aggregate_with_gates_blocks_on_failed_gate():
    executor = DefaultRubricExecutor()
    aggregation = AggregationRule(method="weighted_sum", normalize=True)
    check_scores = {
        "gate": _check("gate", 0.4, is_gate=True),
        "other": _check("other", 1.0),
    }

    score = executor._aggregate_with_gates(aggregation, check_scores)
    assert score == 0.0


def test_aggregate_with_gates_weighted_sum():
    executor = DefaultRubricExecutor()
    aggregation = AggregationRule(method="weighted_sum", normalize=True)
    check_scores = {
        "gate": _check("gate", 0.5, weight=2.0, is_gate=True),
        "other": _check("other", 1.0, weight=1.0),
    }

    score = executor._aggregate_with_gates(aggregation, check_scores)
    assert score == pytest.approx((2.0 * 0.5 + 1.0 * 1.0) / 3.0)


def test_aggregate_with_gates_min():
    executor = DefaultRubricExecutor()
    aggregation = AggregationRule(method="min", normalize=True)
    check_scores = {
        "a": _check("a", 0.7),
        "b": _check("b", 0.2),
    }

    score = executor._aggregate_with_gates(aggregation, check_scores)
    assert score == 0.2


def test_aggregate_with_gates_product():
    executor = DefaultRubricExecutor()
    aggregation = AggregationRule(method="product", normalize=True)
    check_scores = {
        "a": _check("a", 0.5),
        "b": _check("b", 0.2),
    }

    score = executor._aggregate_with_gates(aggregation, check_scores)
    assert score == pytest.approx(0.1)
