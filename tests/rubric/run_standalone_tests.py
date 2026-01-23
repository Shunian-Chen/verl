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
Standalone test runner that avoids ray import issues.

This script uses importlib to import modules directly without triggering
verl/__init__.py which requires ray.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Any

# Get project root
project_root = Path(__file__).parents[2]  # D:\Lab\Agentic-rubric\verl
verl_pkg_root = project_root / "verl"  # The verl package directory

# Don't add to sys.path to avoid importing verl/__init__.py
# Instead, use importlib to load modules directly


def load_module_from_file(module_name: str, file_path: Path, globals_dict: dict = None):
    """Load a Python module from file without going through package __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    # Inject globals if provided
    if globals_dict:
        for key, value in globals_dict.items():
            setattr(module, key, value)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


print("=" * 60)
print("Running Standalone Rubric Tests")
print("=" * 60)

total_passed = 0
total_failed = 0


def run_test(test_name, test_fn):
    """Run a single test and report result."""
    global total_passed, total_failed
    try:
        test_fn()
        print(f"  [PASS] {test_name}")
        total_passed += 1
    except Exception as e:
        print(f"  [FAIL] {test_name}: {e}")
        total_failed += 1


# ================== Schema Tests ==================
print("\n--- Schema Tests ---")

# Load schemas module
schemas_path = verl_pkg_root / "rubric" / "schemas.py"
schemas = load_module_from_file("verl.rubric.schemas", schemas_path)

# Get classes from schemas
VerificationItem = schemas.VerificationItem
EvidencePlan = schemas.EvidencePlan
AggregationRule = schemas.AggregationRule
ExecutableRubric = schemas.ExecutableRubric
EvidenceRecord = schemas.EvidenceRecord
EvidenceStatus = schemas.EvidenceStatus
AggregationMethod = schemas.AggregationMethod
RubricExecutionResult = schemas.RubricExecutionResult


def test_verification_item_defaults():
    item = VerificationItem(
        id="check-1",
        description="basic correctness",
        category="correctness",
    )
    assert item.weight == 1.0
    assert item.is_gate is False


run_test("test_verification_item_defaults", test_verification_item_defaults)


def test_evidence_plan_defaults():
    plan = EvidencePlan(
        check_id="check-1",
        tool_name="sandbox_fusion",
        tool_arguments={"code": "print(1)"},
        argument_extractor="{{ policy_output }}",
    )
    assert plan.timeout_ms == 30000


run_test("test_evidence_plan_defaults", test_evidence_plan_defaults)


def test_aggregation_rule_validates_method():
    AggregationRule(method="weighted_sum", normalize=True)
    try:
        AggregationRule(method="average", normalize=True)
        raise AssertionError("Should have raised ValidationError")
    except Exception as e:
        if "ValidationError" not in type(e).__name__ and "validation" not in str(e).lower():
            raise


run_test("test_aggregation_rule_validates_method", test_aggregation_rule_validates_method)


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


run_test("test_executable_rubric_round_trip", test_executable_rubric_round_trip)


def test_executable_rubric_requires_task_intent():
    try:
        ExecutableRubric(
            id="rubric-1",
            verification_checklist=[],
            evidence_plans=[],
            aggregation=AggregationRule(method="min", normalize=True),
        )
        raise AssertionError("Should have raised ValidationError")
    except Exception as e:
        if "ValidationError" not in type(e).__name__ and "validation" not in str(e).lower():
            raise


run_test("test_executable_rubric_requires_task_intent", test_executable_rubric_requires_task_intent)


def test_evidence_record_creation():
    record = EvidenceRecord(
        check_id="check-1",
        tool_name="test_runner",
        tool_input={"code": "print(1)"},
        tool_output={"passed": 5, "total": 5},
        success=True,
        status=EvidenceStatus.OK,
        start_time_ms=1000.0,
        end_time_ms=1500.0,
        latency_ms=500.0,
    )
    assert record.success is True
    assert record.latency_ms == 500.0


run_test("test_evidence_record_creation", test_evidence_record_creation)


def test_rubric_latency_bucket():
    rubric = ExecutableRubric(
        id="rubric-1",
        task_intent="verify",
        verification_checklist=[],
        evidence_plans=[],
        aggregation=AggregationRule(method="mean", normalize=True),
        estimated_latency_ms=3000,
    )
    # 3000ms should be bucket 1 (1000-5000ms range)
    assert rubric.get_latency_bucket() == 1


run_test("test_rubric_latency_bucket", test_rubric_latency_bucket)


def test_rubric_get_check_by_id():
    item = VerificationItem(
        id="check-1",
        description="test check",
        category="correctness",
    )
    rubric = ExecutableRubric(
        id="rubric-1",
        task_intent="verify",
        verification_checklist=[item],
        evidence_plans=[],
        aggregation=AggregationRule(method="mean", normalize=True),
    )
    found = rubric.get_check_by_id("check-1")
    assert found is not None
    assert found.id == "check-1"
    assert rubric.get_check_by_id("nonexistent") is None


run_test("test_rubric_get_check_by_id", test_rubric_get_check_by_id)


# ================== Latency Sampler Tests ==================
print("\n--- Latency Sampler Tests ---")

# Load sampler module
sampler_path = verl_pkg_root / "experimental" / "dataset" / "latency_bucketed_sampler.py"
sampler = load_module_from_file("verl.experimental.dataset.latency_bucketed_sampler", sampler_path)

LatencyBucketedSampler = sampler.LatencyBucketedSampler
compute_latency_bucket = sampler.compute_latency_bucket
assign_latency_buckets = sampler.assign_latency_buckets


class DummyDataset:
    def __init__(self, buckets):
        self.items = [{"extra_info": {"latency_bucket": bucket}} for bucket in buckets]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_latency_bucketed_sampler_orders_by_bucket():
    dataset = DummyDataset([1, 0, 1, 2])
    s = LatencyBucketedSampler(dataset, num_buckets=3)
    indices = list(iter(s))
    assert indices == [1, 0, 2, 3], f"Expected [1, 0, 2, 3], got {indices}"


run_test("test_latency_bucketed_sampler_orders_by_bucket", test_latency_bucketed_sampler_orders_by_bucket)


def test_assign_bucket_boundaries():
    test_cases = [
        (-1, 0),
        (0, 0),
        (999, 0),
        (1000, 1),
        (4999, 1),
        (5000, 2),
        (14999, 2),
        (15000, 3),
        (29999, 3),
        (30000, 4),
        (999999, 4),
    ]
    for latency_ms, expected_bucket in test_cases:
        bucket = LatencyBucketedSampler.assign_bucket(latency_ms, num_buckets=5)
        assert bucket == expected_bucket, f"For {latency_ms}ms: expected bucket {expected_bucket}, got {bucket}"


run_test("test_assign_bucket_boundaries", test_assign_bucket_boundaries)


def test_compute_latency_bucket():
    assert compute_latency_bucket(500) == 0
    assert compute_latency_bucket(3000) == 1
    assert compute_latency_bucket(10000) == 2
    assert compute_latency_bucket(20000) == 3
    assert compute_latency_bucket(50000) == 4


run_test("test_compute_latency_bucket", test_compute_latency_bucket)


def test_assign_latency_buckets_to_dataset():
    dataset_items = [
        {"extra_info": {"estimated_latency_ms": 500}},
        {"extra_info": {"estimated_latency_ms": 3000}},
        {"extra_info": {"estimated_latency_ms": 50000}},
    ]
    result = assign_latency_buckets(dataset_items)
    assert result[0]["extra_info"]["latency_bucket"] == 0
    assert result[1]["extra_info"]["latency_bucket"] == 1
    assert result[2]["extra_info"]["latency_bucket"] == 4


run_test("test_assign_latency_buckets_to_dataset", test_assign_latency_buckets_to_dataset)


def test_sampler_get_bucket_batches():
    dataset = DummyDataset([0, 0, 0, 1, 1, 2])
    s = LatencyBucketedSampler(dataset, batch_size=2, num_buckets=3)
    batches = list(s.get_bucket_batches())
    # Should have batches: bucket 0 (3 items, 2 batches), bucket 1 (2 items, 1 batch), bucket 2 (1 item, 1 batch)
    assert len(batches) >= 3


run_test("test_sampler_get_bucket_batches", test_sampler_get_bucket_batches)


# ================== Executor Base Tests ==================
print("\n--- Executor Base Tests ---")

# Register verl.rubric.schemas for imports
# Load executor base
executor_base_path = verl_pkg_root / "rubric" / "executor" / "base.py"
executor_base = load_module_from_file("verl.rubric.executor.base", executor_base_path)
sys.modules["verl.rubric.executor.base"] = executor_base

RubricExecutorConfig = executor_base.RubricExecutorConfig


def test_executor_config_defaults():
    config = RubricExecutorConfig()
    assert config.max_parallel_evidence == 5  # Default is 5
    assert config.fail_fast_on_gate is True


run_test("test_executor_config_defaults", test_executor_config_defaults)


def test_executor_config_custom():
    config = RubricExecutorConfig(
        max_parallel_evidence=5,
        fail_fast_on_gate=False,
        default_timeout_ms=60000,
    )
    assert config.max_parallel_evidence == 5
    assert config.fail_fast_on_gate is False
    assert config.default_timeout_ms == 60000


run_test("test_executor_config_custom", test_executor_config_custom)


# ================== Primitives Tests ==================
print("\n--- Primitives Tests ---")

# Load primitives module
primitives_path = verl_pkg_root / "rubric" / "executor" / "primitives.py"
primitives = load_module_from_file("verl.rubric.executor.primitives", primitives_path)
sys.modules["verl.rubric.executor.primitives"] = primitives

ExactMatchPrimitive = primitives.ExactMatchPrimitive
ContainsPrimitive = primitives.ContainsPrimitive
RegexPrimitive = primitives.RegexPrimitive
NumericComparisonPrimitive = primitives.NumericComparisonPrimitive
get_scoring_primitive = primitives.get_scoring_primitive


def test_exact_match_primitive():
    primitive = ExactMatchPrimitive()
    item = VerificationItem(
        id="check-1",
        description="exact match",
        category="correctness",
        scoring_config={"expected_field": "tool_output"},
    )
    evidence = [EvidenceRecord(
        check_id="check-1",
        tool_name="test",
        tool_input={},
        tool_output="hello world",
        success=True,
        status=EvidenceStatus.OK,
        start_time_ms=0,
        end_time_ms=0,
        latency_ms=0,
    )]
    score = primitive.score(item, evidence, "hello world")
    assert score == 1.0


run_test("test_exact_match_primitive", test_exact_match_primitive)


def test_contains_primitive():
    primitive = ContainsPrimitive()
    item = VerificationItem(
        id="check-1",
        description="contains check",
        category="correctness",
        scoring_config={"expected_value": "def fibonacci"},
    )
    policy_output = "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"
    score = primitive.score(item, [], policy_output)
    assert score == 1.0


run_test("test_contains_primitive", test_contains_primitive)


def test_regex_primitive():
    primitive = RegexPrimitive()
    item = VerificationItem(
        id="check-1",
        description="regex check",
        category="correctness",
        scoring_config={"pattern": r"def \w+\("},
    )
    policy_output = "def hello():\n    pass"
    score = primitive.score(item, [], policy_output)
    assert score == 1.0


run_test("test_regex_primitive", test_regex_primitive)


def test_numeric_comparison_primitive():
    primitive = NumericComparisonPrimitive()
    item = VerificationItem(
        id="check-1",
        description="numeric check",
        category="correctness",
        scoring_config={"comparison": "approx", "tolerance": 0.01},
    )
    evidence = [EvidenceRecord(
        check_id="check-1",
        tool_name="test",
        tool_input={},
        tool_output={"expected": 42.0},
        success=True,
        status=EvidenceStatus.OK,
        start_time_ms=0,
        end_time_ms=0,
        latency_ms=0,
    )]
    score = primitive.score(item, evidence, "The answer is 42.001")
    assert score == 1.0  # Within default tolerance


run_test("test_numeric_comparison_primitive", test_numeric_comparison_primitive)


def test_get_scoring_primitive():
    assert isinstance(get_scoring_primitive("exact_match"), ExactMatchPrimitive)
    assert isinstance(get_scoring_primitive("contains"), ContainsPrimitive)
    assert isinstance(get_scoring_primitive("regex"), RegexPrimitive)
    assert isinstance(get_scoring_primitive("numeric_comparison"), NumericComparisonPrimitive)


run_test("test_get_scoring_primitive", test_get_scoring_primitive)


# ================== Default Executor Tests ==================
print("\n--- Default Executor Tests ---")

# Load default executor
default_executor_path = verl_pkg_root / "rubric" / "executor" / "default_executor.py"
default_executor = load_module_from_file("verl.rubric.executor.default_executor", default_executor_path)

DefaultRubricExecutor = default_executor.DefaultRubricExecutor
MockToolExecutor = default_executor.MockToolExecutor


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


run_test("test_aggregate_with_gates_blocks_on_failed_gate", test_aggregate_with_gates_blocks_on_failed_gate)


def test_aggregate_with_gates_weighted_sum():
    executor = DefaultRubricExecutor()
    aggregation = AggregationRule(method="weighted_sum", normalize=True)
    check_scores = {
        "gate": _check("gate", 0.5, weight=2.0, is_gate=True),
        "other": _check("other", 1.0, weight=1.0),
    }
    score = executor._aggregate_with_gates(aggregation, check_scores)
    expected = (2.0 * 0.5 + 1.0 * 1.0) / 3.0
    assert abs(score - expected) < 0.001, f"Expected {expected}, got {score}"


run_test("test_aggregate_with_gates_weighted_sum", test_aggregate_with_gates_weighted_sum)


def test_aggregate_with_gates_min():
    executor = DefaultRubricExecutor()
    aggregation = AggregationRule(method="min", normalize=True)
    check_scores = {
        "a": _check("a", 0.7),
        "b": _check("b", 0.2),
    }
    score = executor._aggregate_with_gates(aggregation, check_scores)
    assert score == 0.2


run_test("test_aggregate_with_gates_min", test_aggregate_with_gates_min)


def test_aggregate_with_gates_product():
    executor = DefaultRubricExecutor()
    aggregation = AggregationRule(method="product", normalize=True)
    check_scores = {
        "a": _check("a", 0.5),
        "b": _check("b", 0.2),
    }
    score = executor._aggregate_with_gates(aggregation, check_scores)
    assert abs(score - 0.1) < 0.001, f"Expected 0.1, got {score}"


run_test("test_aggregate_with_gates_product", test_aggregate_with_gates_product)


def test_aggregate_with_gates_mean():
    executor = DefaultRubricExecutor()
    aggregation = AggregationRule(method="mean", normalize=True)
    check_scores = {
        "a": _check("a", 0.8),
        "b": _check("b", 0.4),
    }
    score = executor._aggregate_with_gates(aggregation, check_scores)
    assert abs(score - 0.6) < 0.001, f"Expected 0.6, got {score}"


run_test("test_aggregate_with_gates_mean", test_aggregate_with_gates_mean)


def test_mock_tool_executor():
    executor = MockToolExecutor(mock_results={
        "test_runner": {"passed": 5, "total": 5, "all_passed": True}
    })
    import asyncio
    result = asyncio.run(executor.execute("test_runner", {"code": "print(1)"}))
    assert result["all_passed"] is True
    assert len(executor.call_history) == 1


run_test("test_mock_tool_executor", test_mock_tool_executor)


# ================== Generator Base Tests ==================
print("\n--- Generator Base Tests ---")

# Load generator base
generator_base_path = verl_pkg_root / "rubric" / "generator" / "base.py"
generator_base = load_module_from_file("verl.rubric.generator.base", generator_base_path)
sys.modules["verl.rubric.generator.base"] = generator_base

ToolSchema = generator_base.ToolSchema
IntentDirection = generator_base.IntentDirection
RubricGeneratorConfig = generator_base.RubricGeneratorConfig


def test_tool_schema_creation():
    schema = ToolSchema(
        name="test_runner",
        description="Run tests",
        parameters={"type": "object", "properties": {}},
    )
    assert schema.name == "test_runner"


run_test("test_tool_schema_creation", test_tool_schema_creation)


def test_intent_direction_creation():
    intent = IntentDirection(
        task_intent="Write a fibonacci function",
        verification_dimensions=["correctness", "style"],
        key_requirements=["must use recursion"],
        expected_output_type="code",
    )
    assert intent.task_intent == "Write a fibonacci function"
    assert len(intent.verification_dimensions) == 2


run_test("test_intent_direction_creation", test_intent_direction_creation)


def test_rubric_generator_config():
    config = RubricGeneratorConfig(
        model_name="gpt-4o",
    )
    assert config.model_name == "gpt-4o"


run_test("test_rubric_generator_config", test_rubric_generator_config)


# ================== Integration Tests ==================
print("\n--- Integration Tests ---")


def test_full_rubric_workflow():
    """Test creating a rubric, evidence, and execution result."""
    # Create a rubric
    rubric = ExecutableRubric(
        id="test-rubric",
        task_intent="Verify code correctness",
        verification_checklist=[
            VerificationItem(
                id="test-passes",
                description="All tests pass",
                category="correctness",
                weight=2.0,
                is_gate=True,
                scoring_primitive="code_execution",
            ),
            VerificationItem(
                id="style-check",
                description="Code follows style guidelines",
                category="style",
                weight=1.0,
                scoring_primitive="contains",
                expected_value="def ",
            ),
        ],
        evidence_plans=[
            EvidencePlan(
                check_id="test-passes",
                tool_name="test_runner",
                tool_arguments={"language": "python"},
                argument_extractor="{{ policy_output }}",
                timeout_ms=30000,
            ),
        ],
        aggregation=AggregationRule(
            method="weighted_sum",
            normalize=True,
            gate_threshold=0.5,
        ),
    )

    # Validate rubric
    assert len(rubric.verification_checklist) == 2
    assert len(rubric.evidence_plans) == 1
    assert rubric.get_check_by_id("test-passes") is not None
    assert rubric.get_check_by_id("test-passes").is_gate

    # Create evidence
    evidence = EvidenceRecord(
        check_id="test-passes",
        tool_name="test_runner",
        tool_input={"code": "def fib(n): ..."},
        tool_output={"passed": 5, "total": 5, "all_passed": True},
        success=True,
        status=EvidenceStatus.OK,
        start_time_ms=1000,
        end_time_ms=2000,
        latency_ms=1000,
    )

    assert evidence.success
    assert evidence.latency_ms == 1000

    # Create execution result
    result = RubricExecutionResult(
        reward_score=0.9,
        rubric_id="test-rubric",
        check_scores={"test-passes": 1.0, "style-check": 0.7},
        evidence_records=[evidence],
        gate_passed=True,
        failed_gates=[],
        total_latency_ms=1500,
    )

    assert result.gate_passed
    assert result.reward_score == 0.9


run_test("test_full_rubric_workflow", test_full_rubric_workflow)


def test_serialization_round_trip():
    """Test that rubrics serialize and deserialize correctly."""
    rubric = ExecutableRubric(
        id="serialization-test",
        task_intent="Test serialization",
        verification_checklist=[
            VerificationItem(
                id="check-1",
                description="Test check",
                category="correctness",
            ),
        ],
        evidence_plans=[
            EvidencePlan(
                check_id="check-1",
                tool_name="test",
                tool_arguments={"key": "value"},
            ),
        ],
        aggregation=AggregationRule(method="mean", normalize=True),
        estimated_latency_ms=5000,
    )

    # Serialize to dict
    data = rubric.model_dump()

    # Deserialize back
    restored = ExecutableRubric.model_validate(data)

    assert restored.id == rubric.id
    assert restored.task_intent == rubric.task_intent
    assert len(restored.verification_checklist) == 1
    assert restored.estimated_latency_ms == 5000


run_test("test_serialization_round_trip", test_serialization_round_trip)


# ================== Summary ==================
print("\n" + "=" * 60)
print(f"Test Summary: {total_passed} passed, {total_failed} failed")
print("=" * 60)

if total_failed > 0:
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
