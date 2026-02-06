#!/usr/bin/env python3
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
Standalone tests for ccplus preprocessing and PerformanceRatioPrimitive.

Uses importlib.util to avoid ray dependency.

Usage:
    python tests/rubric/test_preprocess_ccplus.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# ---- Module loading (no ray) ----
project_root = Path(__file__).parents[2]
verl_pkg_root = project_root / "verl"

total_passed = 0
total_failed = 0


def load_module_from_file(module_name: str, file_path: Path, globals_dict: dict = None):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if globals_dict:
        for key, value in globals_dict.items():
            setattr(module, key, value)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_test(test_name, test_fn):
    global total_passed, total_failed
    try:
        test_fn()
        print(f"  [PASS] {test_name}")
        total_passed += 1
    except Exception as e:
        print(f"  [FAIL] {test_name}: {e}")
        import traceback
        traceback.print_exc()
        total_failed += 1


# Load modules
schemas = load_module_from_file(
    "verl.rubric.schemas", verl_pkg_root / "rubric" / "schemas.py"
)
sampler = load_module_from_file(
    "verl.experimental.dataset.latency_bucketed_sampler",
    verl_pkg_root / "experimental" / "dataset" / "latency_bucketed_sampler.py",
)
primitives = load_module_from_file(
    "verl.rubric.executor.primitives",
    verl_pkg_root / "rubric" / "executor" / "primitives.py",
)
preprocess = load_module_from_file(
    "preprocess_ccplus",
    project_root / "examples" / "agentic_rubric" / "data_preprocess" / "preprocess_ccplus.py",
)

# Import classes
ExecutableRubric = schemas.ExecutableRubric
VerificationItem = schemas.VerificationItem
EvidencePlan = schemas.EvidencePlan
EvidenceRecord = schemas.EvidenceRecord
EvidenceStatus = schemas.EvidenceStatus
ScoringPrimitive = schemas.ScoringPrimitive
VerificationCategory = schemas.VerificationCategory

PerformanceRatioPrimitive = primitives.PerformanceRatioPrimitive
get_scoring_primitive = primitives.get_scoring_primitive

build_prompt = preprocess.build_prompt
build_rubric = preprocess.build_rubric
map_test_cases = preprocess.map_test_cases
build_rubric_id = preprocess.build_rubric_id

# ============================
print("=" * 60)
print("Running CCPlus Preprocessing Tests")
print("=" * 60)

# ---- Prompt tests ----
print("\n--- Prompt Tests ---")


def test_build_prompt():
    prompt = build_prompt("Two Sum", "Given an array...", 2000, 256)
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    assert "Two Sum" in prompt[1]["content"]
    assert "2000ms" in prompt[1]["content"]
    assert "256MB" in prompt[1]["content"]


run_test("test_build_prompt", test_build_prompt)


# ---- Rubric building tests ----
print("\n--- Rubric Building Tests ---")

SAMPLE_TEST_CASES = [
    {"input": "1 2\n", "output": "3\n"},
    {"input": "5 7\n", "output": "12\n"},
]

TIME_OPT_CODE = "import sys\nprint(sum(map(int, sys.stdin.read().split())))"
MEMORY_OPT_CODE = "a, b = map(int, input().split())\nprint(a + b)"


def test_build_rubric_with_both_refs():
    rubric = build_rubric(
        title="A Plus B",
        description="Add two numbers",
        time_limit=2000,
        memory_limit=256,
        test_cases=SAMPLE_TEST_CASES,
        time_opt=TIME_OPT_CODE,
        memory_opt=MEMORY_OPT_CODE,
        latency_factor=0.3,
    )
    # 3 verification items: correctness, time, memory
    assert len(rubric.verification_checklist) == 3
    ids = [v.id for v in rubric.verification_checklist]
    assert "check_correctness" in ids
    assert "check_time" in ids
    assert "check_memory" in ids

    # 5 evidence plans: 1 correctness + 2 time + 2 memory
    assert len(rubric.evidence_plans) == 5

    # Gate check
    correctness = rubric.get_check_by_id("check_correctness")
    assert correctness.is_gate is True
    assert correctness.weight == 0.0

    # Efficiency checks
    time_check = rubric.get_check_by_id("check_time")
    assert time_check.weight == 1.0
    assert time_check.scoring_primitive == ScoringPrimitive.PERFORMANCE_RATIO

    memory_check = rubric.get_check_by_id("check_memory")
    assert memory_check.weight == 1.0
    assert memory_check.scoring_primitive == ScoringPrimitive.PERFORMANCE_RATIO


run_test("test_build_rubric_with_both_refs", test_build_rubric_with_both_refs)


def test_build_rubric_missing_time_opt():
    rubric = build_rubric(
        title="A Plus B",
        description="Add two numbers",
        time_limit=2000,
        memory_limit=256,
        test_cases=SAMPLE_TEST_CASES,
        time_opt=None,
        memory_opt=MEMORY_OPT_CODE,
        latency_factor=0.3,
    )
    # Only correctness + memory
    assert len(rubric.verification_checklist) == 2
    ids = [v.id for v in rubric.verification_checklist]
    assert "check_correctness" in ids
    assert "check_time" not in ids
    assert "check_memory" in ids
    # 1 correctness + 2 memory = 3 evidence plans
    assert len(rubric.evidence_plans) == 3


run_test("test_build_rubric_missing_time_opt", test_build_rubric_missing_time_opt)


def test_build_rubric_missing_both_opts():
    rubric = build_rubric(
        title="A Plus B",
        description="Add two numbers",
        time_limit=2000,
        memory_limit=256,
        test_cases=SAMPLE_TEST_CASES,
        time_opt=None,
        memory_opt=None,
        latency_factor=0.3,
    )
    # Only correctness
    assert len(rubric.verification_checklist) == 1
    assert rubric.verification_checklist[0].id == "check_correctness"
    # 1 evidence plan
    assert len(rubric.evidence_plans) == 1


run_test("test_build_rubric_missing_both_opts", test_build_rubric_missing_both_opts)


# ---- PerformanceRatioPrimitive tests ----
print("\n--- PerformanceRatioPrimitive Tests ---")


def _make_evidence(check_id, role, metric_field, value, success=True):
    return EvidenceRecord(
        check_id=check_id,
        tool_name="test_runner",
        tool_input={"role": role},
        tool_output={metric_field: value},
        success=success,
        status=EvidenceStatus.OK,
    )


def test_performance_ratio_model_faster():
    """Model is faster than reference -> score = 1.0 (capped)."""
    prim = PerformanceRatioPrimitive()
    item = VerificationItem(
        id="check_time",
        description="time check",
        category=VerificationCategory.EFFICIENCY,
        scoring_primitive=ScoringPrimitive.PERFORMANCE_RATIO,
        scoring_config={"metric_field": "execution_time_ms", "direction": "lower_is_better"},
    )
    evidence = [
        _make_evidence("check_time", "model", "execution_time_ms", 50.0),
        _make_evidence("check_time", "reference", "execution_time_ms", 100.0),
    ]
    score = prim.score(item, evidence, "")
    # ref/model = 100/50 = 2.0, capped at 1.0
    assert score == 1.0


run_test("test_performance_ratio_model_faster", test_performance_ratio_model_faster)


def test_performance_ratio_model_slower():
    """Model is 2x slower than reference -> score = 0.5."""
    prim = PerformanceRatioPrimitive()
    item = VerificationItem(
        id="check_time",
        description="time check",
        category=VerificationCategory.EFFICIENCY,
        scoring_primitive=ScoringPrimitive.PERFORMANCE_RATIO,
        scoring_config={"metric_field": "execution_time_ms", "direction": "lower_is_better"},
    )
    evidence = [
        _make_evidence("check_time", "model", "execution_time_ms", 200.0),
        _make_evidence("check_time", "reference", "execution_time_ms", 100.0),
    ]
    score = prim.score(item, evidence, "")
    assert abs(score - 0.5) < 0.001, f"Expected 0.5, got {score}"


run_test("test_performance_ratio_model_slower", test_performance_ratio_model_slower)


def test_performance_ratio_equal():
    """Model equals reference -> score = 1.0."""
    prim = PerformanceRatioPrimitive()
    item = VerificationItem(
        id="check_time",
        description="time check",
        category=VerificationCategory.EFFICIENCY,
        scoring_primitive=ScoringPrimitive.PERFORMANCE_RATIO,
        scoring_config={"metric_field": "execution_time_ms", "direction": "lower_is_better"},
    )
    evidence = [
        _make_evidence("check_time", "model", "execution_time_ms", 100.0),
        _make_evidence("check_time", "reference", "execution_time_ms", 100.0),
    ]
    score = prim.score(item, evidence, "")
    assert abs(score - 1.0) < 0.001, f"Expected 1.0, got {score}"


run_test("test_performance_ratio_equal", test_performance_ratio_equal)


def test_performance_ratio_missing_evidence():
    """Missing model or reference evidence -> score = 0.0."""
    prim = PerformanceRatioPrimitive()
    item = VerificationItem(
        id="check_time",
        description="time check",
        category=VerificationCategory.EFFICIENCY,
        scoring_primitive=ScoringPrimitive.PERFORMANCE_RATIO,
        scoring_config={"metric_field": "execution_time_ms", "direction": "lower_is_better"},
    )
    # Only model evidence, no reference
    evidence = [
        _make_evidence("check_time", "model", "execution_time_ms", 100.0),
    ]
    score = prim.score(item, evidence, "")
    assert score == 0.0


run_test("test_performance_ratio_missing_evidence", test_performance_ratio_missing_evidence)


def test_performance_ratio_higher_is_better():
    """Test higher_is_better direction."""
    prim = PerformanceRatioPrimitive()
    item = VerificationItem(
        id="check_throughput",
        description="throughput check",
        category=VerificationCategory.EFFICIENCY,
        scoring_primitive=ScoringPrimitive.PERFORMANCE_RATIO,
        scoring_config={"metric_field": "throughput", "direction": "higher_is_better"},
    )
    # Model throughput is half of reference
    evidence = [
        _make_evidence("check_throughput", "model", "throughput", 50.0),
        _make_evidence("check_throughput", "reference", "throughput", 100.0),
    ]
    score = prim.score(item, evidence, "")
    assert abs(score - 0.5) < 0.001, f"Expected 0.5, got {score}"


run_test("test_performance_ratio_higher_is_better", test_performance_ratio_higher_is_better)


def test_performance_ratio_memory():
    """Test memory ratio with lower_is_better."""
    prim = PerformanceRatioPrimitive()
    item = VerificationItem(
        id="check_memory",
        description="memory check",
        category=VerificationCategory.EFFICIENCY,
        scoring_primitive=ScoringPrimitive.PERFORMANCE_RATIO,
        scoring_config={"metric_field": "max_memory_mb", "direction": "lower_is_better"},
    )
    # Model uses 3x more memory than reference
    evidence = [
        _make_evidence("check_memory", "model", "max_memory_mb", 300.0),
        _make_evidence("check_memory", "reference", "max_memory_mb", 100.0),
    ]
    score = prim.score(item, evidence, "")
    expected = 100.0 / 300.0  # ~0.333
    assert abs(score - expected) < 0.001, f"Expected {expected:.4f}, got {score}"


run_test("test_performance_ratio_memory", test_performance_ratio_memory)


def test_get_scoring_primitive_performance_ratio():
    """PerformanceRatioPrimitive is registered and retrievable."""
    prim = get_scoring_primitive("performance_ratio")
    assert isinstance(prim, PerformanceRatioPrimitive)


run_test("test_get_scoring_primitive_performance_ratio", test_get_scoring_primitive_performance_ratio)


# ---- Rubric round-trip test ----
print("\n--- Rubric Round-trip Tests ---")


def test_rubric_roundtrip():
    """model_dump() -> model_validate() preserves all data."""
    rubric = build_rubric(
        title="A Plus B",
        description="Add two numbers",
        time_limit=2000,
        memory_limit=256,
        test_cases=SAMPLE_TEST_CASES,
        time_opt=TIME_OPT_CODE,
        memory_opt=MEMORY_OPT_CODE,
        latency_factor=0.3,
    )
    data = rubric.model_dump()
    restored = ExecutableRubric.model_validate(data)
    assert restored.id == rubric.id
    assert len(restored.verification_checklist) == len(rubric.verification_checklist)
    assert len(restored.evidence_plans) == len(rubric.evidence_plans)
    assert restored.estimated_latency_ms == rubric.estimated_latency_ms


run_test("test_rubric_roundtrip", test_rubric_roundtrip)


# ---- Field name mapping test ----
print("\n--- Field Name Mapping Tests ---")


def test_field_name_mapping():
    """ccplus 'output' field maps to 'expected_output'."""
    mapped = map_test_cases([
        {"input": "1\n", "output": "2\n"},
        {"input": "3\n", "output": "4\n"},
    ])
    for tc in mapped:
        assert "expected_output" in tc
        assert "output" not in tc
    assert mapped[0]["expected_output"] == "2\n"
    assert mapped[1]["expected_output"] == "4\n"


run_test("test_field_name_mapping", test_field_name_mapping)


def test_field_name_mapping_already_mapped():
    """If 'expected_output' already present, use it."""
    mapped = map_test_cases([
        {"input": "1\n", "expected_output": "2\n"},
    ])
    assert mapped[0]["expected_output"] == "2\n"


run_test("test_field_name_mapping_already_mapped", test_field_name_mapping_already_mapped)


# ---- Evidence plan dependency tests ----
print("\n--- Evidence Plan Dependency Tests ---")


def test_evidence_plan_dependencies():
    """Efficiency evidence plans depend on correctness."""
    rubric = build_rubric(
        title="A Plus B",
        description="Add two numbers",
        time_limit=2000,
        memory_limit=256,
        test_cases=SAMPLE_TEST_CASES,
        time_opt=TIME_OPT_CODE,
        memory_opt=MEMORY_OPT_CODE,
        latency_factor=0.3,
    )
    correctness_plans = [p for p in rubric.evidence_plans if p.check_id == "check_correctness"]
    time_plans = [p for p in rubric.evidence_plans if p.check_id == "check_time"]
    memory_plans = [p for p in rubric.evidence_plans if p.check_id == "check_memory"]

    # Correctness plan has no dependencies
    for p in correctness_plans:
        assert p.depends_on == [], f"Correctness plan should have no deps, got {p.depends_on}"

    # Time and memory plans depend on correctness
    for p in time_plans:
        assert "check_correctness" in p.depends_on, f"Time plan missing correctness dep: {p.depends_on}"

    for p in memory_plans:
        assert "check_correctness" in p.depends_on, f"Memory plan missing correctness dep: {p.depends_on}"


run_test("test_evidence_plan_dependencies", test_evidence_plan_dependencies)


def test_evidence_plan_roles():
    """Evidence plans have correct roles (model/reference)."""
    rubric = build_rubric(
        title="A Plus B",
        description="Add two numbers",
        time_limit=2000,
        memory_limit=256,
        test_cases=SAMPLE_TEST_CASES,
        time_opt=TIME_OPT_CODE,
        memory_opt=MEMORY_OPT_CODE,
        latency_factor=0.3,
    )
    time_plans = [p for p in rubric.evidence_plans if p.check_id == "check_time"]
    assert len(time_plans) == 2

    roles = {p.tool_arguments.get("role") for p in time_plans}
    assert roles == {"model", "reference"}, f"Expected model+reference roles, got {roles}"

    # Reference plan should have static code
    ref_plan = [p for p in time_plans if p.tool_arguments.get("role") == "reference"][0]
    assert ref_plan.tool_arguments.get("code") == TIME_OPT_CODE

    # Model plan should have argument_extractor (dynamic)
    model_plan = [p for p in time_plans if p.tool_arguments.get("role") == "model"][0]
    assert model_plan.argument_extractor is not None
    assert "policy_output" in model_plan.argument_extractor


run_test("test_evidence_plan_roles", test_evidence_plan_roles)


def test_rubric_id_deterministic():
    """Same title produces same rubric ID."""
    id1 = build_rubric_id("Two Sum")
    id2 = build_rubric_id("Two Sum")
    assert id1 == id2
    assert id1.startswith("rubric_ccplus_")
    # Different title -> different ID
    id3 = build_rubric_id("Three Sum")
    assert id1 != id3


run_test("test_rubric_id_deterministic", test_rubric_id_deterministic)


# ---- Summary ----
print("\n" + "=" * 60)
print(f"Test Summary: {total_passed} passed, {total_failed} failed")
print("=" * 60)

if total_failed > 0:
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
