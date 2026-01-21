# Test Plan for Agentic Rubrics Changes

## Scope and Goals
- Verify environment contract compliance (schema discovery, invocation, provenance).
- Ensure tool trace persistence is available for rubric execution.
- Enforce policy vs verifier tool isolation.
- Validate evidence-driven scoring (no reliance on untrusted policy traces).
- Ensure verification_data (unit-test I/O pairs) is passed into rubric generation and reflected in evidence plans.

## Test Fixtures and Mocks
- FakePolicyTool: deterministic output + ToolExecutionMeta.
- FakeVerifierTool: only in verifier config.
- FakeRubricGenerator: fixed ExecutableRubric output.
- FakeRubricExecutor: constructs EvidenceRecord from tool calls.
- FakeDatasetRow: includes extra_info and tool_extra_fields.
- FakeVerificationData: list of input/output pairs for unit-test style checks.

## Unit Tests
- Tool schema return types:
  - Ensure schema includes returns/output_schema and serializes.
  - Fail or warn when missing (per spec).
- BaseTool.execute provenance meta:
  - Verify latency/timestamps/status/error are provided.
- Tool trace record:
  - Validate ToolTraceRecord fields and serialization.
  - Cover ok/error/timeout states.
- Rubric schemas:
  - VerificationItem/EvidencePlan/AggregationRule/ExecutableRubric validate fields.
  - Defaults (weight, is_gate, timeout_ms, estimated_latency_ms) preserved.
  - Invalid AggregationRule.method rejected.
- EvidenceRecord construction:
  - Build from ToolExecutionMeta and verify fields.
- LatencyBucketedSampler:
  - Use extra_info.latency_bucket to group samples; no duplicates.
  - assign_bucket boundary cases around thresholds.
- Aggregation with gates:
  - Gate failure forces reward to 0.
  - weighted_sum/min/product return expected scores.
- Generator structure:
  - LLMRubricGenerator.generate returns ExecutableRubric.
  - estimated_latency_ms is set by _estimate_latency.
- Generator verification_data wiring:
  - verification_data is forwarded into rubric generation and can populate test_runner evidence_plans.

## Integration Tests
- Policy rollout -> tool trace persistence -> reward loop:
  - ToolAgentLoop emits tool_extra_fields.
  - Reward loop can read tool traces as hints.
- Rubric executor evidence-driven scoring:
  - Score derives from evidence; ignores fabricated traces.
- Rubric reward manager end-to-end:
  - reward_score and reward_extra_info.evidence returned.

## Isolation Tests (Policy vs Verifier Tools)
- Policy loads policy_tools.yaml only:
  - Verifier tools unavailable to policy.
- Verifier loads verification_tools.yaml only:
  - Rubric executor sees verifier tools only.
- Rubric generator uses merged schemas for T_e^rub:
  - Merge policy + verifier tool schemas and deduplicate.

## Negative and Robustness Tests
- Tool timeout/error:
  - EvidenceRecord records status/error; gating applies.
- Missing return schema:
  - Explicit failure or warning (per design).
- Missing or forged tool trace:
  - Rubric executor still verifies using tools.

## Data Contract Tests
- Dataset passes extra_info and tool_extra_fields:
  - RLHFDataset preserves both fields.
- Dataset preserves verification_data for offline rubric synthesis.

## Test Coverage Map (Test Item -> File)
- Tool schema return types -> `tests/tools/test_tool_schema_provenance.py`
- BaseTool.execute provenance meta -> `tests/tools/test_base_tool_meta.py`
- Tool trace record fields/serialization -> `tests/tools/test_tool_trace_record.py` (planned)
- Rubric schemas (VerificationItem/EvidencePlan/AggregationRule/ExecutableRubric) -> `tests/rubric/test_schemas.py`
- EvidenceRecord construction -> `tests/rubric/test_evidence_record.py`
- LatencyBucketedSampler iter + assign_bucket boundaries -> `tests/experimental/dataset/test_latency_bucketed_sampler.py`
- Aggregation with gates + weighted_sum/min/product -> `tests/rubric/test_executor_aggregation.py`
- Generator structure + estimated_latency_ms -> `tests/rubric/test_generator_structure.py`
- Generator uses verification_data -> `tests/rubric/test_generator_verification_data.py` (planned)
- Policy rollout -> tool trace persistence -> reward loop -> `tests/experimental/agent_loop/test_tool_trace_persistence.py`
- Rubric executor evidence-driven scoring -> `tests/rubric/test_executor_evidence_scoring.py` (planned)
- Rubric reward manager end-to-end -> `tests/experimental/reward_loop/test_rubric_manager.py`
- Policy vs verifier tool isolation -> `tests/tools/test_tool_isolation.py`
- Rubric generator merges T_e^rub schemas -> `tests/rubric/test_tool_schema_merge.py` (planned)
- Tool timeout/error handling -> `tests/rubric/test_executor_evidence_errors.py` (planned)
- Missing return schema behavior -> `tests/tools/test_tool_schema_provenance.py` (add negative case)
- Missing/forged tool trace robustness -> `tests/rubric/test_executor_trace_robustness.py` (planned)
- Dataset preserves extra_info/tool_extra_fields -> `tests/utils/dataset/test_rlhf_dataset_tool_fields.py` (planned)
- Dataset preserves verification_data -> `tests/utils/dataset/test_rlhf_dataset_verification_data.py` (planned)

## Suggested Test Files
- tests/tools/test_tool_schema_provenance.py
- tests/tools/test_base_tool_meta.py
- tests/tools/test_tool_trace_record.py
- tests/rubric/test_schemas.py
- tests/rubric/test_evidence_record.py
- tests/rubric/test_executor_aggregation.py
- tests/rubric/test_generator_structure.py
- tests/rubric/test_executor_evidence_scoring.py
- tests/rubric/test_tool_schema_merge.py
- tests/rubric/test_executor_evidence_errors.py
- tests/rubric/test_executor_trace_robustness.py
- tests/experimental/agent_loop/test_tool_trace_persistence.py
- tests/experimental/reward_loop/test_rubric_manager.py
- tests/experimental/dataset/test_latency_bucketed_sampler.py
- tests/tools/test_tool_isolation.py
- tests/utils/dataset/test_rlhf_dataset_tool_fields.py

## Commands
```bash
pytest tests/tools/test_tool_schema_provenance.py
pytest tests/tools/test_base_tool_meta.py
pytest tests/tools/test_tool_trace_record.py
pytest tests/rubric/test_schemas.py
pytest tests/rubric/test_evidence_record.py
pytest tests/rubric/test_executor_aggregation.py
pytest tests/rubric/test_generator_structure.py
pytest tests/rubric/test_executor_evidence_scoring.py
pytest tests/rubric/test_tool_schema_merge.py
pytest tests/rubric/test_executor_evidence_errors.py
pytest tests/rubric/test_executor_trace_robustness.py
pytest tests/experimental/agent_loop/test_tool_trace_persistence.py
pytest tests/experimental/reward_loop/test_rubric_manager.py
pytest tests/experimental/dataset/test_latency_bucketed_sampler.py
pytest tests/tools/test_tool_isolation.py
pytest tests/utils/dataset/test_rlhf_dataset_tool_fields.py
```

## Pass/Fail Criteria
- Must pass: tool trace persistence, schema return types, provenance meta, isolation, evidence-driven scoring.
- Any failure blocks rollout.
