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

from verl.rubric.schemas import EvidenceRecord


def test_evidence_record_fields():
    record = EvidenceRecord(
        id="e1",
        check_id="check-1",
        tool_name="dummy_tool",
        tool_input={"x": "1"},
        tool_output={"result": "ok"},
        latency_ms=12,
        started_at="2025-01-01T00:00:00Z",
        finished_at="2025-01-01T00:00:00Z",
        success=True,
        error=None,
    )

    for field in (
        "id",
        "check_id",
        "tool_name",
        "tool_input",
        "tool_output",
        "latency_ms",
        "started_at",
        "finished_at",
        "success",
        "error",
    ):
        assert hasattr(record, field), f"EvidenceRecord missing field: {field}"
