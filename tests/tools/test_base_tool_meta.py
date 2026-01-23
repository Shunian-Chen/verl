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

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)


class DummyTool(BaseTool):
    pass


@pytest.mark.xfail(reason="BaseTool.execute does not yet return provenance metadata in tool_metrics")
@pytest.mark.asyncio
async def test_base_tool_execute_returns_provenance_meta():
    tool_schema = OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="dummy_tool",
            description="dummy",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={"x": OpenAIFunctionPropertySchema(type="string")},
                required=["x"],
            ),
        ),
    )
    tool = DummyTool(config={"type": "native"}, tool_schema=tool_schema)
    _, _, tool_metrics = await tool.execute("instance-1", {"x": "1"})

    meta = tool_metrics.get("execution_meta", tool_metrics)
    for key in ("latency_ms", "started_at", "finished_at", "status"):
        assert key in meta, f"Missing provenance field: {key}"
