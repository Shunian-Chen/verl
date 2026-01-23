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

import os
import tempfile

from omegaconf import OmegaConf

from verl.tools.base_tool import BaseTool
from verl.tools.utils.tool_registry import initialize_tools_from_config


class PolicyOnlyTool(BaseTool):
    pass


class VerifierOnlyTool(BaseTool):
    pass


def _write_tool_config(path, class_name, tool_name):
    config = {
        "tools": [
            {
                "class_name": class_name,
                "config": {"type": "native"},
                "tool_schema": {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "dummy",
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"type": "string", "description": "x"}},
                            "required": ["x"],
                        },
                    },
                },
            }
        ]
    }
    OmegaConf.save(config, path)


def test_policy_verifier_tool_isolation():
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_path = os.path.join(tmpdir, "policy_tools.yaml")
        verifier_path = os.path.join(tmpdir, "verification_tools.yaml")

        _write_tool_config(
            policy_path,
            "tests.tools.test_tool_isolation.PolicyOnlyTool",
            "policy_tool",
        )
        _write_tool_config(
            verifier_path,
            "tests.tools.test_tool_isolation.VerifierOnlyTool",
            "verifier_tool",
        )

        policy_tools = initialize_tools_from_config(policy_path)
        verifier_tools = initialize_tools_from_config(verifier_path)

        assert [tool.name for tool in policy_tools] == ["policy_tool"]
        assert [tool.name for tool in verifier_tools] == ["verifier_tool"]
