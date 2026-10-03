# Copyright 2026 FlagOS Contributors
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

"""Tests for WriteParamLossGuard (post-inject on write_file missing content)."""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.write_param_loss import WriteParamLossGuard

REGISTRY_ERR = (
    "ERROR: required argument(s) missing for tool 'write_file': content. "
    "Re-issue the call with these arguments (names/types per the tool schema)."
)


def _ctx(tool_name, tool_result):
    return GuardContext(tool_name=tool_name, tool_args={}, tool_result=tool_result)


class TestWriteParamLossGuard:
    def test_fires_on_registry_error(self):
        g = WriteParamLossGuard()
        v = g.check_post(_ctx("write_file", REGISTRY_ERR))
        assert v is not None and v.action == "inject"
        assert "parallel" in v.message
        assert "3000" in v.message

    def test_once_per_turn(self):
        g = WriteParamLossGuard()
        assert g.check_post(_ctx("write_file", REGISTRY_ERR)) is not None
        assert g.check_post(_ctx("write_file", REGISTRY_ERR)) is None

    def test_reset_turn_rearms(self):
        g = WriteParamLossGuard()
        assert g.check_post(_ctx("write_file", REGISTRY_ERR)) is not None
        g.reset_turn()
        assert g.check_post(_ctx("write_file", REGISTRY_ERR)) is not None

    def test_other_tools_never_fire(self):
        g = WriteParamLossGuard()
        assert g.check_post(_ctx("read_file", REGISTRY_ERR)) is None
        assert g.check_post(_ctx("edit_file", "ERROR: required argument(s) missing for tool 'edit_file': old_string.")) is None

    def test_real_write_errors_do_not_fire(self):
        g = WriteParamLossGuard()
        # shrink-guard rejection and a successful write must never trip it
        assert g.check_post(_ctx(
            "write_file",
            "ERROR: mode=write would shrink /x from 1000 to 300 bytes (>50% drop).")) is None
        assert g.check_post(_ctx("write_file", "Wrote 100 chars to /x")) is None

    def test_check_pre_is_noop(self):
        g = WriteParamLossGuard()
        assert g.check_pre(_ctx("write_file", REGISTRY_ERR)) is None

    def test_match_anchors_to_real_registry_output(self):
        """Anti-tautology: the guard's _MATCH must appear in the error string a
        REAL ToolRegistry produces for write_file with content absent."""
        import flagscale_agent.react.guard.write_param_loss as m
        from flagscale_agent.react.tools import ToolRegistry
        from flagscale_agent.react.tools.write_file import WriteFileTool
        reg = ToolRegistry()
        reg.register(WriteFileTool())
        real = reg.execute("write_file", path="/tmp/whatever.txt")
        assert m._MATCH in real
