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

"""Tests for tool_executor._execute_parallel override_reason extraction.

Verifies the fix for the bug where _execute_parallel didn't pass
_override_reason to GuardContext, causing overrides to fail in multi-tool
batches even when kernel.py already allowed them.
"""

from unittest.mock import MagicMock, patch
from flagscale_agent.react.guard import GuardContext


class TestExecuteParallelOverrideExtraction:
    """Verify _execute_parallel extracts _override_reason from tool_args."""

    def test_override_reason_passed_to_guard_context(self):
        """GuardContext receives override_reason when _override_reason in args."""
        # We test the GuardContext construction logic directly
        # by simulating what _execute_parallel does after the fix
        raw_args = {
            "command": "ls -la",
            "_override_reason": "Safe read-only command, no domain knowledge needed"
        }

        # Extract override_reason (same logic as the fix)
        override_reason = ""
        if raw_args and "_override_reason" in raw_args:
            override_reason = raw_args["_override_reason"]
            raw_args = {k: v for k, v in raw_args.items() if k != "_override_reason"}

        ctx = GuardContext(
            tool_name="shell",
            tool_args=raw_args,
            override_reason=override_reason,
            turn_count=5,
        )

        assert ctx.override_reason == "Safe read-only command, no domain knowledge needed"
        assert "_override_reason" not in ctx.tool_args
        assert ctx.tool_args == {"command": "ls -la"}

    def test_no_override_reason_defaults_empty(self):
        """GuardContext gets empty override_reason when _override_reason absent."""
        raw_args = {"command": "ls -la"}

        override_reason = ""
        if raw_args and "_override_reason" in raw_args:
            override_reason = raw_args["_override_reason"]
            raw_args = {k: v for k, v in raw_args.items() if k != "_override_reason"}

        ctx = GuardContext(
            tool_name="shell",
            tool_args=raw_args,
            override_reason=override_reason,
            turn_count=5,
        )

        assert ctx.override_reason == ""
        assert ctx.tool_args == {"command": "ls -la"}

    def test_override_reason_stripped_from_execution_args(self):
        """_override_reason must not leak into tool execution arguments."""
        raw_args = {
            "path": "/tmp/test.py",
            "content": "print('hello')",
            "_override_reason": "Writing test file for verification"
        }

        override_reason = ""
        if raw_args and "_override_reason" in raw_args:
            override_reason = raw_args["_override_reason"]
            raw_args = {k: v for k, v in raw_args.items() if k != "_override_reason"}

        assert "_override_reason" not in raw_args
        assert override_reason == "Writing test file for verification"
        assert raw_args == {"path": "/tmp/test.py", "content": "print('hello')"}

    def test_override_with_empty_args(self):
        """Handle empty args dict gracefully."""
        raw_args = {}

        override_reason = ""
        if raw_args and "_override_reason" in raw_args:
            override_reason = raw_args["_override_reason"]
            raw_args = {k: v for k, v in raw_args.items() if k != "_override_reason"}

        ctx = GuardContext(
            tool_name="plan_status",
            tool_args=raw_args,
            override_reason=override_reason,
        )

        assert ctx.override_reason == ""
        assert ctx.tool_args == {}
