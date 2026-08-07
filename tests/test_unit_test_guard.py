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

"""Tests for UnitTestGuard — reminds to write tests when modifying agent source."""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field

from flagscale_agent.react.guard.unit_test import UnitTestGuard
from flagscale_agent.react.guard import GuardContext


@pytest.fixture
def guard():
    return UnitTestGuard()


@pytest.fixture
def make_ctx():
    """Factory for GuardContext with tool_name and path."""
    def _make(tool_name="write_file", path="", tool_result="ok"):
        ctx = GuardContext()
        ctx.tool_name = tool_name
        ctx.tool_args = {"path": path}
        ctx.tool_result = tool_result
        return ctx
    return _make


class TestUnitTestGuardSourceDetection:
    def test_agent_source_detected(self):
        assert UnitTestGuard._is_agent_source("flagscale_agent/react/kernel.py")
        assert UnitTestGuard._is_agent_source("/workspace/FlagScale-Agent/flagscale_agent/react/guard/file_tool.py")

    def test_non_agent_paths_not_detected(self):
        assert not UnitTestGuard._is_agent_source("/workspace/documents/readme.md")
        assert not UnitTestGuard._is_agent_source("some_other_project/main.py")
        assert not UnitTestGuard._is_agent_source("")

    def test_test_files_excluded_from_source(self):
        assert not UnitTestGuard._is_agent_source("tests/test_kernel.py")
        assert not UnitTestGuard._is_agent_source("flagscale_agent/tests/test_foo.py")

    def test_non_python_excluded(self):
        assert not UnitTestGuard._is_agent_source("flagscale_agent/react/README.md")
        assert not UnitTestGuard._is_agent_source("flagscale_agent/react/config.yaml")

    def test_is_test_file(self):
        assert UnitTestGuard._is_test_file("tests/test_display.py")
        assert UnitTestGuard._is_test_file("tests/unit/test_kernel.py")
        assert UnitTestGuard._is_test_file("/workspace/tests/test_foo.py")
        assert not UnitTestGuard._is_test_file("flagscale_agent/react/kernel.py")


class TestUnitTestGuardBehavior:
    def test_no_reminder_on_first_source_edit(self, guard, make_ctx):
        """Single edit should not fire — avoid noise on small changes."""
        ctx = make_ctx("edit_file", "flagscale_agent/react/kernel.py")
        verdict = guard.check_post(ctx)
        assert verdict is None

    def test_reminder_after_multiple_source_edits(self, guard, make_ctx):
        """After 2+ source file edits without tests, reminder fires."""
        ctx1 = make_ctx("edit_file", "flagscale_agent/react/kernel.py")
        guard.check_post(ctx1)

        ctx2 = make_ctx("write_file", "flagscale_agent/react/display.py")
        verdict = guard.check_post(ctx2)
        assert verdict is not None
        assert verdict.action == "inject"
        assert "unit test" in verdict.message.lower() or "UnitTest" in verdict.message

    def test_no_reminder_when_test_written(self, guard, make_ctx):
        """If test file is written between source edits, no reminder."""
        ctx1 = make_ctx("edit_file", "flagscale_agent/react/kernel.py")
        guard.check_post(ctx1)

        # Write a test
        ctx_test = make_ctx("write_file", "tests/test_kernel.py")
        guard.check_post(ctx_test)

        # Another source edit
        ctx2 = make_ctx("edit_file", "flagscale_agent/react/agent.py")
        verdict = guard.check_post(ctx2)
        # Should not fire because test was written (pending cleared)
        assert verdict is None

    def test_no_reminder_for_non_agent_files(self, guard, make_ctx):
        """Editing non-agent files should never trigger."""
        ctx1 = make_ctx("write_file", "/workspace/documents/analysis.md")
        guard.check_post(ctx1)
        ctx2 = make_ctx("write_file", "/workspace/some_other/file.py")
        verdict = guard.check_post(ctx2)
        assert verdict is None

    def test_reset_turn_clears_test_flag(self, guard, make_ctx):
        """reset_turn clears the test-written flag but keeps pending sources."""
        ctx1 = make_ctx("edit_file", "flagscale_agent/react/kernel.py")
        guard.check_post(ctx1)
        guard._test_written_this_turn = True
        guard.reset_turn()
        assert not guard._test_written_this_turn
        # Pending sources are cleared since test was written
        # Actually pending was cleared when test was detected — let's test fresh
        guard2 = UnitTestGuard()
        ctx = make_ctx("edit_file", "flagscale_agent/react/agent.py")
        guard2.check_post(ctx)
        guard2.reset_turn()
        # Pending sources preserved across turns
        assert "flagscale_agent/react/agent.py" in guard2._pending_sources
