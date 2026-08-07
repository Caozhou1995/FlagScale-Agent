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

"""Integration tests for the new guard system."""
import pytest
from unittest.mock import MagicMock

from flagscale_agent.react.guard import GuardContext, GuardVerdict
from flagscale_agent.react.guard.file_tool import FileToolGuard
from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard


def make_ctx(tool_name="", tool_args=None, tool_result=""):
    """Helper to create a GuardContext."""
    ctx = MagicMock(spec=GuardContext)
    ctx.tool_name = tool_name
    ctx.tool_args = tool_args or {}
    ctx.tool_result = tool_result
    ctx.classify_fn = None
    ctx.current_experiment_name = ""
    ctx.experiment_diff_fn = None
    return ctx



class TestFileToolGuard:
    """Test file truncation detection."""

    def test_detects_truncated_content(self):
        guard = FileToolGuard()
        # Content with unbalanced brackets (looks truncated)
        content = "def foo():\n" + "    x = {\n" * 10 + "    'key': 'value',\n" * 200
        ctx = make_ctx("write_file", {"path": "test.py", "content": content, "mode": "write"})
        result = guard.check_pre(ctx)
        # Should detect unbalanced brackets
        if len(content) > 4000:
            assert result is not None

    def test_no_warning_for_balanced_content(self):
        guard = FileToolGuard()
        content = "x = 1\ny = 2\n" * 400  # Long but balanced
        ctx = make_ctx("write_file", {"path": "test.py", "content": content, "mode": "write"})
        result = guard.check_pre(ctx)
        # Balanced content shouldn't trigger truncation warning
        assert result is None


    def test_memory_discipline_reminder_threshold(self):
        """Memory discipline reminds every 10 non-memory tool calls."""
        guard = MemoryDisciplineGuard()

        # 9 calls — no reminder
        for i in range(9):
            ctx = make_ctx("shell", {"command": f"echo {i}"}, tool_result="ok")
            result = guard.check_pre(ctx)
            assert result is None, f"Unexpected reminder on call {i+1}: {result}"

        # 10th call — triggers reminder, counter resets
        ctx = make_ctx("shell", {"command": "echo 10"}, tool_result="ok")
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "inject"
        assert "10 tool calls" in result.message
        assert guard._calls_since_memory == 0  # Reset after firing

        # Next 9 calls — no reminder again
        for i in range(9):
            ctx = make_ctx("shell", {"command": f"echo {i}"}, tool_result="ok")
            result = guard.check_pre(ctx)
            assert result is None

        # 20th total call (10th since last reminder) — triggers again
        ctx = make_ctx("shell", {"command": "echo again"}, tool_result="ok")
        result = guard.check_pre(ctx)
        assert result is not None
        assert result.action == "inject"

        # memory_read resets counter
        ctx = make_ctx("memory_read", {"key": "test"}, tool_result="value")
        result = guard.check_pre(ctx)
        assert result is None
        assert guard._calls_since_memory == 0

    def test_memory_discipline_staleness_check_on_read(self):
        """After memory_read returns content, inject staleness verification reminder."""
        guard = MemoryDisciplineGuard()

        # memory_read returns substantive content → staleness check fires
        ctx = make_ctx("memory_read", {"key": "some_finding"}, 
                       tool_result="[finding] [global] Some old bug info that might be stale... " * 5)
        result = guard.check_post(ctx)
        assert result is not None
        assert result.action == "inject"
        assert "stale" in result.message.lower() or "supersede" in result.message.lower()

    def test_memory_discipline_staleness_check_on_list(self):
        """After memory_list returns entries, inject staleness verification reminder."""
        guard = MemoryDisciplineGuard()

        ctx = make_ctx("memory_list", {},
                       tool_result="Showing 5/5 entries\n[finding] key1: some old content\n[finding] key2: more old content")
        result = guard.check_post(ctx)
        assert result is not None
        assert result.action == "inject"

    def test_memory_discipline_no_staleness_on_empty_result(self):
        """No staleness reminder if memory_read returns nothing/error."""
        guard = MemoryDisciplineGuard()

        # Short result (likely "not found")
        ctx = make_ctx("memory_read", {"key": "missing"}, tool_result="Key not found")
        result = guard.check_post(ctx)
        assert result is None

    def test_memory_discipline_no_staleness_on_no_entries(self):
        """No staleness reminder if memory_list has no entries."""
        guard = MemoryDisciplineGuard()

        ctx = make_ctx("memory_list", {}, tool_result="No entries found.")
        result = guard.check_post(ctx)
        assert result is None

    def test_memory_discipline_staleness_fires_once_per_batch(self):
        """Staleness reminder fires only once even if multiple reads happen."""
        guard = MemoryDisciplineGuard()

        # First read → fires
        ctx = make_ctx("memory_read", {"key": "key1"}, 
                       tool_result="[finding] big content here... " * 10)
        result = guard.check_post(ctx)
        assert result is not None

        # Second read → suppressed
        ctx = make_ctx("memory_read", {"key": "key2"},
                       tool_result="[finding] more content... " * 10)
        result = guard.check_post(ctx)
        assert result is None

    def test_memory_discipline_staleness_resets_after_write(self):
        """After memory_write, staleness flag resets so next read gets reminder again."""
        guard = MemoryDisciplineGuard()

        # Read → fires
        ctx = make_ctx("memory_read", {"key": "key1"},
                       tool_result="[finding] big content... " * 10)
        result = guard.check_post(ctx)
        assert result is not None

        # Write (supersede) → resets flag
        ctx = make_ctx("memory_write", {"key": "new_key", "supersedes": ["key1"]})
        guard.check_pre(ctx)

        # New read → fires again
        ctx = make_ctx("memory_read", {"key": "key2"},
                       tool_result="[finding] another old entry... " * 10)
        result = guard.check_post(ctx)
        assert result is not None

    def test_memory_discipline_staleness_resets_on_new_turn(self):
        """New user message resets staleness flag."""
        guard = MemoryDisciplineGuard()

        # Read → fires
        ctx = make_ctx("memory_read", {"key": "key1"},
                       tool_result="[finding] big content... " * 10)
        guard.check_post(ctx)

        # New turn
        guard.reset_turn()

        # Read again → fires (fresh turn)
        ctx = make_ctx("memory_read", {"key": "key2"},
                       tool_result="[finding] other content... " * 10)
        result = guard.check_post(ctx)
        assert result is not None

    def test_memory_discipline_no_staleness_for_shell(self):
        """Non-memory tools don't trigger staleness check."""
        guard = MemoryDisciplineGuard()

        ctx = make_ctx("shell", {"command": "ls"}, tool_result="lots of output " * 100)
        result = guard.check_post(ctx)
        assert result is None


class TestMemoryEvolution:
    """Tests for memory self-evolution mechanism in MemoryDisciplineGuard."""

    def test_evolution_reminder_on_task_complete_without_review(self):
        """If agent emits TASK_COMPLETE without any memory_list, remind to review."""
        guard = MemoryDisciplineGuard()

        # Simulate assistant text with TASK_COMPLETE, no tool call
        ctx = MagicMock(spec=GuardContext)
        ctx.tool_name = ""
        ctx.tool_args = {}
        ctx.tool_result = ""
        ctx.assistant_text = "Done. [TASK_COMPLETE]"
        ctx.classify_fn = None

        result = guard.check_pre(ctx)
        assert result is not None
        assert "TASK_COMPLETE" in result.message
        assert "memory_list" in result.message
        assert guard._evolution_reminded is True

    def test_no_evolution_reminder_if_memory_reviewed(self):
        """If agent already did memory_list, no evolution reminder on TASK_COMPLETE."""
        guard = MemoryDisciplineGuard()

        # Simulate a memory_list call
        ctx = MagicMock(spec=GuardContext)
        ctx.tool_name = "memory_list"
        ctx.tool_args = {}
        ctx.tool_result = "entries..."
        ctx.assistant_text = ""
        ctx.classify_fn = None
        guard.check_pre(ctx)

        assert guard._has_memory_review is True

        # Now TASK_COMPLETE — no reminder needed
        ctx2 = MagicMock(spec=GuardContext)
        ctx2.tool_name = ""
        ctx2.tool_args = {}
        ctx2.tool_result = ""
        ctx2.assistant_text = "All done [TASK_COMPLETE]"
        ctx2.classify_fn = None

        result = guard.check_pre(ctx2)
        assert result is None

    def test_evolution_reminder_fires_only_once(self):
        """Evolution reminder should fire at most once per session."""
        guard = MemoryDisciplineGuard()

        ctx = MagicMock(spec=GuardContext)
        ctx.tool_name = ""
        ctx.tool_args = {}
        ctx.tool_result = ""
        ctx.assistant_text = "[TASK_COMPLETE]"
        ctx.classify_fn = None

        result1 = guard.check_pre(ctx)
        assert result1 is not None

        # Second time — no reminder
        result2 = guard.check_pre(ctx)
        assert result2 is None

    def test_memory_read_also_counts_as_review(self):
        """memory_read should also mark _has_memory_review."""
        guard = MemoryDisciplineGuard()

        ctx = MagicMock(spec=GuardContext)
        ctx.tool_name = "memory_read"
        ctx.tool_args = {"key": "fact/cluster/ssh_port"}
        ctx.tool_result = "content..."
        ctx.assistant_text = ""
        ctx.classify_fn = None
        guard.check_pre(ctx)

        assert guard._has_memory_review is True


# ── Override Hint Tests ──

class TestOverrideHint:
    def test_override_hint_format(self):
        """Override hint should contain _override_reason instruction."""
        from flagscale_agent.react.guard import _OVERRIDE_HINT
        assert "_override_reason" in _OVERRIDE_HINT
        assert "OVERRIDE REQUIRED" in _OVERRIDE_HINT

    def test_hint_added_to_block(self):
        """Block verdicts get override hint appended by registry."""
        from flagscale_agent.react.guard import GuardRegistry, Guard, GuardVerdict, GuardContext

        class BlockingGuard(Guard):
            name = "blocker"
            def check_pre(self, ctx):
                return GuardVerdict.block("[Blocked] reason", reason="test", category="test")

        reg = GuardRegistry()
        reg.register(BlockingGuard())
        ctx = GuardContext(tool_name="shell", override_reason="")
        result = reg.check_pre(ctx)
        assert "_override_reason" in result.message
        assert "OVERRIDE REQUIRED" in result.message

    def test_hint_not_re_added_after_rejected_override(self):
        """If override was attempted and rejected, no re-hint."""
        from flagscale_agent.react.guard import GuardRegistry, Guard, GuardVerdict, GuardContext

        class BlockingGuard(Guard):
            name = "blocker"
            def check_pre(self, ctx):
                return GuardVerdict.block("[Blocked] still wrong", reason="test", category="test")
            def accept_override(self, reason, ctx):
                return False  # Always reject

        reg = GuardRegistry()
        reg.register(BlockingGuard())
        ctx = GuardContext(tool_name="shell", override_reason="I already tried")
        result = reg.check_pre(ctx)
        # Override was attempted but rejected — hint should not be re-added
        assert "OVERRIDE REQUIRED" not in result.message

    def test_escalate_hint_added(self):
        """Escalate verdicts get escalate hint telling LLM not to retry."""
        from flagscale_agent.react.guard import GuardRegistry, Guard, GuardVerdict, GuardContext

        class EscalatingGuard(Guard):
            name = "escalator"
            def check_pre(self, ctx):
                return GuardVerdict.escalate("[Escalated] forbidden", reason="test", category="test")

        reg = GuardRegistry()
        reg.register(EscalatingGuard())
        ctx = GuardContext(tool_name="shell", override_reason="")
        result = reg.check_pre(ctx)
        assert "ESCALATED" in result.message
        assert "DO NOT retry" in result.message
        assert "OVERRIDE REQUIRED" not in result.message

    def test_escalate_cannot_be_overridden(self):
        """Escalate ignores override_reason."""
        from flagscale_agent.react.guard import GuardRegistry, Guard, GuardVerdict, GuardContext

        class EscalatingGuard(Guard):
            name = "escalator"
            def check_pre(self, ctx):
                return GuardVerdict.escalate("[Escalated] forbidden", reason="test", category="test")

        reg = GuardRegistry()
        reg.register(EscalatingGuard())
        ctx = GuardContext(tool_name="shell", override_reason="I have a good reason")
        result = reg.check_pre(ctx)
        # Still escalated — override_reason is ignored
        assert result is not None
        assert "ESCALATED" in result.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
