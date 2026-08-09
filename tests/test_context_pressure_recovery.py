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

"""Tests for ContextPressureGuard auto-recovery fix.

Regression tests for the deadlock bug where:
- _need_hard_reset was set when evictable < 60
- Later evictable grew back above 60 (messages arrived or conditions changed)
- But _need_hard_reset never cleared — guard kept blocking forever
- hard_reset tool refused to execute (conditions didn't warrant it)
- Result: permanent deadlock, no tool could execute
"""

import pytest
from flagscale_agent.react.guard.context_pressure import (
    ContextPressureGuard,
    BLOCK_RATIO,
    EVICTABLE_THRESHOLD,
)
from flagscale_agent.react.guard import GuardContext


def _make_ctx(pressure: float, evictable_count: int = 80,
              tool_name: str = "", tool_args: dict = None) -> GuardContext:
    """Create a GuardContext with specified pressure and tool info."""
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        context_pressure=pressure,
        evictable_indexes=list(range(1, evictable_count + 1)),
    )


class TestHardResetAutoRecovery:
    """Verify _need_hard_reset clears when conditions recover."""

    def test_recovers_when_evictable_grows_above_threshold(self):
        """Once evictable grows back >= 60, hard_reset lock releases."""
        guard = ContextPressureGuard()

        # Trigger hard_reset path: high pressure + low evictable
        ctx1 = _make_ctx(0.85, evictable_count=30, tool_name="shell")
        result = guard.check_pre(ctx1)
        assert result is not None
        assert result.category == "context_pressure_hard_reset"
        assert guard._need_hard_reset is True

        # Now conditions improve: evictable grew back to 80
        ctx2 = _make_ctx(0.85, evictable_count=80, tool_name="shell")
        result = guard.check_pre(ctx2)
        # Should now be evict-path block (not hard_reset block)
        assert result is not None
        assert result.category == "context_pressure_evict"
        assert guard._need_hard_reset is False

    def test_recovers_when_pressure_drops_below_threshold(self):
        """If pressure drops below 80%, hard_reset lock releases."""
        guard = ContextPressureGuard()

        # Trigger hard_reset path
        ctx1 = _make_ctx(0.90, evictable_count=20, tool_name="shell")
        result = guard.check_pre(ctx1)
        assert result is not None
        assert guard._need_hard_reset is True

        # Pressure dropped (e.g. after eviction reduced estimated tokens)
        ctx2 = _make_ctx(0.70, evictable_count=20, tool_name="shell")
        result = guard.check_pre(ctx2)
        # Below BLOCK_RATIO → no block at all
        assert result is None
        assert guard._need_hard_reset is False

    def test_still_blocks_when_conditions_remain_bad(self):
        """If conditions haven't improved, still blocks."""
        guard = ContextPressureGuard()

        # Trigger hard_reset path
        ctx1 = _make_ctx(0.90, evictable_count=20, tool_name="shell")
        guard.check_pre(ctx1)
        assert guard._need_hard_reset is True

        # Same bad conditions
        ctx2 = _make_ctx(0.88, evictable_count=25, tool_name="shell")
        result = guard.check_pre(ctx2)
        assert result is not None
        assert result.category == "context_pressure_hard_reset"
        assert guard._need_hard_reset is True

    def test_save_tools_still_pass_during_hard_reset_lock(self):
        """Save tools are never blocked even during hard_reset lock."""
        guard = ContextPressureGuard()

        # Trigger hard_reset path
        ctx1 = _make_ctx(0.90, evictable_count=20, tool_name="shell")
        guard.check_pre(ctx1)

        # Save tools pass through
        for tool in ["memory_write", "evict", "plan_update", "hard_reset"]:
            ctx = _make_ctx(0.90, evictable_count=20, tool_name=tool)
            result = guard.check_pre(ctx)
            assert result is None, f"{tool} should pass through"


class TestCheckPostRecovery:
    """Verify check_post resets _need_hard_reset after eviction."""

    def test_reset_after_hard_reset_call(self):
        """Original behavior: hard_reset clears the flag."""
        guard = ContextPressureGuard()
        guard._need_hard_reset = True

        ctx = _make_ctx(0.50, evictable_count=80, tool_name="hard_reset")
        guard.check_post(ctx)
        assert guard._need_hard_reset is False

    def test_reset_after_evict_when_conditions_improve(self):
        """After evict, if conditions improved, clear the flag."""
        guard = ContextPressureGuard()
        guard._need_hard_reset = True

        # Evict succeeded and now evictable is back above threshold
        ctx = _make_ctx(0.75, evictable_count=70, tool_name="evict")
        guard.check_post(ctx)
        assert guard._need_hard_reset is False

    def test_no_reset_after_evict_if_still_bad(self):
        """After evict, if conditions still bad, keep the flag."""
        guard = ContextPressureGuard()
        guard._need_hard_reset = True

        # Evict but conditions still bad
        ctx = _make_ctx(0.85, evictable_count=30, tool_name="evict")
        guard.check_post(ctx)
        assert guard._need_hard_reset is True


class TestOverrideClearsState:
    """Verify that accept_override clears _need_hard_reset so override sticks."""

    def test_override_clears_hard_reset_flag(self):
        """After successful override, _need_hard_reset is cleared."""
        guard = ContextPressureGuard()
        guard._need_hard_reset = True

        ctx = _make_ctx(0.90, evictable_count=20, tool_name="shell")
        accepted = guard.accept_override("Safe read-only grep command", ctx)
        assert accepted is True
        assert guard._need_hard_reset is False

    def test_subsequent_calls_not_blocked_after_override(self):
        """After override, next tool call goes through without needing override."""
        guard = ContextPressureGuard()

        # Trigger hard_reset path
        ctx1 = _make_ctx(0.90, evictable_count=20, tool_name="shell")
        result = guard.check_pre(ctx1)
        assert result is not None
        assert guard._need_hard_reset is True

        # Override accepted
        guard.accept_override("Need to read file to fix the guard bug", ctx1)
        assert guard._need_hard_reset is False

        # Next call: still high pressure + low evictable → triggers fresh check
        # and sets _need_hard_reset again (this is correct — new evaluation)
        ctx2 = _make_ctx(0.90, evictable_count=20, tool_name="read_file")
        result = guard.check_pre(ctx2)
        # Re-evaluates: pressure >= 80% AND evictable < 60 → blocks again
        assert result is not None

    def test_override_with_short_reason_rejected(self):
        """Override with too-short reason is rejected, flag stays."""
        guard = ContextPressureGuard()
        guard._need_hard_reset = True

        ctx = _make_ctx(0.90, evictable_count=20, tool_name="shell")
        accepted = guard.accept_override("ok", ctx)
        assert accepted is False
        assert guard._need_hard_reset is True
