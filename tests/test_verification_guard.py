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

"""Tests for VerificationGuard — requires verification evidence before step_done."""

import pytest

from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.guard import GuardContext


class TestVerificationGuard:
    """Test VerificationGuard blocks step_done without override_reason."""

    def test_blocks_step_done_without_override_reason(self):
        """step_done without _override_reason should be blocked."""
        guard = VerificationGuard()
        
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 3}
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is not None
        assert verdict.action == "block"
        assert "verification required" in verdict.message.lower()
        assert verdict.reason == "step_done_no_verification"

    def test_allows_step_done_with_override_reason(self):
        """step_done with _override_reason should pass."""
        guard = VerificationGuard()
        
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "step_done",
                "step_id": 3,
                "_override_reason": "grep shows no conflicts, files parseable"
            }
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is None  # Should pass

    def test_allows_empty_override_reason_is_blocked(self):
        """Empty _override_reason should still be blocked."""
        guard = VerificationGuard()
        
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "step_done",
                "step_id": 3,
                "_override_reason": "   "  # whitespace only
            }
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is not None
        assert verdict.action == "block"

    def test_allows_other_plan_update_actions(self):
        """Other plan_update actions (step_doing, add_steps, etc.) should pass."""
        guard = VerificationGuard()
        
        actions = ["step_doing", "step_skip", "add_steps", "complete", "abandon"]
        
        for action in actions:
            ctx = GuardContext(
                tool_name="plan_update",
                tool_args={"action": action, "step_id": 3}
            )
            verdict = guard.check_pre(ctx)
            assert verdict is None, f"Action {action} should not be blocked"

    def test_allows_all_other_tools(self):
        """All other tools should pass through without blocking."""
        guard = VerificationGuard()
        
        tools = [
            "shell", "read_file", "write_file", "edit_file",
            "memory_read", "memory_write", "grep", "evict"
        ]
        
        for tool in tools:
            ctx = GuardContext(tool_name=tool, tool_args={})
            verdict = guard.check_pre(ctx)
            assert verdict is None, f"Tool {tool} should not be blocked"

    def test_post_recovery_inject_on_first_step_doing(self):
        """After notify_recovery(), should inject reminder on first step_doing."""
        guard = VerificationGuard()
        
        # Trigger recovery
        guard.notify_recovery()
        
        # First step_doing should inject reminder
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_doing", "step_id": 2}
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is not None
        assert verdict.action == "inject"
        assert "recovered via hard_reset" in verdict.message.lower()
        assert verdict.reason == "post_recovery_reminder"

    def test_post_recovery_inject_only_once(self):
        """Post-recovery reminder should only fire once."""
        guard = VerificationGuard()
        
        guard.notify_recovery()
        
        # First step_doing - should inject
        ctx1 = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_doing", "step_id": 2}
        )
        verdict1 = guard.check_pre(ctx1)
        assert verdict1 is not None
        assert verdict1.action == "inject"
        
        # Second step_doing - should not inject
        ctx2 = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_doing", "step_id": 3}
        )
        verdict2 = guard.check_pre(ctx2)
        assert verdict2 is None

    def test_post_recovery_does_not_affect_step_done_blocking(self):
        """Post-recovery state should not interfere with step_done blocking."""
        guard = VerificationGuard()
        
        guard.notify_recovery()
        
        # step_done without override_reason should still be blocked
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 3}
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "step_done_no_verification"
