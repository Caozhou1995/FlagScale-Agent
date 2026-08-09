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

"""Tests for SkillReflectionGuard."""

import pytest
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.skill_reflection import SkillReflectionGuard


class TestSkillReflectionGuard:
    """Test SkillReflectionGuard periodic review mechanism."""

    def test_no_trigger_without_skills(self):
        """Should not trigger if no skills loaded."""
        guard = SkillReflectionGuard(review_interval=3)
        ctx = GuardContext(tool_name="shell", tool_args={})
        
        # 5 calls, no skills
        for _ in range(5):
            verdict = guard.check_pre(ctx)
            assert verdict is None

    def test_no_trigger_before_interval(self):
        """Should not trigger before interval reached."""
        guard = SkillReflectionGuard(review_interval=5)
        guard.on_skill_loaded("test-skill")
        
        ctx = GuardContext(tool_name="shell", tool_args={})
        
        # Calls 1-4: no trigger
        for i in range(4):
            verdict = guard.check_pre(ctx)
            assert verdict is None, f"Triggered too early at call {i+1}"
            assert guard.tool_call_count == i + 1

    def test_trigger_at_interval(self):
        """Should trigger at exactly the interval."""
        guard = SkillReflectionGuard(review_interval=3)
        guard.on_skill_loaded("train-env-setup")
        
        ctx = GuardContext(tool_name="shell", tool_args={})
        
        # Calls 1-2: no trigger
        for _ in range(2):
            assert guard.check_pre(ctx) is None
        
        # Call 3: triggers
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "SKILL ADHERENCE REVIEW" in verdict.message
        assert "/train-env-setup" in verdict.message

    def test_trigger_multiple_times(self):
        """Should trigger every N calls."""
        guard = SkillReflectionGuard(review_interval=4)
        guard.on_skill_loaded("skill-a")
        
        ctx = GuardContext(tool_name="read_file", tool_args={})
        
        # First trigger at call 4
        for _ in range(3):
            assert guard.check_pre(ctx) is None
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        
        # Second trigger at call 8
        for _ in range(3):
            assert guard.check_pre(ctx) is None
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert guard.tool_call_count == 8

    def test_multiple_skills_tracked(self):
        """Should list all loaded skills in review prompt."""
        guard = SkillReflectionGuard(review_interval=2)
        guard.on_skill_loaded("skill-a")
        guard.on_skill_loaded("skill-b")
        guard.on_skill_loaded("skill-c")
        
        ctx = GuardContext(tool_name="shell", tool_args={})
        
        # Skip to trigger
        guard.check_pre(ctx)
        verdict = guard.check_pre(ctx)
        
        assert "/skill-a" in verdict.message
        assert "/skill-b" in verdict.message
        assert "/skill-c" in verdict.message

    def test_skill_unload(self):
        """Should remove skill from tracking on unload."""
        guard = SkillReflectionGuard(review_interval=2)
        guard.on_skill_loaded("skill-a")
        guard.on_skill_loaded("skill-b")
        
        assert "skill-a" in guard.loaded_skills
        assert "skill-b" in guard.loaded_skills
        
        guard.on_skill_unloaded("skill-a")
        
        assert "skill-a" not in guard.loaded_skills
        assert "skill-b" in guard.loaded_skills

    def test_disabled_with_zero_interval(self):
        """Interval=0 should disable reviews."""
        guard = SkillReflectionGuard(review_interval=0)
        guard.on_skill_loaded("skill-a")
        
        ctx = GuardContext(tool_name="shell", tool_args={})
        
        # 10 calls, no trigger
        for _ in range(10):
            assert guard.check_pre(ctx) is None

    def test_disabled_with_negative_interval(self):
        """Negative interval should disable reviews."""
        guard = SkillReflectionGuard(review_interval=-1)
        guard.on_skill_loaded("skill-a")
        
        ctx = GuardContext(tool_name="shell", tool_args={})
        
        for _ in range(10):
            assert guard.check_pre(ctx) is None

    def test_empty_tool_name_not_counted(self):
        """Tool calls with empty tool_name should not increment counter."""
        guard = SkillReflectionGuard(review_interval=3)
        guard.on_skill_loaded("skill-a")
        
        # Empty tool_name calls
        for _ in range(5):
            ctx = GuardContext(tool_name="", tool_args={})
            assert guard.check_pre(ctx) is None
        
        # Counter should still be 0
        assert guard.tool_call_count == 0
        
        # Now real tool calls
        ctx = GuardContext(tool_name="shell", tool_args={})
        guard.check_pre(ctx)
        guard.check_pre(ctx)
        assert guard.tool_call_count == 2

    def test_calls_until_next_review(self):
        """Property should correctly report remaining calls."""
        guard = SkillReflectionGuard(review_interval=5)
        guard.on_skill_loaded("skill-a")
        
        ctx = GuardContext(tool_name="shell", tool_args={})
        
        assert guard.calls_until_next_review == 5
        
        guard.check_pre(ctx)
        assert guard.calls_until_next_review == 4
        
        guard.check_pre(ctx)
        assert guard.calls_until_next_review == 3
        
        guard.check_pre(ctx)
        guard.check_pre(ctx)
        assert guard.calls_until_next_review == 1
        
        # Trigger
        guard.check_pre(ctx)
        assert guard.calls_until_next_review == 5  # Reset

    def test_reset_turn_does_not_clear_state(self):
        """reset_turn should keep skills and counter (session-level)."""
        guard = SkillReflectionGuard(review_interval=3)
        guard.on_skill_loaded("skill-a")
        
        ctx = GuardContext(tool_name="shell", tool_args={})
        guard.check_pre(ctx)
        guard.check_pre(ctx)
        
        # User sends new message
        guard.reset_turn()
        
        # State should persist
        assert guard.tool_call_count == 2
        assert "skill-a" in guard.loaded_skills
        assert guard.calls_until_next_review == 1

    def test_review_prompt_structure(self):
        """Review prompt should have correct structure."""
        guard = SkillReflectionGuard(review_interval=1)
        guard.on_skill_loaded("test-skill")
        
        ctx = GuardContext(tool_name="shell", tool_args={})
        verdict = guard.check_pre(ctx)
        
        msg = verdict.message
        
        # Key sections present
        assert "[SKILL ADHERENCE REVIEW]" in msg
        assert "loaded these skills" in msg
        assert "/test-skill" in msg
        assert "review your recent actions" in msg
        assert "conversation history" in msg
        assert "deviated intentionally" in msg
        assert "drifted unintentionally" in msg
        assert "NOT a judgment" in msg

    def test_check_post_always_none(self):
        """check_post should always return None."""
        guard = SkillReflectionGuard(review_interval=1)
        ctx = GuardContext(tool_name="shell", tool_result="output")
        
        assert guard.check_post(ctx) is None

    def test_duplicate_skill_load_ignored(self):
        """Loading same skill twice should not duplicate."""
        guard = SkillReflectionGuard(review_interval=2)
        
        guard.on_skill_loaded("skill-a")
        guard.on_skill_loaded("skill-a")
        guard.on_skill_loaded("skill-a")
        
        assert guard.loaded_skills == ["skill-a"]

    def test_priority_is_low(self):
        """Priority should be 5 (lower than safety, higher than advisory)."""
        guard = SkillReflectionGuard()
        assert guard.priority == 5

    def test_name_is_skill_reflection(self):
        """Guard name should be skill_reflection."""
        guard = SkillReflectionGuard()
        assert guard.name == "skill_reflection"
