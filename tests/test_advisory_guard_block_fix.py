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

"""Test fix for advisory guard block counter reset bug.

Bug: MemoryDisciplineGuard and KnowledgeSkillGuard were resetting their counters
when returning a block verdict, BEFORE checking if LLM provided an override.
This allowed LLM to bypass blocks by simply ignoring them (no override).

Fix: Only reset counter in accept_override() when override succeeds.
Block verdict should NOT reset the counter.
"""

import pytest
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
from flagscale_agent.react.guard.knowledge_skill import KnowledgeSkillGuard


class TestMemoryDisciplineBlockFix:
    """Test MemoryDisciplineGuard block behavior without override."""

    def test_block_without_override_persists(self):
        """Block should persist if LLM doesn't provide override."""
        guard = MemoryDisciplineGuard()
        
        # Make 29 calls (counter will be 29)
        for i in range(29):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
        
        assert guard._calls_since_memory == 29
        
        # 30th call: increment to 30, check >= 30, block
        ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert guard._calls_since_memory == 30  # Incremented before returning block
        
        # Continue without override — should keep blocking
        for i in range(5):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
            assert verdict is not None
            assert verdict.action == "block", f"Call {30+i+1} should still block"
            # Counter keeps incrementing: 31, 32, 33, 34, 35
            assert guard._calls_since_memory == 31 + i

    def test_block_with_override_resets(self):
        """Override should reset counter and allow new cycle."""
        guard = MemoryDisciplineGuard()
        
        # Reach threshold and block
        for i in range(30):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            guard.check_pre(ctx)
        
        ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
        verdict = guard.check_pre(ctx)
        assert verdict.action == "block"
        
        # Provide override
        result = guard.accept_override("Task doesn't need memory yet", ctx)
        assert result is True
        assert guard._calls_since_memory == 0  # Counter reset by override
        
        # New cycle — should not block until threshold again
        for i in range(29):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
            if verdict and verdict.action == "block":
                pytest.fail(f"Shouldn't block at call {i+1} after override")


class TestKnowledgeSkillBlockFix:
    """Test KnowledgeSkillGuard block behavior without override."""

    def test_block_without_override_persists(self):
        """Block should persist if LLM doesn't provide override."""
        guard = KnowledgeSkillGuard()
        
        # Reach threshold (40 calls)
        for i in range(40):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
        
        assert guard._calls_since_knowledge == 40
        
        # Next call: counter increments to 41, then checks >= 40, blocks
        ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert guard._calls_since_knowledge == 41  # Incremented before check
        
        # Continue without override — should keep blocking
        for i in range(5):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
            assert verdict is not None
            assert verdict.action == "block", f"Call {41+i} should still block"
            assert guard._calls_since_knowledge > 40

    def test_block_with_override_resets(self):
        """Override should reset counter and allow new cycle."""
        guard = KnowledgeSkillGuard()
        
        # Reach threshold and block
        for i in range(40):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            guard.check_pre(ctx)
        
        ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
        verdict = guard.check_pre(ctx)
        assert verdict.action == "block"
        
        # Provide override
        result = guard.accept_override("Task is simple, doesn't need knowledge", ctx)
        assert result is True
        assert guard._calls_since_knowledge == 0  # Counter reset by override
        
        # New cycle — should not block until threshold again
        for i in range(39):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
            if verdict and verdict.action == "block":
                pytest.fail(f"Shouldn't block at call {i+1} after override")


class TestInjectStillWorks:
    """Verify inject messages still trigger at correct intervals."""


    def test_knowledge_skill_inject_timing(self):
        """Inject should trigger every 15 calls."""
        guard = KnowledgeSkillGuard()
        
        inject_at = []
        for i in range(1, 41):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
            if verdict and verdict.action == "inject":
                inject_at.append(i)
        
        assert inject_at == [15, 30], f"Inject should trigger at 15/30, got {inject_at}"
    def test_memory_discipline_inject_timing(self):
        """Inject should trigger every 10 calls. Block at 30 supersedes inject."""
        guard = MemoryDisciplineGuard()
        
        inject_at = []
        block_at = []
        for i in range(1, 31):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
            if verdict:
                if verdict.action == "inject":
                    inject_at.append(i)
                elif verdict.action == "block":
                    block_at.append(i)
        
        # At call 30: counter increments to 30, check >= 30 passes, returns block (not inject)
        assert inject_at == [10, 20], f"Inject should trigger at 10/20, got {inject_at}"
        assert block_at == [30], f"Block should trigger at 30, got {block_at}"

    def test_knowledge_skill_inject_timing(self):
        """Inject should trigger every 15 calls."""
        guard = KnowledgeSkillGuard()
        
        inject_at = []
        for i in range(1, 41):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
            if verdict and verdict.action == "inject":
                inject_at.append(i)
        
        assert inject_at == [15, 30], f"Inject should trigger at 15/30, got {inject_at}"
