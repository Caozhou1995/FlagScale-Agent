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

"""Advisory-guard counter semantics for the two call-count guards.

History of this file:
- It began as the regression test for a counter-reset bug: both guards reset
  their counter when returning a block verdict, BEFORE checking whether the LLM
  provided an override, which let the LLM bypass the gate by ignoring it. The
  fix: only accept_override() resets the counter.
- The gates then became ADVISORY: the 30-call (memory) and 80-call (knowledge)
  thresholds still fire on the same call count, but they inject instead of
  blocking. The assertions below keep the ORIGINAL counter/threshold semantics
  and only move the asserted object from the block verdict to the inject
  verdict, so the cadence is still pinned by a test.

Two properties are load-bearing here:
1. the counter still advances exactly as before (both guards' thresholds and
   per-call arithmetic are unchanged), and
2. the escalated mark lands ONCE per cycle — a bare `>= threshold` inject would
   re-fire on every later call and nag harder than the block it replaced.
"""

import pytest
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
from flagscale_agent.react.guard.knowledge_skill import KnowledgeSkillGuard


def _adv_ks(guard, ctx):
    """One real KnowledgeSkill tool cycle: check_pre decision + check_post count
    persistence. KnowledgeSkillGuard advances its counter in check_post (so a
    call that never ran does not inflate it), so accumulation must drive both
    phases. Returns the check_pre verdict."""
    verdict = guard.check_pre(ctx)
    guard.check_post(ctx)
    return verdict


class TestMemoryDisciplineInjectAtThreshold:
    """MemoryDisciplineGuard escalates at 30 calls with an inject, not a block."""

    def test_heavy_mark_is_inject_and_lands_once(self):
        """The 30-call mark injects once; later calls fall back to the 10 rhythm."""
        guard = MemoryDisciplineGuard()

        # 29 calls: below the heavy mark (10 and 20 already fired).
        for i in range(29):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            guard.check_pre(ctx)

        assert guard._calls_since_memory == 29

        # 30th call: increment to 30, cross the mark, inject the escalated text.
        ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "inject"
        assert verdict.reason == "no_memory_ops_30_calls"
        assert guard._calls_since_memory == 30

        # Calls 31-39: no re-fire. A bare `>= 30` inject would nag every call.
        for i in range(9):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
            assert verdict is None, f"Call {31+i} must not re-fire the heavy mark"
            assert guard._calls_since_memory == 31 + i

        # Call 40: the ordinary 10-call rhythm resumes.
        ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
        verdict = guard.check_pre(ctx)
        assert verdict is not None and verdict.action == "inject"
        assert verdict.reason == "no_memory_ops_recently"

    def test_never_blocks(self):
        """No call count ever produces a block from this guard."""
        guard = MemoryDisciplineGuard()
        for i in range(1, 121):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
            assert verdict is None or verdict.action == "inject", (
                f"Call {i} produced {getattr(verdict, 'action', None)}")

    def test_memory_op_rearms_the_heavy_mark(self):
        """A memory op resets counter AND re-arms the latch for the next cycle."""
        guard = MemoryDisciplineGuard()

        for i in range(30):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            guard.check_pre(ctx)
        assert guard._escalated is True

        # A memory read resets and re-arms.
        guard.check_pre(GuardContext(tool_name="memory_read",
                                     tool_args={"key": "pitfall/x/"}))
        assert guard._calls_since_memory == 0
        assert guard._escalated is False

        # Second cycle: the heavy mark fires again at 30.
        reasons = []
        for i in range(1, 31):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
            if verdict:
                reasons.append((i, verdict.reason))
        assert (30, "no_memory_ops_30_calls") in reasons

    def test_override_resets_counter_and_latch(self):
        """accept_override still clears counter and latch (no block to release)."""
        guard = MemoryDisciplineGuard()

        for i in range(30):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            guard.check_pre(ctx)

        ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
        result = guard.accept_override("Task doesn't need memory yet", ctx)
        assert result is True
        assert guard._calls_since_memory == 0
        assert guard._escalated is False


class TestKnowledgeSkillInjectAtThreshold:
    """KnowledgeSkillGuard escalates at 80 calls with an inject, not a block."""

    def test_heavy_mark_is_inject_and_lands_once(self):
        """The 80-call mark injects once; the 30-call rhythm resumes after it."""
        guard = KnowledgeSkillGuard()

        # Drive BLOCK_THRESHOLD-1 executed calls: counter reaches 79.
        for i in range(guard.BLOCK_THRESHOLD - 1):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            _adv_ks(guard, ctx)
        assert guard._calls_since_knowledge == guard.BLOCK_THRESHOLD - 1

        # Next call: projection = 79+1 = 80 >= 80 → escalated inject.
        ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "inject"
        assert verdict.reason == "no_knowledge_load_80_calls"

        # Execute it: 80. Calls 81-89 must stay silent (the projection still
        # reads >= 80, so only the latch keeps it quiet).
        guard.check_post(ctx)
        assert guard._calls_since_knowledge == 80
        for i in range(9):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = _adv_ks(guard, ctx)
            assert verdict is None, f"Call {81+i} must not re-fire the heavy mark"

        # Call 90 (80 executed + 10) is a multiple of 30 → ordinary rhythm again.
        ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
        verdict = guard.check_pre(ctx)
        assert verdict is not None and verdict.action == "inject"
        assert verdict.reason == "no_knowledge_load_recently"

    def test_never_blocks(self):
        """No call count ever produces a block from this guard."""
        guard = KnowledgeSkillGuard()
        for i in range(1, 161):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = _adv_ks(guard, ctx)
            assert verdict is None or verdict.action == "inject", (
                f"Call {i} produced {getattr(verdict, 'action', None)}")

    def test_knowledge_load_rearms_the_heavy_mark(self):
        """A research load resets counter AND re-arms the latch."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.BLOCK_THRESHOLD - 1):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            _adv_ks(guard, ctx)
        guard.check_pre(GuardContext(tool_name="shell", tool_args={"command": "ls"}))
        assert guard._escalated is True

        guard.check_pre(GuardContext(tool_name="web_fetch",
                                    tool_args={"url": "http://x"}))
        assert guard._calls_since_knowledge == 0
        assert guard._escalated is False

    def test_load_knowledge_resets(self):
        """A knowledge/skill load still zeroes the counter (unchanged)."""
        guard = KnowledgeSkillGuard()
        for i in range(10):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            _adv_ks(guard, ctx)
        assert guard._calls_since_knowledge == 10
        guard.check_pre(GuardContext(tool_name="load_knowledge",
                                    tool_args={"name": "know-x"}))
        assert guard._calls_since_knowledge == 0


class TestInjectRhythmUnchanged:
    """The pre-existing inject cadence must be bit-for-bit unchanged."""

    def test_knowledge_skill_inject_timing(self):
        """Inject should still trigger every 30 calls (30 and 60)."""
        guard = KnowledgeSkillGuard()
        inject_at = []
        for i in range(1, 61):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
            if verdict and verdict.action == "inject":
                inject_at.append(i)
            guard.check_post(ctx)  # persist the executed call's increment

        assert inject_at == [30, 60], f"Inject should trigger at 30/60, got {inject_at}"

    def test_memory_discipline_inject_timing(self):
        """Inject still triggers every 10 calls; 30 adds the escalated mark."""
        guard = MemoryDisciplineGuard()

        inject_at = []
        escalated_at = []
        for i in range(1, 31):
            ctx = GuardContext(tool_name="shell", tool_args={"command": "ls"})
            verdict = guard.check_pre(ctx)
            if verdict and verdict.action == "inject":
                if verdict.reason == "no_memory_ops_30_calls":
                    escalated_at.append(i)
                else:
                    inject_at.append(i)

        assert inject_at == [10, 20], f"Ordinary injects at 10/20, got {inject_at}"
        assert escalated_at == [30], f"Escalated mark at 30, got {escalated_at}"
