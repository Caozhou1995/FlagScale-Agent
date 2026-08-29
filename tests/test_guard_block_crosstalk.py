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

"""Regression tests for guard block crosstalk / deadlock (Bug B).

Root cause: GuardRegistry.check_pre returned on the FIRST block regardless of
overridability, and fed the single global override_reason to whichever guard
happened to win. A low-priority OVERRIDABLE block (compile_redirect) surfaced
ahead of a high-priority NON-OVERRIDABLE block (knowledge_skill single-shot
gate); the agent overrode the soft block, retried, then hit the hard wall it
could not override — and kept sending the (wrong-guard) override reason forever.

Fix: collect-all + severity-rank + owner-scoped override. See GuardRegistry._resolve.
"""

from flagscale_agent.react.guard import (
    Guard, GuardVerdict, GuardContext, GuardRegistry,
)


class _Overridable(Guard):
    name = "soft"
    priority = 8  # low number = surfaced first under the OLD first-wins rule

    def check_pre(self, ctx):
        return GuardVerdict.block("[soft] transient?", reason="soft", category="soft")


class _NonOverridable(Guard):
    name = "hard"
    priority = 85  # higher number = would lose the priority race, but is HARDER

    def check_pre(self, ctx):
        return GuardVerdict.block("[hard] single-shot gate", reason="hard",
                                  category="hard", overridable=False)


class _Escalate(Guard):
    name = "stop"
    priority = 50

    def check_pre(self, ctx):
        return GuardVerdict.escalate("[stop] dead end", reason="stop", category="stop")


def _ctx(override=None):
    return GuardContext(tool_name="shell", tool_args={"command": "x"},
                        override_reason=override)


class TestSeverityRanking:
    def test_non_overridable_surfaces_before_overridable(self):
        # Even though the overridable guard has the lower priority number (would
        # win first-wins), the NON-overridable block must surface first.
        reg = GuardRegistry()
        reg.register(_Overridable())
        reg.register(_NonOverridable())
        v = reg.check_pre(_ctx())
        assert v is not None and v.action == "block"
        assert v.guard_name == "hard", "hard (non-overridable) must surface first"
        assert v.overridable is False

    def test_escalate_surfaces_before_any_block(self):
        reg = GuardRegistry()
        reg.register(_Overridable())
        reg.register(_NonOverridable())
        reg.register(_Escalate())
        v = reg.check_pre(_ctx())
        assert v is not None and v.action == "escalate"
        assert v.guard_name == "stop"


class TestOwnerScopedOverride:
    def test_reason_for_soft_does_not_release_hard(self):
        # A reason written while the soft block was showing must NOT release the
        # hard block once it surfaces.
        reg = GuardRegistry()
        reg.register(_Overridable())
        reg.register(_NonOverridable())
        v = reg.check_pre(_ctx(override="this reason was meant for the soft guard"))
        # hard is non-overridable -> still blocks, reason cannot touch it
        assert v is not None and v.action == "block" and v.guard_name == "hard"

    def test_overriding_soft_falls_through_to_hard(self):
        # With only soft+hard, overriding soft should reveal hard the same turn
        # (collect-all), not silently let the call through.
        reg = GuardRegistry()
        reg.register(_Overridable())
        reg.register(_NonOverridable())
        v = reg.check_pre(_ctx(override="release the soft block please"))
        assert v is not None and v.guard_name == "hard"

    def test_two_overridable_reason_scoped_to_surfaced_owner(self):
        # Bug C regression test: an override reason releases ONLY the guard that
        # was last surfaced — never a guard that fired for the first time this
        # turn (the phantom override). Here A and B both always block; the reason
        # can only release whichever was shown last.
        class A(Guard):
            name = "A"; priority = 10
            def check_pre(self, ctx):
                return GuardVerdict.block("[A]", reason="a", category="a")
            def accept_override(self, reason, ctx):
                return "for-A" in reason
        class B(Guard):
            name = "B"; priority = 20
            def check_pre(self, ctx):
                return GuardVerdict.block("[B]", reason="b", category="b")
            def accept_override(self, reason, ctx):
                return "for-B" in reason
        reg = GuardRegistry()
        reg.register(A()); reg.register(B())
        
        # Surface A first (prio 10 sorts first)
        v_surf = reg.check_pre(_ctx(override=""))
        assert v_surf is not None and v_surf.guard_name == "A"
        
        # Reason for A: releases A; next hardest is B
        v = reg.check_pre(_ctx(override="this is for-A only"))
        assert v is not None and v.guard_name == "B", \
            "A released, B surfaces (was NOT released by A's reason)"
        
        # Reason for B (but last-surfaced is still A from earlier): doesn't release B
        v2 = reg.check_pre(_ctx(override="this is for-B only"))
        assert v2 is not None and v2.guard_name == "A", \
            "B's reason doesn't release B because B wasn't last-surfaced"


class TestCollectAllNoLeak:
    def test_all_blocks_gone_only_when_each_released(self):
        # Bug C regression test: with multiple guards that always block, releasing
        # one does NOT silently release others — you see the next one surface.
        class A(Guard):
            name = "A"; priority = 10
            def check_pre(self, ctx):
                return GuardVerdict.block("[A]", reason="a", category="a")
        class B(Guard):
            name = "B"; priority = 20
            def check_pre(self, ctx):
                return GuardVerdict.block("[B]", reason="b", category="b")
        reg = GuardRegistry()
        reg.register(A()); reg.register(B())
        
        # Surface A
        v_surf = reg.check_pre(_ctx(override=""))
        assert v_surf is not None and v_surf.guard_name == "A"
        
        # Release A with a generic reason; B surfaces next
        v1 = reg.check_pre(_ctx(override="release everything now"))
        assert v1 is not None and v1.guard_name == "B", \
            "A released; B surfaces (was NOT silently released by A's reason)"


class TestCompcertSequenceNoDeadlock:
    """End-to-end: the real guards from the compcert trajectory no longer deadlock.

    Original failure: `make --version | head` was mis-flagged by CompileRedirectGuard
    (Bug A), the agent overrode it, retries counted toward the single-shot early gate,
    which then blocked NON-overridably — and the agent kept re-sending the
    compile_redirect override, which could never release the knowledge_skill gate.
    """

    def _reg(self):
        from flagscale_agent.react.guard.compile_redirect import CompileRedirectGuard
        from flagscale_agent.react.guard.knowledge_skill import KnowledgeSkillGuard
        reg = GuardRegistry()
        reg.register(CompileRedirectGuard())
        reg.register(KnowledgeSkillGuard(single_shot=True))
        return reg

    def test_version_checks_not_blocked_by_compile_redirect(self):
        # Bug A: transient version/probe commands must pass compile_redirect.
        reg = self._reg()
        for cmd in ["make --version 2>&1 | head -1", "gcc --version",
                    "which coqc menhir", "opam --version"]:
            ctx = GuardContext(tool_name="shell", tool_args={"command": cmd})
            v = reg.check_pre(ctx)
            # If anything blocks, it must NOT be compile_redirect mis-flagging.
            if v is not None and v.action == "block":
                assert v.guard_name != "compile_redirect", \
                    f"compile_redirect must not block transient cmd: {cmd}"

    def test_single_shot_gate_is_the_only_hard_wall(self):
        # Drive enough non-meta calls to trip the single-shot early gate; when it
        # fires it is non-overridable, and it is presented as the knowledge_skill
        # guard — never masked by a stale compile_redirect block.
        reg = self._reg()
        blocked_by = None
        for i in range(5):
            ctx = GuardContext(tool_name="shell",
                               tool_args={"command": f"gcc --version # {i}"})
            v = reg.check_pre(ctx)
            if v is not None and v.action == "block" and not v.overridable:
                blocked_by = v.guard_name
                break
            # simulate the call executing (post-check advances the counter)
            ctx.tool_result = "ok"
            reg.check_post(ctx)
        assert blocked_by == "knowledge_skill", \
            "the single-shot early gate should be the surfaced non-overridable block"

    def test_compile_override_never_releases_knowledge_gate(self):
        # The core anti-deadlock invariant: a compile_redirect-flavored override
        # reason must never release the knowledge_skill non-overridable gate.
        from flagscale_agent.react.guard.knowledge_skill import KnowledgeSkillGuard
        reg = GuardRegistry()
        ks = KnowledgeSkillGuard(single_shot=True)
        reg.register(ks)
        # fast-forward to just under the threshold
        ks._single_shot_call_count = ks.SINGLE_SHOT_EARLY_THRESHOLD - 1
        ctx = GuardContext(tool_name="shell",
                           tool_args={"command": "gcc --version"},
                           override_reason="this build already writes its own log file")
        v = reg.check_pre(ctx)
        assert v is not None and v.action == "block" and v.overridable is False
        assert v.guard_name == "knowledge_skill"


class TestTieBreakAndPrecedence:
    """Within one severity tier, lower priority number surfaces; severity always
    dominates priority across tiers. Order must not depend on registration order."""

    def _mk(self, nm, prio, action="block", overridable=True):
        class _G(Guard):
            name = nm; priority = prio
            def check_pre(self, ctx):
                if action == "escalate":
                    return GuardVerdict.escalate(nm, reason="r", category=nm)
                return GuardVerdict.block(nm, reason="r", category=nm,
                                          overridable=overridable)
        return _G()

    def test_same_severity_lower_priority_number_wins(self):
        for order in ([90, 20], [20, 90]):
            reg = GuardRegistry()
            for p in order:
                reg.register(self._mk(f"h{p}", p, overridable=False))
            v = reg.check_pre(_ctx())
            assert v.guard_name == "h20", f"prio20 must surface, insert order {order}"

    def test_escalate_dominates_lower_priority_block(self):
        # escalate at the WORST priority number still beats a non-overridable block
        # at the best priority number — severity ranks above priority.
        reg = GuardRegistry()
        reg.register(self._mk("esc", 99, action="escalate"))
        reg.register(self._mk("blk", 20, overridable=False))
        v = reg.check_pre(_ctx())
        assert v.action == "escalate" and v.guard_name == "esc"

    def test_owner_tag_names_the_surfaced_guard(self):
        reg = GuardRegistry()
        reg.register(self._mk("alpha", 10, overridable=False))
        v = reg.check_pre(_ctx())
        assert "[blocked by guard: alpha]" in v.message
