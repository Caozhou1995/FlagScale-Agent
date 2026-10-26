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

"""Regression tests for the gate-liveness defects found in the 2026-10-26
write-compressor trial.

Three distinct ways an ORDER checkpoint could stop the very action it asks
for, each measured on a real trajectory:

1. PREREQUISITE DEADLOCK. `plan` blocks write_file ("call plan_create first")
   while the 25% rung blocked plan_create ("a write must have EXECUTED this
   turn"). Neither could ever run: 11 consecutive identical blocks, the agent
   wrote "deadlock... circular" in its own reasoning, 5% of the trial gone.

2. STARVED ESCAPE VALVE. The 25% rung is lower priority than the plan guard,
   so when both fired its verdict rode along as `also_failing` — shown to the
   agent, refused by it, but `accept_override` is only called for the
   SURFACED block, so `_denials` never grew and `_MAX_DENIALS` never fired.

3. A new user turn must still clear the counters (no leak across turns).
"""

from flagscale_agent.react.guard import GuardContext, GuardRegistry
from flagscale_agent.react.guard.first_e2e_run import FirstE2eRunGuard


def make_ctx(tool_name="shell", history=None):
    ctx = GuardContext()
    ctx.tool_name = tool_name
    if history is not None:
        ctx.recent_tool_history = history
    return ctx


class _Stats:
    def __init__(self, pct=0.0):
        self.pct = pct

    def __call__(self):
        if self.pct is None:
            return None
        budget = 1200.0
        elapsed = budget * self.pct / 100.0
        return {
            "elapsed": elapsed,
            "budget": budget,
            "remaining": budget - elapsed,
            "pct": self.pct,
        }


class TestPrerequisiteToolsNeverBlocked:
    """Defect 1: a gate may not block the call that is its own prerequisite."""

    def test_plan_create_passes_the_25pct_rung(self):
        g = FirstE2eRunGuard(stats_fn=_Stats(30.0))
        assert g.check_pre(make_ctx(tool_name="plan_create")) is None

    def test_plan_update_passes_the_75pct_rung(self):
        # The rung is crossed hard; only the tool identity exempts it.
        g = FirstE2eRunGuard(stats_fn=_Stats(80.0))
        assert g.check_pre(make_ctx(tool_name="plan_update")) is None

    def test_memory_write_passes_a_crossed_rung(self):
        g = FirstE2eRunGuard(stats_fn=_Stats(80.0))
        assert g.check_pre(make_ctx(tool_name="memory_write")) is None

    def test_a_real_work_tool_is_still_gated(self):
        # The exemption is narrow: the write the rung actually asks for must
        # still be demanded, otherwise the checkpoint is dead code.
        g = FirstE2eRunGuard(stats_fn=_Stats(30.0))
        v = g.check_pre(make_ctx(tool_name="read_file"))
        assert v is not None and v.action == "block"
        assert v.reason == "first_e2e_run_25pct"

    def test_write_tool_is_still_gated_before_it_lands(self):
        # F4 contract: the write itself IS the rung's answer, so a write/edit
        # call passes through the sub-75% rungs — but a NON-write call still
        # faces the same question, and the 75% rung is never released by a
        # write (a write does not answer a final-check question).
        g = FirstE2eRunGuard(stats_fn=_Stats(30.0))
        assert g.check_pre(make_ctx(tool_name="write_file")) is None
        v = g.check_pre(make_ctx(tool_name="read_file"))
        assert v is not None and v.reason == "first_e2e_run_25pct"

    def test_exemption_does_not_spend_the_rung(self):
        # Passing a prerequisite must not be read as "checkpoint answered":
        # the next real call still faces the same question.
        g = FirstE2eRunGuard(stats_fn=_Stats(30.0))
        assert g.check_pre(make_ctx(tool_name="plan_create")) is None
        v = g.check_pre(make_ctx(tool_name="read_file"))
        assert v is not None and v.reason == "first_e2e_run_25pct"


class TestMergedRefusalCountsTowardValve:
    """Defect 2: a merged (also_failing) block was delivered but never counted."""

    def _merged_verdict(self, thr=25):
        # pct must actually cross the rung being exercised: 75 needs >75.
        pct = {25: 30.0, 50: 55.0, 75: 80.0}[thr]
        g = FirstE2eRunGuard(stats_fn=_Stats(pct))
        v = g.check_pre(make_ctx(tool_name="read_file"))
        assert v is not None and v.reason == f"first_e2e_run_{thr}pct"
        return g, v

    def test_pending_thr_is_recovered_from_the_reason_slug(self):
        g, v = self._merged_verdict(25)
        assert g._pending_thr_for(v) == 25

    def test_pending_thr_is_none_for_an_unrelated_reason(self):
        g = FirstE2eRunGuard(stats_fn=_Stats(30.0))
        v = GuardContext()
        v.reason = "write_file_without_plan"
        assert g._pending_thr_for(v) is None

    def test_two_merged_refusals_exhaust_the_rung(self):
        # _MAX_DENIALS is 2: after two delivered-but-merged refusals the rung
        # must stop blocking and speak as a non-blocking demand instead.
        g = FirstE2eRunGuard(stats_fn=_Stats(30.0))
        _, v1 = self._merged_verdict(25)
        g.note_merged_refusal(v1)
        _, v2 = self._merged_verdict(25)
        g.note_merged_refusal(v2)
        v = g.check_pre(make_ctx(tool_name="read_file"))
        assert v is not None
        assert v.action == "inject", f"expected inject after 2 merged refusals, got {v.action}"
        assert v.reason == "first_e2e_run_25pct_exhausted"

    def test_one_merged_refusal_still_blocks(self):
        # One is not enough — the agent gets a second look.
        g = FirstE2eRunGuard(stats_fn=_Stats(30.0))
        _, v1 = self._merged_verdict(25)
        g.note_merged_refusal(v1)
        v = g.check_pre(make_ctx(tool_name="read_file"))
        assert v is not None and v.action == "block"

    def test_exhaustion_also_retires_lower_rungs(self):
        # The property that matters: once the HIGHEST rung exhausts, an agent
        # already past 75% must not be re-asked the 50%/25% questions — that is
        # pure noise burning the budget the valve is protecting.
        g = FirstE2eRunGuard(stats_fn=_Stats(80.0))
        _, v = self._merged_verdict(75)
        g.note_merged_refusal(v)
        _, v2 = self._merged_verdict(75)
        g.note_merged_refusal(v2)
        v3 = g.check_pre(make_ctx(tool_name="read_file"))
        assert v3 is not None
        assert v3.action == "inject", "75% must stop blocking after exhaustion"
        # And no lower rung wakes up behind it.
        assert g.check_pre(make_ctx(tool_name="read_file")) is None


class TestRegistryCallsTheHook:
    """The registry must INVOKE the hook — the guard-side counter alone would
    pass the unit tests above while the production path never calls it."""

    def test_merged_block_calls_note_merged_refusal(self):
        from flagscale_agent.react.guard import Guard, GuardVerdict

        calls = []
        blocks = []

        class _Winner(Guard):
            name = "winner"
            priority = 10

            def check_pre(self, ctx):
                return GuardVerdict.block(
                    message="winner blocks", reason="w",
                    category="test", overridable=True,
                )

        class _Loser(Guard):
            name = "loser"
            priority = 20

            def check_pre(self, ctx):
                return GuardVerdict.block(
                    message="loser blocks", reason="l",
                    category="test", overridable=True,
                )

            def note_merged_refusal(self, verdict):
                calls.append(verdict.reason)

        reg = GuardRegistry()
        reg.register(_Winner())
        reg.register(_Loser())
        v = reg.check_pre(GuardContext())
        assert v is not None and v.action == "block"
        assert calls == ["l"], f"merged block never reached the hook: {calls}"


class TestResetTurnClearsCounters:
    def test_new_turn_clears_denials(self):
        g = FirstE2eRunGuard(stats_fn=_Stats(30.0))
        v = g.check_pre(make_ctx(tool_name="read_file"))
        g.note_merged_refusal(v)
        assert g._denials.get(25, 0) >= 1
        g.reset_turn()
        assert g._denials == {}
        v2 = g.check_pre(make_ctx(tool_name="read_file"))
        assert v2 is not None and v2.action == "block"
