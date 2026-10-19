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

"""Regression tests: an outranked 90% hard wall must still reach the agent.

Defect (observed in a live 1800s run): TimeBudgetGuard.check_pre marked the 90
threshold fired while *building* its block. When a rank-1 (non-overridable)
block from another guard won the same resolve, TimeBudgetGuard's block never
reached the agent, yet 90 stayed spent -- so the only hard convergence point of
the turn was silently disabled. The agent then ran 6m34s past the 80% wrap-up
window with no 90% gate ever surfacing.

Two mechanisms now guarantee delivery, and the tests below pin BOTH:
  1. `Guard.block_not_surfaced()` hands the threshold back when a block is
     never delivered (kept for any future dropping path).
  2. M14: the registry MERGES every block that fires at a decision point into
     the one message the agent receives. The 90% gate is therefore delivered
     even when a harder block leads the same resolve — the invariant the agent
     depends on is "the warning reaches it", not "a second round trip re-sends
     it".
"""
from flagscale_agent.react.guard import (
    Guard,
    GuardContext,
    GuardRegistry,
    GuardVerdict,
)
from flagscale_agent.react.guard.time_budget import TimeBudgetGuard


class _Stats:
    def __init__(self, pct):
        self.pct = pct

    def __call__(self):
        budget = 1800.0
        return {
            "elapsed": budget * self.pct / 100.0,
            "budget": budget,
            "remaining": budget - budget * self.pct / 100.0,
            "pct": self.pct,
        }


class _HardBlock(Guard):
    """A rank-1 (non-overridable) blocker, like VerificationGuard's hard walls."""

    name = "hard_block_for_test"

    def __init__(self):
        self.armed = True

    def check_pre(self, ctx):
        if not ctx.tool_name or not self.armed:
            return None
        return GuardVerdict.block(
            "hard wall", reason="hard_wall", category="verification",
            overridable=False,
        )


def _ctx():
    ctx = GuardContext()
    # read_file: the observation class this gate is allowed to stop
    ctx.tool_name = "read_file"
    ctx.tool_args = {}
    ctx.turn_count = 1
    return ctx


def test_90_gate_reaches_agent_even_when_outranked_by_hard_block():
    """The regression itself: the 90% warning must REACH the agent.

    Under M14 the harder wall still leads, and the 90% gate is merged into the
    same message — the agent learns both at once instead of one round trip apart.
    """
    reg = GuardRegistry()
    tb = TimeBudgetGuard(stats_fn=_Stats(91.7))
    hard = _HardBlock()
    reg.register(tb)
    reg.register(hard)

    first = reg.check_pre(_ctx())
    # The harder wall leads (severity ranking is unchanged by merging).
    assert first is not None
    assert first.guard_name == "hard_block_for_test"
    assert first.reason == "hard_wall"
    # ...and the 90% gate is in the SAME message, so it is genuinely delivered.
    assert "CRITICAL CHECKPOINT" in first.message, \
        "90% gate must reach the agent in the merged message, not a later turn"
    assert "time_budget_90pct_block" in first.message


def test_block_not_surfaced_returns_thresholds():
    """Unit-level: the hook must put the pending thresholds back."""
    tb = TimeBudgetGuard(stats_fn=_Stats(91.7))
    verdict = tb.check_pre(_ctx())
    assert verdict is not None
    assert 90 in tb._fired
    tb.block_not_surfaced()
    assert 90 not in tb._fired, "unseen threshold must be re-armable"
    # And it fires again now that it is re-armed.
    again = tb.check_pre(_ctx())
    assert again is not None
    assert "90pct" in again.reason


def test_reset_turn_clears_pending_set():
    tb = TimeBudgetGuard(stats_fn=_Stats(91.7))
    tb.check_pre(_ctx())
    assert tb._pending_block_fired
    tb.reset_turn()
    assert not tb._fired
    assert not tb._pending_block_fired
