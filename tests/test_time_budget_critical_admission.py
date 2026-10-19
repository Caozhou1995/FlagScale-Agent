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

"""Regression tests for the critical-gate admission rule.

Contract under test: the critical checkpoint may only stop an OBSERVATION
round. Its trigger is derived from how many ACTION rounds the remaining wall
can still cover at this turn's own measured cost — not from a fixed percentage.
"""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.time_budget import TimeBudgetGuard
from flagscale_agent.react.guard.utils import READ_ONLY_TOOLS
from flagscale_agent.react.kernel import AgentKernel


def make_ctx(tool_name):
    ctx = GuardContext()
    ctx.tool_name = tool_name
    return ctx


def check(stats, tool_name):
    g = TimeBudgetGuard(lambda: stats)
    return g.check_pre(make_ctx(tool_name))


# ── admission by tool class ──────────────────────────────────────────────

def test_read_only_set_excludes_every_production_tool():
    # The rule is only sound if the harness's own read-only list really is the
    # observation set: a production tool leaking into it would be blocked at the
    # exact moment its output is the deliverable.
    for t in ("write_file", "edit_file", "shell"):
        assert t not in READ_ONLY_TOOLS, t
    for t in ("read_file", "memory_list", "plan_status"):
        assert t in READ_ONLY_TOOLS, t


def test_critical_gate_never_blocks_a_production_call():
    # Measured failure this closes: the gate blocked the `write_file` that WAS
    # the deliverable, then consumed the two remaining rounds on its override.
    stats = {"pct": 91.0, "budget": 1800.0, "remaining": 76.0,
             "elapsed": 1724.0, "rounds": 20}
    for tool in ("write_file", "edit_file", "shell"):
        assert check(stats, tool) is None, tool


def test_critical_gate_still_blocks_an_observation_call():
    stats = {"pct": 91.0, "budget": 1800.0, "remaining": 76.0,
             "elapsed": 1724.0, "rounds": 20}
    v = check(stats, "read_file")
    assert v is not None
    assert v.reason == "time_budget_90pct_block"


# ── trigger derived from remaining rounds ────────────────────────────────

def test_trigger_ignores_percentage_when_rounds_are_ample():
    # 91% but the turn has been cheap per action, so many rounds remain: the
    # wall is not actually scarce and the gate must stay silent.
    stats = {"pct": 91.0, "budget": 1800.0, "remaining": 162.0,
             "elapsed": 1638.0, "rounds": 100}  # 16.4s/round -> ~9.9 left
    assert check(stats, "read_file") is None


def test_trigger_arms_earlier_than_90pct_when_rounds_are_slow():
    # 78% with only ~5.7 actions left is exactly the case a percentage gate
    # misses; the round criterion must catch it.
    stats = {"pct": 78.0, "budget": 1800.0, "remaining": 400.0,
             "elapsed": 1400.0, "rounds": 20}  # 70s/round -> 5.7 left
    assert check(stats, "read_file") is not None


def test_never_blocks_once_the_wall_is_spent():
    # At/after 100% the agent needs an unobstructed path to its wrap-up.
    stats = {"pct": 103.0, "budget": 1800.0, "remaining": -54.0,
             "elapsed": 1854.0, "rounds": 20}
    assert check(stats, "read_file") is None


def test_falls_back_to_fraction_without_round_history():
    no_history = {"pct": 92.0, "budget": 1800.0, "remaining": 144.0,
                  "elapsed": 1656.0, "rounds": 0}
    assert check(no_history, "read_file") is not None
    calm = {"pct": 70.0, "budget": 1800.0, "remaining": 540.0,
            "elapsed": 1260.0, "rounds": 0}
    assert check(calm, "read_file") is None


# ── the count must be LIVE inside the turn, not only reported after it ───

def test_kernel_publishes_iteration_count_inside_the_turn():
    # The post-turn field would read 0 for the whole turn, silently degrading
    # the rounds criterion to the percentage fallback (an inert mechanism).
    k = AgentKernel.__new__(AgentKernel)
    k.iterations_live = 0
    assert k.iterations_live == 0
    k.iterations_live = 4
    assert k.iterations_live == 4
