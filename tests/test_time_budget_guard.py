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

"""Unit tests for TimeBudgetGuard — the agent-side wall-clock awareness guard,
and the prompt-side urgency wording that pairs with it."""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.time_budget import TimeBudgetGuard, _fmt


def make_ctx(tool_name="shell"):
    ctx = GuardContext()
    ctx.tool_name = tool_name
    return ctx


class _Stats:
    """Mutable stats source: set .pct=None to simulate no injected wall."""

    def __init__(self):
        self.pct = 0.0

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


class TestTimeBudgetGuardSilence:
    def test_none_stats_is_silent(self):
        s = _Stats()
        s.pct = None
        g = TimeBudgetGuard(stats_fn=s)
        assert g.check_post(make_ctx()) is None

    def test_stats_fn_none_callable_is_silent(self):
        g = TimeBudgetGuard(stats_fn=None)
        assert g.check_post(make_ctx()) is None

    def test_stats_fn_exception_is_silent(self):
        def boom():
            raise RuntimeError("stats failed")

        g = TimeBudgetGuard(stats_fn=boom)
        # Must swallow the error, never break tool execution.
        assert g.check_post(make_ctx()) is None

    def test_no_tool_name_is_silent(self):
        s = _Stats()
        s.pct = 95.0
        g = TimeBudgetGuard(stats_fn=s)
        assert g.check_post(make_ctx(tool_name="")) is None

    def test_below_first_threshold_is_silent(self):
        # First rung is now 25% — below it the guard stays silent.
        s = _Stats()
        s.pct = 20.0
        g = TimeBudgetGuard(stats_fn=s)
        assert g.check_post(make_ctx()) is None


class TestTimeBudgetGuardThresholds:
    def test_fires_each_threshold_once(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)

        # 25% is the new earliest rung: pacing / front-load guidance.
        s.pct = 30.0
        v = g.check_post(make_ctx())
        assert v is not None and v.action == "inject"
        assert v.reason == "time_budget_25pct"
        # front-load / pacing language, fired while the decision window is open
        assert "front-load" in v.message or "front load" in v.message
        assert "background=true" in v.message
        # Same threshold must not refire.
        assert g.check_post(make_ctx()) is None

        s.pct = 55.0
        v = g.check_post(make_ctx())
        assert v is not None and v.reason == "time_budget_50pct"
        # 50% is now a HEALTH CHECK, not a repeat of the pacing tip.
        assert "HEALTH CHECK" in v.message or "health check" in v.message.lower()
        # Same threshold must not refire.
        assert g.check_post(make_ctx()) is None

        s.pct = 78.0
        v = g.check_post(make_ctx())
        assert v is not None and v.reason == "time_budget_75pct"
        assert g.check_post(make_ctx()) is None

        s.pct = 92.0
        v = g.check_post(make_ctx())
        assert v is not None and v.reason == "time_budget_90pct"
        assert "CRITICAL" in v.message
        assert g.check_post(make_ctx()) is None

    def test_jump_past_multiple_fires_most_severe_only(self):
        # Jumping 30 -> 95 in one step should emit the 90% CRITICAL message,
        # not stack three separate advisories.
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 95.0
        v = g.check_post(make_ctx())
        assert v is not None and v.reason == "time_budget_90pct"
        # Lower thresholds are now considered spent for this turn.
        s.pct = 96.0
        assert g.check_post(make_ctx()) is None

    def test_over_100_does_not_break(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 130.0
        v = g.check_post(make_ctx())
        assert v is not None and v.reason == "time_budget_90pct"

    def test_reset_turn_rearms(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        s.pct = 60.0
        assert g.check_post(make_ctx()) is not None
        assert g.check_post(make_ctx()) is None
        g.reset_turn()
        assert g.check_post(make_ctx()) is not None

    def test_inject_only_never_blocks(self):
        s = _Stats()
        g = TimeBudgetGuard(stats_fn=s)
        for p in (55.0, 78.0, 92.0):
            s.pct = p
            v = g.check_post(make_ctx())
            assert v is None or v.action == "inject"


class TestFmt:
    def test_negative_clamps_to_zero(self):
        assert _fmt(-5) == "0m00s"

    def test_hours_format(self):
        assert _fmt(3720) == "1h02m"

    def test_minutes_format(self):
        assert _fmt(65) == "1m05s"
