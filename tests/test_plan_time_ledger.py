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

"""Tests for the plan time ledger: per-step estimates, measured actuals, and
the running comparison against the injected wall-clock budget."""

import re

import pytest

from flagscale_agent.react.plan import (
    DEFAULT_RESERVE_FRACTION,
    TaskPlan,
    _coerce_sec,
    _fmt_sec,
    is_fixed_step,
)
from flagscale_agent.react.tools.plan_create import PlanCreateTool


@pytest.fixture()
def plan(tmp_path):
    return TaskPlan(str(tmp_path / "plans"))


def _budgeted(plan, budget=900.0):
    plan.set_budget_sec(budget)
    return plan


class TestCoerceSec:
    @pytest.mark.parametrize("value", [0, -5, "abc", None, False, "", []])
    def test_non_positive_or_unparseable_is_no_estimate(self, value):
        assert _coerce_sec(value) is None

    @pytest.mark.parametrize("value,expected", [(30, 30.0), ("45", 45.0), (0.5, 0.5)])
    def test_positive_numbers_survive(self, value, expected):
        assert _coerce_sec(value) == expected

    def test_zero_is_not_treated_as_an_estimate(self):
        # A 0 estimate would silently satisfy any budget check the agent never
        # actually performed, so it must collapse to "no estimate given".
        assert _coerce_sec(0) is None


class TestFmtSec:
    @pytest.mark.parametrize(
        "sec,expected",
        [(45, "45s"), (60, "1m"), (200, "3m20s"), (3900, "1h05m"), (None, "?")],
    )
    def test_readable_durations(self, sec, expected):
        assert _fmt_sec(sec) == expected


class TestStepEstimates:
    def test_estimate_is_persisted_on_string_and_dict_steps(self, plan):
        p = plan.create("t", [
            "plain step",
            {"title": "timed", "est_sec": 120},
        ])
        assert p["steps"][0]["est_sec"] is None
        assert p["steps"][1]["est_sec"] == 120.0

    def test_add_steps_also_accepts_estimates(self, plan):
        plan.create("t", ["a"])
        p = plan.add_steps([{"title": "b", "est_sec": 60}])
        added = [s for s in p["steps"] if s["title"] == "b"][0]
        assert added["est_sec"] == 60.0
        assert added["actual_sec"] is None


class TestActualRecording:
    def test_doing_then_done_records_actual_sec(self, plan):
        plan.create("t", [{"title": "a", "est_sec": 30}])
        plan.update_step(1, "doing")
        assert plan.get_active()["steps"][0]["started_at"] > 0
        plan.update_step(1, "done")
        step = plan.get_active()["steps"][0]
        assert step["ended_at"] > 0
        assert step["actual_sec"] is not None
        assert step["actual_sec"] >= 0.0

    def test_single_shot_close_records_actual_as_unknown(self, plan):
        plan.create("t", [{"title": "a", "est_sec": 30}])
        # closed without ever passing through "doing": this step's own cost was
        # never measured, so it must be recorded as UNKNOWN rather than charged
        # the whole plan's elapsed time (which inflated a batch close by the
        # step count and made the actual total a fabricated baseline).
        plan.update_step(1, "done")
        step = plan.get_active()["steps"][0]
        assert step["actual_sec"] is None
        assert step.get("actual_unknown") is True

    def test_batch_close_does_not_inflate_actual_total(self, plan):
        plan.create("t", [{"title": "a"}, {"title": "b"}, {"title": "c"}])
        for i in (1, 2, 3):
            plan.update_step(i, "done")
        steps = plan.get_active()["steps"]
        assert all(s["actual_sec"] is None for s in steps)
        assert sum(s.get("actual_sec") or 0.0 for s in steps) == 0.0

    def test_actual_is_frozen_on_first_close(self, plan):
        plan.create("t", [{"title": "a", "est_sec": 30}])
        plan.update_step(1, "doing")
        plan.update_step(1, "done")
        first = plan.get_active()["steps"][0]["actual_sec"]
        plan.update_step(1, "done", notes="re-confirmed")
        assert plan.get_active()["steps"][0]["actual_sec"] == first


class TestNoFabricatedDeadline:
    def test_without_budget_no_elapsed_or_remaining_is_rendered(self, plan):
        plan.create("t", [{"title": "a", "est_sec": 60}])
        out = plan.summary()
        assert "已用" not in out
        assert "剩余" not in out
        # the estimate the agent committed to IS still shown
        assert "预计合计" in out

    def test_without_budget_and_without_estimates_ledger_is_silent(self, plan):
        plan.create("t", ["a"])
        assert "时间账" not in plan.summary()

    def test_non_positive_budget_clears_the_ledger(self, plan):
        _budgeted(plan)
        plan.set_budget_sec(0)
        plan.create("t", [{"title": "a", "est_sec": 60}])
        assert "已用" not in plan.summary()


class TestLedgerRendering:
    def test_fitting_plan_reports_slack(self, plan):
        _budgeted(plan, 900.0)
        plan.create("t", [{"title": "a", "est_sec": 300}])
        out = plan.summary()
        assert "计划余量" in out
        assert "超预算" not in out

    def test_overrun_is_flagged_at_plan_time(self, plan):
        _budgeted(plan, 900.0)
        plan.create("t", [{"title": "expensive", "est_sec": 1100}])
        out = plan.summary()
        assert "超预算" in out

    def test_drift_warning_when_remaining_steps_outrun_remaining_time(self, plan):
        _budgeted(plan, 900.0)
        plan.create("t", [{"title": "a", "est_sec": 1100}])
        assert "漂移" in plan.summary()

    def test_reserve_is_held_back_from_step_allocation(self, plan):
        _budgeted(plan, 1000.0)
        plan.create("t", [{"title": "a", "est_sec": 100}])
        expected = round(1000.0 * DEFAULT_RESERVE_FRACTION, 1)
        assert plan.get_active()["time_budget"]["reserve_sec"] == expected

    def test_context_for_prompt_carries_the_ledger(self, plan):
        _budgeted(plan, 900.0)
        plan.create("t", [{"title": "a", "est_sec": 300}])
        ctx = plan.context_for_prompt()
        assert "时间账" in ctx

    def test_unestimated_agent_step_is_nagged(self, plan):
        _budgeted(plan, 900.0)
        plan.create("t", [{"title": "unpriced"}])
        assert "没有估时" in plan.summary()


class TestPerStepRendering:
    def test_estimate_and_actual_show_side_by_side(self, plan):
        _budgeted(plan, 900.0)
        plan.create("t", [{"title": "a", "est_sec": 30}])
        plan.update_step(1, "doing")
        plan.update_step(1, "done")
        out = plan.summary()
        assert "估30s" in out
        assert "实" in out


class TestPlanCreateFrameAccounting:
    def _tool(self, plan):
        return PlanCreateTool(plan, "session")

    def test_overrun_surfaces_at_creation(self, plan):
        _budgeted(plan, 900.0)
        out = self._tool(plan).execute(
            title="t",
            steps=[
                {"title": "reverse layout", "est_sec": 1100},
                {"title": "write output", "est_sec": 400},
            ],
        )
        assert "TIME BUDGET" in out
        assert "DOES NOT FIT" in out

    def test_fitting_plan_reports_slack_not_failure(self, plan):
        _budgeted(plan, 900.0)
        out = self._tool(plan).execute(
            title="t", steps=[{"title": "a", "est_sec": 100}],
        )
        assert "TIME BUDGET" in out
        assert "DOES NOT FIT" not in out

    def test_no_budget_injected_keeps_the_old_behaviour(self, plan):
        out = self._tool(plan).execute(
            title="t", steps=[{"title": "a", "est_sec": 100}],
        )
        assert "TIME BUDGET" not in out

    def test_all_unestimated_is_called_out(self, plan):
        _budgeted(plan, 900.0)
        out = self._tool(plan).execute(title="t", steps=[{"title": "a"}])
        assert "NONE of your steps carries an est_sec" in out

    def test_harness_fixed_steps_are_not_counted_as_unestimated(self, plan):
        _budgeted(plan, 900.0)
        out = self._tool(plan).execute(
            title="t", steps=[{"title": "a", "est_sec": 100}],
        )
        assert "0 unestimated" in out


class TestLegacyPlanCompatibility:
    """A plan written before this feature (no est_sec / time_budget keys) must
    still render a live ledger and still accept updates — a missing key must
    never silently switch time awareness off."""

    def _legacy(self, plan, budget=900.0, created=1700000000.0):
        if budget is not None:
            plan.set_budget_sec(budget)
        legacy = {
            "id": "plan_legacy01",
            "title": "legacy",
            "status": "active",
            "created": created,
            "updated": created,
            "session_id": "s",
            "steps": [
                {"id": 1, "title": "old a", "status": "pending", "notes": "",
                 "depends_on": [], "acceptance": [], "verification": []},
            ],
            "thinking": "",
            "oracle": {},
        }
        plan._save(legacy)
        plan._set_active("plan_legacy01")
        return plan

    def test_legacy_plan_still_renders_a_ledger(self, plan):
        self._legacy(plan, budget=900.0, created=__import__("time").time())
        out = plan.summary()
        assert "时间账" in out, "missing time_budget key must not disable the ledger"

    def test_legacy_plan_updates_without_keyerror(self, plan):
        self._legacy(plan, budget=900.0, created=__import__("time").time())
        plan.update_step(1, "doing")
        plan.update_step(1, "done")
        step = plan.get_active()["steps"][0]
        assert step["actual_sec"] is not None

    def test_resumed_legacy_plan_does_not_print_absurd_percentage(self, plan):
        # A plan written long ago and resumed NOW: the old session's origin must
        # not be used as this session's elapsed origin. Because elapsed is
        # anchored to this session, the absurd five-digit percentage cannot
        # appear (previously it printed 已用 25840h (10336123%)).
        self._legacy(plan, budget=900.0, created=1700000000.0)
        out = plan.summary()
        assert "%" in out
        assert "25840h" not in out, out
        assert "10336123" not in out, out
        # and the repairable state is not reported as unrepairable
        assert "预算已耗尽" not in out, out

    def test_genuinely_exhausted_budget_says_so(self, plan):
        # The legitimate exhausted case: the wall really is spent WITHIN this
        # session (a wall shorter than the time already spent in it).
        import time as _t
        plan.set_budget_sec(0.001)
        plan.create("t", [{"title": "a", "est_sec": 60}])
        _t.sleep(0.05)
        out = plan.summary()
        assert "预算已耗尽" in out, out
        assert "只做收尾" in out, out

    def test_legacy_plan_without_budget_stays_silent_about_time(self, plan):
        self._legacy(plan, budget=None)
        out = plan.summary()
        assert "已用" not in out


class TestNonFiniteInputs:
    """nan/inf parse successfully as floats but are not durations. They used to
    survive `sec <= 0` and then raise inside every render, taking the whole
    ledger (and every plan tool that renders it) down with them."""

    def test_nan_and_inf_are_not_estimates(self):
        import math
        for value in ("nan", float("nan"), "inf", float("inf"), "-inf"):
            assert _coerce_sec(value) is None, value

    def test_nan_and_inf_budget_is_rejected_not_stored(self, plan):
        for value in ("nan", float("nan"), float("inf")):
            plan.set_budget_sec(value)
            assert plan.budget_sec() is None, value

    def test_inf_budget_from_env_does_not_crash_render(self, plan, monkeypatch):
        # The real input path: agent.py parses the env string then admits it
        # when `budget > 0` — true for inf.
        monkeypatch.setenv("FLAGSCALE_AGENT_TIME_BUDGET_SEC", "inf")
        plan.set_budget_sec(float("inf"))
        plan.create("t", [{"title": "a", "est_sec": 60}])
        out = plan.summary()  # must not raise
        assert isinstance(out, str)

    def test_fmt_sec_tolerates_non_finite(self):
        import math
        assert _fmt_sec(float("nan")) == "?"
        assert _fmt_sec(float("inf")) == "?"
        assert _fmt_sec(None) == "?"


class TestResumedPlanAnchorsToThisSession:
    """A plan resumed in a later session carries a `started` from the earlier
    one. Measuring elapsed from it reported the whole gap as spent and told a
    full-budget run to stop — the exact opposite of the feature's purpose."""

    def _legacy_started(self, plan, budget, age_sec):
        import time as _t
        plan.set_budget_sec(budget)
        old = _t.time() - age_sec
        plan._save({
            "id": "plan_resumed01", "title": "resumed", "status": "active",
            "created": old, "updated": old, "session_id": "s",
            "steps": [{"id": 1, "title": "a", "status": "pending", "notes": "",
                       "depends_on": [], "acceptance": [], "verification": [],
                       "est_sec": 100}],
            "thinking": "", "oracle": {},
            "time_budget": {"budget_sec": budget, "started": old,
                            "reserve_sec": round(budget * 0.15, 1)},
        })
        plan._set_active("plan_resumed01")
        return plan

    def test_resumed_plan_does_not_report_exhausted_budget(self, plan):
        self._legacy_started(plan, 720.0, 86400.0)
        out = plan.summary()
        assert "预算已耗尽" not in out, out
        assert "只做收尾" not in out, out

    def test_resumed_plan_reports_fresh_elapsed(self, plan):
        self._legacy_started(plan, 720.0, 86400.0)
        out = plan.summary()
        # elapsed is measured from THIS session, so almost all of the wall is
        # still available and the ledger must say so
        assert "24h" not in out, out
        assert "剩余" in out


class TestOneFixedStepPredicate:
    """The per-turn ledger and the plan_create frame-time report must agree
    about which steps are harness-owned; two divergent predicates meant the
    same step was exempt on one surface and 'unestimated' on the other."""

    def test_predicate_accepts_both_template_phrasings(self):
        assert is_fixed_step({"title": "Review (FlagScale-Agent Plan fixed template)"})
        assert is_fixed_step({"title": "RSI (x — fixed step, final)"})
        assert not is_fixed_step({"title": "reverse the layout"})

    def test_both_surfaces_agree_on_a_fixed_step(self, plan):
        _budgeted(plan, 900.0)
        tool_out = PlanCreateTool(plan, "session").execute(
            title="t",
            steps=[{"title": "a", "est_sec": 100},
                   {"title": "Review (x fixed step)"}],
        )
        assert "0 unestimated" in tool_out, tool_out
        assert "没有估时" not in plan.summary(), plan.summary()


class TestPublicBudgetAccessor:
    def test_budget_is_readable_without_private_attribute(self, plan):
        assert plan.budget_sec() is None
        _budgeted(plan, 900.0)
        assert plan.budget_sec() == 900.0


# plan_create now REQUIRES an oracle declaration: a plan created without one
# arms none of the verification machinery, which is how the measured failure
# (oracle: {} persisted, every downstream gate silently inert) happened. The
# tests in this module exercise OTHER behaviour and merely pass through the
# creation call, so they get a minimal valid declaration injected here. The
# contract itself -- including the refusal -- is covered in
# tests/test_oracle_at_frame.py.
@pytest.fixture(autouse=True)
def _default_oracle_for_creation(monkeypatch):
    from flagscale_agent.react.tools.plan_create import PlanCreateTool

    original = PlanCreateTool.execute

    def execute(self, **kwargs):
        if kwargs.get("oracle") in (None, ""):
            kwargs["oracle"] = {
                "oracle": "test decider",
                "premise_source": "provided",
                "first_exercise": "true",
            }
        return original(self, **kwargs)

    monkeypatch.setattr(PlanCreateTool, "execute", execute)


# Same shim, second required declaration: plan_create now also REFUSES to create
# a plan without a `calibration` (instrument 1). The tests here exercise the time
# ledger and merely pass through creation, so they get a minimal valid
# calibration injected. The refusal contract itself is covered in
# tests/test_instruments_four_levers.py, which drives the tool directly.
@pytest.fixture(autouse=True)
def _default_calibration_for_creation(monkeypatch):
    from flagscale_agent.react.tools.plan_create import PlanCreateTool
    from tests._calibration_fixture import DEFAULT_CALIBRATION

    original = PlanCreateTool.execute

    def execute(self, **kwargs):
        kwargs.setdefault("calibration", dict(DEFAULT_CALIBRATION))
        return original(self, **kwargs)

    monkeypatch.setattr(PlanCreateTool, "execute", execute)
