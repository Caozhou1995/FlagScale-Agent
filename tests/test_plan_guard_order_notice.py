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

"""PlanGuard.check_post — the plan-order notice after plan_create.

The failure this covers is structural, not careless: plan_create returns a plan
that the harness has already modified (Frame prepended, Review + RSI appended),
yet the agent reads that return as "my steps" and works only those, discovering
the fixed steps at the finish line. Frame then gets "closed" long after the
framing decision it was supposed to govern, and Review/RSI get back-filled.

Contract asserted here:
  1. a SUCCESSFUL plan_create yields exactly ONE inject naming the harness steps;
  2. the notice is inject-only (never block/escalate) — it must not gate work;
  3. it does not repeat on a second plan_create (per-run one-shot);
  4. a FAILED plan_create yields no notice (no plan was created to reorder);
  5. a non-plan tool yields no notice;
  6. the settled no-plan obligation behaviour is unchanged by the notice.
"""

from unittest.mock import MagicMock

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.plan import PlanGuard, _PLAN_REORDER_MESSAGE


_OK_RESULT = "Plan created.\n\nNOTE: this is the FlagScale-Agent Plan STANDARD TEMPLATE."
_FAIL_RESULTS = (
    "Error executing tool: boom",
    "[BLOCKED BY GUARD] This tool call was prevented.",
    "ERROR: At least one step is required.",
)


def _ctx(tool_name, args=None, result=None):
    return GuardContext(tool_name=tool_name, tool_args=args or {}, tool_result=result)


def _guard(single_shot=True):
    task_plan = MagicMock()
    task_plan.get_active.return_value = {"title": "t", "steps": []}
    return PlanGuard(task_plan=task_plan, single_shot=single_shot)


# ---------------------------------------------------------------- clause 1 + 2
def test_successful_plan_create_injects_once_and_never_blocks():
    g = _guard()
    v = g.check_post(_ctx("plan_create", {}, _OK_RESULT))
    assert v is not None, "a successful plan_create must surface the order notice"
    assert v.action == "inject", f"notice must be inject-only, got {v.action}"
    assert v.category == "plan_harness_steps_reordered"


def test_notice_names_both_insertions_and_the_order():
    """The notice is worthless if it says 'review your plan' without saying WHAT
    changed. Pin the two insertions and the order verbatim."""
    g = _guard()
    msg = g.check_post(_ctx("plan_create", {}, _OK_RESULT)).message
    assert "PREPENDED" in msg, "must say the Frame step was prepended"
    assert "TWO steps were APPENDED" in msg, "must say two steps were appended"
    assert "Frame FIRST" in msg, "must state the execution order"
    # It must also carry the anti-back-fill rule, which is the actual defect.
    assert "not in a batch at the finish line" in msg


def test_notice_is_the_module_constant():
    """Guards the copy: the injected text must be the reviewed constant, not a
    second hand-written copy that can drift."""
    g = _guard()
    assert g.check_post(_ctx("plan_create", {}, _OK_RESULT)).message == _PLAN_REORDER_MESSAGE


# ------------------------------------------------------------------- clause 3
def test_notice_does_not_repeat_on_second_plan_create():
    g = _guard()
    assert g.check_post(_ctx("plan_create", {}, _OK_RESULT)) is not None
    assert g.check_post(_ctx("plan_create", {}, _OK_RESULT)) is None, "one-shot per run"


# ------------------------------------------------------------------- clause 4
@pytest.mark.parametrize("bad", _FAIL_RESULTS)
def test_failed_plan_create_does_not_inject(bad):
    g = _guard()
    assert g.check_post(_ctx("plan_create", {}, bad)) is None
    # ...and the flag stays unset, so a LATER successful create still notifies.
    assert g._frame_inject_done is False
    assert g.check_post(_ctx("plan_create", {}, _OK_RESULT)) is not None


# ------------------------------------------------------------------- clause 5
def test_other_tools_do_not_inject():
    g = _guard()
    for tool in ("shell", "write_file", "plan_update", "read_file"):
        assert g.check_post(_ctx(tool, {}, "ok")) is None, tool


# ------------------------------------------------------------------- clause 6
def test_no_plan_obligation_still_settled_by_success():
    g = _guard()
    g.check_post(_ctx("plan_create", {}, _OK_RESULT))
    assert g._plan_ever_created is True
    assert g._calls_without_plan == 0


def test_no_plan_obligation_still_live_after_failure():
    g = _guard()
    g.check_post(_ctx("plan_create", {}, "ERROR: nope"))
    assert g._plan_ever_created is False
