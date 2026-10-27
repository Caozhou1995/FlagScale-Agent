
# --- calibration shim (test layer only) -------------------------------------
# plan_create now REFUSES without a `calibration` declaration (instrument 1:
# prove your check works on a case whose answer is already known). The tests in
# this module measure step parsing / rendering / ledgers, not calibration, so
# they supply a default. Production code has no such default -- the refusal is
# load-bearing BECAUSE it cannot be skipped.
_CAL_SHIM = {
    "instrument": "pytest shim default",
    "known_case": "shim known case",
    "observed": "shim observed",
    "matches": "yes",
    "instrument_input": {
        "expression": "value = check(shim_path)",
        "source": "the shim names its own path as the input",
        "mismatch_checked": "recomputed from the same path: identical value",
    },
}


def _plan_create_tool(tp, session_id=""):
    from flagscale_agent.react.tools.plan_create import PlanCreateTool

    class _Shim(PlanCreateTool):
        def execute(self, **kwargs):
            kwargs.setdefault("calibration", dict(_CAL_SHIM))
            return super().execute(**kwargs)

    return _Shim(tp) if not session_id else _Shim(tp, session_id)
# ---------------------------------------------------------------------------

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

"""The plan's NEGATIVE half: ``forbidden`` (不做清单).

A plan says what to DO. These tests cover the symmetric artefact the agent
writes about what NOT to do, and — the part that actually matters — that it is
re-presented every turn rather than stored and forgotten. A prohibition written
once at planning time and never shown again cannot prevent anything: the trap it
describes becomes most tempting exactly when the clock is nearly gone, and that
is the moment the agent is least likely to go re-read its own plan file.
"""

import pytest

from flagscale_agent.react.plan import TaskPlan, _normalize_forbidden
from flagscale_agent.react.tools.plan_create import PlanCreateTool
from flagscale_agent.react.tools.plan_update import PlanUpdateTool





@pytest.fixture
def plan(tmp_path):
    return TaskPlan(str(tmp_path / "plan"))


# ── normalization ──────────────────────────────────────────────────────────


def test_string_items_normalize_to_action_only():
    got = _normalize_forbidden(["不要盲目调参"])
    assert got == [{"action": "不要盲目调参", "why": "", "instead": ""}]


def test_dict_items_keep_all_three_fields():
    got = _normalize_forbidden(
        [{"action": "a", "why": "b", "instead": "c"}]
    )
    assert got == [{"action": "a", "why": "b", "instead": "c"}]


@pytest.mark.parametrize("bad", [None, "", [], {}, [None], [{}], ["   "]])
def test_empty_and_malformed_input_is_dropped_not_crashed(bad):
    # An item with no action carries no prohibition; keeping it would render a
    # blank bullet the agent could read as a real constraint.
    assert _normalize_forbidden(bad) == []


def test_duplicates_are_collapsed_case_and_space_insensitive():
    got = _normalize_forbidden(["Blind tuning", "  blind   TUNING  ", "other"])
    assert [e["action"] for e in got] == ["Blind tuning", "other"]


def test_duplicate_with_richer_payload_wins_over_bare_string():
    # A later structured item adds the 'why'/'instead' the first one lacked;
    # dropping it for being a "duplicate" would lose the useful part.
    got = _normalize_forbidden(["tune blindly", {"action": "tune blindly", "why": "w"}])
    assert len(got) == 1
    assert got[0]["why"] == "w"


# ── persistence through create() ───────────────────────────────────────────


def test_create_persists_forbidden(plan):
    plan.create("t", ["a"], forbidden=[{"action": "x", "why": "y", "instead": "z"}])
    assert plan.get_active()["forbidden"][0]["action"] == "x"


def test_create_without_forbidden_persists_empty_list(plan):
    plan.create("t", ["a"])
    assert plan.get_active()["forbidden"] == []


# ── the mechanism: re-rendered every turn ──────────────────────────────────


def test_context_for_prompt_renders_declared_prohibitions(plan):
    plan.create("t", ["a"], forbidden=["不要在没有读数时连续微调参数"])
    out = plan.context_for_prompt()
    assert "不做清单" in out
    assert "不要在没有读数时连续微调参数" in out


def test_context_for_prompt_renders_why_and_instead(plan):
    plan.create("t", ["a"], forbidden=[
        {"action": "撞墙后继续加大 timeout", "why": "同一命令只会再撞一次",
         "instead": "读部分输出并换方法"},
    ])
    out = plan.context_for_prompt()
    assert "同一命令只会再撞一次" in out
    assert "读部分输出并换方法" in out


def test_absent_prohibitions_are_named_as_a_gap_not_silently_skipped(plan):
    # Silence would read as "nothing to see"; the point is that a plan with no
    # negative half is a plan with no defence, and the agent must be told so.
    plan.create("t", ["a"])
    out = plan.context_for_prompt()
    assert "不做清单" in out
    assert "尚未声明" in out


def test_prohibitions_render_before_the_step_list(plan):
    # Position is the mechanism: the list must be read with the time ledger,
    # ahead of the to-do steps, not buried after them where a long plan hides it.
    plan.create("t", ["step one"], forbidden=["不要抛光已有产物"])
    out = plan.context_for_prompt()
    assert out.index("不做清单") < out.index("step one")


def test_rendering_is_not_gated_on_remaining_time(plan):
    # A prohibition must be visible BEFORE the state that would trip it. Gating
    # it on a low-clock condition would show it only once it is already too late.
    plan.set_budget_sec(3600)
    plan.create("t", ["a"], forbidden=["不要抛光已有产物"])
    assert "不要抛光已有产物" in plan.context_for_prompt()


# ── mid-run append (traps are discovered during the work) ──────────────────


def test_add_forbidden_appends_and_persists(plan):
    plan.create("t", ["a"])
    res = plan.add_forbidden(["新发现的坑"])
    assert res["added"] == 1
    assert res["total"] == 1
    assert plan.get_active()["forbidden"][0]["action"] == "新发现的坑"


def test_add_forbidden_rejects_duplicates_without_wiping_the_list(plan):
    plan.create("t", ["a"], forbidden=["已有条目"])
    res = plan.add_forbidden(["已有条目"])
    assert res["added"] == 0
    assert res["total"] == 1
    assert plan.get_active()["forbidden"][0]["action"] == "已有条目"


# ── tool layer (the agent's only way in) ───────────────────────────────────


def test_plan_create_tool_accepts_forbidden_and_echoes_it(plan):
    out = _plan_create_tool(plan).execute(
        title="t", steps=["a"], forbidden=["不要盲目调参"],
    )
    assert "已写下 1 条" in out
    assert "不要盲目调参" in out


def test_plan_create_tool_flags_an_empty_forbidden_list(plan):
    out = _plan_create_tool(plan).execute(title="t", steps=["a"])
    assert "为空" in out
    assert "add_forbidden" in out


def test_plan_update_tool_add_forbidden_action(plan):
    plan.create("t", ["a"])
    out = PlanUpdateTool(plan).execute(action="add_forbidden", forbidden=["中途发现的坑"])
    assert "Appended 1" in out
    assert "中途发现的坑" in out


def test_plan_update_tool_add_forbidden_requires_items(plan):
    plan.create("t", ["a"])
    assert "ERROR" in PlanUpdateTool(plan).execute(action="add_forbidden")


def test_plan_update_schema_exposes_forbidden_action(plan):
    # The tool description is the agent's only documentation; an action missing
    # from the enum is an action the model will never emit.
    from flagscale_agent.react.tools.plan_update import PlanUpdateTool as P
    assert "add_forbidden" in P.parameters["properties"]["action"]["enum"]
    assert "forbidden" in P.parameters["properties"]


def test_plan_create_schema_exposes_forbidden(plan):
    from flagscale_agent.react.tools.plan_create import PlanCreateTool as P
    assert "forbidden" in P.parameters["properties"]


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
