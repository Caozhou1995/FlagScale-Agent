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

"""Regression tests for the reviewer-adjudication fixes (F4-F9).

The first landing (commit f86789f) shipped the mechanism green under its own
tests but INERT in production: the render had no production caller and several
edge states were silently swallowed. The reviewer found what the author's tests
could not because the tests called the function under test directly instead of
the production entry point. Each test here pins one adjudicated defect at the
layer the agent actually exercises.
"""

import pytest

from flagscale_agent.react.plan import TaskPlan, _normalize_forbidden
from flagscale_agent.react.guard.plan_update import PlanUpdateGuard
from flagscale_agent.react.tools.plan_create import PlanCreateTool
from flagscale_agent.react.tools.plan_update import PlanUpdateTool


@pytest.fixture
def plan(tmp_path):
    return TaskPlan(str(tmp_path / "plan"))


# ── F8: a JSON-encoded structure must decode, not become one junk line ─────


def test_json_encoded_list_string_decodes_to_entries():
    items = _normalize_forbidden(
        ['[{"action": "do not spin", "why": "w", "instead": "i"}]']
    )
    assert items == [{"action": "do not spin", "why": "w", "instead": "i"}]


def test_json_encoded_dict_string_decodes_to_entry():
    items = _normalize_forbidden(['{"action": "no blind rerun"}'])
    assert items == [{"action": "no blind rerun", "why": "", "instead": ""}]


def test_non_json_bracketed_sentence_stays_raw():
    # A plain sentence that merely opens with a bracket is prose, not JSON —
    # keep it verbatim instead of corrupting it with a failed decode.
    items = _normalize_forbidden(["[sic] never re-derive by hand"])
    assert items[0]["action"] == "[sic] never re-derive by hand"


# ── F9/F5: malformed entries are dropped but never silently ─────────────────


def test_dict_without_action_is_dropped_and_counted(plan):
    p = plan.create(
        "t", ["s"], forbidden=[{"why": "no action text"}, {"action": "ok", "why": "w"}]
    )
    assert [e["action"] for e in p["forbidden"]] == ["ok"]
    assert p["forbidden_meta"]["requested"] == 2
    assert p["forbidden_meta"]["dropped"] == 1


def test_clean_create_has_zero_dropped(plan):
    p = plan.create("t", ["s"], forbidden=[{"action": "a", "why": "w"}])
    assert p["forbidden_meta"]["dropped"] == 0


def test_creation_echo_discloses_dropped_entries(plan):
    p = plan.create(
        "t", ["s"], forbidden=[{"why": "no action text"}, {"action": "ok"}]
    )
    out = PlanCreateTool(plan).execute(title="t", steps=["s"],
                                       forbidden=[{"why": "no action"}, {"action": "ok"}])
    assert "被丢弃" in out
    assert "{'action': ...}" in out  # the fix is named, not just the count


def test_creation_echo_clean_when_nothing_dropped(plan):
    out = PlanCreateTool(plan).execute(title="t", steps=["s"], forbidden=["real one"])
    assert "被丢弃" not in out


# ── F6: cap-full and duplicate are different states with different advice ───


def _fill_to_cap(plan_obj):
    plan_obj.create("t", ["s"], forbidden=[f"prohibition number {i}" for i in range(8)])
    return plan_obj


def test_add_forbidden_at_cap_gets_cap_message_not_duplicate_message(plan):
    p = _fill_to_cap(plan)
    out = PlanUpdateTool(p).execute(action="add_forbidden", forbidden=["brand new"])
    assert "已满" in out
    assert "unchanged" not in out


def test_add_forbidden_duplicate_gets_duplicate_message(plan):
    plan.create("t", ["s"], forbidden=["only one trap"])
    out = PlanUpdateTool(plan).execute(action="add_forbidden", forbidden=["only one trap"])
    assert "unchanged" in out
    assert "已满" not in out


def test_add_forbidden_reaching_cap_exactly_warns_on_success(plan):
    # 7 existing + 1 append lands exactly at cap: the append SUCCEEDS (added=1)
    # yet the reply must warn that further appends will be refused.
    plan.create("t", ["s"], forbidden=[f"prohibition number {i}" for i in range(7)])
    out = PlanUpdateTool(plan).execute(action="add_forbidden", forbidden=["seventh"])
    assert "Appended 1" in out
    assert "NOTE" in out
    assert "further appends will be refused" in out


# ── F7: a rejected add_forbidden must not clear the stall escalation ────────


def _ctx(tool_args, tool_result):
    from types import SimpleNamespace
    return SimpleNamespace(
        tool_name="plan_update", tool_args=tool_args, tool_result=tool_result,
        messages=[], llm_responded=False, classify_fn=None, judge=None,
    )


def test_rejected_add_forbidden_is_not_progress():
    ctx = _ctx({"action": "add_forbidden", "forbidden": ["x"]},
               "No new prohibitions added (3 already recorded). The list on disk is unchanged")
    assert PlanUpdateGuard._progress_action_landed(ctx) is False
    assert PlanUpdateGuard._is_substantive_update(ctx) is False
    assert PlanUpdateGuard._clears_escalated_block(ctx) is False


def test_landed_add_forbidden_is_progress():
    ctx = _ctx({"action": "add_forbidden", "forbidden": ["x"]},
               "Appended 1 prohibition(s); 4 now on record:")
    assert PlanUpdateGuard._progress_action_landed(ctx) is True
    assert PlanUpdateGuard._is_substantive_update(ctx) is True
    assert PlanUpdateGuard._clears_escalated_block(ctx) is True


def test_other_progress_actions_keep_unconditional_credit():
    ctx = _ctx({"action": "step_done", "step_id": 1}, "Step 1 done.")
    assert PlanUpdateGuard._progress_action_landed(ctx) is True


def test_add_forbidden_with_no_result_yet_is_not_credited():
    # Pre-execution context (result None) must not credit — the reply is the
    # only witness that an append actually landed.
    ctx = _ctx({"action": "add_forbidden", "forbidden": ["x"]}, None)
    assert PlanUpdateGuard._progress_action_landed(ctx) is False


# ── F4: cross-call dedupe matches on the folded key, not raw text ───────────


def test_cross_call_case_variant_is_duplicate_not_new(plan):
    plan.create("t", ["s"], forbidden=["Do NOT rerun the whole suite"])
    res = plan.add_forbidden(["do not rerun the WHOLE suite"])
    assert res["added"] == 0
    assert res["skipped_duplicates"] == 1


def test_add_forbidden_result_carries_cap_state(plan):
    plan.create("t", ["s"], forbidden=["a", "b"])
    res = plan.add_forbidden(["c"])
    assert res["added"] == 1 and res["total"] == 3 and res["at_cap"] is False
