"""Regression tests for the 6 defects an independent reviewer found in the
parallel-aware time ledger AFTER the author's 51 unit tests were green.

Each test pins one defect by its OBSERVED symptom, so a revert turns it red.
"""
import os
import tempfile
import time

from flagscale_agent.react.plan import (
    TaskPlan,
    _critical_path_secs,
    _id_list,
)


def _plan():
    return TaskPlan(os.path.join(tempfile.mkdtemp(), "p.yaml"))


def _ledger(plan, steps, budget=900.0):
    plan._budget_sec = budget
    doc = {
        "created": time.time(),
        "steps": steps,
        "time_budget": {
            "budget_sec": budget,
            "reserve_sec": round(budget * 0.15, 1),
            "started": time.time(),
        },
    }
    return "\n".join(plan._time_ledger_lines(doc))


def _st(i, est, deps, status="pending"):
    return {"id": i, "title": f"s{i}", "est_sec": est,
            "depends_on": deps, "status": status}


# F3 (reviewer HIGH): a finished step must not be named the constraint, and the
# same render must not say "over budget" in one line and "to spare" in another.
def test_done_step_is_not_the_constraint_and_render_does_not_contradict_itself():
    text = _ledger(_plan(), [
        _st(1, 900, [], "done"),
        _st(2, 100, [1]),
        _st(3, 100, [2]),
    ])
    assert "超" not in text, text          # 200s of live work against 900s fits
    assert "第 1 步"  not in text, text      # the finished 900s step is not a target
    assert "余量" in text


# F3b: the reserve must be deducted EXACTLY once. Double-counting made a
# fan-out plan read "over by N" while its own drift block said it fits.
def test_reserve_is_not_counted_twice():
    # budget 400s, reserve 60s -> usable clock 340s. Critical path = 100+200 =
    # 300s, so it FITS (300 <= 340). Double-counting the reserve inside the
    # workload gives 360 > 340 and falsely reports an overrun.
    text = _ledger(_plan(), [
        _st(1, 100, []),
        _st(2, 200, [1]),
        _st(3, 200, [1]),
    ], budget=400.0)
    assert "关键路径 5m" in text, text        # 300s, reserve NOT added
    assert "计划余量" in text, text           # 40s to spare
    assert "超" not in text, text


# F2 (reviewer HIGH): an unestimated step was priced at 0.0 and the plan was
# then certified as fitting on the strength of the work it had not priced.
def test_unestimated_steps_do_not_certify_the_plan_as_fitting():
    text = _ledger(_plan(), [
        _st(1, None, []),
        _st(2, None, [1]),
        _st(3, 400, [2]),
        _st(4, 400, [3]),
    ], budget=1300.0)
    assert "按计划走可完成" not in text, text
    assert "未估时的步骤未计入" in text, text


# F4 (reviewer MED): add_steps discarded the caller's depends_on, so a later
# step could not declare independence ([] = start immediately).
def test_add_steps_honours_caller_depends_on():
    p = _plan()
    p.create("t", steps=[{"title": "a", "est_sec": 100},
                         {"title": "b", "est_sec": 100}])
    plan = p.add_steps([
        {"title": "x", "depends_on": []},
        {"title": "y", "depends_on": [1]},
    ])
    deps = {s["id"]: s["depends_on"] for s in plan["steps"]}
    assert deps[3] == []      # caller said "start now" — preserved
    assert deps[4] == [1]     # caller named its own predecessor


def test_add_steps_without_depends_on_keeps_the_serial_chain():
    p = _plan()
    p.create("t", steps=[{"title": "a"}])
    plan = p.add_steps([{"title": "x"}, {"title": "y"}])
    deps = {s["id"]: s["depends_on"] for s in plan["steps"]}
    assert deps[2] == [1]
    assert deps[3] == [2]


# F5 (reviewer MED): duplicate ids collapsed the estimate map and silently
# dropped work (10s + 500s reported as 500).
def test_duplicate_ids_do_not_drop_work():
    r = _critical_path_secs([_st(1, 10, []), _st(1, 500, [])])
    assert r["serial"] == 510.0, r


# F7 / F8 (reviewer LOW): crash-free helpers.
def test_empty_id_list_renders_empty_string():
    assert _id_list([]) == ""


def test_critical_path_tolerates_non_dict_entries():
    r = _critical_path_secs(["a", "b"])   # must not raise AttributeError
    assert r["serial"] == 0.0
