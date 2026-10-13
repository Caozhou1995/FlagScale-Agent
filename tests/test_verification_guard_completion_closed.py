"""Unit tests for the completion-boundary closure arm (VerificationGuard).

A text-only [TASK_COMPLETE] must obey the SAME closure rule that
plan_update(action="complete") already enforces: it may not release with plan
steps still open. Without this, an agent under deadline pressure can skip every
plan_update and declare done, so the fixed convention / deliverable_reading
record (the declaration of the decisive reading it shipped under) is never
written. Overridable, exactly like the plan_update route.

See verification.py check_pre (completion-boundary arm) and _unfinished_steps.
"""
import tempfile
import shutil

from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.plan import TaskPlan


def _plan_with_statuses(statuses):
    """Build a plan whose steps carry the given statuses in order."""
    d = tempfile.mkdtemp()
    plan = TaskPlan(d)
    plan.create("T", [{"title": f"s{i}", "acceptance": [f"a{i}"]}
                      for i in range(len(statuses))])
    for i, st in enumerate(statuses, start=1):
        if st != "pending":
            plan.update_step(i, st)
    return d, plan


def _ctx(text, override="", llm=True, tool_name=""):
    return GuardContext(tool_name=tool_name, tool_args={}, tool_result=None,
                        assistant_text=text, override_reason=override,
                        llm_responded=llm)


def _close(d):
    shutil.rmtree(d, ignore_errors=True)


def test_tc_with_pending_step_blocks():
    d, plan = _plan_with_statuses(["pending", "done"])
    try:
        g = VerificationGuard(plan=plan)
        v = g.check_pre(_ctx("[TASK_COMPLETE]"))
        assert v is not None and v.reason == "completion_with_unfinished_steps"
    finally:
        _close(d)


def test_tc_with_doing_step_blocks():
    d, plan = _plan_with_statuses(["doing"])
    try:
        g = VerificationGuard(plan=plan)
        v = g.check_pre(_ctx("[TASK_COMPLETE]"))
        assert v is not None and v.reason == "completion_with_unfinished_steps"
    finally:
        _close(d)


def test_tc_all_done_passes():
    d, plan = _plan_with_statuses(["done", "skipped"])
    try:
        g = VerificationGuard(plan=plan)
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        _close(d)


def test_tc_decorated_markdown_blocks():
    d, plan = _plan_with_statuses(["pending"])
    try:
        g = VerificationGuard(plan=plan)
        v = g.check_pre(_ctx("**[TASK_COMPLETE]**"))
        assert v is not None and v.reason == "completion_with_unfinished_steps"
    finally:
        _close(d)


def test_tc_override_releases():
    d, plan = _plan_with_statuses(["pending"])
    try:
        g = VerificationGuard(plan=plan)
        assert g.check_pre(_ctx("[TASK_COMPLETE]", override="abandoning, obsolete")) is None
    finally:
        _close(d)


def test_need_user_input_never_blocks():
    d, plan = _plan_with_statuses(["pending"])
    try:
        g = VerificationGuard(plan=plan)
        assert g.check_pre(_ctx("[NEED_USER_INPUT]")) is None
    finally:
        _close(d)


def test_no_plan_never_blocks():
    g = VerificationGuard(plan=None)
    assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None


def test_stale_scan_not_blocked():
    """The top-of-loop check_pre (llm_responded=False) scans prior-turn text — the
    completion arm must not fire on that stale scan (only on fresh output)."""
    d, plan = _plan_with_statuses(["pending"])
    try:
        g = VerificationGuard(plan=plan)
        assert g.check_pre(_ctx("[TASK_COMPLETE]", llm=False)) is None
    finally:
        _close(d)


def test_tool_path_arm_inert():
    """With a real tool_name this arm stays inert (the plan_update path owns it)."""
    d, plan = _plan_with_statuses(["pending"])
    try:
        g = VerificationGuard(plan=plan)
        assert g.check_pre(_ctx("ok", tool_name="plan_update")) is None
    finally:
        _close(d)
