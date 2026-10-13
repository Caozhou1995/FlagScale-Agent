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


# ── override-FORM completion (kernel's SECOND completion shape) ──────────────
# The kernel accepts `trailing == "[TASK_COMPLETE]" or has_override_completion(t)`
# (kernel.py:378). The override form has text after the marker, so trailing_sentinel
# returns None; the arm must test the SAME predicate or this form leaks.

def test_tc_override_form_with_reason_releases():
    """Marker + inline _override_reason line with a substantive reason releases.

    On the real path the kernel parses the inline reason (kernel._extract_text_override)
    and puts it on ctx.override_reason before consulting the guard, so the arm sees a
    populated reason. Here we supply that as the override kwarg."""
    d, plan = _plan_with_statuses(["pending"])
    try:
        g = VerificationGuard(plan=plan)
        text = "[TASK_COMPLETE]\n_override_reason: remaining step is obsolete now"
        assert g.check_pre(_ctx(text, override="remaining step is obsolete now")) is None
    finally:
        _close(d)


def test_tc_override_form_empty_reason_blocks():
    """Marker + _override_reason with NO text must still block (the leak F1)."""
    d, plan = _plan_with_statuses(["pending"])
    try:
        g = VerificationGuard(plan=plan)
        for text in ("[TASK_COMPLETE]\n_override_reason:",
                     "[TASK_COMPLETE]\n_override_reason:   "):
            v = g.check_pre(_ctx(text))
            assert v is not None and v.reason == "completion_with_unfinished_steps"
    finally:
        _close(d)


def test_tc_override_bar_matches_tool_route():
    """A too-short override (<6 chars) must NOT release the text route — the same
    bar accept_override applies on the tool route."""
    d, plan = _plan_with_statuses(["pending"])
    try:
        g = VerificationGuard(plan=plan)
        v = g.check_pre(_ctx("[TASK_COMPLETE]", override="ok"))
        assert v is not None and v.reason == "completion_with_unfinished_steps"
    finally:
        _close(d)


def test_unfinished_steps_id_fallback():
    """A step dict without an 'id' must not render 'None' in the block message."""
    d, plan = _plan_with_statuses(["pending"])
    try:
        g = VerificationGuard(plan=plan)
        ids = g._unfinished_steps()
        assert all(i is not None for i in ids), ids
    finally:
        _close(d)
