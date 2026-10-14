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


# ── completion-boundary GRADED-READING arm ──────────────────────────────────
# The deliverable_reading gate is otherwise opt-outable: `{"deliverable": "none"}`
# satisfies it with no reading enumerated. The completion arm forces at least one
# REAL reading before [TASK_COMPLETE] when the task text is groundable.

def _ctx_tc(text, override="", task_text="Read /app/x and write /app/y."):
    """ctx for the text-completion path with a groundable task text bound."""
    return _ctx(text, override=override), task_text


def _guard_with_task(plan, task_text):
    return VerificationGuard(plan=plan, task_text_fn=lambda: task_text)


def test_tc_no_graded_reading_blocks():
    """A [TASK_COMPLETE] with all steps done but no real deliverable_reading ever
    declared must block (the opt-out bypass)."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        v = g.check_pre(_ctx("[TASK_COMPLETE]"))
        assert v is not None and v.reason == "completion_without_graded_reading", v
    finally:
        _close(d)


def test_tc_graded_reading_seen_passes():
    """Once a REAL reading was declared (flag set), completion passes."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        _close(d)


def test_tc_graded_reading_override_releases():
    """A substantive override releases the graded-reading arm."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        assert g.check_pre(
            _ctx("[TASK_COMPLETE]", override="genuinely ungraded task, no artifact")
        ) is None
    finally:
        _close(d)


def test_tc_ungroundable_task_fails_open():
    """When the harness could not recover the task text, the arm must not block."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "(unavailable) task text not recovered")
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        _close(d)


def test_tc_truncated_task_fails_open():
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "some task ...[truncated]")
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        _close(d)


def test_tc_empty_task_fails_open():
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "")
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        _close(d)


def test_tc_no_task_text_fn_fails_open():
    """task_text_fn is optional (None) -> fail-open."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = VerificationGuard(plan=plan)
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        _close(d)


def test_tc_need_user_input_not_blocked_by_graded_arm():
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        assert g.check_pre(_ctx("[NEED_USER_INPUT]")) is None
    finally:
        _close(d)


def test_declares_graded_reading_predicate():
    """Unit-test the predicate: none-opt-out is False; a real name is True."""
    from flagscale_agent.react.guard.verification import _declares_graded_reading
    assert _declares_graded_reading({"deliverable": "none"}) is False
    assert _declares_graded_reading({"deliverable": "  None "}) is False
    assert _declares_graded_reading("none") is False
    assert _declares_graded_reading(None) is False
    assert _declares_graded_reading({}) is False
    assert _declares_graded_reading({"deliverable": "/app/y"}) is True


def test_declares_graded_reading_rejects_grader_visible_claim():
    """A `grader_visible: True` claim enumerates NO reading (it short-circuits the
    readings/hedge requirement upstream), so it must NOT satisfy the completion arm
    — otherwise a self-asserted visibility re-opens the opt-out."""
    from flagscale_agent.react.guard.verification import _declares_graded_reading
    assert _declares_graded_reading(
        {"deliverable": "/app/y", "grader_visible": True}) is False
    # A visible grader still closes through the override, so a real reading with
    # grader_visible=False (or absent) remains a valid declaration.
    assert _declares_graded_reading(
        {"deliverable": "/app/y", "grader_visible": False}) is True


def test_tc_grader_visible_claim_still_blocks():
    """End-to-end: a gv=True claim without a real reading must not set the flag,
    so [TASK_COMPLETE] still blocks."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        assert g._graded_reading_seen is False
        v = g.check_pre(_ctx("[TASK_COMPLETE]"))
        assert v is not None and v.reason == "completion_without_graded_reading", v
    finally:
        _close(d)


def test_tc_task_text_containing_marker_still_groundable():
    """Positional marker matching: real task text that merely CONTAINS '(unavailable'
    or '...[truncated]' as a substring must remain groundable — only the producer's
    exact whole-text form disables the arm."""
    from flagscale_agent.react.guard.verification import _task_text_groundable
    # Whole-text unavailable / ending-truncation forms -> not groundable.
    assert _task_text_groundable("(unavailable) task text not recovered") is False
    assert _task_text_groundable("some task ...[truncated]") is False
    assert _task_text_groundable("") is False
    # A substring occurrence inside real text -> still groundable.
    assert _task_text_groundable(
        "the log shows '(unavailable' as a literal token") is True
    assert _task_text_groundable(
        "quote ...[truncated] appears mid-sentence here") is True


def test_tc_composite_wiring_truncated_segment_not_groundable():
    """The guard is wired with a COMPOSITE (agent.py: orig + "\\n\\n" + last turn).
    A truncated or unavailable SEGMENT of that composite must disable grounding
    too — the marker sits at a segment edge, not at the composite's ends. A
    regression here would fail-CLOSED (spurious reading_not_grounded) against a
    partial corpus."""
    from flagscale_agent.react.guard.verification import _task_text_groundable
    # Truncated original segment then a real last-turn segment -> not groundable.
    assert _task_text_groundable(
        "real task head line one\n...[truncated]\n\nfinal user turn text") is False
    # Unavailable original segment then a real last-turn segment -> not groundable.
    assert _task_text_groundable(
        "(unavailable: original not recovered)\n\nfinal user turn text") is False
    # A truly complete composite (no partial segment) -> groundable.
    assert _task_text_groundable(
        "complete task text here\n\nfinal user turn text") is True
