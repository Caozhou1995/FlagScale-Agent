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


def _bind_oracle(g):
    """Satisfy the oracle-BIND + oracle-EXERCISE completion preconditions.

    The oracle-bound arm fires when NO external-premise oracle was bound this
    run. A test that targets a LATER completion arm (graded-reading, acceptance-
    observation) must not be blocked by the oracle arms, so it binds an oracle
    (and marks its first exercise run) up front -- exactly the state a
    well-behaved agent reaches at framing."""
    g._oracle_bound = {"premise_source": "provided",
                       "oracle": "the task's reference artifact",
                       "first_exercise": "cat /app/y"}
    g._oracle_exercised_seen = True


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
        g._acceptance_observed = True
        _bind_oracle(g)
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        _close(d)


def test_tc_graded_reading_override_does_not_release():
    """M2: the graded-reading arm is NON-overridable -- a prose override cannot
    release it. The exit is the ACTION of declaring a real deliverable_reading
    on a step_done (which sets the latch), never a sentence."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        v = g.check_pre(
            _ctx("[TASK_COMPLETE]", override="genuinely ungraded task, no artifact")
        )
        assert v is not None and v.reason == "completion_without_graded_reading", v
        assert v.overridable is False
    finally:
        _close(d)


def test_tc_graded_reading_action_exit_clears():
    """The M2 exit is the ACTION: once a real deliverable_reading was declared
    (latch set), the same completion passes."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        g._acceptance_observed = True
        _bind_oracle(g)
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
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

# ── completion-boundary ACCEPTANCE-OBSERVATION arm (R8) ─────────────────────
# A declared graded reading is not an OBSERVATION of the acceptance. The arm
# forces, at the boundary, either a real observed acceptance act (command +
# output, corpus-checked) or an explicit `unobservable` declaration — silence
# blocks. See verification.py _acceptance_act_recorded + the completion arm.

def _ctx_with_messages(text, messages, override=""):
    c = _ctx(text, override=override)
    c.messages = messages
    return c


def _tool_result_msg(observed_text):
    return [{"role": "tool", "content": observed_text}]


def test_tc_no_acceptance_act_blocks():
    """All steps done, graded reading seen, but NO acceptance act recorded ->
    the acceptance-observation arm blocks."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        v = g.check_pre(_ctx("[TASK_COMPLETE]"))
        assert v is not None and v.reason == "completion_without_acceptance_observation", v
    finally:
        _close(d)


def test_tc_acceptance_observed_flag_passes():
    """Once an observed acceptance act was recorded (flag set), completion passes."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        g._acceptance_observed = True
        _bind_oracle(g)
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        _close(d)


def test_tc_acceptance_override_does_not_release():
    """M2: the acceptance-observation arm is NON-overridable -- the exit is the
    recorded acceptance_act (an observed act, or the UNOBSERVABLE form carrying
    the probe that was actually run), not prose."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        v = g.check_pre(
            _ctx("[TASK_COMPLETE]", override="no observable acceptance; artifact is prose")
        )
        assert v is not None and v.reason == "completion_without_acceptance_observation", v
        assert v.overridable is False
    finally:
        _close(d)


def test_tc_acceptance_ungroundable_task_fails_open():
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "(unavailable) task text not recovered")
        g._graded_reading_seen = True
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        _close(d)


def test_tc_acceptance_need_user_input_not_blocked():
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        assert g.check_pre(_ctx("[NEED_USER_INPUT]")) is None
    finally:
        _close(d)


def test_acceptance_act_recorded_predicate():
    """Unit-test __acceptance_act_recorded across shapes."""
    from flagscale_agent.react.guard.verification import _acceptance_act_recorded
    corpus = "ok wrote 5 lines"
    # A BARE unobservable declaration (no attempt) is refused, even fail-open corpus.
    assert _acceptance_act_recorded({"unobservable": "hidden corpus"}, None) is False
    assert _acceptance_act_recorded({"unobservable": "hidden corpus"}, corpus) is False
    # An UNOBSERVABLE ATTEMPT carrying a real recorded probe is admissible.
    assert _acceptance_act_recorded(
        {"unobservable": "hidden corpus", "command": "ls /grader",
         "observed": "ok wrote 5 lines"}, corpus) is True
    # ...but a claimed-invisible probe not present in the corpus is refused.
    assert _acceptance_act_recorded(
        {"unobservable": "hidden corpus", "command": "ls /grader",
         "observed": "not-in-corpus"}, corpus) is False
    # observed act present in the corpus -> True.
    assert _acceptance_act_recorded(
        {"command": "wc -l /app/y", "observed": "5"}, corpus) is True
    # observed act NOT in the corpus -> False.
    assert _acceptance_act_recorded(
        {"command": "wc -l /app/y", "observed": "not-in-corpus"}, corpus) is False
    # fail-open only when corpus unavailable (None) but shape is well-formed.
    assert _acceptance_act_recorded(
        {"command": "wc -l /app/y", "observed": "5"}, None) is True
    # malformed / silent -> False.
    assert _acceptance_act_recorded(None, corpus) is False
    assert _acceptance_act_recorded({}, corpus) is False
    assert _acceptance_act_recorded({"command": "x"}, corpus) is False
    assert _acceptance_act_recorded({"observed": "   "}, corpus) is False
    assert _acceptance_act_recorded({"unobservable": ""}, corpus) is False


def test_acceptance_observed_flag_set_via_tool_result():
    """End-to-end: a step_done carrying an observed acceptance act whose output
    really appears in a session tool result sets the flag, so completion passes."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        g._step_done_recheck_reminded = True  # skip the first-step_done premise latch
        # Simulate the step_done gate running with a real corpus hit.
        ctx = _ctx_with_messages(
            "done", _tool_result_msg("wrote 5 lines to /app/y\n"),
            override="")
        ctx.tool_name = "plan_update"
        ctx.tool_args = {
            "action": "step_done", "assumptions": {"unverified": "none"}, "step_id": 1,
            "verification": ["ran it"],
            "convention": {"name": "none"},
            "deliverable_reading": {"deliverable": "none"},
            "acceptance_act": {"command": "cat /app/y | wc -l", "observed": "wrote 5 lines to /app/y"},
        }
        # No acceptance criteria on this plan's step? give it one so Mode 1 runs.
        assert g.check_pre(ctx) is None, "step_done should pass with act"
        assert g._acceptance_observed is True
    finally:
        _close(d)


def _step_done_ctx(acceptance_act, corpus_text="wrote 5 lines to /app/y\n"):
    """A step_done ctx carrying `acceptance_act`, over a tool-result corpus."""
    c = _ctx_with_messages("done", _tool_result_msg(corpus_text), override="")
    c.tool_name = "plan_update"
    c.tool_args = {
        "action": "step_done", "assumptions": {"unverified": "none"}, "step_id": 1,
        "verification": ["ran it"],
        "convention": {"name": "none"},
        "deliverable_reading": {"deliverable": "none"},
        "acceptance_act": acceptance_act,
    }
    return c


def _guard_ready_for_acceptance_act(plan):
    g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
    g._graded_reading_seen = True
    g._step_done_recheck_reminded = True  # skip the first-step_done premise latch
    _bind_oracle(g)
    return g


def test_tc_bare_unobservable_does_not_clear_boundary():
    """SEMANTIC INVERSION FALSIFIER: a bare `{"unobservable": ...}` declaration
    (no attempt) must NOT set the observed flag. If `_acceptance_act_recorded`
    is reverted to its free-pass form, this test FAILS. This is the exact
    loophole the change closes: declaring a dimension invisible without trying."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_ready_for_acceptance_act(plan)
        assert g.check_pre(
            _step_done_ctx({"unobservable": "grader corpus is hidden"})
        ) is None
        assert g._acceptance_observed is False, \
            "a bare unobservable declaration must not clear the boundary"
        # ...so completion is still blocked by the acceptance-observation arm.
        v = g.check_pre(_ctx("[TASK_COMPLETE]"))
        assert v is not None and v.reason == "completion_without_acceptance_observation", v
    finally:
        _close(d)


def test_tc_unobservable_with_recorded_attempt_clears_boundary():
    """The INCONCLUSIVE form is still admissible WHEN it carries a real,
    corpus-recorded attempt: declaration + the probe the agent ran."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_ready_for_acceptance_act(plan)
        assert g.check_pre(
            _step_done_ctx({
                "unobservable": "grader corpus is hidden",
                "command": "ls /grader",
                "observed": "wrote 5 lines to /app/y",
            })
        ) is None
        assert g._acceptance_observed is True
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        _close(d)


def test_tc_unobservable_with_fabricated_attempt_still_blocks():
    """An unobservable declaration plus an `observed` string that is NOT in the
    recorded corpus is refused — the attempt must be real, not asserted."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_ready_for_acceptance_act(plan)
        assert g.check_pre(
            _step_done_ctx({
                "unobservable": "grader corpus is hidden",
                "command": "ls /grader",
                "observed": "I could not find the grader",
            })
        ) is None
        assert g._acceptance_observed is False
    finally:
        _close(d)


# ── completion-boundary ORACLE-BIND arm (M1) ────────────────────────────────
# The frame step mandates naming an EXTERNAL-premise oracle (provided /
# differential / consumer) and passing plan_create(oracle=...). Observed failure
# mode (0/76 plans in a full TB2.1 run ever bind): agents satisfy that mandate in
# PROSE on the frame note and never bind, so the bind->exercise->completion chain
# is dead by construction and the agent closes on a purely self-authored
# verification. This arm fires when NO oracle was ever bound. See verification.py
# _COMPLETION_ORACLE_NOT_BOUND + the completion arm.

def test_tc_no_oracle_bound_blocks():
    """All steps done, graded reading + acceptance observed, but NO oracle was
    ever bound -> the oracle-bound arm blocks."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        g._acceptance_observed = True
        assert g._bound_oracle() == {}
        v = g.check_pre(_ctx("[TASK_COMPLETE]"))
        assert v is not None and v.reason == "completion_without_oracle_bound", v
    finally:
        _close(d)


def test_tc_oracle_bound_exercised_passes():
    """Once an external-premise oracle was bound AND its first exercise run,
    completion passes."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        g._acceptance_observed = True
        _bind_oracle(g)
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        _close(d)


def test_tc_oracle_bound_not_exercised_blocks_on_exercise():
    """An oracle was bound but its first exercise was never run -> the EXERCISE
    arm blocks (the bind arm does not fire, since an oracle IS bound)."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        g._acceptance_observed = True
        g._oracle_bound = {"premise_source": "provided",
                           "oracle": "the task's reference artifact",
                           "first_exercise": "cat /app/y"}
        v = g.check_pre(_ctx("[TASK_COMPLETE]"))
        assert v is not None and v.reason == "completion_without_oracle_exercise", v
    finally:
        _close(d)


def test_tc_oracle_bind_override_does_not_release():
    """M2: the oracle-bind arm is NON-overridable -- the exit is the ACTION of
    binding via plan_create(oracle={...}) (any non-empty dict latches), then
    exercising it. Even a weak/partial oracle beats an unanchored close."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        g._acceptance_observed = True
        v = g.check_pre(
            _ctx("[TASK_COMPLETE]", override="no external authority can anchor this axis")
        )
        assert v is not None and v.reason == "completion_without_oracle_bound", v
        assert v.overridable is False
    finally:
        _close(d)


def test_tc_oracle_bind_action_exit_clears():
    """The M2 exit: binding the oracle through plan_create latches it in
    check_pre, and the completion then passes the bind arm (moves on to the
    exercise arm, which the helper marks exercised)."""
    from flagscale_agent.react.tools.plan_create import PlanCreateTool
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        g._acceptance_observed = True
        # The real ACTION exit: bind via plan_create (not a hand-set latch).
        PlanCreateTool(plan).execute(
            title="re-bind", steps=["bind"],
            oracle={"premise_source": "provided", "oracle": "task reference",
                    "first_exercise": "cat /app/y"},
        )
        ctx = _ctx("[TASK_COMPLETE]")
        ctx.tool_name = "plan_create"   # first consult: capture the bind
        assert g.check_pre(ctx) is None
        # now the completion-path consult sees the latch
        ctx2 = _ctx("[TASK_COMPLETE]")
        v = g.check_pre(ctx2)
        assert v is None or v.reason != "completion_without_oracle_bound", v
    finally:
        _close(d)


def test_tc_oracle_bind_ungroundable_task_fails_open():
    """When the harness could not recover the task text, the arm must not block."""
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "(unavailable) task text not recovered")
        g._graded_reading_seen = True
        g._acceptance_observed = True
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        _close(d)


def test_tc_oracle_bind_need_user_input_not_blocked():
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        g._graded_reading_seen = True
        g._acceptance_observed = True
        assert g.check_pre(_ctx("[NEED_USER_INPUT]")) is None
    finally:
        _close(d)


def test_registry_does_not_release_nonwaivable_completion_block():
    """REGISTRY-level M2 contract: even when the completion block is the one
    surfaced last turn and a valid override reason is re-supplied, a
    non-overridable verdict is NOT released (guard/__init__.py _resolve requires
    verdict.overridable). The only exit is the corrective action."""
    from flagscale_agent.react.guard import GuardRegistry
    d, plan = _plan_with_statuses(["done"])
    try:
        g = _guard_with_task(plan, "Read /app/x and emit /app/y exactly.")
        reg = GuardRegistry()
        reg.register(g)
        ctx = _ctx("[TASK_COMPLETE]",
                   override="genuinely ungraded task, no artifact")
        v1 = reg.check_pre(ctx)
        assert v1 is not None and v1.action == "block", v1
        # same block, reason now on record -> still blocked (non-waivable)
        v2 = reg.check_pre(ctx)
        assert v2 is not None and v2.action == "block", v2
        assert v2.reason == "completion_without_graded_reading", v2
        assert v2.overridable is False
    finally:
        _close(d)
