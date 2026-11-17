"""SystemPromptRemindGuard: ONE block per completed step, then it passes.

The beat semantics (revised 2026-11-10): the FIRST submission of a completion
declaration is blocked once — forcing a scientific self-review — and the
re-issued SAME declaration passes only when its own evidence/notes carry the
six STRUCTURAL review labels (W0/W1/W2/W3/W4/RISK) plus three FORM
sub-slots — W3's two-slot probe, W2's UNSET-FIELDS line, and W2's
UNIT-FORMULA line (added 2026-11-16 per user ruling: an unset convention must
be THOUGHT about, not just headed, and a unit mismatch must be CONVERTED, not
just reported — raman-fitting shipped nm as cm^-1 three runs running). The guard still never
grades the review: a wrong-but-filled answer passes, an empty one does not.
Measured motivation for the revision (job 2026-11-10__00-11-22): 72/72 beaten
trials released the block with a 5+ char `_override_reason` and produced no
review at all — the old "re-issue passes untouched" form was a no-op gate.
Tests pin both sides: every shape that declares a step done gets exactly one
beat, and every neighbouring shape (other tools, non-declaration
plan_updates) stays silent.
"""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.system_prompt_remind import (
    SystemPromptRemindGuard,
    _declares_completion,
)


def _ctx(tool_name="plan_update", args=None, result="Plan updated"):
    return GuardContext(
        tool_name=tool_name,
        tool_args=args or {},
        tool_result=result,
    )


def _filled(**over):
    """The six review slots, structurally filled. The CONTENT is deliberately
    junk: these tests must pass whether or not the answers are any good, which
    is how the guard's "structure only, no semantics" boundary is pinned."""
    slots = [
        "W0 unverified layer: x",
        "W1 settled-too-early pattern: x",
        "W2 trap inventory walk: x; UNSET-FIELDS: none - every delivered field "
        "is fixed by the statement; UNIT-FORMULA: none - no unit conversion "
        "was in play; LIST-FLOOR: none - no allowlist in the statement; "
        "READBACK-STATE: none - no shared read-back object; "
        "ABSENCE-AS-EVIDENCE: none - no inference from silence; "
        "DEFINING-FIELD: none - no category assignment in this task; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated",
        "W3 cmd=id -> output=uid=0 verdict=pass bar=<1",
        "W4 instance: x",
        "RISK SWEEP: r1 -> e1 -> side",
    ]
    slots.extend(over.get("extra", []))
    return slots


def _pre(g, args, tool_name="plan_update", llm_responded=None):
    ctx = _ctx(tool_name, args)
    if llm_responded is not None:
        ctx.llm_responded = llm_responded
    return g.check_pre(ctx)


def test_first_step_done_is_blocked_with_beat_message():
    g = SystemPromptRemindGuard()
    v = _pre(g, {"action": "step_done", "step_id": 3})
    assert v is not None and v.action == "block"
    low = v.message.lower()
    assert "blocked once" in low
    assert "re-issue" in low
    # The beat is a review instruction, not a pass — re-issuing is the ONLY
    # escape, and the harness states the re-issue passes once the evidence
    # lines are present.
    assert "passes when" in low


def test_beat_message_carries_the_three_parts_in_order():
    """The beat's parts must appear in the message, in this order: the
    unverified-layer prompt, the failure receipt, the trap-inventory walk,
    then the fresh probe, and finally the risk sweep."""
    g = SystemPromptRemindGuard()
    v = _pre(g, {"action": "step_done", "step_id": 1})
    low = v.message.lower()
    i1 = low.index("unverified layer")
    i2 = low.index("failure receipt")
    i3 = low.index("trap inventory walk")
    i4 = low.index("fresh probe")
    i5 = low.index("risk sweep")
    assert i1 < i2 < i3 < i4 < i5, "the beat parts must appear in the stated order"
    # the walk must demand a real output, and the sweep must demand settlement
    assert "real output" in low
    assert "blocks this declaration" in low or "settle it now" in low


def test_old_seven_question_payload_does_not_return():
    """Anti-decay in the OTHER direction: the 7 numbered questions were removed
    on 2026-11-01 because they had decayed into a checklist that was answered
    without changing anything. They must not creep back."""
    g = SystemPromptRemindGuard()
    v = _pre(g, {"action": "step_done", "step_id": 2})
    low = v.message.lower()
    for dead in (
        "observed vs inferred",
        "counterfactual",
        "verifier must be able to fail",
        "strike count",
        "historical claims",
        "kill-class",
    ):
        assert dead not in low, f"removed 7-question item came back: {dead}"


def test_reissue_passes_only_with_the_six_review_slots():
    g = SystemPromptRemindGuard()
    args = {"action": "step_done", "step_id": 3}
    assert _pre(g, args) is not None  # beat
    # A re-issue that adds prose but no slots is STILL refused — this is the
    # whole point of the revision: the old form let any wording through.
    assert _pre(g, {"action": "step_done", "step_id": 3, "notes": "self-review done"}) is not None
    # With the slots present (still junk content) it passes, and the signature
    # must not change (evidence fields are not identity).
    assert _pre(g, {"action": "step_done", "step_id": 3, "verification": _filled()}) is None


def test_readback_state_reaches_private_endpoints():
    """SIXTH AXIS (2026-11-16): the checker may dial a PRIVATE endpoint the
    statement never names.

    Measured: one task lost 11 trials; the newest passed 3 of 4 assertions and
    failed only where the checker connects to an address the statement left
    unset while the agent listened on a reasonable-looking different one.
    Reporting "what the checker reads" is not enough when the address itself is
    unset — the declaration must say either that a working endpoint is exposed
    at EVERY address the checker could read, or which rival address was left
    empty and why that is safe. Pinned here so the axis cannot decay into a
    label that is named and never discharged.
    """
    others = ("UNSET-FIELDS: none; UNIT-FORMULA: none; LIST-FLOOR: none; "
              "ABSENCE-AS-EVIDENCE: none; DEFINING-FIELD: none; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated")

    def shell(w2):
        return ["W0 unverified layer: x", "W1 settled: x", w2,
                "W3 cmd=id -> uid=0 verdict=pass bar=<1",
                "W4 instance: x", "RISK SWEEP: r1 -> e1 -> side"]

    g = SystemPromptRemindGuard()
    assert _pre(g, {"action": "step_done", "step_id": 7}) is not None  # beat

    # absent sub-slot -> refused, and the message names it
    v = _pre(g, {"action": "step_done", "step_id": 7,
                 "verification": shell(f"W2 trap inventory walk: x; {others}")})
    assert v is not None and v.action == "block", (
        "a W2 with no READBACK-STATE was accepted — the sub-slot is decoration")
    assert "READBACK-STATE" in v.message

    # the explicit no-op is legitimate
    assert _pre(g, {"action": "step_done", "step_id": 7, "verification": shell(
        f"W2 trap inventory walk: x; {others}; "
        "READBACK-STATE: none - no shared read-back object; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated")}) is None

    # the coverage declaration discharges the axis
    assert _pre(g, {"action": "step_done", "step_id": 7, "verification": shell(
        f"W2 trap inventory walk: x; {others}; "
        "READBACK-STATE: exposed a working endpoint at every address the "
        "checker could read")}) is None

    # and so does an explicit leave-empty declaration
    assert _pre(g, {"action": "step_done", "step_id": 7, "verification": shell(
        f"W2 trap inventory walk: x; {others}; "
        "READBACK-STATE: no private address left open; my probe wrote none")}) is None


def test_unit_formula_subslot_is_not_decoration():
    """ABLATION for the W2 UNIT-FORMULA sub-slot (2026-11-16).

    The sub-slot exists because raman-fitting went 3-for-3 the same way: the
    reading block reported `unit or scale` as UNSET and named the affected
    field, and the agent still shipped the raw axis value. Reporting an unset
    convention does not discharge it — the CONVERSION has to be written out.

    This test pins that the label can actually turn the gate red. A
    five-trap-layer W2 that carries UNSET-FIELDS but no conversion must be
    REFUSED, so the sub-slot is a structural requirement rather than a word
    in the reminder text. If this test goes green-by-accident (i.e. the
    declaration is accepted), the label has become decoration.
    """
    g = SystemPromptRemindGuard()
    no_conv = [
        "W0 unverified layer: x",
        "W1 settled-too-early pattern: x",
        "W2 trap inventory walk: ENUMERATE/PARSE-SCHEMA/RANK/RUN-TIME; "
        "UNSET-FIELDS: field F -> raw 1.9e4 -> an outside source fixes it -> rival 1.6e3; "
        "LIST-FLOOR: no allowlist in the statement; READBACK-STATE: no shared read-back object; "
        "DEFINING-FIELD: no category assignment in this task; ; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated"
        "ABSENCE-AS-EVIDENCE: none - no inference from silence; "
        "DEFINING-FIELD: none - no category assignment in this task; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated",
        "W3 cmd=id -> output=uid=0",
        "W4 instance: x",
        "RISK SWEEP: r1 -> e1 -> side",
    ]
    assert _pre(g, {"action": "step_done", "step_id": 9}) is not None  # beat
    v = _pre(g, {"action": "step_done", "step_id": 9, "verification": no_conv})
    assert v is not None and v.action == "block", (
        "a W2 with UNSET-FIELDS but no unit conversion was accepted — "
        "the UNIT-FORMULA sub-slot is decoration"
    )
    assert "UNIT-FORMULA" in v.message

    # The explicit no-op declaration is legitimate and must pass (a task with
    # no unit in play must not be punished for saying so).
    with_none = list(no_conv)
    with_none[2] += "; UNIT-FORMULA: no unit conversion was in play"
    with_none[2] += "; DEFINING-FIELD: no category assignment in this task; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated"
    assert _pre(g, {"action": "step_done", "step_id": 9, "verification": with_none}) is None
    # So is the literal sub-label carrying an actual conversion.
    with_conv = list(no_conv)
    with_conv[2] += "; UNIT-FORMULA: field F -> x_B = f(x_A) -> 1.9e4 in A vs 1.6e3 in B"
    assert _pre(g, {"action": "step_done", "step_id": 9, "verification": with_conv}) is None


def test_list_floor_subslot_is_not_decoration():
    """ABLATION for the W2 LIST-FLOOR sub-slot (2026-11-16).

    Same shape as the UNIT-FORMULA ablation one task later. The statement
    enumerated what a deliverable may consist of; the agent read that list as a
    CEILING (nothing off-list), shipped a deliverable that never contained one
    of the enumerated items, and its own scan confirmed only the ceiling — so
    its self-test was green while the checker's set-membership test failed.

    Pins that the label can turn the gate red: a W2 carrying UNSET-FIELDS and
    UNIT-FORMULA but no per-item probe must be REFUSED. If this goes green by
    accident, the label is decoration.
    """
    g = SystemPromptRemindGuard()
    no_probe = [
        "W0 unverified layer: x",
        "W1 settled-too-early pattern: x",
        "W2 trap inventory walk: ENUMERATE/PARSE-SCHEMA/RANK/RUN-TIME; "
        "UNSET-FIELDS: none; UNIT-FORMULA: none - no unit conversion in play; "
        "DEFINING-FIELD: no category assignment in this task; ; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated"
        "READBACK-STATE: none - no shared read-back object; "
        "ABSENCE-AS-EVIDENCE: none - no inference from silence; "
        "DEFINING-FIELD: none - no category assignment in this task; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated",
        "W3 cmd=id -> output=uid=0",
        "W4 instance: x",
        "RISK SWEEP: r1 -> e1 -> side",
    ]
    assert _pre(g, {"action": "step_done", "step_id": 11}) is not None  # beat
    v = _pre(g, {"action": "step_done", "step_id": 11, "verification": no_probe})
    assert v is not None and v.action == "block", (
        "a W2 with no per-item probe of an enumerated list was accepted — "
        "the LIST-FLOOR sub-slot is decoration"
    )
    assert "LIST-FLOOR" in v.message

    # An explicit no-op is legitimate: a statement that enumerates nothing must
    # not be punished for saying so.
    with_none = list(no_probe)
    with_none[2] += "; LIST-FLOOR: no allowlist in the statement; READBACK-STATE: no shared read-back object"
    with_none[2] += "; DEFINING-FIELD: no category assignment in this task; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated"
    assert _pre(g, {"action": "step_done", "step_id": 11, "verification": with_none}) is None
    # And so is the literal label carrying a real per-item result.
    with_probe = list(no_probe)
    with_probe[2] += ("; LIST-FLOOR: item a -> absent from delivered bytes -> "
                      "floor reading, fixing")
    assert _pre(g, {"action": "step_done", "step_id": 11, "verification": with_probe}) is None


def test_readback_state_subslot_is_not_decoration():
    """ABLATION for the W2 READBACK-STATE sub-slot (2026-11-16, third axis).

    Measured on a task whose agent ran the deliverable itself to check it. That
    probe CREATED the very object the checker waits on, and left it behind. The
    checker's wait loop is `while not <path>.exists()`, so the residue made it
    fall through instantly, terminate the process before the expected start-up
    line was printed, and compare its expected image against a frame produced by
    a LONG-RUNNING run rather than the start-up screen. One leftover file failed
    two of three assertions: one on CONTENT, one on TIMING. The agent's own
    self-test was green the whole time, because it only checked that the object
    it had produced was well formed.

    This is the third instance of one shape: report an unset convention without
    discharging it / report an enumeration without probing each item / probe the
    artifact without asking whether the probe CHANGED what the checker reads.

    Pins that the label can turn the gate red: a W2 carrying the other two
    sub-slots but no read-back inventory must be REFUSED. If this goes green by
    accident, the label is decoration.
    """
    g = SystemPromptRemindGuard()
    no_readback = [
        "W0 unverified layer: x",
        "W1 settled-too-early pattern: x",
        "W2 trap inventory walk: ENUMERATE/PARSE-SCHEMA/RANK/RUN-TIME; "
        "UNSET-FIELDS: none; UNIT-FORMULA: none - no unit conversion in play; "
        "DEFINING-FIELD: no category assignment in this task; ; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated"
        "LIST-FLOOR: none - no allowlist in the statement; "
        "ABSENCE-AS-EVIDENCE: none - no inference from silence; "
        "DEFINING-FIELD: none - no category assignment in this task; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated",
        "W3 cmd=id -> output=uid=0",
        "W4 instance: x",
        "RISK SWEEP: r1 -> e1 -> side",
    ]
    assert _pre(g, {"action": "step_done", "step_id": 12}) is not None  # beat
    v = _pre(g, {"action": "step_done", "step_id": 12, "verification": no_readback})
    assert v is not None and v.action == "block", (
        "a W2 that never inventoried the state left at the checker's read-back "
        "path was accepted — the READBACK-STATE sub-slot is decoration"
    )
    assert "READBACK-STATE" in v.message

    # An explicit no-op is legitimate: a task whose probes touch no object the
    # checker later reads must not be punished for saying so.
    with_none = list(no_readback)
    with_none[2] += "; READBACK-STATE: no shared read-back object"
    with_none[2] += "; DEFINING-FIELD: no category assignment in this task; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated"
    assert _pre(g, {"action": "step_done", "step_id": 12,
                    "verification": with_none}) is None
    # And so is the literal label carrying a real inventory, INCLUDING the case
    # where the residue was real and had to be cleaned.
    with_probe = list(no_readback)
    with_probe[2] += ("; READBACK-STATE: /out/artifact -> my own probe created "
                      "it -> removed before finishing, because the checker waits "
                      "on its existence and would have skipped its wait")
    assert _pre(g, {"action": "step_done", "step_id": 12,
                    "verification": with_probe}) is None


def test_different_step_gets_its_own_beat():
    g = SystemPromptRemindGuard()
    assert _pre(g, {"action": "step_done", "step_id": 1}) is not None
    assert _pre(g, {"action": "step_done", "step_id": 1, "verification": _filled()}) is None
    assert _pre(g, {"action": "step_done", "step_id": 2}) is not None  # new beat


def test_complete_is_one_beat_for_whole_plan():
    g = SystemPromptRemindGuard()
    assert _pre(g, {"action": "complete"}) is not None
    assert _pre(g, {"action": "complete", "verification": _filled()}) is None
    assert _pre(g, {"action": "complete", "verification": _filled()}) is None


def test_batch_beat_keys_on_done_row_set():
    g = SystemPromptRemindGuard()
    b1 = {
        "action": "batch",
        "updates": [
            {"step_id": 1, "status": "doing"},
            {"step_id": 2, "status": "done"},
        ],
    }
    assert _pre(g, b1) is not None  # beat on {2}
    assert _pre(g, b1) is not None  # same done-set, still no slots -> refused
    assert _pre(g, dict(b1, verification=_filled())) is None  # slots present
    # A batch that declares a NEW done row is a different beat.
    b2 = {
        "action": "batch",
        "updates": [
            {"step_id": 2, "status": "done"},
            {"step_id": 3, "status": "done"},
        ],
    }
    assert _pre(g, b2) is not None


def test_silent_on_step_doing_and_step_skip():
    g = SystemPromptRemindGuard()
    assert _pre(g, {"action": "step_doing", "step_id": 3}) is None
    assert _pre(g, {"action": "step_skip", "step_id": 3}) is None


def test_silent_on_batch_without_done_rows():
    g = SystemPromptRemindGuard()
    args = {
        "action": "batch",
        "updates": [{"step_id": 1, "status": "doing"}],
    }
    assert _pre(g, args) is None


def test_beat_is_compact_and_machine_checkable():
    """The beat must stay short enough to be read inside a 900s task's budget,
    and its W3 slot must ask for the three values a reviewer can check without
    re-reading prose: the command's real output, the bar, and a verdict.

    Measured motivation (2026-11-07 v75fails19 replay, 19 trials): the beat is
    paid on EVERY step of EVERY trial, so its length is a per-trial cost; the
    previous wording (3497 chars, mostly explanatory) produced declarations that
    satisfied the labels without producing a new reading. The compact form is
    1480 chars and names the output/bar/verdict triple explicitly.
    """
    g = SystemPromptRemindGuard()
    v = _pre(g, {"action": "step_done", "step_id": 1})
    msg = v.message
    # USER RULING (2026-11-08, final round before the 89-task full run): the
    # budget was raised and the beat may carry a TEACHING block that taxes the
    # agent to name its own instance of each recurring trap shape. That block is
    # pure mechanism (no numbers, no task names), so the guard against answer
    # leakage still holds; only the hard length bar moves, 2000 -> 2400.
    #
    # USER RULING (2026-11-16): the bar moves again, 2400 -> 2900, because the
    # beat grew a THIRD measured enforcement layer (READBACK-STATE) and the
    # first two each flipped a reward=0 task to reward=1.0 (a unit-conversion
    # task and an enumerated-list task). The budget was raised on a measured
    # cost, not on convenience: across the 89-task batch the beat fires 7.32
    # times per trial on average (549 fires over the 75 trials that hit it,
    # max 11), so 2697 chars is ~674 tokens per fire and ~4936 tokens per
    # trial -- 2.5% of a 200k context, paid to enforce three distinct failure
    # families that each cost a whole task when unenforced. Each further layer
    # must clear the same bar: a measured failure it prevents, and no task
    # literals in its text.
    #
    # 2900 -> 3100 (same bar, cleared the same way). STATED-VS-SHIPPED stopped
    # being enough as a description: measured over three independent trials of
    # one task, the agent's review line asserted the constraint was satisfied in
    # words while the delivered artifact omitted the very element it described.
    # The slot now demands the characters QUOTED off the shipped file, so a
    # true-sounding sentence cannot fill it. The added text names no task, no
    # number and no domain term (leak gate: 0 hits), and it prevents a failure
    # that has now cost three whole trials.
    assert len(msg) < 3100, f"beat grew back to {len(msg)} chars; keep it under 3100"
    low = msg.lower()
    # the W3 slot must demand a machine-checkable triple
    assert "verdict=pass|fail" in low, "the W3 verdict slot lost its machine-readable form"
    assert "bar=<" in low, "the W3 slot no longer asks for the bar value"
    # and the answers must be form-filled, not paragraphs
    assert "<command>" in low

    g = SystemPromptRemindGuard()
    args = {"action": "step_done", "step_id": 1}
    assert _pre(g, args, tool_name="shell") is None
    assert _pre(g, args, tool_name="edit_file") is None


def test_check_post_never_injects():
    """The beat moved to pre-side; post must be silent (no double reminder)."""
    g = SystemPromptRemindGuard()
    args = {"action": "step_done", "step_id": 1}
    ctx = _ctx("plan_update", args)
    assert g.check_post(ctx) is None
    assert g.check_post(ctx) is None


def test_declares_completion_ignores_unknown_actions():
    assert not _declares_completion({"action": "add_steps"})
    assert not _declares_completion({})
    assert _declares_completion({"action": "batch", "updates": [{"status": "DONE"}]})


def test_absence_as_evidence_subslot_is_not_decoration():
    """ABLATION for the W2 ABSENCE-AS-EVIDENCE sub-slot (2026-11-16, fourth axis).

    Measured on a retrieval task whose statement names a component whose
    behaviour ships with the named library: the reference invocation passes a
    usage-typed argument that selects it. The agent inspected a DIFFERENT
    artifact, found it silent about that behaviour, and concluded the behaviour
    did not exist -- while its own note had already recorded that applying it
    changed the answer. One artifact's silence was treated as a negative
    observation about another component.

    This is the fourth instance of one shape: report an unset convention
    without discharging it / report an enumeration without probing each item /
    probe the artifact without asking what state it left / infer a NEGATIVE
    from an artifact that merely never mentioned the subject.

    Pins that the label can turn the gate red: a W2 carrying every other
    sub-slot but no silence-vs-observation line must be REFUSED. If this goes
    green by accident, the label is decoration.
    """
    g = SystemPromptRemindGuard()
    no_absence = [
        "W0 unverified layer: x",
        "W1 settled-too-early pattern: x",
        "W2 trap inventory walk: ENUMERATE/PARSE-SCHEMA/RANK/RUN-TIME; "
        "UNSET-FIELDS: none; UNIT-FORMULA: none - no unit conversion in play; "
        "DEFINING-FIELD: no category assignment in this task; ; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated"
        "LIST-FLOOR: none - no allowlist in the statement; "
        "READBACK-STATE: none - no shared read-back object",
        "W3 cmd=id -> output=uid=0",
        "W4 instance: x",
        "RISK SWEEP: r1 -> e1 -> side",
    ]
    assert _pre(g, {"action": "step_done", "step_id": 13}) is not None  # beat
    v = _pre(g, {"action": "step_done", "step_id": 13, "verification": no_absence})
    assert v is not None and v.action == "block", (
        "a W2 that weighed a reading without saying whether its support was an "
        "observation or an artifact's silence was accepted — the "
        "ABSENCE-AS-EVIDENCE sub-slot is decoration"
    )
    assert "ABSENCE-AS-EVIDENCE" in v.message

    # An explicit no-op is legitimate: a task whose readings all rest on
    # components the agent actually exercised must not be punished.
    with_none = list(no_absence)
    with_none[2] += "; ABSENCE-AS-EVIDENCE: no inference from silence"
    with_none[2] += "; DEFINING-FIELD: no category assignment in this task; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated"
    assert _pre(g, {"action": "step_done", "step_id": 13,
                    "verification": with_none}) is None
    # So is the literal label carrying a real inventory of the two readings.
    with_probe = list(no_absence)
    with_probe[2] += ("; ABSENCE-AS-EVIDENCE: a config file was silent about the "
                      "behaviour -> I had inferred the component did not do it -> "
                      "I ran the component itself and observed it does")
    assert _pre(g, {"action": "step_done", "step_id": 13,
                    "verification": with_probe}) is None

def test_defining_field_subslot_is_not_decoration():
    """The DEFINING-FIELD sub-slot must be able to turn the gate RED.

    Same discipline as the other four sub-slot ablation tests: if removing it
    changes nothing, it is decoration. Pins both directions — a W2 that assigns
    a category from proxy evidence without naming the DEFINING field is
    rejected, and an explicit no-op for a task with no category assignment is
    accepted.
    """
    g = SystemPromptRemindGuard()
    base = [
        "W0 unverified layer: x",
        "W1 settled-too-early pattern: x",
        "W2 trap inventory walk: x; UNSET-FIELDS: none; "
        "UNIT-FORMULA: no unit conversion in play; "
        "LIST-FLOOR: no allowlist in the statement; "
        "READBACK-STATE: no shared read-back object; "
        "ABSENCE-AS-EVIDENCE: no inference from silence"
        "; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated",
        "W3 cmd=id -> output=uid=0",
        "W4 instance: x",
        "RISK SWEEP: r1 -> e1 -> side",
    ]

    v = _pre(g, {"action": "step_done", "step_id": 14, "verification": base})
    assert v is not None and v.action == "block", (
        "a W2 with every other sub-slot but no DEFINING-FIELD was accepted — "
        "the DEFINING-FIELD sub-slot is decoration"
    )
    assert "DEFINING-FIELD" in v.message

    # An explicit no-op is legitimate: a task whose artifact no consumer
    # classifies must not be punished for saying so.
    with_none = list(base)
    with_none[2] += "; DEFINING-FIELD: no category assignment in this task; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated"
    assert _pre(g, {"action": "step_done", "step_id": 14,
                    "verification": with_none}) is None

    # The filled form must pass too, so the sub-slot cannot be satisfied only
    # by a negation.
    with_field = list(base)
    with_field[2] += ("; DEFINING-FIELD: the field that defines the category "
                      "-> proxy -> +1 on a POSITIVE example, -1 on a NEGATIVE one")
    with_field[2] += "; OBSERVED-VS-CONSUMED: none - consumer reads the same quantity here; STATED-VS-SHIPPED: no constraint I stated"
    assert _pre(g, {"action": "step_done", "step_id": 14,
                    "verification": with_field}) is None


def test_observed_vs_consumed_subslot_is_not_decoration():
    """The OBSERVED-VS-CONSUMED sub-slot must be able to turn the gate RED.

    The failure it pins: a check can pass on a quantity that is NOT the one the
    consumer reads (an exit code instead of the pixel delta it stands for), and
    every earlier layer is silent about that. If removing the sub-slot changes
    nothing, the layer is decoration.
    """
    g = SystemPromptRemindGuard()
    base = [
        "W0 unverified layer: x",
        "W1 settled-too-early pattern: x",
        "W2 trap inventory walk: x; UNSET-FIELDS: none; "
        "UNIT-FORMULA: no unit conversion in play; "
        "LIST-FLOOR: no allowlist in the statement; "
        "READBACK-STATE: no shared read-back object; "
        "ABSENCE-AS-EVIDENCE: no inference from silence; "
        "DEFINING-FIELD: no category assignment in this task; "
        "STATED-VS-SHIPPED: no constraint I stated",
        "W3 cmd=id -> output=uid=0",
        "W4 instance: x",
        "RISK SWEEP: r1 -> e1 -> side",
    ]

    v = _pre(g, {"action": "step_done", "step_id": 14, "verification": base})
    assert v is not None and v.action == "block", (
        "a W2 with every other sub-slot but no OBSERVED-VS-CONSUMED was "
        "accepted — the sub-slot is decoration"
    )
    assert "OBSERVED-VS-CONSUMED" in v.message

    # An explicit no-op is legitimate: a task whose own check and whose
    # consumer read the same quantity must not be punished for saying so.
    with_none = list(base)
    with_none[2] += (
        "; OBSERVED-VS-CONSUMED: none - my check and the consumer read the "
        "same quantity here")
    assert _pre(g, {"action": "step_done", "step_id": 14,
                    "verification": with_none}) is None

    # And the filled form must pass too, so a negation alone cannot satisfy it.
    with_pair = list(base)
    with_pair[2] += (
        "; OBSERVED-VS-CONSUMED: my check read an exit code -> the consumer "
        "reads a rendered delta -> not the same quantity; I tied them with a "
        "probe that produced the delta directly")
    assert _pre(g, {"action": "step_done", "step_id": 14,
                    "verification": with_pair}) is None

def test_a_completion_marker_is_never_blocked_by_this_guard():
    """The completion sentinel is out of scope for this guard, by design.

    This guard's pressure lives on declarations: it beats a step_done once and
    demands the review slots on the re-issue. Completion is a different channel
    — the agent ending its run must not be gated on a review it may have had no
    opportunity to give, and a gate that can hold completion open is a livelock
    risk the design rules out. Pinned here so a later "carry the unpaid beat to
    completion" idea cannot silently re-introduce it.
    """
    g = SystemPromptRemindGuard()
    # Open a beat and never answer it.
    assert _pre(g, {"action": "step_done", "step_id": 71}) is not None
    # The completion-path consultation (tool_name="", post-LLM) must pass.
    assert _pre(g, {}, tool_name="", llm_responded=True) is None
    # ...and so must the top-of-loop pre-check that also carries tool_name="".
    assert _pre(g, {}, tool_name="", llm_responded=False) is None
