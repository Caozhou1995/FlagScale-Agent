"""SystemPromptRemindGuard: ONE block per completed step, then it passes.

The beat semantics (revised 2026-11-10): the FIRST submission of a completion
declaration is blocked once — forcing a scientific self-review — and the
re-issued SAME declaration passes only when its own evidence/notes carry the
six STRUCTURAL review labels (W0/W1/W2/W3/W4/RISK). The guard still never
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
        "W2 trap inventory walk: x",
        "W3 cmd=id -> output=uid=0 verdict=pass bar=<1",
        "W4 instance: x",
        "RISK SWEEP: r1 -> e1 -> side",
    ]
    slots.extend(over.get("extra", []))
    return slots


def _pre(g, args, tool_name="plan_update"):
    return g.check_pre(_ctx(tool_name, args))


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
    assert len(msg) < 2400, f"beat grew back to {len(msg)} chars; keep it under 2400"
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
