"""STATED-VS-SHIPPED must invoice EVERY constraint the agent stated, not a subset.

Measured motivation (2026-11-17, task with five generations of the same failure):
the slot itself already fired and was answered substantively 11 times in one
run -- yet the run shipped the same violating bytes. Reading the agent's own log
showed WHY: it had stated the decisive constraint verbatim early in the run, but
its STATED list enumerated a DIFFERENT, self-chosen subset of constraints, all of
which agreed. The gate judged that a reconciliation HAD HAPPENED; it never asked
whether the reconciled set covered what the agent itself had stated.

So the wording carries the coverage requirement, in both places the agent reads:
the beat (every step_done) and the refusal message (when the slot is absent).
A structural gate cannot grade coverage -- it can only make the requirement
unmissable, which is what these assertions pin. The wording stays abstract: no
task name, constant, sequence or filename appears here.
"""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.system_prompt_remind import (
    SystemPromptRemindGuard,
    _REMIND_MESSAGE,
    _missing_labels,
    _review_labels_present,
)

OTHER_SUBSLOTS = (
    "UNSET-FIELDS: none; UNIT-FORMULA: none; LIST-FLOOR: none; "
    "READBACK-STATE: none; ABSENCE-AS-EVIDENCE: none; DEFINING-FIELD: none; "
    "OBSERVED-VS-CONSUMED: none"
)


def _shell(w2):
    return [
        "W0 unverified layer: x",
        "W1 settled-too-early pattern: x",
        w2,
        "W3 cmd=id -> output=uid=0 verdict=pass bar=<1",
        "W4 instance: x",
        "RISK SWEEP: r1 -> e1 -> side",
    ]


def _pre(g, args):
    return g.check_pre(
        GuardContext(tool_name="plan_update", tool_args=args, tool_result="ok")
    )


def test_beat_asks_for_every_stated_constraint():
    low = _REMIND_MESSAGE.lower()
    assert "stated-vs-shipped" in low, "the beat lost the sub-slot"
    assert "every constraint i stated" in low, (
        "the beat must invoice EVERY stated constraint; '<a constraint I stated>' "
        "lets the agent reconcile a self-chosen subset and go green"
    )
    # The omission clause must be present too: naming the set is not enough,
    # leaving one out has to be stated as leaving the slot unfilled.
    assert "left off" in low, (
        "the beat must say that omitting a stated constraint leaves the slot "
        "unfilled, or an omission reads as a complete answer"
    )


def test_refusal_message_asks_for_every_stated_constraint():
    g = SystemPromptRemindGuard()
    v = _pre(g, {"action": "step_done", "step_id": 3})
    assert v is not None and v.action == "block"  # first submission: the beat
    w2 = f"W2 trap inventory walk: x; {OTHER_SUBSLOTS}"
    args = {"action": "step_done", "step_id": 3, "verification": _shell(w2)}
    v = _pre(g, args)
    assert v is not None and v.action == "block", (
        "a W2 with no STATED-VS-SHIPPED was accepted — decoration"
    )
    assert "EVERY constraint" in v.message, (
        "the refusal message must carry the coverage requirement, not only the "
        "Beat's first delivery of it"
    )


def test_subslot_accepts_the_noop_and_a_filled_answer():
    g = SystemPromptRemindGuard()
    for sid in (4, 5):
        _pre(g, {"action": "step_done", "step_id": sid})  # the beat, once per step

    explicit_none = (
        f"W2 trap inventory walk: x; {OTHER_SUBSLOTS}; "
        "STATED-VS-SHIPPED: no constraint I stated"
    )
    args = {"action": "step_done", "step_id": 4, "verification": _shell(explicit_none)}
    assert _pre(g, args) is None, "an explicit 'no constraint I stated' must pass"

    filled = (
        f"W2 trap inventory walk: x; {OTHER_SUBSLOTS}; "
        "STATED-VS-SHIPPED: every constraint I stated -> delivered bytes, "
        "measured -> agree"
    )
    args = {"action": "step_done", "step_id": 5, "verification": _shell(filled)}
    assert _pre(g, args) is None, "a filled list must pass"


def test_subslot_is_not_decoration():
    """Ablation: filling every OTHER sub-slot and omitting this one is refused."""
    g = SystemPromptRemindGuard()
    _pre(g, {"action": "step_done", "step_id": 6})
    w2 = f"W2 trap inventory walk: x; {OTHER_SUBSLOTS}"
    args = {"action": "step_done", "step_id": 6, "verification": _shell(w2)}
    v = _pre(g, args)
    assert v is not None and v.action == "block"
    cov = _review_labels_present(args)
    assert cov.has_stated_shipped_subslot is False
    assert any(
        "STATED-VS-SHIPPED" in m for m in _missing_labels(cov)
    ), "the refusal must name the missing sub-slot"
