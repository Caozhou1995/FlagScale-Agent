"""The acceptance-observation obligation must fire on EVERY route to "done".

Measured escape this closes (dna-assembly trial, 2664 log lines):

  * the agent marked every step done through `plan_update(action="batch")`, never
    a single `step_done`;
  * that batch call reached the fixed-field list carrying only the three earlier
    declarations, and its rejection message named exactly those three:
    convention_declaration_missing, deliverable_reading_missing,
    assumptions_declaration_missing;
  * `acceptance_act` appeared ZERO times anywhere in the trial log, so the
    requirement-coverage sweep never ran on any route.

The defect class is generic and had already cost this harness twice before
(render-into-a-string != string-consumed; an affordance hint rendered only in the
empty-list branch). The structural fix is the one under test here: the obligation
is a PURE FUNCTION of the act it is handed, so "which routes are covered" is a
visible list of call sites instead of a hidden default buried in the reader.

These tests are deliberately parameterised over the ROUTE, not the act: an act
that satisfies the obligation must pass on BOTH routes, and an act that is absent
must block on BOTH. A test that only exercises `step_done` is exactly the test
that let the batch hole ship.
"""

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.verification import VerificationGuard


TASK = (
    "Design Golden Gate primers for the given plasmid. The primers must "
    "amplify the insert with the specified overhangs, each primer must be at "
    "least 18 nt, and the Tm of each annealing region must be between 58 and 72 "
    "degrees C."
)

CORPUS = "13.2\n18nt ok\nTM RANGE 58-72\n"

FIXED = {
    "convention": {"name": "none"},
    "deliverable_reading": {"deliverable": "none"},
    "assumptions": {"unverified": "none"},
}

ACT = {
    "command": "python3 design.py",
    "observed": "TM RANGE 58-72",
    "constraints": [
        {"quote": "primers must amplify the insert with the specified overhangs",
         "command": "python3 design.py", "observed": "13.2"},
        {"quote": "each primer must be at least 18 nt",
         "command": "python3 design.py", "observed": "18nt ok"},
        {"quote": "Tm of each annealing region must be between 58 and 72 degrees C",
         "command": "python3 design.py", "observed": "TM RANGE 58-72"},
    ],
}


class _Judge:
    def constraints(self, text):
        return [c["quote"] for c in ACT["constraints"]]

    def deferred_requirements(self, text):
        return []

    def constraint_coverage(self, quotes, stated, text):
        return set(range(len(stated)))


def _guard():
    g = VerificationGuard()
    g._task_text = lambda: TASK
    g._step_done_recheck_reminded = True   # isolate the field under test
    return g


def _ctx(args):
    return GuardContext(
        tool_name="plan_update",
        tool_args=dict(args),
        tool_result=CORPUS,
        judge=_Judge(),
        messages=[],
        override_reason="observed the design table in the recorded tool result",
    )


def _batch(act=None, nest=True):
    """A batch call marking one step done, with `act` on the item (or not at all)."""
    item = {"step_id": 1, "status": "done",
            "verification": ["ran design.py and read the Tm table"]}
    if act is not None and nest:
        item["acceptance_act"] = act
    args = {"action": "batch", "updates": [item], **FIXED}
    if act is not None and not nest:
        args["acceptance_act"] = act      # top-level fallback -- must also work
    return args


# --------------------------------------------------- the missing-act case

def test_batch_done_without_acceptance_act_is_blocked():
    """THE measured hole: every step marked done, zero acts, sweep never ran."""
    v = _guard().check_pre(_ctx(_batch(act=None)))
    assert v is not None, (
        "a batch that marks a step done commits the same claim as a single "
        "step_done, so it must carry the same acceptance_act"
    )
    assert "acceptance_act" in v.message


def test_batch_block_names_the_nested_field():
    """The message must tell the agent WHERE to put it (on the item), not just that
    something is missing -- the batch route's natural spot is the item itself."""
    v = _guard().check_pre(_ctx(_batch(act=None)))
    assert "updates[].acceptance_act" in v.message, (
        "on the batch route the field belongs on the item; naming it is what "
        "makes the obligation actionable rather than a dead end"
    )


# ----------------------------------------------------- the satisfied case

@pytest.mark.parametrize("nest", [True, False])
def test_batch_done_with_act_passes_on_either_placement(nest):
    """Item-level first, top level as fallback -- the same precedence the three
    earlier declarations already use. Both placements must satisfy the bar, or an
    agent that writes the field at the top level is blocked for a formal reason
    while its observation was real."""
    assert _guard().check_pre(_ctx(_batch(act=ACT, nest=nest))) is None


# ------------------------------------------------- the single-route parity

def test_step_done_route_still_blocks_and_still_passes():
    """Parity guard: the refactor to a pure function must not change the original
    route in either direction."""
    g = _guard()
    blocked = g.check_pre(_ctx({
        "action": "step_done", "step_id": 1,
        "verification": ["ran design.py"], **FIXED}))
    assert blocked is not None and "acceptance_act" in blocked.message

    g2 = _guard()
    assert g2.check_pre(_ctx({
        "action": "step_done", "step_id": 1,
        "verification": ["ran design.py"],
        "acceptance_act": ACT, **FIXED})) is None
