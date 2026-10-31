"""When the act WAS received but its `observed` is not in the tool-result corpus,
the block must say THAT -- never "the field is missing".

Measured loss this closes (dna-assembly trial 19-21-11, final minutes):

  the agent submitted a well-formed `acceptance_act` whose `observed` was a
  paraphrase of what it had run. The guard rejected it with the "no
  `acceptance_act`" message. The agent's own thinking, verbatim: "The guard says
  acceptance_act missing but I did pass it. Possibly the field name must be
  inside... it says RECEIVED acceptance_act = {...} then says missing." It then
  guessed at exactness twice while the wall ran out.

The gate was RIGHT to reject -- corpus membership is the whole point, otherwise
the field is a free pass. The MESSAGE was wrong, and the message is what the
agent acts on. Two distinguishable states must produce two distinguishable
messages:

  * field absent / malformed      -> "you did not send a usable acceptance_act"
  * field present, text ungrounded -> "you sent it, this text is not in your
                                      recorded output, here is the longest real
                                      slice of YOURS that is"

`_evidence_observed_diagnosis` already existed for exactly this -- five other
gates append it. The acceptance obligation was the one gate that did not.
"""

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.verification import VerificationGuard


TASK = (
    "Design Golden Gate primers. Each primer must be at least 18 nt, the Tm of "
    "each annealing region must be between 58 and 72 degrees C, and the primers "
    "must amplify the insert with the specified overhangs."
)

# The ONLY text the harness recorded for this session.
CORPUS = "13.2\n18nt ok\nTM RANGE 58-72\nviolations=none\n"

FIXED = {
    "convention": {"name": "none"},
    "deliverable_reading": {"deliverable": "none"},
    "assumptions": {"unverified": "none"},
}


class _Judge:
    def constraints(self, text):
        return [
            "primers must amplify the insert with the specified overhangs",
            "each primer must be at least 18 nt",
            "Tm of each annealing region must be between 58 and 72 degrees C",
        ]

    def deferred_requirements(self, text):
        return []

    def constraint_coverage(self, quotes, stated, text):
        return set(range(len(stated)))


def _guard():
    g = VerificationGuard()
    g._task_text = lambda: TASK
    g._step_done_recheck_reminded = True
    return g


def _ctx(act):
    # `messages` must carry the recorded tool-result text: `_observed()` returns
    # None when the message window and the full log are both empty, and a None
    # corpus makes the grounding check FAIL OPEN -- which would let every
    # paraphrase through and make these tests measure nothing.
    return GuardContext(
        tool_name="plan_update",
        tool_args={"action": "step_done", "step_id": 1,
                   "verification": ["ran the audit and read its verdict"],
                   "acceptance_act": act, **FIXED},
        tool_result=CORPUS,
        judge=_Judge(),
        messages=[{"role": "user",
                   "content": [{"type": "tool_result", "content": CORPUS}]}],
        override_reason="read the audit verdict in the recorded tool result",
    )


PARAPHRASED = {
    "command": "python3 audit.py",
    # Real work happened; this text is a summary of it, not a slice of it.
    "observed": "all 13 clauses passed, Tm values within the 58-72 band",
    "constraints": [],
}


def test_paraphrased_observed_is_rejected():
    """Corpus membership stays mandatory -- the fix is the message, not the bar."""
    v = _guard().check_pre(_ctx(PARAPHRASED))
    assert v is not None, (
        "a paraphrased `observed` must not count as an acceptance record; "
        "otherwise the field is a free pass"
    )


def test_message_says_received_not_missing():
    """THE defect: telling an agent its field is absent when it is present sends
    it into a blind loop re-adding a field that is already there."""
    v = _guard().check_pre(_ctx(PARAPHRASED))
    assert "WAS received" in v.message, (
        "the block must acknowledge the field arrived -- an agent that reads "
        "'missing' about a field it just sent can only guess"
    )
    assert "no `acceptance_act`" not in v.message, (
        "must not tell the agent the field is missing when it was received"
    )


def test_message_names_the_fix_as_a_paste():
    """The actionable half: show the longest real slice of the agent's OWN text
    that does occur in the corpus, so the repair is a copy, not a re-guess."""
    v = _guard().check_pre(_ctx(PARAPHRASED))
    diagnostic = ("how to fix" in v.message
                  or "does not occur" in v.message
                  or "No tool-result text" in v.message)
    assert diagnostic, (
        "an ungrounded `observed` is unactionable without the diagnosis that "
        "says whether a real slice exists and what it is"
    )


def test_grounded_observed_still_passes():
    """Negative control in the other direction: the fix must not start rejecting
    a verbatim observation."""
    grounded = {
        "command": "python3 audit.py",
        "observed": "TM RANGE 58-72",
        "constraints": [
            {"quote": q, "command": "python3 audit.py", "observed": obs}
            for q, obs in zip(_Judge().constraints(TASK),
                              ["13.2", "18nt ok", "TM RANGE 58-72"])
        ],
    }
    assert _guard().check_pre(_ctx(grounded)) is None


def test_absent_field_still_gets_the_missing_message():
    """The two states must stay distinguishable in BOTH directions: a genuinely
    absent field keeps the tutorial message that teaches the shape."""
    ctx = GuardContext(
        tool_name="plan_update",
        tool_args={"action": "step_done", "step_id": 1,
                   "verification": ["ran the audit"], **FIXED},
        tool_result=CORPUS, judge=_Judge(),
        messages=[{"role": "user",
                   "content": [{"type": "tool_result", "content": CORPUS}]}],
        override_reason="read the audit verdict in the recorded tool result",
    )
    v = _guard().check_pre(ctx)
    assert v is not None
    assert "no `acceptance_act`" in v.message
    assert "WAS received" not in v.message
