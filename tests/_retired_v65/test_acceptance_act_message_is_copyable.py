"""The two step_done messages must SHOW a filled-in, copyable example.

Measured failure this prevents: in a live trial the agent was told about the
`acceptance_act` field and then spent ~12 minutes guessing its exact shape --
whether `command` was parsed, whether whitespace mattered, whether it should
join commands, whether `observed` had to be verbatim. The guard already knew
the answers; the message just did not say them. Every ambiguity that costs a
turn is a defect in the message, not in the agent.

WHY THESE TESTS RENDER THROUGH THE GUARD AND NOT THROUGH A LOCAL HELPER.
A first version of this file called `CONSTANT.replace("{field}", ...)` directly.
The red-phase sweep showed why that is worthless: mutating the guard's real
renderer back to `str.format` left the suite GREEN, because the test's own
helper did the substitution itself. A test that re-implements the code under
test measures its own arithmetic. So every assertion below obtains the message
by driving `VerificationGuard.check_pre` on a real plan_update call and reading
`verdict.message` -- the exact bytes an agent would receive.
"""
import json
import re

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


class _Judge:
    def constraints(self, text):
        return []

    def deferred_requirements(self, text):
        return []

    def constraint_coverage(self, quotes, stated, text):
        return set()


def _guard():
    g = VerificationGuard()
    g._task_text = lambda: TASK
    g._step_done_recheck_reminded = True
    return g


def _ctx(args):
    return GuardContext(
        tool_name="plan_update",
        tool_args=dict(args),
        tool_result=CORPUS,
        judge=_Judge(),
        messages=[],
        override_reason="ran the design script and read the Tm table",
    )


def _message_missing_act():
    """The message an agent gets when it reports done with NO acceptance_act."""
    args = {"action": "step_done", "step_id": 1,
            "verification": ["ran design.py"], **FIXED}
    v = _guard().check_pre(_ctx(args))
    assert v is not None and "acceptance_act" in v.message, v
    return v.message


def _message_ungrounded_act():
    """The message an agent gets when it DID submit an act whose `observed` text
    is not in the recorded corpus -- the exact live-trial confusion.

    The corpus must be non-empty for this to be observable at all: `_observed`
    returns None when there is no message window AND no full log, and a None
    corpus makes the evidence check FAIL OPEN (nothing to compare against). A
    fixture with `messages=[]` therefore silently satisfies the gate and the
    ungrounded branch is never reached -- which is what this test caught.
    """
    act = {"command": "python3 design.py", "observed": "made up text not in corpus"}
    args = {"action": "step_done", "step_id": 1,
            "verification": ["ran design.py"], "acceptance_act": act, **FIXED}
    ctx = GuardContext(
        tool_name="plan_update",
        tool_args=args,
        tool_result=CORPUS,
        judge=_Judge(),
        messages=[{"role": "user", "content": CORPUS}],
        override_reason="ran the design script and read the Tm table",
    )
    g = _guard()
    g._full_log_fn = lambda: CORPUS
    v = g.check_pre(ctx)
    assert v is not None, "an ungrounded act must block when a corpus exists"
    return v.message


def _example_block(body):
    return body.split("COPY THIS SHAPE")[1].split("HOW EXACT IS")[0]


# ------------------------------------------------------------------ shape

def test_no_placeholder_angle_brackets_in_example():
    """A copyable example cannot contain <...> slots."""
    leftovers = re.findall(r"<[^<>\n]{1,80}>", _example_block(_message_missing_act()))
    assert not leftovers, f"example still has placeholders: {leftovers}"


def test_example_is_working_json():
    """The example the agent copies must be literal JSON it can paste."""
    block = _example_block(_message_missing_act())
    block = block[block.index("acceptance_act = {") + len("acceptance_act = "):]
    depth, end = 0, None
    for i, ch in enumerate(block):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end is not None, "example object is not balanced"
    parsed = json.loads(block[:end].replace("'", '"'))
    assert set(parsed) == {"command", "observed", "constraints"}
    assert parsed["constraints"]
    assert all(set(r) == {"quote", "command", "observed"} for r in parsed["constraints"])


def test_four_exactness_rules_are_stated():
    """Each ambiguity that cost turns in the live trial must be answered here."""
    body = _message_missing_act()
    for phrase in [
        "FREE TEXT",                              # command/observed are not parsed
        "Whitespace and letter case are ignored",  # what the matcher normalizes
        "paraphrase",                             # what it does NOT normalize
        "do not retype it from memory",           # copy, don't recall
    ]:
        assert phrase in body, f"exactness rule missing: {phrase!r}"


def test_unobservable_form_is_also_filled_in():
    """The fallback is the branch an agent reaches when stuck -- it must not be
    the vaguest thing in the message."""
    seg = _message_missing_act().split("UNOBSERVABLE form")[1].split("Cover the requirements")[0]
    assert '"unobservable"' in seg
    assert not re.findall(r"<[^<>\n]{1,80}>", seg), "unobservable example has slots"
    assert "ls /grader" in seg


# ------------------------------------------------------- renderer integrity

def test_field_substitution_really_happened():
    """The examples contain real JSON braces, so the renderer must not be
    str.format: that would raise KeyError at report time, or -- worse -- leave a
    raw `{field}` in front of the agent. Asserting on the RENDERED message is the
    only form of this test that can fail when the renderer is wrong."""
    for body in (_message_missing_act(), _message_ungrounded_act()):
        assert "{field}" not in body
        assert "`acceptance_act`" in body


def test_batch_route_label_reaches_the_same_example():
    """The batch route must get the SAME copyable example, with its own field
    label -- the batch hole let this message go missing entirely."""
    item = {"step_id": 1, "status": "done", "verification": ["ran design.py"]}
    v = _guard().check_pre(_ctx({"action": "batch", "updates": [item], **FIXED}))
    assert v is not None
    assert "updates[].acceptance_act" in v.message
    assert "COPY THIS SHAPE" in v.message, (
        "the batch route must carry the worked example too, not a bare demand"
    )


# --------------------------------------------------- the lie that cost time

def test_ungrounded_message_says_shape_is_fine_and_only_text_is_wrong():
    """The live trial's exact confusion: 'guard says missing but I did pass it'."""
    body = _message_ungrounded_act()
    assert "WAS received" in body
    assert "shape is fine" in body
    assert "Do not re-send the field and do not change its shape" in body
    # and it must NOT reuse the "missing" wording that sent the agent into a loop
    assert "with no `acceptance_act`" not in body
