"""Instrument-input construction gate (instrument 1b).

`instrument` binds WHICH check; this binds WHAT is fed into it. The measured
losses were all input constructions, never broken instruments: the quantity was
read over a span the agent chose, while the grader's construction of the same
quantity crossed the stated bound.

Two legal shapes, both a real declaration:
  * construction form: expression + source + mismatch_checked (all non-blank)
  * not-applicable form: not_applicable stating why no input is constructed
Forbidden: absent, blank, a bare string, unknown/n-a placeholders, or any of
the three construction keys left blank or vague.

The rejection message ships a FILLED-IN example (copy-pasteable, per the
2026-10-27 user directive: give the finished shape, do not make the agent
guess the format), and the example carries NO task-specific tokens (leak gate).
"""
import re

from flagscale_agent.react.instruments import check_instrument_input
from flagscale_agent.react.tools.plan_create import PlanCreateTool


FULL = {
    "expression": "reading = measure(data[fs + 4 : fs + 4 + bind_len])",
    "source": "task text fixes the span start after the 4-position prefix",
    "mismatch_checked": "second construction with prefix INCLUDED read 61.8 vs 67.4",
}



def test_missing_field_is_refused():
    reason, _msg = check_instrument_input({"instrument": "x", "matches": "yes"})
    assert reason == "calibration_instrument_input_missing"


def test_string_form_is_refused_not_coerced():
    reason, msg = check_instrument_input(
        {"instrument_input": "I will measure the right span"})
    assert reason == "calibration_instrument_input_not_object"
    assert "must be an object" in msg


def test_full_declaration_passes():
    assert check_instrument_input({"instrument_input": dict(FULL)}) is None


def test_not_applicable_with_reason_passes():
    r = check_instrument_input({"instrument_input": {
        "not_applicable": "the check runs against delivered.bin, the path the task names"}})
    assert r is None


def test_na_placeholder_is_refused():
    reason, _ = check_instrument_input(
        {"instrument_input": {"not_applicable": "n/a"}})
    assert reason == "calibration_instrument_input_vague"


def test_missing_keys_named_in_message():
    reason, msg = check_instrument_input(
        {"instrument_input": {"expression": "x[a:b]"}})
    assert reason == "calibration_instrument_input_incomplete"
    for key in ("source", "mismatch_checked"):
        assert key in msg, f"message must name the missing key {key}"


def test_vague_values_refused():
    reason, msg = check_instrument_input({"instrument_input": {
        "expression": "unknown",
        "source": FULL["source"],
        "mismatch_checked": FULL["mismatch_checked"],
    }})
    assert reason == "calibration_instrument_input_vague"
    assert "unknown" in msg


def test_rejection_message_carries_filled_example_and_no_task_tokens():
    """User directive 2026-10-27: give the finished shape, never slots. And the
    leak gate applies to agent-facing text -- the example must stay neutral."""
    _reason, msg = check_instrument_input({})
    assert "COPY THIS SHAPE" in msg
    assert "expression" in msg and "mismatch_checked" in msg
    example = msg.split("COPY THIS SHAPE")[1].split("What each key is for")[0]
    assert not re.findall(r"<[^<>\n]{1,80}>", example)
    for token in ("oligotm", "primer", "mteb", "anneal"):
        assert token not in msg.lower()


# ------------------------------------------------------------ plan_create gate

def test_plan_create_refuses_without_instrument_input():
    """The refusal is the mechanism: a plan whose calibration lacks the input
    construction is refused AT CREATION, with the filled example in the reply."""
    import tempfile
    from flagscale_agent.react.plan import TaskPlan
    d = tempfile.mkdtemp()
    tool = PlanCreateTool(TaskPlan(d), "t-inp")
    from tests._calibration_fixture import DEFAULT_CALIBRATION
    cal = dict(DEFAULT_CALIBRATION)
    cal.pop("instrument_input", None)
    assert "instrument_input" not in cal
    out = tool.execute(
        title="t", steps=[{"title": "s", "acceptance": ["a"]}],
        oracle={"premise_source": "consumer",
                "oracle": "none: test-only plan",
                "first_exercise": "true"},
        calibration=dict(cal))
    assert out.startswith("ERROR:")
    assert "instrument_input" in out
    assert "COPY THIS SHAPE" in out


def test_plan_create_accepts_full_declaration():
    import tempfile
    from flagscale_agent.react.plan import TaskPlan
    d = tempfile.mkdtemp()
    tool = PlanCreateTool(TaskPlan(d), "t-inp-full")
    from tests._calibration_fixture import DEFAULT_CALIBRATION
    cal = dict(DEFAULT_CALIBRATION)
    cal["instrument_input"] = {
        "expression": "score = check(out_path)",
        "source": "task names out_path as the deliverable to check",
        "mismatch_checked": "recomputed from the rendered file: same value",
    }
    out = tool.execute(
        title="t", steps=[{"title": "s", "acceptance": ["a"]}],
        oracle={"premise_source": "consumer",
                "oracle": "none: test-only plan",
                "first_exercise": "true"},
        calibration=cal)
    assert not out.startswith("ERROR"), out[:200]
