"""The mechanical gates must TELL the model what it actually sent, and must
report every wrong required field in ONE round instead of one per round.

Design contract under test (JSON-Schema/Pydantic error-reporting convention):
  (a) a block names the rejected field path and echoes the value received;
  (b) independent required fields are reported together, so N wrong fields cost
      one round, not N;
  (c) a field that is valid produces NO accusation (no over-reporting), and
      all-valid produces no block at all.

Why this is a mechanism, not a message tweak: each round the model spends
re-discovering which field is wrong is a full model call. With 3 fixed required
fields and a short-circuit gate, a fully malformed step_done cost 3 rounds;
measured in a live trial the same class cost 42 blocks on one slug.
"""
import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.verification import VerificationGuard

BASE = {"action": "step_done", "step_id": 1,
        "verification": ["ran the thing, saw output"]}

OK = {
    "convention": {"name": "none"},
    "deliverable_reading": {"deliverable": "none"},
    "assumptions": {"unverified": "none"},
}
BAD_CONV = {"convention": {"nam": "my-convention"}}
BAD_DR = {"deliverable_reading": {"pat": "/tmp/x"}}
BAD_ASM = {"assumptions": {"verified": []}}


def _gate(args):
    """Drive the REAL dispatch path (check_pre on plan_update/step_done)."""
    g = VerificationGuard()
    ctx = GuardContext(
        tool_name="plan_update", tool_args=args, assistant_text="",
        llm_responded=True,
        # the first step_done is intercepted by the one-shot premise recheck;
        # any override reaches the field gates this test is about.
        override_reason="premises re-checked before this step")
    return g.check_pre(ctx)


@pytest.mark.parametrize("bad,field", [
    (BAD_CONV, "convention"),
    (BAD_DR, "deliverable_reading"),
    (BAD_ASM, "assumptions"),
])
def test_block_echoes_the_value_actually_received(bad, field):
    v = _gate(dict(BASE, **dict(OK, **bad)))
    assert v is not None and v.action == "block"
    assert "[RECEIVED] %s" % field in v.message
    # the sent payload appears verbatim, so the typo is visible without guessing
    sent = list(bad.values())[0]
    assert repr(sent) in v.message


def test_all_three_wrong_fields_reported_in_one_message():
    """The decisive case: three independent malformed fields, ONE round."""
    v = _gate(dict(BASE, **dict(BAD_CONV, **BAD_DR, **BAD_ASM)))
    assert v is not None and v.action == "block"
    for f in ("convention", "deliverable_reading", "assumptions"):
        assert "[RECEIVED] %s" % f in v.message, f
    assert "[REJECTED FIELDS]" in v.message


def test_valid_fields_are_not_accused():
    """No over-reporting: one bad field must not blame the two valid ones."""
    v = _gate(dict(BASE, **dict(OK, **BAD_CONV)))
    assert v is not None
    assert "[RECEIVED] convention" in v.message
    assert "[RECEIVED] deliverable_reading" not in v.message
    assert "[RECEIVED] assumptions" not in v.message


def test_all_valid_fields_do_not_block_at_the_shape_gate():
    """The echo/merge must not turn a valid call into a block."""
    v = _gate(dict(BASE, **OK))
    if v is not None:
        # further down the cascade may still block, but it must not be the
        # shape gate for these three fields
        assert "[RECEIVED] convention" not in v.message
        assert "[RECEIVED] deliverable_reading" not in v.message
        assert "[RECEIVED] assumptions" not in v.message
