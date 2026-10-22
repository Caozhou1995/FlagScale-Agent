"""Regression tests: a VERIFIED acceptance act must latch even when the SAME
step_done call is blocked on an unrelated FIXED-FIELD declaration.

DEFECT (measured, session eea7fbdc, 2026-10-21):
The step_done route evaluated the fixed-field declarations first and did
`if bad is not None: return bad` (verification.py:2361-2367) BEFORE the block
that latches `_acceptance_observed` / `_oracle_exercised_seen`
(verification.py:2368-2400). Consequence: any round whose step_done also
missed a cheap formal field (`convention`, `assumptions`, `deliverable_reading`)
or the Mode-1 evidence list discarded an acceptance act whose `observed` text
had ALREADY passed the non-forgeable corpus-membership check. The agent then
re-ran the probe and re-submitted, and the latch still never set — the
observable signature is many correct probe runs and a completion boundary that
never releases.

The two questions are INDEPENDENT:
  * is this claim formally complete?      -> the declarations must be fixed
  * did an observation really happen?     -> corpus-checked evidence is durable
Blocking on the first must not destroy the answer to the second.
"""
import tempfile
import shutil

from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.plan import TaskPlan

CORPUS = "root@host:/app# echo probe-ok\nprobe-ok\n"


def _plan_one_step():
    d = tempfile.mkdtemp()
    plan = TaskPlan(d)
    plan.create("T", [{"title": "s1", "acceptance": ["a1"]}])
    return d, plan


def _guard(plan, task_text="Read /app/x and emit /app/y exactly."):
    g = VerificationGuard(plan=plan, task_text_fn=lambda: task_text)
    # Skip the one-shot premise re-check so the test targets the field/act split.
    g._step_done_recheck_reminded = True
    return g


def _ctx(tool_args):
    """A step_done ctx carrying a REAL corpus (membership check is live)."""
    return GuardContext(
        tool_name="plan_update",
        tool_args=tool_args,
        tool_result=None,
        assistant_text="",
        override_reason="",
        llm_responded=True,
        messages=[{"role": "tool", "content": CORPUS}],
    )


def _step_done_args(**extra):
    args = {
        "action": "step_done",
        "step_id": 1,
        "verification": ["probe-ok observed"],
        "acceptance_act": {"command": "echo probe-ok", "observed": "probe-ok"},
    }
    args.update(extra)
    return args


def test_verified_acceptance_act_latches_despite_missing_convention():
    """The act is corpus-verified; the missing `convention` field must still
    block the call, but must NOT discard the verified evidence."""
    d, plan = _plan_one_step()
    try:
        g = _guard(plan)
        v = g.check_pre(_ctx(_step_done_args()))
        assert v is not None, "the missing convention field must still block"
        assert v.reason == "convention_declaration_missing", v.reason
        assert g._acceptance_observed is True, (
            "a corpus-verified acceptance act must latch even when the same "
            "call is blocked on a formal field")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_unverified_acceptance_act_does_not_latch():
    """Discriminating contrast: an act whose `observed` is NOT in the corpus is
    a claim, not a record — it must not latch, blocked or not."""
    d, plan = _plan_one_step()
    try:
        g = _guard(plan)
        args = _step_done_args()
        args["acceptance_act"] = {"command": "echo nope",
                                  "observed": "never-produced-string"}
        g.check_pre(_ctx(args))
        assert g._acceptance_observed is False, (
            "an observation absent from the corpus must never latch")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_verified_acceptance_act_still_latches_when_declarations_complete():
    """Preserve the passing path: complete declarations latch AND release."""
    d, plan = _plan_one_step()
    try:
        g = _guard(plan)
        args = _step_done_args(
            convention={"name": "none"},
            deliverable_reading={"deliverable": "none"},
            assumptions={"unverified": "none"},
        )
        v = g.check_pre(_ctx(args))
        assert v is None, v.reason if v else None
        assert g._acceptance_observed is True
    finally:
        shutil.rmtree(d, ignore_errors=True)
