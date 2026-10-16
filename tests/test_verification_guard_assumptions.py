"""Unit tests for the fixed `assumptions` field gate (structured, no regex).

The gate forces a step to SPLIT its reasoning: what it TESTED against the world
vs what it merely REASONED into its trace and treated as settled. It is a
shape + corpus check — it never judges whether a premise is truly non-affecting
(the harness has no oracle); it refuses to let an untested premise pass
SILENTLY. EVERY step_done must carry the field; "none" (or an empty list) is
itself an explicit owned claim. An AFFECTING premise must carry a real probe +
observed output (found in the session corpus); a NON-affecting one must say WHY.
There is no marker/tag/regex. See verification.py
`_check_unverified_assumptions_field`.
"""
import tempfile

from flagscale_agent.react.guard.verification import (
    VerificationGuard, _check_unverified_assumptions_field,
)
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.plan import TaskPlan


def _guard_with_step(acceptance=("A1",)):
    d = tempfile.mkdtemp()
    plan = TaskPlan(d)
    plan.create("Test", [{"title": "Step A", "acceptance": list(acceptance)}])
    guard = VerificationGuard(plan=plan)
    guard._step_done_recheck_reminded = True  # isolate from one-shot premise re-check
    return d, guard


def _ctx(assumptions, verification=("proof A1",)):
    return GuardContext(
        tool_name="plan_update",
        tool_args={
            "action": "step_done",
            "step_id": 1,
            "verification": list(verification),
            "deliverable_reading": {"deliverable": "none"},
            "convention": {"name": "none"},
            "assumptions": assumptions,
        },
    )


# --- the fixed field fires UNCONDITIONALLY ----------------------------------

def test_missing_field_blocks_even_on_plain_step():
    """A plain step with no assumptions tag still requires the field — the gate
    is not opt-in."""
    d, guard = _guard_with_step()
    try:
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 1,
                       "verification": ["proof A1"],
                       "deliverable_reading": {"deliverable": "none"},
                       "convention": {"name": "none"}},
        )
        v = guard.check_pre(ctx)
        assert v is not None and v.reason == "assumptions_declaration_missing"
    finally:
        pass


# --- the explicit not-applicable forms pass ---------------------------------

def test_string_none_passes():
    assert _check_unverified_assumptions_field("none") is None


def test_dict_none_passes():
    assert _check_unverified_assumptions_field({"unverified": "none"}) is None


def test_empty_list_passes():
    assert _check_unverified_assumptions_field({"unverified": []}) is None
    assert _check_unverified_assumptions_field([]) is None


# --- a declared AFFECTING premise needs a real, OBSERVED test ---------------

def test_affecting_without_probe_blocks():
    v = _check_unverified_assumptions_field(
        {"unverified": [{"claim": "field is 4 bytes", "affects_result": "yes"}]})
    assert v is not None and v.reason == "assumptions_affecting_without_test"


def test_affecting_with_observed_probe_passes():
    from flagscale_agent.react.guard.verification import _normalize_for_grounding as N
    corpus = N(" $ readelf -S a.out\n Section sh_addr 0x1060")
    v = _check_unverified_assumptions_field(
        {"unverified": [{"claim": "base", "affects_result": "yes",
                         "probe": "readelf -S a.out", "probe_output": "sh_addr 0x1060"}]},
        corpus)
    assert v is None


def test_affecting_with_UNOBSERVED_probe_blocks():
    """A claimed-but-never-run probe does not count — the output is not in the
    session corpus."""
    from flagscale_agent.react.guard.verification import _normalize_for_grounding as N
    corpus = N("some other tool output, nothing about sections")
    v = _check_unverified_assumptions_field(
        {"unverified": [{"claim": "base", "affects_result": "yes",
                         "probe": "readelf -S a.out", "probe_output": "sh_addr 0x1060"}]},
        corpus)
    assert v is not None and v.reason == "assumptions_affecting_without_test"


def test_affecting_fails_open_when_corpus_unavailable():
    """corpus None = source unavailable -> shape-checked, membership skipped."""
    v = _check_unverified_assumptions_field(
        {"unverified": [{"claim": "base", "affects_result": "yes",
                         "probe": "readelf -S a.out", "probe_output": "sh_addr 0x1060"}]},
        None)
    assert v is None


# --- a declared NON-affecting premise needs a REASON ------------------------

def test_nonaffecting_without_reason_blocks():
    v = _check_unverified_assumptions_field(
        {"unverified": [{"claim": "log line order", "affects_result": "no"}]})
    assert v is not None and v.reason == "assumptions_nonaffecting_without_reason"


def test_nonaffecting_with_reason_passes():
    v = _check_unverified_assumptions_field(
        {"unverified": [{"claim": "log line order", "affects_result": "no",
                         "reason": "the grader compares sets, not order"}]})
    assert v is None


# --- malformed shapes block -------------------------------------------------

def test_bad_affects_value_blocks():
    v = _check_unverified_assumptions_field(
        {"unverified": [{"claim": "x", "affects_result": "maybe", "reason": "r"}]})
    assert v is not None and v.reason == "assumptions_affects_result_missing"


def test_claim_missing_blocks():
    v = _check_unverified_assumptions_field(
        {"unverified": [{"affects_result": "no", "reason": "r"}]})
    assert v is not None and v.reason == "assumptions_item_malformed"


def test_dict_without_unverified_key_blocks():
    v = _check_unverified_assumptions_field({"foo": "bar"})
    assert v is not None and v.reason == "assumptions_unverified_key_missing"


# --- wired into the step_done cascade (through the guard, not just the helper) ---

def test_none_declaration_reaches_guard_pass():
    d, guard = _guard_with_step()
    try:
        assert guard.check_pre(_ctx({"unverified": "none"})) is None
    finally:
        import shutil
        shutil.rmtree(d)


def test_missing_reaches_guard_block_with_correct_reason():
    d, guard = _guard_with_step()
    try:
        ctx = _ctx({"unverified": "none"})
        del ctx.tool_args["assumptions"]
        v = guard.check_pre(ctx)
        assert v is not None and v.reason == "assumptions_declaration_missing"
    finally:
        import shutil
        shutil.rmtree(d)
