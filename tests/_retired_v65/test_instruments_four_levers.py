"""Four scientific instruments as FORCED declarations — gates + data layer.

WHY THESE TESTS EXIST (measured, two full TB2.1 runs): every mechanism that could
be IGNORED failed to change behaviour; every mechanism that BLOCKED the next
action changed it. The negative-doctrine list reached 90% uptake with
task-specific content and the agent still churned to death. So each of these four
instruments is tested at its REFUSAL, not at its advisory text — a test that only
checks the message exists would pass on the exact build that failed in production.

The four instruments, and the measured loss each closes:

  CALIBRATION      my own Tm probe returned a wrong value because it searched one
                   template record; only a re-run against a KNOWN answer caught it.
  DIFFERENTIAL     dna-assembly measured Tm on the annealing region (70.98, PASS
                   by its own definition) while the grader's construction of the
                   same quantity included the overhang match (73.39 > 72, FAIL).
                   It never computed the second number.
  PRE-REGISTRATION a prediction on this harness was left unsettled after its run
                   was stopped; nothing surfaced it.
  NEGATIVE CONTROL an uncalibrated check has only ever been seen agreeing with
                   itself, so its PASS carries no information.

Each gate is tested in BOTH directions: it fires on the shape that must be
refused, and it stays silent on the shape that must be allowed. The second half
is not decoration — a gate that fires on everything gets routed around wholesale,
which is the failure the scoping (named convention only; no_known_case only) is
designed to avoid.
"""

import tempfile

from flagscale_agent.react.instruments import (
    check_calibration,
    check_differential,
    check_negative_control,
    calibration_needs_negative_control,
    negative_control_fired,
    differential_satisfied,
    check_bound_margin,
    bound_margin_decided,
)
from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.tools.plan_create import PlanCreateTool
from flagscale_agent.react.tools.plan_update import PlanUpdateTool


ORACLE = {
    "oracle": "oligotm Tm of the grader-defined region",
    "premise_source": "provided",
    "first_exercise": "oligotm -mv 50 -dv 2 -n 0.2 -d 50 <seq>",
}


def _mk_plan(**kw):
    d = tempfile.mkdtemp()
    p = TaskPlan(d)
    p.create(title="t", steps=["a", "b"], **kw)
    return p


# ---------------------------------------------------------------- CALIBRATION
def test_calibration_missing_is_refused_by_plan_create():
    """A plan with no calibration declaration must not be created at all."""
    d = tempfile.mkdtemp()
    p = TaskPlan(d)
    tool = PlanCreateTool(p, "s1")
    out = tool.execute(title="x", steps=["a"], oracle=dict(ORACLE))
    assert out.startswith("ERROR:")
    assert "calibration" in out
    # the refusal must not leave a half-created plan behind
    assert p.get_active() is None


def test_calibration_contradicted_is_refused():
    """matches="no" means the instrument disagreed with a KNOWN answer -> broken."""
    d = tempfile.mkdtemp()
    p = TaskPlan(d)
    tool = PlanCreateTool(p, "s1")
    out = tool.execute(
        title="x", steps=["a"], oracle=dict(ORACLE),
        calibration={"instrument": "i", "known_case": "k", "observed": "v",
                     "matches": "no"},
    )
    assert out.startswith("ERROR:")
    assert p.get_active() is None


def test_calibration_no_known_case_is_accepted_and_recorded():
    """An honest 'no known case' is legal — it changes obligations, not legality."""
    d = tempfile.mkdtemp()
    p = TaskPlan(d)
    tool = PlanCreateTool(p, "s1")
    out = tool.execute(
        title="x", steps=["a"], oracle=dict(ORACLE),
        calibration={"instrument": "oligotm on annealing region",
                     "matches": "no_known_case",
                     "instrument_input": {
                         "expression": "tm = run(seq[start:end])",
                         "source": "the span the task text fixes",
                         "mismatch_checked": "second span read 61.8 vs 67.4",
                     }},
    )
    assert out.startswith("Plan created.")
    stored = p.get_active()["calibration"]
    assert stored["matches"] == "no_known_case"
    assert calibration_needs_negative_control(stored) is True


def test_calibration_blank_case_with_yes_is_refused():
    """'yes' with an empty known_case is a declaration that claims nothing."""
    verdict = check_calibration({"instrument": "p", "known_case": "  ",
                                 "observed": "v", "matches": "yes"})
    assert verdict is not None and verdict[0] == "calibration_blank_fields"


# ---------------------------------------------------------------- DIFFERENTIAL
ARBITER_ORACLE = {
    "oracle": "oligotm Tm of the grader-defined region",
    "premise_source": "provided",
    "first_exercise": "oligotm -mv 50 -dv 2 -n 0.2 -d 50 <seq>",
    "named_arbiter": "oligotm",
}


def _guard(plan):
    return VerificationGuard(plan=plan)


def _mk_plan_with_arbiter(**kw):
    """Plan whose oracle NAMES an arbiter -- the precondition for instrument 2.

    The differential gate is scoped to this case on purpose: with no named
    decider there is no rival definition of the quantity, so two paths would be
    ceremony. Tests that exercise the gate must therefore set up the condition
    the gate is about, not work around it.
    """
    kw.setdefault("oracle", dict(ARBITER_ORACLE))
    return _mk_plan(**kw)


NAMED_CONV = {"name": "Tm scope", "example": "annealing only",
              "delivered": "70.98", "conflict": "no"}


def test_differential_missing_on_named_convention_is_refused():
    """A step resting on a quantity the agent DEFINED must recompute it."""
    g = _guard(_mk_plan_with_arbiter(
        calibration={"instrument": "i", "matches": "no_known_case"}))
    v = g._instrument_obligation(NAMED_CONV, None, None)
    assert v is not None
    assert v.reason == "differential_missing"
    # the message must name the convention so the agent knows WHICH quantity
    assert "Tm scope" in v.message


def test_differential_unresolved_is_refused():
    """Paths that disagree beyond tolerance are the finding, not a formality."""
    g = _guard(_mk_plan_with_arbiter(
        calibration={"instrument": "i", "matches": "no_known_case"}))
    v = g._instrument_obligation(
        NAMED_CONV,
        {"quantity": "Tm", "path_a": "annealing only = 70.98",
         "path_b": "with overhang match = 73.39", "delta": "2.41",
         "within_tolerance": "no", "tolerance_basis": "task states none"},
        None,
    )
    assert v is not None and v.reason == "differential_unresolved"


def test_differential_resolved_passes():
    """Agreement between two independent paths must NOT block."""
    g = _guard(_mk_plan_with_arbiter(
        calibration={"instrument": "i", "matches": "no_known_case"}))
    v = g._instrument_obligation(
        NAMED_CONV,
        {"quantity": "Tm", "path_a": "a=70.98", "path_b": "b=70.97",
         "delta": "0.01", "within_tolerance": "yes",
         "tolerance_basis": "task states none"},
        {"check": "c", "must_fail_input": "m", "observed": "FAILED 80 > 72"},
    )
    assert v is None


def test_differential_not_required_when_convention_is_none():
    """{"name": "none"} is an owned claim that no derived convention governs the
    step — forcing a differential there is the 'abuse of guards' failure."""
    g = _guard(_mk_plan(calibration={"instrument": "i", "matches": "no_known_case"}))
    v = g._instrument_obligation(
        {"name": "none"}, None,
        {"check": "c", "must_fail_input": "m", "observed": "FAILED 80 > 72"},
    )
    assert v is None


def test_differential_not_required_without_a_named_arbiter():
    """SCOPING TEST — pins the second half of the trigger.

    A named convention alone must NOT demand a differential: with no tool named
    by the task as the decider there is no rival definition of the quantity, and
    forcing two paths there is ceremony (the 'abuse of guards' failure). The
    measured dna-assembly loss had a named arbiter, and that is the shape the
    gate is for.
    """
    g = _guard(_mk_plan(calibration={
        "instrument": "probe", "known_case": "k", "observed": "k",
        "matches": "yes"}))
    assert g._instrument_obligation(NAMED_CONV, None, None) is None


def test_differential_cannot_compute_is_an_accepted_gap():
    """An explicit 'cannot_compute' is recorded, not treated as disagreement."""
    ok = {"quantity": "q", "path_a": "a", "path_b": "cannot run here",
          "delta": "n/a", "within_tolerance": "cannot_compute",
          "tolerance_basis": "task states none"}
    assert differential_satisfied(ok) is True
    assert check_differential(ok) is None


# ---------------------------------------------------- BOUND MARGIN (instrument 5)
def test_bound_margin_tight_is_refused_by_the_real_guard():
    """THE MEASURED CASE through the REAL guard: value 44 vs bound 45, rival
    shift 4 (the grader's construction of the same quantity). The self-check
    said PASS; the arithmetic says UNDECIDED."""
    g = _guard(_mk_plan(calibration={"instrument": "i", "matches": "yes"}))
    bm = {"value": 44, "bound": 45, "direction": "max", "rival_shift": 4,
          "rival_definition": "annealing tract incl. the overhang match"}
    v = g._instrument_obligation(None, None, None, bound_margin=bm)
    assert v is not None and v.reason == "bound_margin_too_tight"
    # refusal echoes both numbers so the next attempt can self-correct
    assert "44" in v.message and "45" in v.message


def test_bound_margin_roomy_passes_the_real_guard():
    """The negative control for the gate: margin 15 > shift 4 must NOT block."""
    g = _guard(_mk_plan(calibration={"instrument": "i", "matches": "yes"}))
    bm = {"value": 30, "bound": 45, "direction": "max", "rival_shift": 4,
          "rival_definition": "same rival"}
    assert g._instrument_obligation(None, None, None, bound_margin=bm) is None


def test_bound_margin_absent_does_not_block():
    """Declaration creates the obligation: no declaration, no gate. A step that
    makes no bound claim has nothing to decide this way."""
    g = _guard(_mk_plan(calibration={"instrument": "i", "matches": "yes"}))
    assert g._instrument_obligation(None, None, None, bound_margin=None) is None


def test_bound_margin_missing_required_field_is_refused():
    g = _guard(_mk_plan(calibration={"instrument": "i", "matches": "yes"}))
    v = g._instrument_obligation(None, None, None,
                                 bound_margin={"value": 44, "bound": 45})
    assert v is not None and v.reason == "bound_margin_field_missing"


def test_bound_margin_exposed_in_tool_schema():
    """The tool the model calls must DECLARE the parameter -- the calibration
    outage (forced field, undeclared schema) is the failure this pins."""
    props = PlanUpdateTool.parameters["properties"]
    assert "bound_margin" in props
    assert props["bound_margin"]["type"] == "object"
    for f in ("value", "bound", "direction", "rival_shift", "rival_definition"):
        assert f in props["bound_margin"]["description"], f


# ------------------------------------------------------------ NEGATIVE CONTROL
def test_negative_control_required_when_calibration_had_no_known_case():
    g = _guard(_mk_plan(calibration={"instrument": "i", "matches": "no_known_case"}))
    v = g._instrument_obligation({"name": "none"}, None, None)
    assert v is not None and v.reason == "negative_control_missing"


def test_negative_control_that_stayed_green_is_refused():
    """A control that reports 'ok' proves the check cannot detect failure."""
    assert negative_control_fired({"observed": "ok"}) is False
    v = check_negative_control({"check": "range", "must_fail_input": "Tm 80",
                                "observed": "ok"})
    assert v is not None and v[0] == "negative_control_did_not_fire"


def test_negative_control_that_fired_passes():
    assert negative_control_fired(
        {"observed": "FAILED (expected Tm <= 72, got 80)"}) is True
    assert check_negative_control({"check": "range", "must_fail_input": "Tm 80",
                                   "observed": "rejected: 80 > 72"}) is None


def test_negative_control_not_required_when_calibration_succeeded():
    """A calibrated instrument already has its evidence — no second hoop."""
    g = _guard(_mk_plan(calibration={
        "instrument": "probe", "known_case": "primer X = 67.46",
        "observed": "67.46", "matches": "yes"}))
    assert g._instrument_obligation({"name": "none"}, None, None) is None


# ---------------------------------------------------------- PRE-REGISTRATION
def test_predict_records_and_settle_clears():
    p = _mk_plan(calibration={"instrument": "i", "matches": "no_known_case"})
    tool = PlanUpdateTool(p)
    assert tool.execute(action="predict").startswith("ERROR:")
    out = tool.execute(action="predict",
                       prediction_statement="restart dna, oracle declared",
                       prediction_expected="non-empty")
    assert "Pre-registered" in out
    pending = p.unsettled_predictions()
    assert len(pending) == 1
    pid = pending[0]["id"]
    assert tool.execute(action="settle", prediction_id=pid,
                        outcome="maybe").startswith("ERROR:")
    assert tool.execute(action="settle", prediction_id=pid, outcome="hit",
                        prediction_observed="oracle non-empty").startswith("Settled")
    assert p.unsettled_predictions() == []


def test_plan_complete_refuses_with_unsettled_prediction():
    """Instrument 3's teeth: the plan cannot close while a prediction dangles."""
    p = _mk_plan(calibration={"instrument": "i", "matches": "no_known_case"},
                 oracle=dict(ORACLE))
    g = _guard(p)
    ctx = GuardContext(tool_name="plan_update",
                       tool_args={"action": "complete"}, override_reason="")
    p.predict(statement="expect 42", expected="42")
    reasons = [v.reason for v in g._plan_complete_obligations(ctx)]
    assert "plan_complete_unsettled_predictions" in reasons
    p.settle(p.unsettled_predictions()[0]["id"], "miss", "got 37")
    reasons = [v.reason for v in g._plan_complete_obligations(ctx)]
    assert "plan_complete_unsettled_predictions" not in reasons


def test_unsettled_prediction_blocks_even_with_no_oracle_declared():
    """PLACEMENT TEST — pins where the check lives.

    `_plan_complete_obligations` returns early when no oracle was declared, so an
    obligation added AFTER that return would be invisible for every plan without
    an oracle (exactly the shape that failed in production: oracle: {}). This test
    is the only one that can tell the two insertion points apart.
    """
    d = tempfile.mkdtemp()
    p = TaskPlan(d)
    p.create(title="t", steps=["a"],
             calibration={"instrument": "i", "matches": "no_known_case"})  # no oracle
    g = _guard(p)
    ctx = GuardContext(tool_name="plan_update",
                       tool_args={"action": "complete"}, override_reason="")
    p.predict(statement="predict a number before the run", expected="42")
    reasons = [v.reason for v in g._plan_complete_obligations(ctx)]
    assert reasons == ["plan_complete_unsettled_predictions"]


def test_not_run_is_a_legal_settlement():
    """A run cut short by the wall must be reportable honestly, not dropped."""
    p = _mk_plan(calibration={"instrument": "i", "matches": "no_known_case"})
    tool = PlanUpdateTool(p)
    tool.execute(action="predict", prediction_statement="expect the run to finish")
    pid = p.unsettled_predictions()[0]["id"]
    assert tool.execute(action="settle", prediction_id=pid,
                        outcome="not_run").startswith("Settled")
    assert p.unsettled_predictions() == []
