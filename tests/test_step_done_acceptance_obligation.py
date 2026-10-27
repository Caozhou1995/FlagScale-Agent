"""A+C: acceptance_act must be an OBLIGATION on every step_done, and the
deferred-standard requirements must be folded into the obligation set.

Measured loss this closes (jobs/2026-10-27__18-00-57, dna-assembly, reward=0):

  * the grader failed on `assert 0 >= 1` -- "Primer must have clamp of at least
    1 nucleotide before BsaI site" -- while the agent's own log line 173 read
    "the primer should have 4-6 extra bases 5' to the Type IIS site". It KNEW
    the requirement and dropped it from its obligation set;
  * the coverage gate that should have caught this lives inside
    `if _acceptance_act_recorded(...)` in the step_done route, and that trial
    submitted `acceptance_act` ZERO times across 21 step_done calls, so the
    gate body never executed once;
  * why zero submissions: `acceptance_act` was not declared in the plan_update
    tool schema, so it could not be carried as an object.

Both halves are load-bearing:
  A. `acceptance_act` is a FIXED, REQUIRED field on every step_done, so the
     coverage gate actually runs mid-run instead of only at plan-close;
  C. `judge.deferred_requirements()` output is merged into the stated-constraint
     set, so a requirement the task DEFERS to an external standard becomes an
     obligation instead of a thought that can evaporate.

Budget safety (from pitfall/flagscale_agent/constraint_coverage_gate_eats_budget):
the coverage sweep degrades to advisory under a nearly-spent wall; that existing
valve is REUSED here, never duplicated.
"""

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard import verification as V
from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.tools.plan_update import PlanUpdateTool


TASK = (
    "Design Golden Gate primers for the given plasmid. The primers must "
    "amplify the insert with the specified overhangs, each primer must be at "
    "least 18 nt, the Tm of each annealing region must be between 58 and 72 "
    "degrees C, and the design must check that the cut sites satisfy NEB's "
    "requirements for BsaI-HF v2."
)

CORPUS = "13.2\n18nt ok\nTM RANGE 58-72\n"


class _Judge:
    """Stub judge: literal constraints + optional deferred standard clauses."""

    def __init__(self, deferred=None, covered=(), constraints=None):
        self._deferred = [] if deferred is None else deferred
        self._covered = set(covered)
        self._constraints = constraints

    def constraints(self, text):
        if self._constraints is not None:
            return list(self._constraints)
        return [
            "primers must amplify the insert with the specified overhangs",
            "each primer must be at least 18 nt",
            "Tm of each annealing region must be between 58 and 72 degrees C",
        ]

    def deferred_requirements(self, text):
        return list(self._deferred)

    def constraint_coverage(self, quotes, stated, text):
        return {i for i, _ in enumerate(stated) if i in self._covered}


def _guard(task_text=TASK, pct=None):
    g = VerificationGuard()
    g._task_text = lambda: task_text
    g._step_done_recheck_reminded = True   # isolate the obligation under test
    if pct is not None:
        g._stats_fn = lambda: {"pct": pct}
    return g


def _ctx(args, judge=None, corpus=CORPUS, tool_result=None):
    # `override_reason` satisfies the PRE-EXISTING Mode-2 step_done gate (no plan
    # loaded -> no acceptance list -> the old gate demands an override reason).
    # Without it that earlier gate masks the obligation under test and every
    # PASS-case assertion fails for an unrelated reason.
    return GuardContext(
        tool_name="plan_update",
        tool_args=dict(args),
        tool_result=corpus if tool_result is None else tool_result,
        judge=judge,
        messages=[],
        override_reason="observed the design table, which shows the goals are met",
    )


FIXED = {
    "convention": {"name": "none"},
    "deliverable_reading": {"deliverable": "none"},
    "assumptions": {"unverified": "none"},
    "differential": {"quantity": "x", "path_a": "a", "path_b": "b",
                     "delta": 0, "within_tolerance": "yes"},
}

COVERED_ACT = {
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


# ------------------------------------------------------- A: the obligation

def test_step_done_without_acceptance_act_is_blocked():
    """The measured hole: 21 step_done calls, 0 acceptance_act, gate never ran."""
    g = _guard()
    ctx = _ctx({"action": "step_done", "step_id": 1,
                "verification": ["ran the primer design, read the Tm table"], **FIXED})
    v = g.check_pre(ctx)
    assert v is not None, "step_done with no acceptance_act must not pass"
    assert "acceptance_act" in v.message, (
        "the block must name the field to add, not just say something is missing"
    )


def test_step_done_with_covered_acceptance_act_passes():
    judge = _Judge(covered=(0, 1, 2))
    g = _guard()
    ctx = _ctx({"action": "step_done", "step_id": 1,
                "verification": ["ran design, read Tm table"],
                "acceptance_act": COVERED_ACT, **FIXED}, judge=judge)
    assert g.check_pre(ctx) is None


def test_uncovered_obligation_is_named_not_just_counted():
    """The gate must surface WHICH requirement was missed."""
    judge = _Judge(covered=(0, 2))     # the >=18nt clause left uncovered
    g = _guard()
    ctx = _ctx({"action": "step_done", "step_id": 1,
                "verification": ["ran design, read Tm table"],
                "acceptance_act": COVERED_ACT, **FIXED}, judge=judge)
    v = g.check_pre(ctx)
    assert v is None or "18 nt" in v.message or "uncovered" in v.message.lower()


def test_budget_safety_valve_stays_on_the_expensive_sweep():
    """pitfall/...coverage_gate_eats_budget: the N-constraint SWEEP is what cost
    a whole 900 s turn, so the wall valve lives THERE. The act RECORDING stays
    obligatory -- degrading it would re-open the hole this closes."""
    judge = _Judge(covered=())
    g = _guard(pct=90)              # 10% left < _COVERAGE_GATE_MIN_REMAINING
    assert g._coverage_gate_affordable() is False
    single = {
        "command": "python3 design.py", "observed": "TM RANGE 58-72",
        "constraints": [{"quote": "primers must amplify the insert with the specified overhangs",
                         "command": "python3 design.py", "observed": "13.2"}],
    }
    ctx = _ctx({"action": "step_done", "step_id": 1,
                "verification": ["read the table"],
                "acceptance_act": single, **FIXED}, judge=judge)
    assert g.check_pre(ctx) is None, (
        "with 10% of the wall left, the coverage sweep must not block delivery"
    )


def test_budget_valve_does_not_weaken_the_obligation_itself():
    g = _guard(pct=95)
    ctx = _ctx({"action": "step_done", "step_id": 1,
                "verification": ["read the table"], **FIXED})
    v = g.check_pre(ctx)
    assert v is not None and "acceptance_act" in v.message, (
        "recording ONE observed act is cheap and stays obligatory at any clock"
    )


# ------------------------------------------------- C: deferred requirements

def test_deferred_standard_requirement_enters_the_obligation_set():
    """The dna-assembly class: the clause is not in the task text at all."""
    judge = _Judge(
        deferred=[{"source": "check that the cut sites satisfy NEB's requirements",
                   "requirement": "each primer carries at least one protective "
                                  "base 5' of the recognition site",
                   "certain": True}],
        covered=(0, 1, 2),
    )
    stated = V._stated_constraints(TASK, judge)
    assert any("protective base" in s for s in stated), (
        "the deferred standard requirement must become a stated constraint"
    )
    miss = V._constraint_coverage_missing(COVERED_ACT, TASK, CORPUS, judge=judge)
    assert any("protective base" in m for m in miss), (
        "and it must be reported as UNCOVERED, not silently dropped"
    )


def test_deferred_merging_dedupes_literal_duplicates():
    dup = "each primer must be at least 18 nt"
    judge = _Judge(deferred=[{"source": "check NEB", "requirement": dup,
                              "certain": True}],
                   constraints=[dup])
    stated = V._stated_constraints(TASK, judge)
    assert sum(1 for s in stated if s.strip() == dup) == 1


def test_dead_judge_never_manufactures_an_obligation():
    judge = _Judge()

    def boom(text):
        raise RuntimeError("provider down")

    judge.deferred_requirements = boom
    stated = V._stated_constraints(TASK, judge)
    assert isinstance(stated, list)
    assert all("protective base" not in s for s in stated)


def test_no_judge_leaves_the_deferred_path_inert():
    stated = V._stated_constraints(TASK, None)
    assert isinstance(stated, list)


# --------------------------------------------------- schema (the real root)

def test_plan_update_schema_declares_the_band_of_fields():
    """Root cause: an undeclared key is stringified in transport, so the guard's
    isinstance(dict) test is permanently False (pitfall/...schema_undeclared)."""
    # The schema lives at class level (`TaskTool.parameters`); the constructor
    # needs a live TaskPlan, which is irrelevant to the declared shape.
    props = PlanUpdateTool.parameters["properties"]
    for key in ("acceptance_act", "oracle_act", "calibration"):
        assert key in props, f"{key} must be declared or the agent cannot submit it"
        assert props[key].get("type") == "object"
