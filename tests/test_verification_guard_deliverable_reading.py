"""Unit tests for the fixed `deliverable_reading` field gate (structured, no regex).

Second fixed, REQUIRED field on every step_done. It forces the agent to declare
whether the step emits a graded artifact and, under an INVISIBLE grader, to
enumerate the readings the task admits (each GROUNDED verbatim in the task text)
and CONSTRUCT a hedge or prove none exists. Shape-only: it never judges whether
the reading is semantically right. See verification.py
`_check_deliverable_reading_field`.
"""
import tempfile
import shutil

from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.plan import TaskPlan

TASK = ("Output a JSON of addresses. Every 4-byte word in a PT_LOAD segment must "
        "appear. Do not alter the formatting.")


def _guard_with_step(acceptance=("A1",), task_text=None):
    d = tempfile.mkdtemp()
    plan = TaskPlan(d)
    plan.create("Test", [{"title": "Step A", "acceptance": list(acceptance)}])
    guard = VerificationGuard(plan=plan, task_text_fn=(lambda: task_text) if task_text else None)
    guard._step_done_recheck_reminded = True  # isolate from one-shot premise re-check
    return d, guard


def _ctx(dr, verification=("proof A1",)):
    return GuardContext(
        tool_name="plan_update",
        tool_args={
            "action": "step_done",
            "step_id": 1,
            "verification": list(verification),
            "convention": {"name": "none"},
            "deliverable_reading": dr,
        },
    )


GROUNDED = {"reading": "file offset base 0",
            "source": "Every 4-byte word in a PT_LOAD segment must appear",
            "acceptance_test": "pytest t.py"}


# --- the field fires UNCONDITIONALLY ----

def test_missing_field_blocks():
    d, guard = _guard_with_step(["A1"])
    try:
        ctx = GuardContext(tool_name="plan_update",
                           tool_args={"action": "step_done", "step_id": 1,
                                      "verification": ["p"], "convention": {"name": "none"}})
        v = guard.check_pre(ctx)
        assert v is not None and v.action == "block"
        assert v.reason == "deliverable_reading_missing"
    finally:
        shutil.rmtree(d)


def test_explicit_none_passes():
    d, guard = _guard_with_step(["A1"])
    try:
        assert guard.check_pre(_ctx({"deliverable": "none"})) is None
    finally:
        shutil.rmtree(d)


def test_grader_visible_skips_analysis():
    d, guard = _guard_with_step(["A1"])
    try:
        assert guard.check_pre(_ctx({"deliverable": "out.json", "grader_visible": True})) is None
    finally:
        shutil.rmtree(d)


def test_grader_visible_requires_bool():
    d, guard = _guard_with_step(["A1"])
    try:
        v = guard.check_pre(_ctx({"deliverable": "out.json", "grader_visible": "yes"}))
        assert v is not None and v.reason == "deliverable_reading_malformed"
    finally:
        shutil.rmtree(d)


# --- invisible grader: readings required + grounded ----

def test_invisible_without_readings_blocks():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        v = guard.check_pre(_ctx({"deliverable": "out.json", "grader_visible": False}))
        assert v is not None and v.reason == "deliverable_reading_malformed"
    finally:
        shutil.rmtree(d)


def test_grounded_reading_with_union_hedge_passes():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [GROUNDED],
              "hedge": {"shape": "union", "output": "...", "evidence": "$ cmd -> ok",
                        "extras_direction": "ignored"}}
        assert guard.check_pre(_ctx(dr)) is None
    finally:
        shutil.rmtree(d)


def test_ungrounded_source_blocks():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [{"reading": "r", "source": "NOT IN THE TASK AT ALL",
                            "acceptance_test": "x"}],
              "hedge": {"shape": "union", "output": "a", "evidence": "b",
                        "extras_direction": "ignored"}}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "reading_not_grounded"
    finally:
        shutil.rmtree(d)


def test_grounding_skipped_when_task_unavailable():
    """Fail-open: if the harness could NOT recover the task text, the guard must
    not block a reading it cannot check. Structural checks still run."""
    d, guard = _guard_with_step(["A1"], "(unavailable: no user turn found in full log)")
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [{"reading": "r", "source": "NOT IN TASK", "acceptance_test": "x"}],
              "hedge": {"shape": "union", "output": "a", "evidence": "b",
                        "extras_direction": "ignored"}}
        assert guard.check_pre(_ctx(dr)) is None
    finally:
        shutil.rmtree(d)


def test_missing_acceptance_test_blocks():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [{"reading": "r", "source": "Do not alter the formatting"}],
              "hedge": {"shape": "union", "output": "a", "evidence": "b",
                        "extras_direction": "ignored"}}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "deliverable_reading_malformed"
    finally:
        shutil.rmtree(d)


# --- hedge shape/direction consistency + proof ----

def test_union_under_penalized_blocks():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [GROUNDED],
              "hedge": {"shape": "union", "output": "a", "evidence": "b",
                        "extras_direction": "penalized"}}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "hedge_shape_mismatch"
    finally:
        shutil.rmtree(d)


def test_intersection_under_ignored_passes():
    """An intersection hedge (one output in EVERY reading's set) is conservatively
    safe even when extras are ignored — blocking it was over-strict (review F3)."""
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [GROUNDED],
              "hedge": {"shape": "intersection", "output": "a", "evidence": "b",
                        "extras_direction": "ignored"}}
        assert guard.check_pre(_ctx(dr)) is None
    finally:
        shutil.rmtree(d)


def test_blank_string_form_blocks():
    """A blank / 'n/a' / 'null' STRING is NOT a declaration of not-applicable —
    only the literal "none" is (review F1: the wide no-value set was a bypass)."""
    d, guard = _guard_with_step(["A1"])
    try:
        for s in ("", "  ", "n/a", "null", "nil", "-"):
            v = guard.check_pre(_ctx(s))
            assert v is not None and v.reason == "deliverable_reading_malformed", s
        assert guard.check_pre(_ctx("none")) is None
    finally:
        shutil.rmtree(d)


def test_hedge_without_output_blocks():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [GROUNDED],
              "hedge": {"shape": "union", "extras_direction": "ignored"}}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "hedge_unproven"
    finally:
        shutil.rmtree(d)


def test_none_shape_requires_if_none():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [GROUNDED], "hedge": {"shape": "none"}}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "hedge_unproven"
    finally:
        shutil.rmtree(d)


def test_none_shape_complete_passes():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [GROUNDED], "hedge": {"shape": "none"},
              "if_none": {"why_intersection_empty": "mutually exclusive bases",
                          "readings_rejected": [{"reading": "flat offset",
                                                 "source": "Do not alter the formatting",
                                                 "evidence": "ref output uses VA"}],
                          "bet_on": "seg VA", "bet_loses_when": "grader uses flat",
                          "why_safe": "ref proves VA"}}
        assert guard.check_pre(_ctx(dr)) is None
    finally:
        shutil.rmtree(d)


def test_rejected_without_evidence_blocks():
    """A rejected reading MUST carry independent evidence — removing a candidate
    on suspicion alone is what cost a prior run."""
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [GROUNDED], "hedge": {"shape": "none"},
              "if_none": {"why_intersection_empty": "exclusive",
                          "readings_rejected": [{"reading": "flat offset",
                                                 "source": "Do not alter the formatting"}],
                          "bet_on": "seg VA", "bet_loses_when": "flat",
                          "why_safe": "ref"}}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "hedge_unproven"
    finally:
        shutil.rmtree(d)


def test_rejected_ungrounded_blocks():
    """A rejected candidate's `source` must also ground verbatim — a fabricated
    rejection cannot manufacture an empty intersection (review F3)."""
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [GROUNDED], "hedge": {"shape": "none"},
              "if_none": {"why_intersection_empty": "exclusive",
                          "readings_rejected": [{"reading": "flat offset",
                                                 "source": "INVENTED SOURCE",
                                                 "evidence": "ref"}],
                          "bet_on": "seg VA", "bet_loses_when": "flat",
                          "why_safe": "ref"}}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "reading_not_grounded"
    finally:
        shutil.rmtree(d)


# --- batch path parity ---

def test_batch_done_requires_field():
    d, guard = _guard_with_step(["A1"])
    try:
        ctx = GuardContext(tool_name="plan_update",
                           tool_args={"action": "batch",
                                      "updates": [{"step_id": 1, "status": "done",
                                                   "verification": ["p"]}],
                                      "convention": {"name": "none"}},
                           override_reason="rechecked each done step")
        v = guard.check_pre(ctx)
        assert v is not None and v.reason == "deliverable_reading_missing"
    finally:
        shutil.rmtree(d)
