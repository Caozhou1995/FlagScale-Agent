"""Unit tests for the fixed `convention` field gate (structured, no regex).

The gate is a SHAPE check: it never judges whether a delivered value is
semantically right (the harness has no oracle). It refuses to accept a step_done
SILENTLY — EVERY step_done must carry a structured `convention` object, and
"not applicable" is itself an explicit declaration ({"name": "none"} or the
string "none"). There is NO `derived` boolean — the trigger is the NAME, so a
step whose output rests on an uncertain convention cannot be classified away
from the gate. The old design was opt-in (fired only when a step's acceptance
literally contained a text marker) and parsed that marker with regex — both are
gone. See verification.py `_check_convention_field`.
"""
import tempfile
import shutil

from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.plan import TaskPlan


def _guard_with_step(acceptance=("A1",)):
    d = tempfile.mkdtemp()
    plan = TaskPlan(d)
    plan.create("Test", [{"title": "Step A", "acceptance": list(acceptance)}])
    guard = VerificationGuard(plan=plan)
    guard._step_done_recheck_reminded = True  # isolate from one-shot premise re-check
    return d, guard


def _ctx(convention, verification=("proof A1",)):
    return GuardContext(
        tool_name="plan_update",
        tool_args={
            "action": "step_done",
            "step_id": 1,
            "verification": list(verification),
            "deliverable_reading": {"deliverable": "none"},
            "convention": convention,
        },
    )


GOOD = {
    "name": "output field convention",
    "example": "sample-1",
    "delivered": "canonical-1",
    "conflict": "yes",
    "probe": "measure --field",
    "probe_output": "field=canonical-1",
}


# --- the fixed field fires UNCONDITIONALLY (the whole point of the change) ----

def test_missing_field_blocks_even_on_plain_step():
    """A plain step with NO convention tag still requires the field — the gate
    must not be opt-in (this is the regression the redesign fixes)."""
    d, guard = _guard_with_step(["A1", "A2"])
    try:
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 1,
                       "verification": ["proof A1"]},  # no `convention`
        )
        v = guard.check_pre(ctx)
        assert v is not None and v.action == "block"
        assert v.reason == "convention_declaration_missing"
        assert "convention" in v.message
    finally:
        shutil.rmtree(d)


def test_explicit_not_applicable_passes():
    """`{"name": "none"}` is the explicit not-applicable declaration."""
    d, guard = _guard_with_step(["A1"])
    try:
        assert guard.check_pre(_ctx({"name": "none"})) is None
    finally:
        shutil.rmtree(d)


def test_string_none_passes_as_explicit_no_value():
    d, guard = _guard_with_step(["A1"])
    try:
        assert guard.check_pre(_ctx("None")) is None
    finally:
        shutil.rmtree(d)


def test_derived_false_escape_blocks():
    """THE escape this redesign closes: an agent wrote a nameless
    object with a self-declared `derived: false` on the step whose OUTPUT rested
    on the uncertain address convention. With no `name`, the gate now BLOCKS it —
    the old design accepted it because `derived is False` returned None."""
    d, guard = _guard_with_step(["A1"])
    try:
        v = guard.check_pre(_ctx({"derived": False}))
        assert v is not None and v.reason == "convention_missing_field"
        v2 = guard.check_pre(_ctx({"derived": True}))
        assert v2 is not None and v2.reason == "convention_missing_field"
    finally:
        shutil.rmtree(d)


def test_derived_key_is_no_longer_a_trigger():
    """There is no `derived` key: an object that carries `derived` but no `name`
    is missing the fixed field's required key, not a valid not-applicable form."""
    d, guard = _guard_with_step(["A1"])
    try:
        v = guard.check_pre(_ctx({"derived": "maybe"}))
        assert v is not None and v.reason == "convention_missing_field"
    finally:
        shutil.rmtree(d)


# --- shape rules for a NAMED convention ----------------------------------------

def test_missing_required_field_blocks():
    d, guard = _guard_with_step(["A1"])
    try:
        bad = dict(GOOD)
        del bad["delivered"]
        v = guard.check_pre(_ctx(bad))
        assert v is not None and v.reason == "convention_missing_field"
    finally:
        shutil.rmtree(d)


def test_null_values_are_legal_but_keys_must_exist():
    """The doc's "or None" fields (example/delivered/conflict) accept a JSON
    null — an agent serialising to JSON emits null, not the string "None".
    But a key that is ABSENT is a silent omission and still blocks."""
    d, guard = _guard_with_step(["A1"])
    try:
        ok = dict(GOOD, conflict="no", probe=None, probe_output=None,
                  hedge="union", hedge_output="union set covers all readings, 0 mismatches",
                  extras_direction="ignored")
        ok["example"] = None
        ok["delivered"] = None
        assert guard.check_pre(_ctx(ok)) is None, (
            "null values for example/delivered (and probe under conflict=no) "
            "must be accepted — doc says 'or None' — provided a hedge falsifier "
            "is present"
        )
        absent = dict(GOOD)
        del absent["example"]
        v = guard.check_pre(_ctx(absent))
        assert v is not None and v.reason == "convention_missing_field"
    finally:
        shutil.rmtree(d)


def test_conflict_without_probe_blocks():
    d, guard = _guard_with_step(["A1"])
    try:
        bad = dict(GOOD, probe="None", probe_output="None")
        v = guard.check_pre(_ctx(bad))
        assert v is not None and v.reason == "convention_conflict_without_probe"
    finally:
        shutil.rmtree(d)


def test_conflict_no_probe_none_passes():
    d, guard = _guard_with_step(["A1"])
    try:
        ok = dict(GOOD, conflict="no", probe="None", probe_output="None")
        # conflict=no still needs a FALSIFIER (probe or hedge) — probe here.
        ok["probe"] = "measure --field"
        ok["probe_output"] = "field=canonical-1"
        assert guard.check_pre(_ctx(ok)) is None
    finally:
        shutil.rmtree(d)


def test_bad_conflict_blocks():
    d, guard = _guard_with_step(["A1"])
    try:
        v = guard.check_pre(_ctx(dict(GOOD, conflict="maybe")))
        assert v is not None and v.reason == "convention_bad_conflict"
    finally:
        shutil.rmtree(d)


def test_valid_declaration_passes():
    d, guard = _guard_with_step(["A1"])
    try:
        assert guard.check_pre(_ctx(GOOD)) is None
    finally:
        shutil.rmtree(d)


# --- deleting a candidate reading on suspicion -----------

def test_rejected_candidate_without_evidence_blocks():
    """Naming a REJECTED candidate reading requires an independent artifact —
    'removed it on an inferred penalty rule' is the exact observed failure."""
    d, guard = _guard_with_step(["A1"])
    try:
        bad = dict(GOOD, rejected="alternative reading", rejected_evidence="None")
        v = guard.check_pre(_ctx(bad))
        assert v is not None and v.reason == "convention_rejected_without_evidence"
    finally:
        shutil.rmtree(d)


def test_rejected_candidate_with_evidence_passes():
    d, guard = _guard_with_step(["A1"])
    try:
        ok = dict(GOOD, rejected="segment-aware key space",
                  rejected_evidence="reference.json omits those keys")
        assert guard.check_pre(_ctx(ok)) is None
    finally:
        shutil.rmtree(d)


def test_no_rejected_field_is_fine():
    """rejected is optional; its absence is not a rejection claim."""
    d, guard = _guard_with_step(["A1"])
    try:
        assert guard.check_pre(_ctx(GOOD)) is None
    finally:
        shutil.rmtree(d)


# --- the gate must not disturb the pre-existing evidence requirements --------

def test_no_acceptance_still_needs_override_reason():
    """The convention field is checked AFTER the existing mode checks — a step
    with no acceptance and no override_reason still blocks on the old reason."""
    d = tempfile.mkdtemp()
    plan = TaskPlan(d)
    plan.create("Test", ["Plain step"])  # no acceptance list
    guard = VerificationGuard(plan=plan)
    guard._step_done_recheck_reminded = True
    try:
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 1,
                       "deliverable_reading": {"deliverable": "none"},
                       "convention": {"name": "none"}},
        )
        v = guard.check_pre(ctx)
        assert v is not None and v.reason == "step_done_no_verification"
    finally:
        shutil.rmtree(d)


def test_acceptance_without_verification_still_blocks():
    d, guard = _guard_with_step(["A1"])
    try:
        ctx = _ctx({"name": "none"}, verification=[])
        v = guard.check_pre(ctx)
        assert v is not None and v.reason == "step_done_with_acceptance_no_verification"
    finally:
        shutil.rmtree(d)


# --- the structural fix: a NAMED convention MUST carry a falsifier ------------------

def test_named_bare_single_reading_blocks():
    """THE observed failure shape: a derived convention with a single
    reading, conflict=no, and NO probe and NO hedge must BLOCK — that is exactly
    how a value locked onto one example, or a guessed output form,
    shipped wrongly."""
    d, guard = _guard_with_step(["A1"])
    try:
        bare = {"name": "output field convention",
                "example": "sample-1", "delivered": "sample-1", "conflict": "no"}
        v = guard.check_pre(_ctx(bare))
        assert v is not None and v.reason == "convention_no_falsifier", (
            "a bare derived single reading must be blocked"
        )
    finally:
        shutil.rmtree(d)


def test_named_bare_conflict_none_blocks():
    """Same shape with conflict=None (a guessed output form)."""
    d, guard = _guard_with_step(["A1"])
    try:
        bare = {"name": "output form",
                "example": None, "delivered": "guessed", "conflict": None}
        v = guard.check_pre(_ctx(bare))
        assert v is not None and v.reason == "convention_no_falsifier"
    finally:
        shutil.rmtree(d)


def test_named_with_real_probe_passes():
    """A derived reading backed by a real discriminating probe is admissible —
    even with conflict=no (a probe that could have refuted the reading)."""
    d, guard = _guard_with_step(["A1"])
    try:
        ok = {"name": "output field convention",
              "example": "sample-1", "delivered": "canonical-1",
              "conflict": "no",
              "probe": "measure --field",
              "probe_output": "field=canonical-1"}
        assert guard.check_pre(_ctx(ok)) is None
    finally:
        shutil.rmtree(d)


def test_named_probe_with_none_output_blocks():
    """Naming a probe but leaving its output None is the silent skip the gate
    exists to stop — still blocks."""
    d, guard = _guard_with_step(["A1"])
    try:
        bad = {"name": "output field", "example": "x",
               "delivered": "y", "conflict": "no",
               "probe": "measure --field", "probe_output": "None"}
        v = guard.check_pre(_ctx(bad))
        assert v is not None and v.reason == "convention_no_falsifier"
    finally:
        shutil.rmtree(d)


def test_named_with_proved_union_hedge_passes():
    """A proved superset hedge (union, extras ignored) is a valid falsifier —
    the shape a superset hedge realises."""
    d, guard = _guard_with_step(["A1"])
    try:
        ok = {"name": "output field convention",
              "example": "sample-1", "delivered": "union of candidate forms",
              "conflict": "no",
              "hedge": "union",
              "hedge_output": "union set: full recall, 0 mismatches",
              "extras_direction": "ignored"}
        assert guard.check_pre(_ctx(ok)) is None
    finally:
        shutil.rmtree(d)


def test_named_with_proved_intersection_hedge_passes():
    d, guard = _guard_with_step(["A1"])
    try:
        ok = {"name": "base", "example": "x",
              "delivered": "one output", "conflict": "no",
              "hedge": "intersection",
              "hedge_output": "single output in every reading's set",
              "extras_direction": "penalized"}
        assert guard.check_pre(_ctx(ok)) is None
    finally:
        shutil.rmtree(d)


def test_union_hedge_under_penalized_blocks():
    """A union/superset is a hedge ONLY when extras are ignored; declared under
    'penalized' it is self-contradictory (the exact observed reasoning)."""
    d, guard = _guard_with_step(["A1"])
    try:
        bad = {"name": "base", "example": "x",
               "delivered": "y", "conflict": "no",
               "hedge": "union", "hedge_output": "...",
               "extras_direction": "penalized"}
        v = guard.check_pre(_ctx(bad))
        assert v is not None and v.reason == "convention_hedge_shape_mismatch"
    finally:
        shutil.rmtree(d)


def test_hedge_without_output_blocks():
    d, guard = _guard_with_step(["A1"])
    try:
        bad = {"name": "base", "example": "x",
               "delivered": "y", "conflict": "no",
               "hedge": "union", "hedge_output": None,
               "extras_direction": "ignored"}
        v = guard.check_pre(_ctx(bad))
        assert v is not None and v.reason == "convention_no_falsifier"
    finally:
        shutil.rmtree(d)


def test_not_applicable_unaffected_by_falsifier_requirement():
    """The mechanical / not-applicable path must NOT be touched by the new gate."""
    d, guard = _guard_with_step(["A1"])
    try:
        assert guard.check_pre(_ctx({"name": "none"})) is None
        assert guard.check_pre(_ctx("None")) is None
    finally:
        shutil.rmtree(d)


# --- reviewer findings (a195a6f4e458): name-token + step-id asymmetry ---------

def test_blank_or_alias_name_is_not_not_applicable():
    """F1: ONLY the literal token "none" means not-applicable. A blank or
    alias name ("", "  ", "n/a", "null", "nil", "-") is a forgotten name, NOT a
    declaration — accepting it would silently downgrade a named step to
    not-applicable and re-open the escape this gate closes."""
    d, guard = _guard_with_step(["A1"])
    try:
        for nm in ["", "  ", "n/a", "na", "n.a.", "null", "nil", "-", "--"]:
            v = guard.check_pre(_ctx({"name": nm}))
            assert v is not None and v.reason == "convention_missing_field", (
                f"name={nm!r} must NOT be treated as not-applicable"
            )
        # the ONE legal token, case/space-insensitive, still passes
        for nm in ["none", "NONE", "None", "  none  "]:
            assert guard.check_pre(_ctx({"name": nm})) is None
    finally:
        shutil.rmtree(d)


def test_step_id_bool_does_not_bypass_acceptance_gate():
    """F2: a bool step_id must NOT let a step with acceptance skip the Mode-1
    evidence gate. The guard and the handler must disagree on nothing: both
    normalize True/False to None."""
    from flagscale_agent.react.tools.plan_update import _parse_step_id
    from flagscale_agent.react.guard.verification import _normalize_step_id
    for raw in (True, False):
        assert _normalize_step_id(raw) is None
        assert _parse_step_id(raw) is None
    d, guard = _guard_with_step(["A1", "A2"])
    try:
        v = guard.check_pre(_ctx({"name": "none"}, verification=[]))
        # control: a real step id still enforces acceptance evidence
        assert v is not None and v.reason == "step_done_with_acceptance_no_verification"
        ctx_bool = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": True,
                       "verification": ["p"], "convention": {"name": "none"}},
        )
        v2 = guard.check_pre(ctx_bool)
        # True normalizes to None, so it is NOT step 1: the guard must treat it
        # as acceptance-less (Mode 2), never as the step that has acceptance.
        assert v2 is not None and v2.reason == "step_done_no_verification", (
            "a bool step_id must not resolve to the acceptance-bearing step"
        )
        assert v2.reason != "step_done_with_acceptance_no_verification"
    finally:
        shutil.rmtree(d)


# --- reviewer findings (a4233be908a4): evidence must be a TYPED string -------

def test_non_string_probe_is_not_a_real_probe():
    """F1: a list/dict probe is not an observed command — str() coercion must
    not let it masquerade as a real falsifier."""
    d, guard = _guard_with_step(["A1"])
    try:
        bad = {"name": "base", "example": "x",
               "delivered": "y", "conflict": "no",
               "probe": [1], "probe_output": [2]}
        v = guard.check_pre(_ctx(bad))
        assert v is not None and v.reason == "convention_no_falsifier"
        bad2 = dict(bad, probe={"cmd": "x"}, probe_output={"out": "y"})
        v2 = guard.check_pre(_ctx(bad2))
        assert v2 is not None and v2.reason == "convention_no_falsifier"
    finally:
        shutil.rmtree(d)


def test_rejected_false_does_not_block():
    """F2: an explicit 'nothing rejected' (bool False / empty list) is NOT a
    named candidate and must NOT require evidence."""
    d, guard = _guard_with_step(["A1"])
    try:
        ok = {"name": "base", "example": "x",
              "delivered": "y", "conflict": "no",
              "probe": "measure --field", "probe_output": "field=canonical-1",
              "rejected": False}
        assert guard.check_pre(_ctx(ok)) is None
        ok2 = dict(ok, rejected=[])
        assert guard.check_pre(_ctx(ok2)) is None
        # a REAL candidate still demands evidence
        bad = dict(ok, rejected="alternative reading")
        v = guard.check_pre(_ctx(bad))
        assert v is not None and v.reason == "convention_rejected_without_evidence"
    finally:
        shutil.rmtree(d)


def test_non_string_convention_name_blocks():
    """F3: an int `convention` name is not a meaningful label."""
    d, guard = _guard_with_step(["A1"])
    try:
        bad = {"name": 0, "example": "x",
               "delivered": "y", "conflict": "no",
               "probe": "a", "probe_output": "b"}
        v = guard.check_pre(_ctx(bad))
        assert v is not None and v.reason == "convention_missing_field"
    finally:
        shutil.rmtree(d)


def test_conflict_yes_non_string_probe_blocks():
    """F4: the conflict=yes arm shares the typed check."""
    d, guard = _guard_with_step(["A1"])
    try:
        bad = {"name": "c", "example": "x",
               "delivered": "y", "conflict": "yes",
               "probe": [1], "probe_output": [2]}
        v = guard.check_pre(_ctx(bad))
        assert v is not None and v.reason == "convention_conflict_without_probe"
    finally:
        shutil.rmtree(d)
