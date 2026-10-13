"""Unit tests for the ConventionGuard shape-check (`convention-check` tag).

The gate is a SHAPE check: it never judges whether a delivered value is
semantically right (the harness has no oracle). It only refuses to accept a
convention-dependent step SILENTLY — the declaration line must be present, have
all 5 fields non-empty (`None` = explicit no-value), and a conflict=yes must
carry a real probe output.
"""
import tempfile
import shutil

from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.plan import TaskPlan


def _guard_with_step(acceptance):
    d = tempfile.mkdtemp()
    plan = TaskPlan(d)
    plan.create("Test", [{"title": "Step A", "acceptance": acceptance}])
    guard = VerificationGuard(plan=plan)
    guard._step_done_recheck_reminded = True  # isolate from one-shot premise re-check
    return d, guard


def _ctx(verification):
    return GuardContext(
        tool_name="plan_update",
        tool_args={"action": "step_done", "step_id": 1, "verification": verification},
    )


GOOD = ("convention-check: example=0x400000 | delivered=0x0 | conflict=yes | "
        "probe=readelf -S a.out | probe_output=sh_addr=0x1060")


def test_tagged_step_without_declaration_blocks():
    d, guard = _guard_with_step(["convention-check: derived base"])
    try:
        v = guard.check_pre(_ctx(["criterion 1 - observed: 4079 keys"]))
        assert v is not None and v.action == "block"
        assert v.reason == "convention_check_missing"
        assert "convention-check:" in v.message  # tells the agent the exact shape
    finally:
        shutil.rmtree(d)


def test_empty_field_blocks():
    d, guard = _guard_with_step(["convention-check: derived base"])
    try:
        bad = ("convention-check: example=0x400000 | delivered=0x0 | conflict=yes | "
               "probe=readelf -S a.out | probe_output=")
        v = guard.check_pre(_ctx([bad]))
        assert v is not None and v.action == "block"
        assert v.reason == "convention_check_empty_field"
    finally:
        shutil.rmtree(d)


def test_missing_field_blocks():
    d, guard = _guard_with_step(["convention-check: derived base"])
    try:
        bad = "convention-check: example=0x400000 | delivered=0x0 | conflict=yes | probe=readelf"
        v = guard.check_pre(_ctx([bad]))
        assert v is not None and v.action == "block"
        assert v.reason == "convention_check_malformed"
    finally:
        shutil.rmtree(d)


def test_all_none_passes_as_explicit_no_value():
    d, guard = _guard_with_step(["convention-check: derived base"])
    try:
        line = ("convention-check: example=None | delivered=None | conflict=None | "
                "probe=None | probe_output=None")
        v = guard.check_pre(_ctx([line]))
        assert v is None  # explicit no-value is legal
    finally:
        shutil.rmtree(d)


def test_conflict_without_probe_blocks():
    d, guard = _guard_with_step(["convention-check: derived base"])
    try:
        bad = ("convention-check: example=0x400000 | delivered=0x0 | conflict=yes | "
               "probe=None | probe_output=None")
        v = guard.check_pre(_ctx([bad]))
        assert v is not None and v.action == "block"
        assert v.reason == "convention_check_conflict_without_probe"
    finally:
        shutil.rmtree(d)


def test_conflict_no_probe_none_passes():
    d, guard = _guard_with_step(["convention-check: derived base"])
    try:
        line = ("convention-check: example=0x400000 | delivered=0x400000 | conflict=no | "
                "probe=None | probe_output=None")
        v = guard.check_pre(_ctx([line]))
        assert v is None  # no conflict => probe genuinely absent => None legal
    finally:
        shutil.rmtree(d)


def test_valid_declaration_passes():
    d, guard = _guard_with_step(["convention-check: derived base"])
    try:
        assert guard.check_pre(_ctx([GOOD])) is None
    finally:
        shutil.rmtree(d)


def test_untagged_step_unaffected():
    """A step WITHOUT the convention-check tag passes on any non-empty
    verification — the check must not leak onto ordinary steps."""
    d, guard = _guard_with_step(["A1", "A2"])
    try:
        assert guard.check_pre(_ctx(["Proof A1", "Proof A2"])) is None
    finally:
        shutil.rmtree(d)


def test_declaration_may_be_prefixed_by_other_text():
    """The line may be one entry among several; the regex finds it."""
    d, guard = _guard_with_step(["convention-check: derived base"])
    try:
        assert guard.check_pre(_ctx(["criterion 1 - observed: 5", GOOD])) is None
    finally:
        shutil.rmtree(d)


def test_multiline_entry_with_trailing_text_still_matches():
    """A declaration that is NOT the last line of a multi-line verification entry
    must still be found (the regex is line-anchored, not $-anchored)."""
    d, guard = _guard_with_step(["convention-check: derived base"])
    try:
        multi = "crit 1 - observed: ok\n" + GOOD + "\nextra note here"
        assert guard.check_pre(_ctx([multi])) is None
    finally:
        shutil.rmtree(d)
