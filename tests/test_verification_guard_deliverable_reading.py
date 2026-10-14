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
            "acceptance_test": "pytest t.py",
            "normalized_side": "reference"}


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
              "hedge": {"shape": "union", "output": "...",
                        "evidence": {"command": "cmd", "observed": "ok"},
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
              "readings": [{"reading": "r", "source": "NOT IN TASK", "acceptance_test": "x", "normalized_side": "reference"}],
              "hedge": {"shape": "union", "output": "a",
                        "evidence": {"command": "c", "observed": "o"},
                        "extras_direction": "ignored"}}
        assert guard.check_pre(_ctx(dr)) is None
    finally:
        shutil.rmtree(d)


def test_missing_acceptance_test_blocks():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [{"reading": "r", "source": "Do not alter the formatting"}],
              "hedge": {"shape": "union", "output": "a",
                        "evidence": {"command": "c", "observed": "o"},
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
              "hedge": {"shape": "union", "output": "a",
                        "evidence": {"command": "c", "observed": "o"},
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
              "hedge": {"shape": "intersection", "output": "a",
                        "evidence": {"command": "c", "observed": "o"},
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
                          "extras_direction": "ignored",
                          "extras_direction_source": "Do not alter the formatting.",
                          "union_attempt": {"command": "c", "observed": "o"},
                          "readings_rejected": [{"reading": "flat offset",
                                                 "source": "Do not alter the formatting",
                                                 "evidence": {"command": "c",
                                                              "observed": "o"}}],
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
                          "extras_direction": "ignored",
                          "extras_direction_source": "Do not alter the formatting.",
                          "union_attempt": {"command": "c", "observed": "o"},
                          "readings_rejected": [{"reading": "flat offset",
                                                 "source": "Do not alter the formatting"}],
                          "bet_on": "seg VA", "bet_loses_when": "flat",
                          "why_safe": "ref"}}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "hedge_rejection_unobserved"
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
                          "extras_direction": "ignored",
                          "extras_direction_source": "Do not alter the formatting.",
                          "union_attempt": {"command": "c", "observed": "o"},
                          "readings_rejected": [{"reading": "flat offset",
                                                 "source": "INVENTED SOURCE",
                                                 "evidence": {"command": "c",
                                                              "observed": "o"}}],
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


# --- observed-evidence arms: a claim must paste text the harness can find in the
#     session's REAL tool results; an asserted-but-never-run falsifier is blocked.

def _ctx_obs(dr, msgs, verification=("proof A1",)):
    return GuardContext(
        tool_name="plan_update",
        tool_args={"action": "step_done", "step_id": 1,
                   "verification": list(verification),
                   "convention": {"name": "none"},
                   "deliverable_reading": dr},
        messages=msgs,
    )


def _corpus(text):
    return [{"role": "tool", "content": text}]


OBSERVED_LINE = "found 698 addresses coverage 92.9%"


def test_union_hedge_with_observed_evidence_passes():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [GROUNDED],
              "hedge": {"shape": "union", "output": "...",
                        "evidence": {"command": "python -c c",
                                     "observed": OBSERVED_LINE},
                        "extras_direction": "ignored"}}
        assert guard.check_pre(_ctx_obs(dr, _corpus("x\n" + OBSERVED_LINE + "\ny"))) is None
    finally:
        shutil.rmtree(d)


def test_union_hedge_with_fabricated_evidence_blocks():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [GROUNDED],
              "hedge": {"shape": "union", "output": "...",
                        "evidence": {"command": "c",
                                     "observed": "a made-up line 12345 zzz"},
                        "extras_direction": "ignored"}}
        v = guard.check_pre(_ctx_obs(dr, _corpus(OBSERVED_LINE)))
        assert v is not None and v.reason == "hedge_evidence_unobserved"
    finally:
        shutil.rmtree(d)


def test_union_hedge_with_missing_evidence_blocks():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [GROUNDED],
              "hedge": {"shape": "union", "output": "...",
                        "extras_direction": "ignored"}}
        v = guard.check_pre(_ctx_obs(dr, _corpus(OBSERVED_LINE)))
        assert v is not None and v.reason == "hedge_evidence_unobserved"
    finally:
        shutil.rmtree(d)


def test_union_hedge_evidence_failopen_when_corpus_unavailable():
    """No message window and no full log -> cannot check -> fail-open (no block)."""
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [GROUNDED],
              "hedge": {"shape": "union", "output": "...",
                        "evidence": {"command": "c", "observed": "whatever"},
                        "extras_direction": "ignored"}}
        assert guard.check_pre(_ctx(dr)) is None
    finally:
        shutil.rmtree(d)


def _none_dr(if_none):
    if "extras_direction" not in if_none:
        if_none = {**if_none,
                   "extras_direction": "ignored",
                   "extras_direction_source": "Do not alter the formatting."}
    return {"deliverable": "out.json", "grader_visible": False,
            "readings": [GROUNDED],
            "hedge": {"shape": "none", "output": None, "evidence": None,
                      "extras_direction": "ignored"},
            "if_none": if_none}


def test_none_shape_requires_union_attempt():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        v = guard.check_pre(_ctx_obs(_none_dr({
            "why_intersection_empty": "readings differ",
            "readings_rejected": [{"reading": "r", "source": GROUNDED["source"],
                                   "evidence": {"command": "c",
                                                "observed": OBSERVED_LINE}}],
            "bet_on": "a", "bet_loses_when": "b", "why_safe": "c"}),
            _corpus(OBSERVED_LINE)))
        assert v is not None and v.reason == "hedge_union_unattempted"
    finally:
        shutil.rmtree(d)


def test_none_shape_fabricated_union_attempt_blocks():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        v = guard.check_pre(_ctx_obs(_none_dr({
            "why_intersection_empty": "readings differ",
            "union_attempt": {"command": "c", "observed": "never ran zzz"},
            "readings_rejected": [{"reading": "r", "source": GROUNDED["source"],
                                   "evidence": {"command": "c",
                                                "observed": OBSERVED_LINE}}],
            "bet_on": "a", "bet_loses_when": "b", "why_safe": "c"}),
            _corpus(OBSERVED_LINE)))
        assert v is not None and v.reason == "hedge_union_unattempted"
    finally:
        shutil.rmtree(d)


def test_none_shape_with_observed_union_attempt_passes():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = _none_dr({
            "why_intersection_empty": "readings differ",
            "union_attempt": {"command": "python -c c", "observed": OBSERVED_LINE},
            "readings_rejected": [{"reading": "r", "source": GROUNDED["source"],
                                   "evidence": {"command": "c",
                                                "observed": OBSERVED_LINE}}],
            "bet_on": "a", "bet_loses_when": "b", "why_safe": "c"})
        assert guard.check_pre(_ctx_obs(dr, _corpus(OBSERVED_LINE))) is None
    finally:
        shutil.rmtree(d)


def test_rejected_entry_fabricated_evidence_blocks():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        v = guard.check_pre(_ctx_obs(_none_dr({
            "why_intersection_empty": "readings differ",
            "union_attempt": {"command": "c", "observed": OBSERVED_LINE},
            "readings_rejected": [{"reading": "r", "source": GROUNDED["source"],
                                   "evidence": {"command": "c",
                                                "observed": "invented qqq"}}],
            "bet_on": "a", "bet_loses_when": "b", "why_safe": "c"}),
            _corpus(OBSERVED_LINE)))
        assert v is not None and v.reason == "hedge_rejection_unobserved"
    finally:
        shutil.rmtree(d)


# --- direction-of-comparison: at least one ONE-SIDED reading (unless all-neither) ---

def _union_hedge():
    return {"shape": "union", "output": "...",
            "evidence": {"command": "cmd", "observed": "ok"},
            "extras_direction": "ignored"}


def test_all_symmetric_readings_block():
    """The retry16 trap: a reading set that is ALL 'both' is direction-blind."""
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        r = dict(GROUNDED, normalized_side="both")
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [r, dict(r)], "hedge": _union_hedge()}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "reading_direction_uniform"
    finally:
        shutil.rmtree(d)


def test_missing_normalized_side_blocks():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        r = {"reading": "r", "source": GROUNDED["source"], "acceptance_test": "x"}
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [r], "hedge": _union_hedge()}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "reading_direction_missing"
    finally:
        shutil.rmtree(d)


def test_invalid_normalized_side_blocks():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        r = dict(GROUNDED, normalized_side="symmetric")
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [r], "hedge": _union_hedge()}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "reading_direction_missing"
    finally:
        shutil.rmtree(d)


def test_asymmetric_side_passes():
    """A one-sided (reference) reading alongside a symmetric one passes."""
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [dict(GROUNDED, reading="reading A", normalized_side="reference"),
                           dict(GROUNDED, reading="reading B", normalized_side="both")],
              "hedge": _union_hedge()}
        assert guard.check_pre(_ctx(dr)) is None
    finally:
        shutil.rmtree(d)


def test_neither_side_passes():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [dict(GROUNDED, reading="reading A", normalized_side="neither"),
                           dict(GROUNDED, reading="reading B", normalized_side="candidate")],
              "hedge": _union_hedge()}
        assert guard.check_pre(_ctx(dr)) is None
    finally:
        shutil.rmtree(d)


def test_all_neither_passes_exact_equality_task():
    """An all-'neither' set declares a pure byte/value-equality task with NO
    normalization dimension (e.g. an exact-JSON grader) — admissible. This is the
    extract-elf case (reward=1 with normalized_side='neither'); requiring a
    one-sided direction unconditionally would false-block it."""
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [dict(GROUNDED, reading="reading A", normalized_side="neither"),
                           dict(GROUNDED, reading="reading B", normalized_side="neither")],
              "hedge": _union_hedge()}
        assert guard.check_pre(_ctx(dr)) is None
    finally:
        shutil.rmtree(d)


def test_neither_plus_both_blocks():
    """The retry19 trap: a set mixing 'neither' with 'both' but omitting the
    one-sided case is direction-blind — it asserts a normalization is in play yet
    carries no comparison that can reject a wrong output serialization."""
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [dict(GROUNDED, reading="reading A", normalized_side="neither"),
                           dict(GROUNDED, reading="reading B", normalized_side="both")],
              "hedge": _union_hedge()}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "reading_direction_uniform"
    finally:
        shutil.rmtree(d)


def test_conflicting_normalized_side_blocks():
    """The SAME reading re-declared with a different direction is deception:
    an agent padding an all-'both' set with a duplicate 'reference' entry to
    fake asymmetry must be blocked (reviewer F2/F3)."""
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [dict(GROUNDED, normalized_side="both"),
                           dict(GROUNDED, normalized_side="both"),
                           dict(GROUNDED, normalized_side="reference")],
              "hedge": _union_hedge()}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "reading_direction_conflict"
    finally:
        shutil.rmtree(d)


def test_non_string_normalized_side_is_malformed():
    """A present-but-non-string normalized_side is malformed, not merely
    'missing' (reviewer F4 — the str() coercion hid a type error)."""
    for bad in (["reference"], 1, True, {"x": "reference"}):
        d, guard = _guard_with_step(["A1"], TASK)
        try:
            dr = {"deliverable": "out.json", "grader_visible": False,
                  "readings": [dict(GROUNDED, normalized_side=bad)],
                  "hedge": _union_hedge()}
            v = guard.check_pre(_ctx(dr))
            assert v is not None and v.reason == "deliverable_reading_malformed", bad
        finally:
            shutil.rmtree(d)


# --- awareness must be NON-WAIVABLE by prose (not merely prompted) ---
# The direction blocks are released ONLY by the concrete corrective action
# (declare a real normalized_side / a one-sided or all-neither set), never by an
# _override_reason. This is what converts "occasionally aware" into "structurally
# forced aware" — the observed failure was filter-js retry19, where the
# reading_direction_uniform block FIRED and was then OVERRIDDEN with prose.

def test_direction_uniform_block_is_non_overridable():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        r = dict(GROUNDED, normalized_side="both")
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [r, dict(r)], "hedge": _union_hedge()}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "reading_direction_uniform"
        assert v.overridable is False, "direction-blind block must be non-overridable"
    finally:
        shutil.rmtree(d)


def test_direction_missing_block_is_non_overridable():
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        r = {"reading": "r", "source": GROUNDED["source"], "acceptance_test": "x"}
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [r], "hedge": _union_hedge()}
        v = guard.check_pre(_ctx(dr))
        assert v is not None and v.reason == "reading_direction_missing"
        assert v.overridable is False
    finally:
        shutil.rmtree(d)


def test_prose_override_does_not_release_direction_block():
    """End-to-end through the registry: a prose _override_reason authored for the
    direction block must NOT release it (the pre-change behaviour released it)."""
    from flagscale_agent.react.guard import GuardRegistry

    def _reg_ctx(dr, override):
        c = _ctx(dr)
        c.override_reason = override
        return c

    d, guard = _guard_with_step(["A1"], TASK)
    try:
        reg = GuardRegistry()
        reg.register(guard)
        r = dict(GROUNDED, normalized_side="both")
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [r, dict(r)], "hedge": _union_hedge()}
        v1 = reg.check_pre(_reg_ctx(dr, ""))          # surfaces the block
        v2 = reg.check_pre(_reg_ctx(dr, "trust me, it is verified"))  # prose override
        assert v1 is not None and v1.reason == "reading_direction_uniform"
        assert v2 is not None and v2.reason == "reading_direction_uniform", \
            "prose override must NOT release the non-overridable direction block"
    finally:
        shutil.rmtree(d)


def test_corrective_action_releases_direction_requirement():
    """The concrete corrective action — a one-sided direction — satisfies the
    direction requirement (the block was never about rewarding prose)."""
    d, guard = _guard_with_step(["A1"], TASK)
    try:
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [dict(GROUNDED, normalized_side="reference")],
              "hedge": _union_hedge()}
        assert guard.check_pre(_ctx(dr)) is None
    finally:
        shutil.rmtree(d)


def test_hedge_arm_block_is_non_overridable():
    """The real retry15 instance released the HEDGE arm with prose after the
    agent had CONSTRUCTED the union (a value-clause cited as a presence-clause).
    Every block in the deliverable_reading gate is now non-overridable."""
    from flagscale_agent.react.guard import GuardRegistry

    def _reg_ctx(dr, override):
        c = _ctx(dr)
        c.override_reason = override
        return c

    d, guard = _guard_with_step(["A1"], TASK)
    try:
        reg = GuardRegistry()
        reg.register(guard)
        dr = {"deliverable": "out.json", "grader_visible": False,
              "readings": [dict(GROUNDED, normalized_side="reference")],
              "hedge": {"shape": "none"}}          # incomplete if_none
        v1 = reg.check_pre(_reg_ctx(dr, ""))
        v2 = reg.check_pre(_reg_ctx(
            dr, "the example fixes only the FORM of the base; penalize-wrong-values "
                "makes union inadmissible"))
        assert v1 is not None and v1.reason == "hedge_unproven"
        assert v1.overridable is False, "hedge arm must be non-overridable"
        assert v2 is not None, "prose override must NOT release the hedge block"
    finally:
        shutil.rmtree(d)

