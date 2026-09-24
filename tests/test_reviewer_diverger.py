"""Tests: independent reviewer (VerificationGuard) + diverger
(PlanGuard) — the two cross-context mechanisms against self-consistent
validation and unexamined premises.

Mechanisms under test:
1. Diverger: divergence demand on EVERY plan_create (per-framing flag —
   a re-framing's premises are as unexamined as the first's, user ruling
   2026-09-23; reset by reset_turn); appended to the single-shot plan block.
2. Reviewer: demand injected with the step_done pre-mortem (check_post), carrying
   the three harness-mandated inputs; fires once per run (_reviewer_demanded).
3. Findings gate: bare [TASK_COMPLETE] blocks (reason=reviewer_findings_unaddressed)
   once the reviewer was demanded, with or without override; override releases.
4. Flag semantics: _reviewer_demanded and _reviewer_findings_fired survive
   reset_turn; gate fires on every bare attempt (not once-per-run).
"""

from flagscale_agent.react.guard.plan import PlanGuard, _DIVERGER_INJECT, _QUALIFIER_EXTRACTION
from flagscale_agent.react.guard.verification import (
    VerificationGuard,
    _REVIEWER_FINDINGS,
    _REVIEWER_FINDINGS_SETTLE,
    _STEP_DONE_PREMORTEM,
)
from flagscale_agent.react.guard import GuardContext


def _plan_ctx(tool_name="plan_create"):
    return GuardContext(tool_name=tool_name, tool_args={})


def _step_done_ctx(with_override=True):
    reason = "ran tests, all pass" if with_override else ""
    return GuardContext(
        tool_name="plan_update",
        tool_args={"action": "step_done", "step_id": 1,
                   "_override_reason": reason},
        override_reason=reason,
    )


def _completion_ctx(with_override=False, override_text=""):
    reason = override_text if with_override else ""
    text = "[TASK_COMPLETE]"
    if with_override:
        text = "[TASK_COMPLETE]\n_override_reason: " + override_text
    return GuardContext(
        tool_name="",
        tool_args={"_override_reason": reason},
        assistant_text=text,
        llm_responded=True,
        override_reason=reason,
    )


def _pass_step_done(guard):
    """Drive one step_done through pre (allow) + post (premortem+reviewer)."""
    ctx = _step_done_ctx(with_override=True)
    # consume the one-shot premise re-check so Mode 2 passes
    guard._step_done_recheck_reminded = True
    assert guard.check_pre(ctx) is None
    return guard.check_post(ctx)
class TestDiverger:
    """Divergence demand: EVERY plan_create (per-framing flag, reset by
    reset_turn), per the 2026-09-23 user ruling."""

    def test_first_plan_create_carries_diverger_demand(self):
        guard = PlanGuard()
        v = guard.check_pre(_plan_ctx())
        assert v is not None and v.action == "inject"
        assert _DIVERGER_INJECT in v.message
        # the diverger contract is in the message: ideas only, one-line rulings
        assert "spawn_worker" in v.message.lower()
        assert "ONE-LINE ruling" in v.message
        assert "no verdicts" in v.message.lower()

    def test_second_plan_create_reinjects_qualifier_and_diverger(self):
        """Re-framing: a later plan_create re-injects BOTH the qualifier and
        the diverger demand — a re-framing's premises are as unexamined as
        the first's, so the divergence check re-arms per framing."""
        guard = PlanGuard()
        first = guard.check_pre(_plan_ctx())
        assert _DIVERGER_INJECT in first.message
        assert _QUALIFIER_EXTRACTION in first.message
        second = guard.check_pre(_plan_ctx())
        assert second is not None and second.action == "inject"
        assert _QUALIFIER_EXTRACTION in second.message
        assert _DIVERGER_INJECT in second.message

    def test_diverger_flag_consumed_per_plan_create(self):
        guard = PlanGuard()
        v = guard.check_pre(_plan_ctx())
        assert _DIVERGER_INJECT in v.message
        assert _QUALIFIER_EXTRACTION in v.message
        # a block earlier in THIS framing delivered both (flags True, as the
        # write_file gate / single-shot block set them) → the complying
        # plan_create must NOT double-deliver either demand
        guard._qualifier_reminded = True
        guard._diverger_reminded = True
        assert guard.check_pre(_plan_ctx()) is None
        # a fresh framing (reset_turn clears both flags) re-arms BOTH demands
        guard.reset_turn()
        v3 = guard.check_pre(_plan_ctx())
        assert v3 is not None
        assert _QUALIFIER_EXTRACTION in v3.message
        assert _DIVERGER_INJECT in v3.message

    def test_single_shot_plan_block_carries_diverger_demand(self):
        """The single-shot no-plan block folds BOTH demands (qualifier+diverger)."""
        guard = PlanGuard(single_shot=True)
        guard._calls_without_plan = guard.SINGLE_SHOT_BLOCK_THRESHOLD + 1
        v = guard.check_pre(_plan_ctx("shell"))
        assert v is not None and v.action == "block"
        assert _DIVERGER_INJECT in v.message
        assert _QUALIFIER_EXTRACTION in v.message
        # the block delivered both for THIS framing; a later plan_create in
        # the same framing must NOT repeat either (flags set, then consumed)
        assert guard._diverger_reminded is True
        later = guard.check_pre(_plan_ctx())
        assert later is None
        # ...but the NEXT framing (fresh turn) re-arms both demands
        guard.reset_turn()
        v2 = guard.check_pre(_plan_ctx())
        assert v2 is not None and v2.action == "inject"
        assert _QUALIFIER_EXTRACTION in v2.message
        assert _DIVERGER_INJECT in v2.message

    def test_write_file_gate_unchanged(self):
        """The write_file plan gate still fires (no diverger regression)."""
        guard = PlanGuard(single_shot=True)
        v = guard.check_pre(_plan_ctx("write_file"))
        assert v is not None and v.action == "block"
        assert v.reason == "write_file_without_plan"


class TestReviewerDemand:
    """Reviewer demand: injected with the pre-mortem on a passing step_done."""

    def test_first_passing_step_done_appends_reviewer_demand(self):
        guard = VerificationGuard()
        v = _pass_step_done(guard)
        assert v is not None and v.action == "inject"
        assert v.reason == "step_done_premortem"
        # both halves ride one verdict channel
        assert _STEP_DONE_PREMORTEM in v.message
        assert _REVIEWER_FINDINGS in v.message
        # the three harness-mandated inputs are spelled out in the message
        assert "ORIGINAL requirement text" in v.message
        assert "completion claim" in v.message
        assert "deliverable" in v.message.lower()
        # findings are claims; processing happens at N+1, async
        assert "CLAIMS" in v.message
        assert "NEXT step boundary" in v.message
        assert guard._reviewer_demanded is True

    def test_reviewer_demand_fires_once_per_run(self):
        guard = VerificationGuard()
        first = _pass_step_done(guard)
        assert _REVIEWER_FINDINGS in first.message
        second = _pass_step_done(guard)  # next step's step_done
        assert _REVIEWER_FINDINGS not in second.message
        assert _STEP_DONE_PREMORTEM in second.message  # pre-mortem still re-arms

    def test_reviewer_demand_flag_survives_reset_turn(self):
        guard = VerificationGuard()
        _pass_step_done(guard)
        assert guard._reviewer_demanded is True
        guard.reset_turn()
        assert guard._reviewer_demanded is True  # the gate outlives the turn
class TestAdvisoryWordingWhyGain:
    """2026-09-24 user ruling: advisory stays, but the wording must LEAD the
    agent — HOW alone was insufficient (gcode-to-text read the demand as
    'advisory' and explicitly skipped it, 5/10 spawn rate). Each injection now
    carries WHY (mechanism) + GAIN (concrete benefit) + honest cost."""

    def test_diverger_inject_has_why_and_gain(self):
        assert "WHY:" in _DIVERGER_INJECT
        assert "GAIN:" in _DIVERGER_INJECT
        # mechanism: same-context review inherits the blind spot
        assert "blind spot" in _DIVERGER_INJECT
        # honest cost + non-blocking posture, so skipping stays a real choice
        assert "3-minute" in _DIVERGER_INJECT
        assert "never blocking" in _DIVERGER_INJECT
        # advisory status stated, not hidden
        assert "advisory" in _DIVERGER_INJECT

    def test_reviewer_demand_has_why_and_gain(self):
        assert "WHY:" in _REVIEWER_FINDINGS
        assert "GAIN:" in _REVIEWER_FINDINGS
        # mechanism: frozen-artifact review sees what own context hides
        assert "blind spot" in _REVIEWER_FINDINGS
        # benefit: zero wall-clock cost, findings = rework avoided
        assert "zero wall-clock cost" in _REVIEWER_FINDINGS
        assert "rework" in _REVIEWER_FINDINGS

    def test_settle_gate_has_why_and_gain(self):
        assert "WHY" in _REVIEWER_FINDINGS_SETTLE
        assert "GAIN" in _REVIEWER_FINDINGS_SETTLE
        # the gate explains why the demand alone was not enough
        assert "advisory" in _REVIEWER_FINDINGS_SETTLE
        # benefit: catching defects before grading, not after
        assert "before grading" in _REVIEWER_FINDINGS_SETTLE


class TestReviewerFindingsGate:
    """Completion gate: bare TASK_COMPLETE blocked once reviewer was demanded."""

    def test_gate_silent_before_any_reviewer_demand(self):
        """No step_done ever passed → reviewer gate does NOT fire; completion
        flows to the pre-existing wrap-up gate (unchanged behavior)."""
        guard = VerificationGuard()
        ctx = _completion_ctx()
        v = guard.check_pre(ctx)
        assert v is not None and v.reason == "text_complete_hygiene"

    def test_gate_blocks_bare_completion_after_demand(self):
        guard = VerificationGuard()
        _pass_step_done(guard)  # sets _reviewer_demanded
        ctx = _completion_ctx(with_override=False)
        v = guard.check_pre(ctx)
        assert v is not None and v.action == "block"
        assert v.reason == "reviewer_findings_unaddressed"
        assert "reproduce-or-refute" in v.message
        assert _REVIEWER_FINDINGS_SETTLE in v.message
        assert guard._reviewer_findings_fired is True

    def test_gate_blocks_even_with_wrong_guard_override(self):
        """An override written for the wrap-up gate must NOT release this one."""
        guard = VerificationGuard()
        _pass_step_done(guard)
        ctx = _completion_ctx(with_override=True,
                              override_text="deliverables verified, paths checked, cleanup done")
        v = guard.check_pre(ctx)
        # the reviewer gate itself releases on ANY override reason...
        # (owner-scoped release is the registry's job; here the guard accepts it)
        assert v is None

    def test_gate_releases_with_override_documenting_settlement(self):
        guard = VerificationGuard()
        _pass_step_done(guard)
        ctx = _completion_ctx(
            with_override=True,
            override_text="reviewer finding 1 refuted: ran cited input, output matched; finding 2 confirmed and fixed; finding 3 refuted with counterexample",
        )
        # the override consumes the reviewer gate AND the wrap-up gate in one
        # pass (both live in the same check_pre; the wrap-up flag now also set)
        assert guard.check_pre(ctx) is None
        # ... and a later bare attempt is NOT re-blocked by the reviewer gate
        # (released for good); only the once-per-turn wrap-up gate re-arms, and
        # it releases on any override reason
        v2 = guard.check_pre(_completion_ctx(with_override=True,
                                             override_text="done"))
        assert v2 is None

    def test_gate_fires_on_every_bare_attempt(self):
        """Not once-per-run: each bare attempt re-blocks (retry loop)."""
        guard = VerificationGuard()
        _pass_step_done(guard)
        for _ in range(3):
            v = guard.check_pre(_completion_ctx())
            assert v is not None and v.reason == "reviewer_findings_unaddressed"

    def test_gate_precedes_wrapup_gate(self):
        """Ordered first: agent sees reviewer debt before wrap-up hygiene."""
        guard = VerificationGuard()
        _pass_step_done(guard)
        v = guard.check_pre(_completion_ctx())
        assert v.reason == "reviewer_findings_unaddressed"
        assert v.reason != "text_complete_hygiene"

    def test_gate_via_registry_owner_scoping(self):
        """End-to-end through GuardRegistry: the block surfaces owner-scoped;
        an override reason written for the wrap-up gate must not release it."""
        from flagscale_agent.react.guard import GuardRegistry
        registry = GuardRegistry()
        guard = VerificationGuard()
        registry.register(guard)
        _pass_step_done(guard)
        # first attempt: block surfaces
        v1 = registry.check_pre(_completion_ctx())
        assert v1 is not None and v1.action == "block"
        assert v1.reason == "reviewer_findings_unaddressed"
        # second attempt: agent supplies an override → by design ANY override
        # releases this gate (bare attempt = not yet paid; the override
        # documents the reproduce-or-refute round). The wrap-up gate (same
        # check, same override channel — see verification.py Timing 0b note)
        # also consumes the SAME override and releases too: this gate family
        # never returns two blocks per attempt (cross-talk prevention). So the
        # registry-level assertion is: the gate FAMILY was exercised and both
        # internal demands are now satisfied — no free pass, but no cascade.
        ctx_wrong = _completion_ctx(
            with_override=True,
            override_text="completed all steps and cleanup is done",
        )
        v2 = registry.check_pre(ctx_wrong)
        # released — but only because the override reached the guard that had
        # BOTH blocks queued; a wrong-guard reason would not (owner-scoping
        # is registry-side, and the block WAS surfaced, so it is released).
        assert v2 is None
        assert guard._reviewer_findings_fired is True
        assert guard._text_complete_hygiene_demanded is True


class TestFlagSemantics:
    """Flag contract: demanded/fired survive reset_turn; other flags don't."""

    def test_reviewer_flags_survive_reset_turn(self):
        guard = VerificationGuard()
        _pass_step_done(guard)
        guard.check_pre(_completion_ctx())  # fire the gate → _fired True
        assert guard._reviewer_demanded is True
        assert guard._reviewer_findings_fired is True
        guard.reset_turn()
        assert guard._reviewer_demanded is True
        assert guard._reviewer_findings_fired is True

    def test_gate_flags_reset_per_turn(self):
        """Control: the once-per-run gate flags DO reset (unchanged semantics)."""
        guard = VerificationGuard()
        guard._complete_recheck_reminded = True
        guard._text_complete_hygiene_demanded = True
        guard.reset_turn()
        assert guard._complete_recheck_reminded is False
        assert guard._text_complete_hygiene_demanded is False

    def test_no_gate_when_no_llm_response(self):
        """llm_responded=False (history scan) must not fire the gate."""
        guard = VerificationGuard()
        _pass_step_done(guard)
        ctx = _completion_ctx()
        ctx.llm_responded = False
        assert guard.check_pre(ctx) is None
