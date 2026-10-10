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
   once the reviewer was demanded; an override reason releases it for that
   attempt (no latch — a later bare attempt re-blocks).
4. Flag semantics: _reviewer_demanded survives reset_turn; the wrap-up gate
   delivers its own checklist on ITS first arrival even on an attempt whose
   override released the reviewer gate.
"""

import json

from flagscale_agent.react.guard.plan import (
    PlanGuard, _DIVERGER_INJECT, _DIVERGER_ENFORCE, _QUALIFIER_EXTRACTION,
    _dispatch_includes_diverger,
)
from flagscale_agent.react.multi_agent.spawn import (
    _is_diverger_goal, maybe_auto_inject_diverger,
)
from flagscale_agent.react.guard.verification import (
    VerificationGuard,
    _REVIEWER_FINDINGS,
    _REVIEWER_FINDINGS_SETTLE,
    _STEP_DONE_PREMORTEM,
)
from flagscale_agent.react.guard import (
    Guard, GuardContext, GuardRegistry, GuardVerdict,
)


def _plan_ctx(tool_name="plan_create", tool_args=None):
    return GuardContext(tool_name=tool_name, tool_args=tool_args or {})


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

    def test_reviewer_demand_demands_skip_declaration(self):
        """Skipping the reviewer must force an oracle blind-spot declaration —
        not a bare 'my oracle is strongest' exemption (prop_7e1d594a)."""
        flat = " ".join(_REVIEWER_FINDINGS.split())  # templates are line-wrapped
        assert "ORACLE BLIND-SPOT DECLARATION" in flat
        # the classes only a non-author reader would catch
        assert "constraint / mapping / format classes" in flat

    def test_settle_gate_demands_closure_record(self):
        """Settlement record must include task_id + report path, or an explicit
        skip declaration with the blind-spot classes (prop_7e1d594a)."""
        flat = " ".join(_REVIEWER_FINDINGS_SETTLE.split())  # survives rewrap (review F2, task 9a0776106c17)
        assert "reviewer task_id" in flat
        assert "report path" in flat
        assert "skip declaration" in flat
        assert "oracle blind-spot classes" in flat


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

    def test_wrong_guard_override_releases_reviewer_but_wrapup_still_delivers(self):
        """An override written for the wrap-up gate releases the reviewer gate
        (any non-empty reason documents the round) — but it must NOT also
        silence the wrap-up delivery: the wrap-up gate blocks with its full
        checklist on THIS attempt (its first arrival), and passes on the next."""
        guard = VerificationGuard()
        _pass_step_done(guard)
        ctx = _completion_ctx(with_override=True,
                              override_text="deliverables verified, paths checked, cleanup done")
        v = guard.check_pre(ctx)
        # reviewer gate released (override) → wrap-up gate's FIRST arrival
        # delivers the checklist unconditionally
        assert v is not None and v.action == "block"
        assert v.reason == "text_complete_hygiene"
        assert guard._text_complete_hygiene_demanded is True
        # the next override attempt passes through — checklist delivered once
        assert guard.check_pre(_completion_ctx(with_override=True,
                                               override_text="done")) is None
        # ... but a later BARE attempt re-blocks (reviewer gate has no latch)
        assert guard.check_pre(_completion_ctx()).reason == "reviewer_findings_unaddressed"

    def test_gate_releases_with_override_documenting_settlement(self):
        guard = VerificationGuard()
        _pass_step_done(guard)
        ctx = _completion_ctx(
            with_override=True,
            override_text="reviewer finding 1 refuted: ran cited input, output matched; finding 2 confirmed and fixed; finding 3 refuted with counterexample",
        )
        # the override releases the reviewer gate for this attempt; the wrap-up
        # gate still delivers its checklist on ITS first arrival
        v1 = guard.check_pre(ctx)
        assert v1 is not None and v1.reason == "text_complete_hygiene"
        # ... and a later attempt is NOT re-blocked by the reviewer gate (no
        # latch on an override retry), and the wrap-up gate — already
        # delivered — passes through
        assert guard.check_pre(_completion_ctx(with_override=True,
                                               override_text="done")) is None
        # ... but a later BARE attempt re-blocks (reviewer gate has no latch)
        assert guard.check_pre(_completion_ctx()).reason ==             "reviewer_findings_unaddressed"

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
        # second attempt: agent supplies an override → it releases the
        # reviewer gate (any non-empty reason documents the round), but the
        # wrap-up gate — same guard, later in the same check_pre — delivers
        # its own checklist on THIS attempt (first arrival), regardless of
        # the override. One override releases exactly the gate it answers;
        # it never also swallows the other gate's only delivery.
        ctx_wrong = _completion_ctx(
            with_override=True,
            override_text="completed all steps and cleanup is done",
        )
        v2 = registry.check_pre(ctx_wrong)
        assert v2 is not None and v2.action == "block"
        assert v2.reason == "text_complete_hygiene"
        assert guard._text_complete_hygiene_demanded is True


class TestFlagSemantics:
    """Flag contract: demanded survives reset_turn; other flags don't."""

    def test_reviewer_demanded_survives_reset_turn(self):
        guard = VerificationGuard()
        _pass_step_done(guard)
        assert guard._reviewer_demanded is True
        guard.reset_turn()
        assert guard._reviewer_demanded is True

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


class TestDivergerDispatchBlock:
    """The diverger demand is ENFORCED: after a plan_create that carried the
    demand, the FIRST non-plan tool call BLOCKS once (overridable) unless a
    diverger was dispatched. Exempt for reviewer/diverger worker processes
    (user ruling: only non-reviewer, non-diverger are gated)."""

    def test_block_fires_on_first_tool_after_plan_create(self):
        guard = PlanGuard()
        v = guard.check_pre(_plan_ctx())
        assert v is not None and v.action == "inject"  # plan_create advisory
        assert guard._diverger_pending is True
        b = guard.check_pre(_plan_ctx("shell"))
        assert b is not None and b.action == "block"
        assert b.reason == "diverger_not_dispatched"
        assert b.overridable is True
        assert _DIVERGER_INJECT in b.message

    def test_block_fires_only_once_then_releases(self):
        guard = PlanGuard()
        guard.check_pre(_plan_ctx())
        first = guard.check_pre(_plan_ctx("shell"))
        assert first is not None and first.action == "block"
        assert guard._diverger_pending is False
        # the gate is one-shot: the next tool call is not blocked
        assert guard.check_pre(_plan_ctx("shell")) is None

    def test_complying_spawn_as_first_action_is_not_blocked(self):
        """A compliant agent whose FIRST action after plan_create IS the
        dispatch must not be blocked on that very call."""
        guard = PlanGuard()
        guard.check_pre(_plan_ctx())
        assert guard.check_pre(_plan_ctx("spawn_worker")) is None
        assert guard.check_pre(_plan_ctx("dispatch_many")) is None

    def test_spawn_worker_clears_pending(self):
        """A spawn whose goal IS a diverger clears the gate. Exercise the REAL
        kernel order: check_pre (per tool) THEN check_post."""
        guard = PlanGuard()
        guard.check_pre(_plan_ctx())
        spawn = _plan_ctx(
            "spawn_worker",
            {"goal": "Propose 2-3 genuinely different framings of the task."})
        assert guard.check_pre(spawn) is None  # dispatch tool never blocked
        guard.check_post(spawn)
        assert guard._diverger_pending is False
        assert guard.check_pre(_plan_ctx("shell")) is None

    def test_dispatch_many_clears_pending(self):
        """A dispatch_many carrying a diverger spec clears the gate."""
        guard = PlanGuard()
        guard.check_pre(_plan_ctx())
        dm = _plan_ctx(
            "dispatch_many",
            {"specs": [{"goal": "Propose 2-3 alternative framings."}]})
        assert guard.check_pre(dm) is None
        guard.check_post(dm)
        assert guard._diverger_pending is False
        assert guard.check_pre(_plan_ctx("shell")) is None

    def test_non_diverger_spawn_does_not_clear(self):
        """A spawn for ANY other purpose must NOT satisfy the demand (F1):
        the gate stays armed and the next tool still blocks. Real kernel order:
        check_pre THEN check_post (check_pre must not erase the gate first)."""
        guard = PlanGuard()
        guard.check_pre(_plan_ctx())
        spawn = _plan_ctx(
            "spawn_worker", {"goal": "Port the model to Megatron."})
        assert guard.check_pre(spawn) is None
        guard.check_post(spawn)
        assert guard._diverger_pending is True
        assert guard.check_pre(_plan_ctx("shell")) is not None

    def test_non_diverger_dispatch_many_does_not_clear(self):
        guard = PlanGuard()
        guard.check_pre(_plan_ctx())
        dm = _plan_ctx(
            "dispatch_many",
            {"specs": [{"goal": "Train config A."},
                       {"goal": "Train config B."}]})
        assert guard.check_pre(dm) is None
        guard.check_post(dm)
        assert guard._diverger_pending is True
        assert guard.check_pre(_plan_ctx("shell")) is not None

    def test_plan_update_and_status_never_blocked(self):
        """plan_update / plan_status are plan tools: they must not trip the
        dispatch gate (and must not clear it either)."""
        guard = PlanGuard()
        guard.check_pre(_plan_ctx())
        assert guard.check_pre(_plan_ctx("plan_update")) is None
        assert guard.check_pre(_plan_ctx("plan_status")) is None
        assert guard._diverger_pending is True  # still armed
        assert guard.check_pre(_plan_ctx("shell")) is not None

    def test_reset_turn_clears_pending(self):
        guard = PlanGuard()
        guard.check_pre(_plan_ctx())
        assert guard._diverger_pending is True
        guard.reset_turn()
        assert guard._diverger_pending is False
        assert guard.check_pre(_plan_ctx("shell")) is None

    def test_reviewer_and_diverger_workers_exempt(
            self, tmp_path, monkeypatch):
        """The gate is parent-side only: a spawned reviewer OR diverger worker
        (both carry constraints.reviewer=True) is never blocked."""
        from flagscale_agent.react.multi_agent.wiring import (
            CONTRACT_PATH_ENV, TASK_ID_ENV,
        )
        task_dir = tmp_path / "task_x"
        task_dir.mkdir()
        (task_dir / "contract.prompt").write_text("plain", encoding="utf-8")
        (task_dir / "contract.json").write_text(
            json.dumps({"constraints": {"reviewer": True, "writable": []}}),
            encoding="utf-8")
        monkeypatch.setenv(TASK_ID_ENV, "task_x")
        monkeypatch.setenv(CONTRACT_PATH_ENV, str(task_dir / "contract.prompt"))
        guard = PlanGuard()
        guard._diverger_pending = True
        assert guard.check_pre(_plan_ctx("shell")) is None


class TestDivergerGoalAutoTag:
    """A diverger GOAL is auto-tagged with the SEPARATE `diverger` flag at the
    spawn layer (never `reviewer` — that would enter the reviewer preset)."""

    def test_diverger_goal_detected(self):
        assert _is_diverger_goal(
            "Propose 2-3 genuinely different framings of the task.")
        assert _is_diverger_goal("Act as a diverger for this plan.")
        assert _is_diverger_goal("Offer alternative framings.")

    def test_non_diverger_goal_not_detected(self):
        assert not _is_diverger_goal("Port the model to Megatron.")
        assert not _is_diverger_goal("Review the diff for bugs.")

    def test_injects_separate_flag_not_reviewer(self):
        cons = {}
        maybe_auto_inject_diverger(
            "Propose 2-3 genuinely different framings of the task.", cons)
        assert cons.get("diverger") is True
        assert "reviewer" not in cons

    def test_explicit_tag_suppresses_injection(self):
        cons = {"reviewer": True}
        maybe_auto_inject_diverger(
            "Propose 2-3 genuinely different framings.", cons)
        assert "diverger" not in cons


class TestDispatchIncludesDiverger:
    """F1: only an ACTUAL diverger dispatch satisfies the demand."""

    def test_spawn_worker_diverger_goal(self):
        assert _dispatch_includes_diverger(
            "spawn_worker", {"goal": "Propose 2-3 different framings."})

    def test_spawn_worker_diverger_tag(self):
        assert _dispatch_includes_diverger(
            "spawn_worker", {"goal": "x", "constraints": {"diverger": True}})

    def test_spawn_worker_plain(self):
        assert not _dispatch_includes_diverger(
            "spawn_worker", {"goal": "Port the model."})

    def test_dispatch_many_any_spec(self):
        assert _dispatch_includes_diverger(
            "dispatch_many",
            {"specs": [{"goal": "Train A."},
                       {"goal": "Propose alternative framings."}]})

    def test_dispatch_many_none(self):
        assert not _dispatch_includes_diverger(
            "dispatch_many",
            {"specs": [{"goal": "Train A."}, {"goal": "Train B."}]})


class TestDivergerBlockMessageConsistency:
    """F2: the block message must not leave the legacy inject wording
    ('advisory / never blocking') standing as this call's own description."""

    def test_block_message_marks_enforced_and_disambiguates(self):
        guard = PlanGuard()
        guard.check_pre(_plan_ctx())
        b = guard.check_pre(_plan_ctx("shell"))
        assert b is not None and b.action == "block"
        # the reused inject text is still present (back-compat) ...
        assert _DIVERGER_INJECT in b.message
        assert _DIVERGER_ENFORCE in b.message
        low = b.message.lower()
        # ... but the enforcement header disambiguates it.
        assert "reused verbatim" in low
        assert "enforced" in low


class TestExemptionSeparateFlags:
    """The exemption covers BOTH reviewer (`reviewer: True`) and diverger
    (`diverger: True`) workers — two DISTINCT structured flags."""

    def _worker_ctx(self, tmp_path, monkeypatch, constraints):
        from flagscale_agent.react.multi_agent.wiring import (
            CONTRACT_PATH_ENV, TASK_ID_ENV,
        )
        task_dir = tmp_path / "task_x"
        task_dir.mkdir()
        (task_dir / "contract.prompt").write_text("plain", encoding="utf-8")
        (task_dir / "contract.json").write_text(
            json.dumps({"constraints": constraints}), encoding="utf-8")
        monkeypatch.setenv(TASK_ID_ENV, "task_x")
        monkeypatch.setenv(CONTRACT_PATH_ENV, str(task_dir / "contract.prompt"))
        guard = PlanGuard()
        guard._diverger_pending = True
        return guard

    def test_reviewer_flag_exempt(self, tmp_path, monkeypatch):
        guard = self._worker_ctx(tmp_path, monkeypatch, {"reviewer": True})
        assert guard.check_pre(_plan_ctx("shell")) is None

    def test_diverger_flag_exempt(self, tmp_path, monkeypatch):
        guard = self._worker_ctx(tmp_path, monkeypatch, {"diverger": True})
        assert guard.check_pre(_plan_ctx("shell")) is None

    def test_plain_worker_not_exempt(self, tmp_path, monkeypatch):
        guard = self._worker_ctx(tmp_path, monkeypatch, {})
        assert guard.check_pre(_plan_ctx("shell")) is not None


class TestDivergerBlockNotSurfacedReArms:
    """F4: when a HIGHER-ranked block wins the resolve, PlanGuard's diverger
    block never reaches the agent, so its one-shot consumption of
    `_diverger_pending` must be UNDONE (registry calls block_not_surfaced) and
    the demand re-fires once the winning block clears."""

    class _HardBlock(Guard):
        """Non-overridable block on shell — outranks the overridable diverger."""
        name = "hardblock"
        priority = 0

        def __init__(self, active=True):
            self.active = active

        def check_pre(self, ctx):
            if self.active and ctx.tool_name == "shell":
                return GuardVerdict.block(
                    message="hard", reason="hard_reason",
                    category="x", overridable=False)
            return None

    def _registry(self):
        registry = GuardRegistry()
        hard = self._HardBlock()
        registry.register(hard)
        plan = PlanGuard()
        registry.register(plan)
        return registry, hard, plan

    def test_outranked_diverger_demand_survives_and_refires(self):
        registry, hard, plan = self._registry()
        registry.check_pre(_plan_ctx())  # arms the gate
        assert plan._diverger_pending is True
        v = registry.check_pre(_plan_ctx("shell"))
        # the hard block wins the ranking; the diverger block is not surfaced
        assert v is not None and v.reason == "hard_reason"
        # ... yet the demand is NOT consumed (F4 fix): it must re-fire later
        assert plan._diverger_pending is True
        hard.active = False  # the hard wall clears
        v2 = registry.check_pre(_plan_ctx("shell"))
        assert v2 is not None and v2.reason == "diverger_not_dispatched"

    def test_surfaced_block_still_consumes_once(self):
        """Control: with no competitor the block surfaces and the one-shot
        demand IS consumed exactly once (the fix did not break the normal path)."""
        registry, hard, plan = self._registry()
        hard.active = False
        registry.check_pre(_plan_ctx())
        v = registry.check_pre(_plan_ctx("shell"))
        assert v is not None and v.reason == "diverger_not_dispatched"
        assert plan._diverger_pending is False
        v_next = registry.check_pre(_plan_ctx("shell"))
        assert not (v_next is not None and v_next.action == "block")

    def test_other_blocks_do_not_spuriously_rearm(self):
        """A non-diverger-origin block losing the ranking must NOT re-arm the
        dispatch gate (the marker is scoped to the diverger block branch)."""
        guard = PlanGuard()
        guard._diverger_pending = False
        guard.block_not_surfaced()  # no diverger block consumed this resolve
        assert guard._diverger_pending is False
