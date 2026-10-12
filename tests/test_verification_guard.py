# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for VerificationGuard — requires verification evidence before step_done."""

import pytest

from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.guard import GuardContext


class TestVerificationGuard:
    """Test VerificationGuard blocks step_done without override_reason."""

    def test_blocks_step_done_without_override_reason(self):
        """step_done without _override_reason should be blocked (Mode 2)."""
        guard = VerificationGuard()
        # isolate the Mode 2 evidence check: consume the one-shot premise re-check
        # block that now fires ahead of it on the first step_done of the run.
        guard._step_done_recheck_reminded = True

        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 3}
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is not None
        assert verdict.action == "block"
        # reconstructed message poses a self-check question (what did you OBSERVE)
        # rather than the old "verification required" assertion-style phrasing.
        assert "observe" in verdict.message.lower()
        assert verdict.reason == "step_done_no_verification"

    def test_allows_step_done_with_override_reason(self):
        """step_done with _override_reason should pass."""
        guard = VerificationGuard()
        
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "step_done",
                "step_id": 3,
                "_override_reason": "grep shows no conflicts, files parseable"
            },
            override_reason="grep shows no conflicts, files parseable"
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is None  # Should pass

    def test_allows_empty_override_reason_is_blocked(self):
        """Empty _override_reason should still be blocked."""
        guard = VerificationGuard()
        
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "step_done",
                "step_id": 3,
                "_override_reason": "   "  # whitespace only
            }
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is not None
        assert verdict.action == "block"

    def test_allows_other_plan_update_actions(self):
        """Other plan_update actions (step_doing, add_steps, etc.) should pass."""
        guard = VerificationGuard()
        
        # "complete" now has its own one-shot re-check block (see
        # TestTaskCompleteRecheck), so it is intentionally excluded here.
        actions = ["step_doing", "step_skip", "add_steps", "abandon"]
        
        for action in actions:
            ctx = GuardContext(
                tool_name="plan_update",
                tool_args={"action": action, "step_id": 3}
            )
            verdict = guard.check_pre(ctx)
            assert verdict is None, f"Action {action} should not be blocked"

    def test_allows_all_other_tools(self):
        """All other tools should pass through without blocking."""
        guard = VerificationGuard()
        
        tools = [
            "shell", "read_file", "write_file", "edit_file",
            "memory_read", "memory_write", "grep", "evict"
        ]
        
        for tool in tools:
            ctx = GuardContext(tool_name=tool, tool_args={})
            verdict = guard.check_pre(ctx)
            assert verdict is None, f"Tool {tool} should not be blocked"

    def test_first_step_done_blocks_for_premise_recheck(self):
        """The first step_done of a run is blocked once for a premise re-check,
        even when the agent already supplied verification. This is the finalize-time
        counterpart to the plan_create qualifier block: it forces the agent to
        re-examine the premises it operated under (a convenient reading of a term,
        a "close enough", a value taken as given) before any evidence is accepted —
        the exact blind spot that a plan-time block cannot reach, because a premise
        is re-interpreted during execution and only surfaces at completion."""
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "step_done",
                "step_id": 1,
                "verification": ["did the thing"],
            },
        )
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "step_done_premise_recheck"

    def test_premise_recheck_released_by_override_then_fires_evidence_check(self):
        """Override releases the premise re-check, but the step must STILL carry
        evidence — the re-check does not exempt the step from the Mode 1/2 checks.
        A second step_done no longer hits the premise block (fires once per run)."""
        guard = VerificationGuard()
        # first step_done: override releases the premise re-check, then falls through
        # to Mode 2 (no acceptance) which is satisfied by the same override_reason.
        ctx1 = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "step_done",
                "step_id": 1,
                "_override_reason": "re-read criteria, re-examined premises",
            },
            override_reason="re-read criteria, re-examined premises",
        )
        assert guard.check_pre(ctx1) is None
        assert guard._step_done_recheck_reminded is True

        # second step_done without override: premise block is spent, so this now
        # falls straight to the Mode 2 evidence check and is blocked for THAT reason.
        ctx2 = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 2},
        )
        verdict2 = guard.check_pre(ctx2)
        assert verdict2 is not None
        assert verdict2.reason == "step_done_no_verification"

    def test_premise_recheck_reminder_is_task_agnostic_and_premise_focused(self):
        """The reminder must target premise re-examination in general terms, never
        naming a specific constraint kind (time/version/subset) — that would read as
        case-by-case gaming rather than a general finalize discipline."""
        from flagscale_agent.react.guard.verification import _STEP_DONE_RECHECK_REMINDER
        low = _STEP_DONE_RECHECK_REMINDER.lower()
        # names the core move: test belief against result, not restate it
        assert "premise" in low
        assert "restating" in low or "repeats the assumptions" in low
        # names convenience-premise as the trap
        assert "convenience" in low
        # points back at the user's actual ask — off-target competence still misses
        assert "user actually asked" in low or "the user actually asked" in low
        assert "off-target" in low
        assert "wording wins" in low
        # forcing checkpoint, not content check
        assert "not a content check" in low
        assert "_override_reason" in _STEP_DONE_RECHECK_REMINDER
        # no task specifics, no case-by-case constraint kinds
        for w in ("august", "mteb", "scandinavian", "leaderboard", "time boundary"):
            assert w not in low

    def test_premise_recheck_demands_concrete_anchor_not_summary(self):
        """The override_reason instruction must require a CONCRETE ANCHOR (value/command/
        path) or explicit "no evidence", not accept summary-of-diligence ("I re-checked").
        This defends against self-audit degenerating into empty affirmation."""
        from flagscale_agent.react.guard.verification import _STEP_DONE_RECHECK_REMINDER
        low = _STEP_DONE_RECHECK_REMINDER.lower()
        # demands concrete anchor in override_reason
        assert "concrete anchor" in low
        # explicitly names the empty answer this replaces
        assert "empty answer" in low
        assert "re-checked and it's correct" in low or "i re-checked" in low
        # offers specific anchor types: value observed vs expected, premise tested
        assert "value you observed" in low or "observed vs" in low
        assert "premise you re-tested" in low or "re-tested and what" in low
        # provides escape hatch for untestable premise — but must SAY SO explicitly
        assert "no evidence" in low
        # preserves override mechanism — any reason passes, but lack of anchor IS signal
        assert "not a content check" in low
        assert "any override_reason lets" in low or "any override_reason" in low
        assert "reason with no anchor" in low

    def test_premise_recheck_carries_under_specified_clause_discipline(self):
        """The step_done premise re-check must also carry the under-specified-clause
        discipline (the same rule in SYSTEM_PROMPT): when a requirement admitted two
        defensible readings and the checker is invisible, the agent must name the
        reading it produced against / the tie it broke, rather than settle silently.
        Guards against ambiguous-clause coin-flips surviving to delivery."""
        from flagscale_agent.react.guard.verification import _STEP_DONE_RECHECK_REMINDER
        low = _STEP_DONE_RECHECK_REMINDER.lower()
        # names the under-specified condition
        assert "under-specified" in low
        # naming the chosen reading / the bet, not settling silently
        assert "reading" in low
        # deterministic tie-break ladder present (wording-literal first)
        assert "explicitly grants" in low
        assert "mutually exclusive" in low
        # still a forcing checkpoint over an anchor, not a content check
        assert "reading you chose" in low
        assert "not a content check" in low

    def test_post_recovery_inject_on_first_step_doing(self):
        """After notify_recovery(), should inject reminder on first step_doing."""
        guard = VerificationGuard()
        
        # Trigger recovery
        guard.notify_recovery()
        
        # First step_doing should inject reminder
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_doing", "step_id": 2}
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is not None
        assert verdict.action == "inject"
        assert "recovered via hard_reset" in verdict.message.lower()
        assert verdict.reason == "post_recovery_reminder"

    def test_post_recovery_inject_only_once(self):
        """Post-recovery reminder should only fire once."""
        guard = VerificationGuard()
        
        guard.notify_recovery()
        
        # First step_doing - should inject
        ctx1 = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_doing", "step_id": 2}
        )
        verdict1 = guard.check_pre(ctx1)
        assert verdict1 is not None
        assert verdict1.action == "inject"
        
        # Second step_doing - should not inject
        ctx2 = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_doing", "step_id": 3}
        )
        verdict2 = guard.check_pre(ctx2)
        assert verdict2 is None

    def test_post_recovery_does_not_affect_step_done_blocking(self):
        """Post-recovery state should not interfere with step_done blocking."""
        guard = VerificationGuard()
        
        guard.notify_recovery()
        # isolate the Mode 2 evidence check from the one-shot premise re-check block
        guard._step_done_recheck_reminded = True

        # step_done without override_reason should still be blocked
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 3}
        )
        verdict = guard.check_pre(ctx)
        
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "step_done_no_verification"


class TestConstraintGuidanceBlockedComputation:
    """Regression: the six SITUATIONAL delivery traps (byte-immutability,
    reloaded-artifact, noisy-margin, best-so-far write-through, exact-contents,
    backup/rollback) were moved OUT of _ACCEPTANCE_GUIDANCE (plan-time) and into
    _TASK_COMPLETE_DELIVERY_HYGIENE (completion-time), where they are actionable.
    These tests assert each concept survives in the completion-gate text and no
    task-specific noun leaks in. The old mode-I 'no-progress run' bullet was
    deleted (compute-as-defined guidance retired), so no test asserts it."""

    def test_qualifier_reminder_separates_subject_from_boundary(self):
        # Regression: a task asked for the best model "as of" a past date; the
        # agent did solid work on the subject (correct benchmark, correct metric)
        # but silently dropped the time qualifier and answered from the current
        # full snapshot — the run completed and returned a confident, out-of-bounds
        # answer. The qualifier guidance now lives in _QUALIFIER_EXTRACTION in
        # plan.py (delivered at plan-framing time via PlanGuard, not post) and must
        # name qualifiers (time point / version / subset / metric definition) as
        # first-class constraints and forbid substituting most-available/current data.
        from flagscale_agent.react.guard.plan import _QUALIFIER_EXTRACTION
        low = _QUALIFIER_EXTRACTION.lower()
        # separates the subject/thing-to-produce from its bounding qualifier
        assert "subject" in low and "qualifier" in low
        # names qualifier categories in the abstract, not any task-specific value
        assert "point in time" in low
        assert "version" in low
        assert "subset" in low
        # the danger: work completes and yields a confident yet out-of-bounds answer
        assert "confident answer" in low
        # forbids letting the most available / most current data stand in
        assert "most available" in low or "most up-to-date data" in low
        # must fold qualifiers into the plan while framing, not after
        assert "plan" in low
        # a qualifier is verified by an unfakeable external grader, not self-report:
        # declaring done does not move the scored verdict; unmet qualifier => FAIL
        assert "machine-verified" in low or "machine-checked" in low
        assert "byte-for-byte" in low
        assert "fail" in low
        # must not hardcode the originating task's specifics
        assert "august" not in low
        assert "mteb" not in low

    def test_qualifier_reminder_demands_correct_interpretation_not_just_extraction(self):
        # Regression: the block forced the agent to extract the time qualifier into
        # the plan, but the agent then read a past-point time bound as "data recent
        # enough" and used the latest snapshot — extracting a qualifier is not the
        # same as reading its boundary in the direction the task means. Guidance
        # must demand interpreting the boundary, name the "newest/most available is
        # best" bias as the trap, and flag the two-way ambiguity of a time bound
        # (state at a past point vs state now). Stays task-agnostic and must NOT
        # quote the originating task's phrasing (e.g. "as of") — that reads like
        # leaderboard-gaming rather than general guidance.
        from flagscale_agent.react.guard.plan import _QUALIFIER_EXTRACTION
        low = _QUALIFIER_EXTRACTION.lower()
        # extraction is only half — must read the boundary the task means
        assert "only half" in low
        assert "boundary" in low
        # state each qualifier's meaning before building on it
        assert "in your own words" in low
        # names the convenience/recency bias as the failure mode
        assert "convenient" in low
        assert "newest" in low or "most available is best" in low
        # a time bound is two-way: state at a past point vs state now
        assert "bound on time" in low
        assert "as it stood at a" in low
        assert "the state right now" in low
        assert "came later out of scope" in low
        # must not default to the freshest data
        assert "most up-to-date data" in low
        # still no task specifics AND no verbatim task phrasing that reads as gaming
        assert "as of" not in low
        assert "august" not in low
        assert "scandinavian" not in low

    def test_qualifier_content_moved_out_of_acceptance_guidance(self):
        # The qualifier-extraction guidance must NOT remain in the post-timing
        # _ACCEPTANCE_GUIDANCE — its useful moment is at plan framing (pre), and
        # duplicating it would fire it too late again. This asserts the move was
        # clean (no stale copy left behind).
        from flagscale_agent.react.guard.verification import _ACCEPTANCE_GUIDANCE
        low = _ACCEPTANCE_GUIDANCE.lower()
        assert "qualifier" not in low
        assert "scandinavian" not in low

class TestBatchStepDoneGate:
    """Timing 1b: a batch update marking any step 'done' commits the same claim as
    step_done and must clear the same bar. Without this, batch is a silent hole
    that skips every verification check. Non-done batches pass freely."""

    def test_batch_with_done_blocked_without_override(self):
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "batch",
                "updates": [
                    {"step_id": 1, "status": "done"},
                    {"step_id": 2, "status": "doing"},
                ],
            },
        )
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert verdict.reason == "batch_step_done_no_verification"

    def test_batch_with_done_allowed_with_override(self):
        guard = VerificationGuard()
        reason = "each done step re-checked against its acceptance, outputs read"
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "batch",
                "updates": [{"step_id": 1, "status": "done"}],
                "_override_reason": reason,
            },
            override_reason=reason,
        )
        verdict = guard.check_pre(ctx)
        assert verdict is None

    def test_batch_without_done_passes_freely(self):
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "batch",
                "updates": [
                    {"step_id": 1, "status": "doing"},
                    {"step_id": 2, "status": "skipped"},
                ],
            },
        )
        verdict = guard.check_pre(ctx)
        assert verdict is None

    def test_batch_empty_override_still_blocked(self):
        guard = VerificationGuard()
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "batch",
                "updates": [{"step_id": 1, "status": "done"}],
                "_override_reason": "   ",
            },
            override_reason="   ",
        )
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.reason == "batch_step_done_no_verification"

