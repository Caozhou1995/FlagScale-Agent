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

"""VerificationGuard — requires verification evidence when marking steps complete.

Design principles:
- Block plan_update(action="step_done") if:
  * Step has acceptance criteria AND no verification provided
  * Step is complex (no acceptance defined) AND no _override_reason provided
- Other tool calls (read_file/shell/grep) are completely unaffected
- LLM can freely perform verification operations after being blocked
- Once verified, LLM calls with verification=["..."] or _override_reason to pass
- Does not check verification content — any non-empty list passes

Two verification modes:
1. Structured: step has acceptance → must provide verification=["proof1", "proof2"]
2. Override: no acceptance (simple step) → must provide _override_reason="checked X"

Why this works:
- Acceptance criteria define WHAT to verify
- Verification list records HOW it was verified
- Override_reason for simple steps maintains backward compatibility

Execution flow (structured):
1. LLM: plan_update(action="step_done", step_id=3)
2. Guard: BLOCK - step has acceptance, verification required

3. LLM: OK, let me verify acceptance criteria
4. LLM: shell("pytest tests/")  ← executes normally
5. LLM: read_file("output.log")  ← executes normally

6. LLM: plan_update(action="step_done", step_id=3, verification=["all tests passed", "log shows no errors"])
7. Guard: Has verification, allow ✓
"""

from __future__ import annotations

import os
import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.multi_agent.wiring import CONTRACT_PATH_ENV, is_worker
from flagscale_agent.react.display import has_override_completion, trailing_sentinel


def _contract_constraints() -> dict:
    """The spawned worker's `constraints` block, read from the structured
    contract JSON — NOT the rendered contract.prompt prose.

    The rendered prompt is prose: a contract that merely QUOTES a phrase (a
    reviewer contract quoting "## Reviewer discipline", or any body mentioning
    "framings of the task") would falsely suppress the demand. The sibling
    contract.json carries the authoritative `constraints` mapping, so gating on
    constraints.reviewer is exact. Errors fail silent-{} (callers fall through
    to the demand as usual).
    """
    try:
        path = os.environ.get(CONTRACT_PATH_ENV, "")
        if not path:
            return {}
        import json
        with open(os.path.join(os.path.dirname(path), "contract.json"),
                  "r", encoding="utf-8") as f:
            data = json.load(f)
        cons = data.get("constraints") or {}
        return cons if isinstance(cons, dict) else {}
    except Exception:
        return {}


_VERIFICATION_REQUIRED_WITH_ACCEPTANCE = """[VerificationGuard] Before this step is done — answer three questions per criterion.

Do NOT describe what you did ("ran the tests") — that is your activity, not
evidence the criterion holds. For EACH criterion below answer:
  1. OBSERVED — the concrete value/output you saw (a number, count, exit code,
     exact string — not "it worked").
  2. EXPECTED — what the criterion required.
  3. GAP — observed vs expected: match, or off by how much?
No concrete OBSERVED value means the criterion is NOT verified yet — go get the
value, do not invent one.

Retry with one verification entry per criterion, each carrying its observed value:
  plan_update(action="step_done", step_id=N, verification=[
    "criterion 1 — observed: <value>; expected: <value>; gap: <match/diff>"])

Acceptance criteria for this step:
{acceptance}.
"""

_VERIFICATION_REQUIRED_NO_ACCEPTANCE = """[VerificationGuard] Before this step is done — answer, do not assert.

No acceptance criteria here, so answer in _override_reason: what did you
OBSERVE that shows the goal is met — a concrete value/output you saw (count,
exit code, exact string), not "it worked". No observed value means not done — go
get one.

  plan_update(action="step_done", step_id=N, _override_reason="observed <value/output> which shows <goal> is met")
"""

_ACCEPTANCE_FROZEN = """[VerificationGuard] Acceptance change blocked — criteria are frozen.

Step {step_id} is already "{status}" — implementation has begun. Changing its
acceptance criteria now lets the standard be shaped by what you produced.

Define acceptance BEFORE implementing. If the original criteria were genuinely
wrong, retry with _override_reason explaining what was wrong with them and why
the new criteria are still independent of your implementation choices.
"""

_ACCEPTANCE_GUIDANCE = """[VerificationGuard] Writing acceptance criteria — a note on what makes them useful.

Acceptance describes an externally observable end state — a file at a path, a
command exiting 0, an artifact with an expected property. The check must be
independent of your implementation, or it cannot fail.

Re-read the task statement and list every operation it explicitly asks for; each
maps to a criterion. "I know how to do X, so X is done" is not a criterion — it is
the signal to execute X.

Traps:
  • Cover every input the task gives you, not just the one you developed against.
  • Verify against the judge's measurement, not your proxy.
  • A failure inside your declared scope is yours to fix, regardless of origin.

Delivery-time traps (fresh reload, backup+rollback, exact-contents,
byte-immutability, noisy margin) are enforced at completion — do not carry them into acceptance.
"""

# NOTE: The framing-time reminders (qualifier extraction, pre-mortem, wall-aware
# first cut, divergence check) lived in plan.py as inject blocks, then were
# consolidated into the fixed FIRST plan step's acceptance criteria
# (_FRAME_STEP_ACCEPTANCE in tools/plan_create.py). Reason: injecting on
# plan_create caused guard override_reason cross-talk — PlanGuard's single-shot
# block trained the agent to pass _override_reason, which then silently
# satisfied other guards' blocks too, so they never actually fired.

_BATCH_STEP_DONE_BLOCK = """[VerificationGuard] Batch update marks a step done — same bar as a single step_done.

Marking done in bulk must not skip the verification each step would otherwise require.
Re-issue the batch with "_override_reason" affirming that for EACH step you are marking
done, you re-checked the result against what that step was supposed to satisfy.

If a step deserves real evidence, mark it via plan_update(step_done, verification=[...])
individually rather than folding it into a batch."""

_STEP_DONE_RECHECK_REMINDER = """[VerificationGuard] Before this step is marked done — check your premises once.

The last point where a wrong turn is cheap to catch. Verification is not
restating what you believed while working — it is testing that belief against
what you got.

For each criterion the step had to satisfy, read it as written and ask whether
your result lands inside it. Then examine the premises you leaned on: any reading
of a term, any "this is close enough", any value taken as given rather than
confirmed. A premise adopted for convenience is the one most likely wrong — treat
it as the thing under test, not settled ground.

Finally, check the target: re-read what the user actually asked for and confirm
this step moves toward THAT, not a nearby goal your framing swapped in. Verifying
the wrong question thoroughly is still off-target. When your reading and the
task's plain wording pull apart, the wording wins.

If the step's requirement was UNDER-SPECIFIED — the wording admitted two
defensible readings and the checker is invisible — do not settle it by feel. At
this boundary, name the reading you produced against and, if a candidate output
CLAIMED to satisfy BOTH readings, the probe output that demonstrated it — an
unprobed hedge is not evidence — and neither is an introspected "no hedge
possible": bet only once the intersection is CONSTRUCTIVELY shown empty. If the
readings are mutually exclusive you are betting, so name the tie you broke (the
wording the task EXPLICITLY grants first; otherwise the branch strongest on the
axes the task states WITHOUT ambiguity). A reading chosen silently is the one that fails.

To proceed, re-issue plan_update(step_done) with "_override_reason" naming a
CONCRETE ANCHOR, not a summary of diligence. "I re-checked and it's correct" is
the empty answer this replaces. Point at one of: the value you OBSERVED vs the
value the criterion required (report both); the exact premise you re-tested and
what retesting showed; the reading you chose for an under-specified clause and
the evidence for it; or — if a premise cannot be tested here — say so in those
words. An anchor you can point at, or an explicit "no evidence for X"; not an
adjective. This is not a content check — any override_reason lets the step
through; a reason with no anchor is itself the signal you did not look."""

_POST_RECOVERY_REMINDER = """[VerificationGuard] Context was just recovered via hard_reset.

Before continuing work:
1. Read key files to confirm current state
2. Check recent changes (git status, grep for markers, file checksums)
3. Verify assumptions from pre-recovery context still hold

The goal: avoid propagating stale assumptions into new work."""


# Second-stage task-complete gate. The first complete-recheck (Timing 0b) fires
# regardless of content and any override_reason releases it — it only forces a
# pause. But the pause is defeated by the exact failure the prompt's
# observation-vs-argument litmus warns about: the agent writes a reason that
# ARGUES its method is sound ("thresholds are relative / conservative / should
# generalize") instead of REPORTING an observation it made (ran something, read
# a value, compared output to a known answer). A confident rationale for a wrong
# answer is still a wrong answer. So when the release reason is pure argument with
# zero observation signal, we bite ONCE more and demand a concrete observation.
# Fires at most once (independent of the first gate) so it stays a checkpoint.

# Argument markers: method-defence vocabulary the litmus names explicitly. Their
class VerificationGuard(Guard):
    """Requires verification evidence when marking steps complete.
    
    Key design:
    - Only blocks plan_update(action="step_done"), other tool calls unaffected
    - Two modes:
      * Step has acceptance → must provide verification=["..."]
      * Step has no acceptance → must provide _override_reason="..."
    - LLM can freely execute verification operations after being blocked
    - Does not check verification/override_reason content
    
    Also injects a reminder after hard_reset recovery.
    """
    
    name = "verification"
    priority = 55
    
    def __init__(self, plan=None, proposals=None):
        self._plan = plan
        self._proposals = proposals
        self._post_recovery = False
        self._recovery_reminded = False
        self._acceptance_guidance_given = False
        self._step_done_recheck_reminded = False

    def reset_turn(self):
        """Reset per-turn state on a new user message — clears the step_done
        re-check latch so a fresh task starts with the premise re-check armed.
        """
        self._step_done_recheck_reminded = False


    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # The step_done pre-mortem and the independent-reviewer demand now
        # live in the plan's fixed review step (see
        # tools/plan_create.py _REVIEW_STEP_ACCEPTANCE). VerificationGuard
        # keeps only the plan step_done evidence gate in check_pre below.
        return None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # The pure-text [TASK_COMPLETE] finish-path gate stack and the
        # plan_update(action="complete") completion-gate cascade previously
        # lived here. They are SUPERSEDED by the plan's fixed review step plus
        # the retained plan step_done evidence gate (Timing 1 below), so the
        # terminal completion gates are removed.

        # Timing 0: acceptance is frozen once a step leaves "pending".
        # Once implementation has begun, letting the criteria change means the
        # standard can be contaminated by the outcome — verification would no
        # longer be independent of the work. Overridable: if the original
        # criteria were genuinely wrong, the agent must say so explicitly.
        if ctx.tool_name == "plan_update":
            if ctx.tool_args.get("action") == "update_acceptance":
                step_id = ctx.tool_args.get("step_id")
                status = self._step_status(step_id)
                if status is not None and status != "pending":
                    return GuardVerdict.block(
                        message=_ACCEPTANCE_FROZEN.format(step_id=step_id, status=status),
                        reason="acceptance_frozen_after_implementation_started",
                        category="acceptance_frozen",
                    )
                # Still pending — criteria may be edited freely.
                return None

        # Timing 1: step_done requires verification evidence
        if ctx.tool_name == "plan_update":
            action = ctx.tool_args.get("action")
            
            # Only check on step_done, other actions (step_doing/add_steps) pass through
            if action == "step_done":
                step_id = ctx.tool_args.get("step_id")
                verification = ctx.tool_args.get("verification", [])
                override_reason = ctx.override_reason.strip()  # Use ctx.override_reason, not tool_args

                # Timing 1a: premise re-check, forced on the FIRST step_done of the
                # run. Symmetric with the plan_create qualifier block (check_pre):
                # BLOCKS once, any override_reason releases it, content is not
                # checked. Its purpose is different from the verification/override
                # checks below — those ensure evidence *exists*; this forces the
                # agent to stop and re-examine the premises it operated under before
                # any evidence is accepted. The plan-time qualifier block cannot
                # reach this failure: a qualifier can be captured correctly into the
                # plan yet be quietly re-interpreted during execution ("close
                # enough", "this value is current", a convenient reading of a term),
                # and that re-interpretation only surfaces at finalize time. Firing
                # once (not every step) keeps it a genuine checkpoint, not noise.
                # Fires ahead of the Mode 1/2 evidence checks so the agent re-reads
                # its premises before writing the verification it will be judged on.
                if not self._step_done_recheck_reminded:
                    if not override_reason:
                        return GuardVerdict.block(
                            message=_STEP_DONE_RECHECK_REMINDER,
                            reason="step_done_premise_recheck",
                            category="verification_required",
                        )
                    # Override provided → agent paused to re-check. Release for good,
                    # then fall through to the Mode 1/2 evidence checks below: the
                    # re-check does not exempt the step from still carrying evidence.
                    self._step_done_recheck_reminded = True
                
                # Get step's acceptance criteria if plan is available
                acceptance = []
                if self._plan and step_id:
                    try:
                        plan_data = self._plan.get_active()
                        if plan_data:
                            for step in plan_data.get("steps", []):
                                if step.get("id") == step_id:
                                    acceptance = step.get("acceptance", [])
                                    break
                    except Exception:
                        # If plan lookup fails, fall back to simple check
                        pass
                
                # Mode 1: Step has acceptance → require verification list
                if acceptance:
                    if not verification:
                        msg = _VERIFICATION_REQUIRED_WITH_ACCEPTANCE.format(
                            acceptance="\n".join(f"  • {a}" for a in acceptance)
                        )
                        return GuardVerdict.block(
                            message=msg,
                            reason="step_done_with_acceptance_no_verification",
                            category="verification_required"
                        )
                    # Has verification, allow.
                    return None
                
                # Mode 2: No acceptance (simple step) → require override_reason
                else:
                    if not override_reason:
                        return GuardVerdict.block(
                            message=_VERIFICATION_REQUIRED_NO_ACCEPTANCE,
                            reason="step_done_no_verification",
                            category="verification_required"
                        )
                    # Has override_reason, allow.
                    return None

            # Timing 1b: batch action marking any step done bypasses the per-step
            # step_done gate above (action=="batch", not "step_done"). A batch done
            # commits the same claim; without this it is a silent hole through which
            # every verification check is skipped. Require _override_reason when any
            # update sets status=="done". Non-done batches (doing/skipped) pass freely.
            if action == "batch":
                updates = ctx.tool_args.get("updates") or []
                has_done = any(
                    isinstance(u, dict) and u.get("status") == "done"
                    for u in updates
                )
                if has_done and not ctx.override_reason.strip():
                    return GuardVerdict.block(
                        message=_BATCH_STEP_DONE_BLOCK,
                        reason="batch_step_done_no_verification",
                        category="verification_required",
                    )

        # Timing 2: post-recovery, inject reminder on first step_doing
        if self._post_recovery and not self._recovery_reminded:
            if ctx.tool_name == "plan_update":
                action = ctx.tool_args.get("action")
                if action == "step_doing":
                    self._recovery_reminded = True
                    return GuardVerdict.inject(
                        message=_POST_RECOVERY_REMINDER,
                        reason="post_recovery_reminder",
                        category="post_recovery"
                    )
        
        # Timing 3: soft guidance when acceptance criteria are first defined.
        # Fires once — a nudge toward externally observable criteria, never blocks.
        if not self._acceptance_guidance_given and self._defines_acceptance(ctx):
            self._acceptance_guidance_given = True
            return GuardVerdict.inject(
                message=_ACCEPTANCE_GUIDANCE,
                reason="acceptance_quality_guidance",
                category="acceptance_guidance",
            )

        # All other tools (read_file/shell/edit_file) completely unaffected
        return None

    @staticmethod
    def _defines_acceptance(ctx: GuardContext) -> bool:
        """True if this tool call is defining acceptance criteria on any step.

        Covers plan_create/add_steps (steps list carrying "acceptance") and
        plan_update(action="update_acceptance", acceptance=[...]).
        """
        if ctx.tool_name == "plan_update":
            if ctx.tool_args.get("action") == "update_acceptance":
                return bool(ctx.tool_args.get("acceptance"))
            return False
        if ctx.tool_name in ("plan_create", "add_steps"):
            steps = ctx.tool_args.get("steps") or ctx.tool_args.get("new_steps") or []
            if isinstance(steps, list):
                for s in steps:
                    if isinstance(s, dict) and s.get("acceptance"):
                        return True
        return False

    def _step_status(self, step_id) -> str | None:
        """Return the status of a step in the active plan, or None if unknown.

        None means we cannot determine the status (no plan, no step_id, or
        lookup failed) — in that case the caller does not block, staying
        permissive when state is uncertain.
        """
        if not self._plan or not step_id:
            return None
        try:
            plan_data = self._plan.get_active()
            if not plan_data:
                return None
            for step in plan_data.get("steps", []):
                if step.get("id") == step_id:
                    return step.get("status")
        except Exception:
            return None
        return None

    def notify_recovery(self):
        """Called by hard_reset logic to signal recovery."""
        self._post_recovery = True
        self._recovery_reminded = False
