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

"""PlanGuard — reminds agent to create a plan for long tasks."""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Folded into the plan-framing gate on purpose: qualifier extraction is a
# plan-framing concern, and PlanGuard forces the plan into existence. One gate,
# one override, no laundering — a separate block caused cross-talk where the
# override_reason for the plan also released the qualifier block.
_QUALIFIER_EXTRACTION = """

While framing this plan — extract the task's qualifiers, not just its subject.

A task names a subject and usually qualifies it: a point in time, a version, a
subset, a specific metric definition. The subject is obvious; the qualifier is
easy to drop — work completes with a confident answer whether or not you honored
it. A dropped qualifier silently answers a nearby question the task did not ask.

Re-read the task statement; list every qualifier as a first-class item. Fold each
into the plan as a step or acceptance criterion. A qualifier not in the plan is one
execution silently skips. But extraction is only half the job — you must also read
what boundary each qualifier draws, in the direction the task means. State each
qualifier's meaning in your own words. The common failure is bending a fixed
boundary toward what is convenient: widening to the larger, fresher, or more
available thing. A "newest / most available is best" instinct overwrites a bounded
qualifier. When the task fixes a boundary, the answer inside it is right even if a
better candidate exists outside. A bound on time is especially easy to invert: it
can mean the state as it stood at a given point (with anything that came later out of scope), or the state right now; these select different answers — let the task's words decide, not defaulting to the most up-to-date data.

Before committing to shape, answer:
  1. CLASS — what known problem class is this an instance of? Name it. "It's its
     own thing" means you haven't placed it yet.
  2. STANDARD METHOD — the established technique for that class and its first step.

Generic steps ("analyze", "implement", "test") mean the class was never identified.
A brute-force sweep is the tell you skipped this.

Some qualifiers pin you to a concrete tool instance (specific version, revision
hash, named model). That instance may have usage requirements that differ from the
generic API — the difference can take ANY form (preprocessing, defaults, precision,
prefix). When pinned, add a step to consult that instance's own documentation
(README, model card, release notes) before writing the call. But consulting is
only the near half — the far half is APPLYING what the doc says. The task's VERB
selects which documented usage applies; the documented way is the DEFAULT, the
plainer call is the DEVIATION. Don't read the doc, see the requirement, then skip
it with "the task didn't explicitly ask for that" — a literal-minimal reading
silently swaps the task for an easier one."""

_COMPLETION_NO_PLAN = (
    "[Plan] BLOCKED — you are trying to complete without ever creating a plan. "
    "This gate CANNOT be overridden by text. A bare [TASK_COMPLETE] will be "
    "blocked again, and so will any _override_reason you add inline.\n"
    "\n"
    "The ONLY way forward is to call the plan_create() tool now — a real tool "
    "call, not text. Frame the task as ordered steps with acceptance criteria "
    "that name what THIS task requires. Even a genuinely small task gets a "
    "one- or two-step plan: the plan is the structural supervisor for an "
    "unsupervised run, and completing without one leaves the work ungoverned.\n"
    "\n"
    "Do this next: call plan_create(title=..., steps=[...]). After the plan "
    "exists, your [TASK_COMPLETE] will pass this gate."
)


class PlanGuard(Guard):
    """Nudges (interactive) or requires (single-shot) an active plan.

    Interactive: after REMIND_THRESHOLD tool calls without a plan, injects a
    periodic reminder. Never blocks.

    Single-shot: the plan stands in as structural supervisor and completion gate.
    After SINGLE_SHOT_BLOCK_THRESHOLD calls without a plan, blocks further
    non-plan tools until plan_create. Overridable for genuinely-trivial tasks.
    """

    name = "plan"
    priority = 35

    REMIND_THRESHOLD = 15
    # Single-shot: allow this many observation/exploration calls before requiring
    # a plan. Set generously — an unsupervised run legitimately needs to probe the
    # environment (paths, configs, GPU state, prior findings) before it can frame a
    # plan whose steps land on real checkpoints. Blocking too early forces a plan
    # written before understanding, which is worse than no plan.
    SINGLE_SHOT_BLOCK_THRESHOLD = 20

    def __init__(self, task_plan=None, single_shot: bool = False):
        self._task_plan = task_plan
        self._calls_without_plan = 0
        self._single_shot = single_shot
        # Whether plan_create was ever called — completion gate checks this
        # (distinct from get_active(): a plan may be created then deactivated).
        self._plan_ever_created = False
        # Qualifier extraction delivered exactly once: rides the single-shot
        # block or injects on first plan_create, whichever fires first.
        self._qualifier_reminded = False

    def set_single_shot(self, enabled: bool = True):
        """Enable single-shot enforcement at runtime (set once run mode known)."""
        self._single_shot = enabled

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            # Completion gate (single-shot): block [TASK_COMPLETE] when no plan
            # was ever created. NON-OVERRIDABLE — the only exit is an actual
            # plan_create() tool call, which sets _plan_ever_created and releases
            # the gate on the next completion attempt. A text-inline
            # _override_reason no longer releases it: the model could not reliably
            # emit the inline form, so a bare [TASK_COMPLETE] livelocked the loop
            # (re-firing this gate on stale text every iteration). Forcing a real
            # tool call structurally breaks that text-only spin. Never fires in
            # interactive mode.
            if (self._single_shot
                    and not self._plan_ever_created
                    and ctx.assistant_text
                    and "[TASK_COMPLETE]" in ctx.assistant_text):
                return GuardVerdict.block(
                    message=_COMPLETION_NO_PLAN,
                    reason="single_shot_completion_without_plan",
                    category="plan_required",
                    overridable=False,
                )
            return None

        # Plan-related tools don't count toward the no-plan budget. The FIRST
        # plan_create is the plan-framing moment — inject qualifier-extraction
        # demand if not yet delivered.
        if ctx.tool_name == "plan_create":
            if not self._qualifier_reminded:
                self._qualifier_reminded = True
                return GuardVerdict.inject(
                    message="[Plan] Framing the plan." + _QUALIFIER_EXTRACTION,
                    reason="qualifier_extraction",
                    category="plan_required",
                )
            return None
        if ctx.tool_name in ("plan_update", "plan_status"):
            return None

        # If plan exists, nothing to do
        if self._task_plan and self._task_plan.get_active():
            return None

        self._calls_without_plan += 1

        # Single-shot: after observation budget, require a plan (block).
        if self._single_shot and self._calls_without_plan > self.SINGLE_SHOT_BLOCK_THRESHOLD:
            # Block carries the qualifier demand; mark delivered to prevent
            # double-injection on the subsequent plan_create.
            self._qualifier_reminded = True
            return GuardVerdict.block(
                message=(
                    f"[Plan] Pause. {self._calls_without_plan} tool calls in this "
                    f"unsupervised run without a plan. Do you understand the problem's "
                    f"structure well enough to plan it?\n"
                    f"— If yes: call plan_create() now. Steps must land on real "
                    f"checkpoints, acceptance criteria naming what THIS task requires, "
                    f"not generic \"works correctly\".\n"
                    f"— If no: override and keep investigating. A plan written before "
                    f"understanding is worse than no plan — it locks in a shape you'll "
                    f"fight later."
                    + _QUALIFIER_EXTRACTION
                ),
                reason="single_shot_plan_required",
                category="plan_required",
            )

        # Periodic reminder every REMIND_THRESHOLD calls (both modes)
        if self._calls_without_plan % self.REMIND_THRESHOLD == 0:
            return GuardVerdict.inject(
                message=(
                    f"[Plan] {self._calls_without_plan} tool calls without a plan. "
                    f"Consider plan_create() to organize a long task."
                ),
                reason="plan_reminder",
                category="plan_needed",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name == "plan_create":
            self._calls_without_plan = 0
            self._plan_ever_created = True
        return None

    def reset_turn(self):
        """New user message resets counter.

        Note: _plan_ever_created is intentionally NOT reset here. In single-shot
        mode there is only one turn, so it never matters; in interactive mode the
        completion gate never fires anyway. Keeping it sticky avoids a spurious
        block if reset_turn is ever called mid-single-shot-run.
        """
        self._calls_without_plan = 0
