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

"""PlanGuard — nudges (interactive) or requires (single-shot) an active plan.

Scope note: the PLAN-FRAMING discipline (divergence check, qualifier/facet
extraction, wall-aware first cut, premortem, standard-method first) no longer
lives here as a plan_create-time inject. It is carried by the fixed FIRST step
of the plan standard template (tools/plan_create.py `_FRAME_STEP_*`), which the
(retained) step_done evidence gate enforces — one systematic home, no separate
divergence guard. This guard now only forces the plan into existence: the
write_file gate in single-shot mode and the periodic nudge.
"""

from __future__ import annotations

import os

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


def _tool_result_indicates_failure(result) -> bool:
    """True when a tool_result shows the call did NOT really succeed: a kernel/
    executor exception wrapper ("Error executing tool: ..."), a guard block
    ("[BLOCKED BY GUARD] ..."), or the tool's own "ERROR: ..." return.

    `None` is treated as NOT-a-failure: callers that don't plumb a result (e.g.
    unit tests, or a path where the result was not captured) must keep their
    historical success semantics. Only a POSITIVE error signal flips this.
    """
    if result is None:
        return False
    text = str(result).lstrip()
    return (
        text.startswith("Error executing tool")
        or text.startswith("[BLOCKED BY GUARD]")
        or text.startswith("ERROR")
    )


_WRITE_FILE_NO_PLAN = (
    "[Plan] You are about to write or edit a file — the concrete signal that you have "
    "started producing a deliverable. A plan must exist before you produce output.\n"
    "\n"
    "Call plan_create() now: frame the task as ordered steps with acceptance "
    "criteria naming what THIS task requires. A budget-order first step ('land a "
    "crude but complete, scorable deliverable at the required path, then refine') "
    "keeps an unsupervised run from spending its whole budget with nothing on disk.\n"
    "\n"
    "If this write is a genuinely throwaway scratch/draft file (not the "
    "deliverable), override with a one-line reason and proceed."
)


class PlanGuard(Guard):
    """Nudges (interactive) or requires (single-shot) an active plan.

    Interactive: after REMIND_THRESHOLD tool calls without a plan, injects a
    periodic reminder. Never blocks.

    Single-shot: the plan stands in as structural supervisor. Enforcement is
    front-loaded to the "start producing" moment: the first write_file with no
    plan blocks (overridable) to force plan_create before deliverable output.
    After SINGLE_SHOT_BLOCK_THRESHOLD calls without a plan, blocks further
    non-plan tools until plan_create. Overridable for genuinely-trivial tasks.
    There is NO completion-time block: completing without a plan is never a hard
    failure (that punished weak models with a livelock kill); the plan is forced
    earlier, at write_file, where it is actionable rather than punitive.
    """

    name = "plan"
    priority = 35

    # Interactive: periodic nudge cadence. Single-shot uses the tighter
    # SINGLE_SHOT_REMIND cadence — an unsupervised run should be reminded sooner,
    # since the write_file gate (not a completion block) is the real enforcement.
    REMIND_THRESHOLD = 15
    SINGLE_SHOT_REMIND = 5
    # Single-shot: allow this many observation/exploration calls before requiring
    # a plan. Set generously — an unsupervised run legitimately needs to probe the
    # environment (paths, configs, GPU state, prior findings) before it can frame a
    # plan whose steps land on real checkpoints. Blocking too early forces a plan
    # written before understanding, which is worse than no plan.
    SINGLE_SHOT_BLOCK_THRESHOLD = 20

    def __init__(self, task_plan=None, single_shot: bool = False):
        self._task_plan = task_plan
        self._single_shot = single_shot

        # No-plan nag budget. [per-turn]
        # Armed: incremented by every non-plan check_pre. Discharged: a
        # plan_create (check_post resets to 0 — the framing happened) or
        # reset_turn().
        self._calls_without_plan = 0

        # A plan was actually framed at least once. [per-run]
        # Distinct from task_plan.get_active(): a plan may later be completed/
        # abandoned, but the "you have not planned yet" obligation stays
        # discharged. Set ONLY by a plan_create whose result shows success —
        # a blocked/errored attempt leaves the obligation live.
        self._plan_ever_created = False

    def set_single_shot(self, enabled: bool = True):
        """Enable single-shot enforcement at runtime (set once run mode known)."""
        self._single_shot = enabled

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            # No completion-time gate. Completing without a plan is intentionally
            # NOT blocked: the old NON-OVERRIDABLE completion block livelocked weak
            # models (they neither reliably override nor plan_create, so the loop
            # burned MAX_CONSECUTIVE_COMPLETION_BLOCKS and auto-killed the task →
            # reward 0). Enforcement moved earlier to write_file, where forcing a
            # plan is actionable instead of punitive.
            return None

        # write_file gate (single-shot): the first attempt to write a file with
        # no plan is the "start producing a deliverable" moment. Block once
        # (overridable) to force plan_create before deliverable output. Only
        # write_file triggers — read_file/shell exploration never does, so the
        # investigation phase is not disturbed. Once a plan exists this never
        # fires.
        if ((ctx.tool_name == "write_file" or ctx.tool_name == "edit_file")
                and self._single_shot
                and not self._plan_ever_created
                and not (self._task_plan and self._task_plan.get_active())):
            return GuardVerdict.block(
                message=_WRITE_FILE_NO_PLAN,
                reason="write_file_without_plan",
                category="plan_required",
                overridable=True,
            )

        # Plan-related tools don't count toward the no-plan budget and never
        # block. The plan_create moment itself is a plain pass-through: the
        # framing discipline is delivered by the plan's fixed FIRST step now,
        # not by an inject here.
        if ctx.tool_name in ("plan_create", "plan_update", "plan_status"):
            return None

        # If plan exists, nothing to do
        if self._task_plan and self._task_plan.get_active():
            return None

        # A plan that was created and then COMPLETED/ABANDONED clears the active
        # pointer, so get_active() returns None and the check above reads it as
        # "never planned" — re-incrementing the counter and re-nagging after the
        # plan is already done. In single-shot there is exactly one turn, so the
        # sticky _plan_ever_created flag is equivalent to per-turn: once a plan
        # has SUCCESSFULLY been framed, the "you have not planned yet" obligation
        # is discharged and the no-plan budget must not re-fire. (The flag covers
        # any later plan state — completed, abandoned, or paused — because the
        # nag's purpose is to get a FIRST plan framed, which has happened.)
        # Scoped to single_shot ONLY: an interactive multi-turn session must keep
        # nagging a fresh later turn that has not planned yet.
        if self._single_shot and self._plan_ever_created:
            return None

        self._calls_without_plan += 1

        # Single-shot: after observation budget, require a plan (block).
        if self._single_shot and self._calls_without_plan > self.SINGLE_SHOT_BLOCK_THRESHOLD:
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
                ),
                reason="single_shot_plan_required",
                category="plan_required",
            )

        # Periodic reminder. Single-shot uses the tighter cadence (5) since the
        # write_file gate is the real enforcement and an unsupervised run should
        # be nudged toward a plan sooner.
        cadence = self.SINGLE_SHOT_REMIND if self._single_shot else self.REMIND_THRESHOLD
        if self._calls_without_plan % cadence == 0:
            return GuardVerdict.inject(
                message=(
                    f"[Plan] {self._calls_without_plan} tool calls without a plan. "
                    f"If environment exploration is done and you've begun producing "
                    f"output, you understand the structure — call plan_create() to "
                    f"freeze it into steps with acceptance criteria."
                ),
                reason="plan_reminder",
                category="plan_needed",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Settlement site for the no-plan obligation.

        Nothing here is surfaced to the LLM: check_post runs AFTER the tool, and
        its verdicts are advisory-only (this method always returns None). Its job
        is to SETTLE the no-plan obligation by OBSERVING that the completed
        plan_create actually produced a plan — never by the appearance of it.
        """
        if ctx.tool_name == "plan_create":
            self._calls_without_plan = 0
            # Only a plan_create that ACTUALLY produced a plan may discharge the
            # no-plan obligation. A failed/blocked call (exception wrapper, guard
            # block, or the tool's own "ERROR: ..." return) leaves `tool_result`
            # carrying a failure marker — do NOT set the sticky flag, or the
            # single-shot no-plan block would be silently disabled for the rest of
            # the run by a mere ATTEMPT. `None` (no result plumbed) keeps the
            # historical success semantics.
            if not _tool_result_indicates_failure(ctx.tool_result):
                self._plan_ever_created = True
        return None

    def reset_turn(self):
        """New user message resets the no-plan nag budget.

        _plan_ever_created is intentionally NOT reset here. In single-shot
        mode there is only one turn, so it never matters; in interactive mode the
        completion gate never fires anyway. Keeping it sticky avoids a spurious
        block if reset_turn is ever called mid-single-shot-run.
        """
        self._calls_without_plan = 0
