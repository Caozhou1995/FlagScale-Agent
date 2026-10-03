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

Treat every qualifier as machine-verified by a grader you cannot reach or fool: it
re-parses the time boundary, diffs the exact version string, compares the output file
byte-for-byte against a reference, re-counts the entries. Reporting the task done does
not feed that check — if a qualifier is not literally satisfied, the scored result is
FAIL no matter how finished the work looks or how you word the completion. So a
qualifier is not a nicety to mention; it is a hard pass/fail condition to build the
plan around.

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

Before committing to shape, apply P1's framework: name the problem CLASS, identify the STANDARD METHOD, and ensure the plan includes a SMALL-SAMPLE step that validates the method on the smallest meaningful input before scaling to the full task — the small run also gives a time estimate, and each debugging iteration at full scale costs 10-100x more. Generic steps ("analyze", "implement", "test") mean the class was never identified — a brute-force sweep is the tell.

Some qualifiers pin you to a concrete tool instance (specific version, revision, named model). The instance may have usage requirements that differ from the generic API in ANY form (preprocessing, defaults, precision, prefix). When pinned, add a step to consult that instance's own documentation before writing the call — per P1's TOOL INSTANCE rule, the task's VERB selects which documented usage applies, and the documented way is the DEFAULT. Consulting is the near half; APPLYING what the doc says is the far half — a literal-minimal reading silently swaps the task for an easier one."""

_WRITE_FILE_NO_PLAN = (
    "[Plan] You are about to write or edit a file — the concrete signal that you have "
    "started producing a deliverable. A plan must exist before you produce "
    "output.\n"
    "\n"
    "Call plan_create() now: frame the task as ordered steps with acceptance "
    "criteria naming what THIS task requires. A budget-order first step ('land a "
    "crude but complete, scorable deliverable at the required path, then refine') "
    "keeps an unsupervised run from spending its whole budget with nothing on "
    "disk.\n"
    "\n"
    "If this write is a genuinely throwaway scratch/draft file (not the "
    "deliverable), override with a one-line reason and proceed."
)


# Divergence demand, delivered on EVERY plan_create — per framing
# (the premise-inherited-unexamined half: "我想窄了"). A plan written from a
# framing locks in that framing: every later step inherits premises that
# were never inspected against alternatives — and a LATER framing (re-plan)
# inherits its own unexamined premises just the same, so the demand re-arms
# per framing. This inject does NOT judge the plan —
# it demands a divergence check by a SEPARATE reasoning process: spawn ONE
# diverger worker (read-only, no verdict, no agreement) that proposes alternative
# framings of the task, then the parent issues a one-line ruling for EACH
# alternative (adopt → revise the plan, or reject → why). Rulings are the
# deliverable: they force the unexamined premises into the open. Fire once
# PER FRAMING (every plan_create — a re-framing re-arms the demand) and only
# via plan_create —
# same folded-into-one-gate pattern as _QUALIFIER_EXTRACTION, never a separate
# block (cross-talk rule, see verification.py Timing 0b note).
_DIVERGER_INJECT = """

Divergence check — before this plan hardens, probe what it made invisible.

WHY: you framed this plan, so its premises are invisible to you — the same
context that produced a framing cannot reliably judge it (self-review
inherits the blind spot that produced the claim; Huang et al. 2024). A wrong
framing is the most expensive defect class: every step, test and hour
downstream inherits it, and it surfaces only as late-stage dead-ends
("why is this not working") when the budget is nearly gone. The diverger
reads the task WITHOUT your plan, so it sees framings you did not consider.

GAIN: one of two concrete outcomes — (a) a genuinely better framing adopted
while changing it costs minutes, or (b) alternatives rejected against fresh
eyes, which upgrades "I think this framing is right" into "I checked what
else it could be". Cost: one ~3-minute read-only worker, spawn-and-continue
(never blocking). Skipping is legal — this is advisory — but it saves 3
minutes against the risk that the whole plan rests on an unexamined premise.

Spawn ONE diverger worker (spawn_worker) RIGHT AFTER framing this plan — do not
wait for step 2. Contract:
  - goal: "Propose 2-3 genuinely different framings of the task — different
    method-class, different decomposition, or a different reading of an ambiguous
    term — NOT refinements of this plan."
  - constraints: read-only (writable: [one scratch dir that contains output_ptr —
    an empty list is rejected by validation]; forbidden: modify any file),
    max_minutes: 3. Give it the task's ORIGINAL statement, not your plan.
  - The diverger proposes IDEAS ONLY: no verdicts, no ranking, no agreeing with
    you. It answers "what else could this problem be", not "is this plan good".
When it returns, issue a ONE-LINE ruling for EACH alternative it proposed —
adopt (then actually revise the plan) or reject (name WHY in one sentence, e.g.
"incompatible with X constraint"). A ruling with no reason is not a ruling; an
unruled alternative is an unexamined premise. This fires per framing — at
every plan_create, where a wrong framing is cheapest to fix."""


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
        self._calls_without_plan = 0
        self._single_shot = single_shot
        # Whether plan_create was ever called — completion gate checks this
        # (distinct from get_active(): a plan may be created then deactivated).
        self._plan_ever_created = False
        # Qualifier extraction delivered once per FRAMING: set by whichever
        # block carries it (write_file / single-shot), or injected on a
        # plan_create; a plan_create consumes it again so re-planning re-injects.
        self._qualifier_reminded = False
        # Divergence demand, like the qualifier, is PER FRAMING: injected on
        # EVERY plan_create (a later re-framing's premises are as unexamined
        # as the first's — user ruling 2026-09-23), consumed on injection,
        # re-armed by reset_turn. Anti-ritualization lives in the contract
        # itself (a one-line ruling for EVERY alternative), not in a
        # once-per-run flag.
        self._diverger_reminded = False

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
            self._qualifier_reminded = True
            return GuardVerdict.block(
                message=_WRITE_FILE_NO_PLAN + _QUALIFIER_EXTRACTION,
                reason="write_file_without_plan",
                category="plan_required",
                overridable=True,
            )

        # Plan-related tools don't count toward the no-plan budget. plan_create is
        # the plan-framing moment — deliver the qualifier-extraction demand on
        # every framing (re-armed by reset_turn), and fold the divergence demand
        # into EVERY framing as well (both ride the same one-verdict-per-check_pre
        # channel).
        if ctx.tool_name == "plan_create":
            # Both demands belong to EACH framed plan (user ruling 2026-09-23:
            # a re-framing's premises are as unexamined as the first's, so the
            # diverger re-arms per framing exactly like the qualifier). A block
            # earlier in THIS framing (write_file gate / single-shot) may
            # already have carried them — consume the flags unconditionally
            # either way, so the NEXT framing re-arms them.
            inject_qualifier = not self._qualifier_reminded
            inject_diverger = not self._diverger_reminded
            self._qualifier_reminded = False
            self._diverger_reminded = False
            if not inject_qualifier and not inject_diverger:
                return None
            msg = "[Plan] Framing the plan."
            if inject_qualifier:
                msg += _QUALIFIER_EXTRACTION
            if inject_diverger:
                msg += _DIVERGER_INJECT
            return GuardVerdict.inject(
                message=msg,
                reason="qualifier_extraction",
                category="plan_required",
            )
        if ctx.tool_name in ("plan_update", "plan_status"):
            return None

        # If plan exists, nothing to do
        if self._task_plan and self._task_plan.get_active():
            return None

        self._calls_without_plan += 1

        # Single-shot: after observation budget, require a plan (block).
        if self._single_shot and self._calls_without_plan > self.SINGLE_SHOT_BLOCK_THRESHOLD:
            # Block carries BOTH demands (qualifier + diverger); mark both
            # delivered to prevent double-injection on the subsequent plan_create.
            self._qualifier_reminded = True
            self._diverger_reminded = True
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
                    + _DIVERGER_INJECT
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
        if ctx.tool_name == "plan_create":
            self._calls_without_plan = 0
            self._plan_ever_created = True
        return None

    def reset_turn(self):
        """New user message resets counter.

        _qualifier_reminded and _diverger_reminded are both reset: both
        demands are per-framing, so a new turn's plan_create must re-inject
        them (a prior block may have carried them earlier in the same turn —
        that is why they are not cleared at the top).

        Anti-ritualization note: re-arming per turn does not ritualize the
        checks — the qualifier's value is its framing-time questions, and the
        diverger's is its one-line-ruling contract, both of which survive
        repetition. What was ritual-prone was firing the SAME plan shape
        again, which re-framing is not.

        Note: _plan_ever_created is intentionally NOT reset here. In single-shot
        mode there is only one turn, so it never matters; in interactive mode the
        completion gate never fires anyway. Keeping it sticky avoids a spurious
        block if reset_turn is ever called mid-single-shot-run.
        """
        self._calls_without_plan = 0
        self._qualifier_reminded = False
        self._diverger_reminded = False
