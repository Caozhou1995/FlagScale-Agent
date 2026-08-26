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


# Folded into the plan-framing gate on purpose. Qualifier extraction is a
# plan-framing concern, and PlanGuard is the guard that actually forces the plan
# into existence — so the demand to extract qualifiers belongs in the same
# message that requires the plan, sharing PlanGuard's single override channel.
# Keeping it as a separate block in another guard caused cross-talk: an
# override_reason the agent supplied to satisfy *this* plan requirement also
# silently released the separate qualifier block, so the qualifier demand never
# actually bit. One gate, one override, no laundering.
_QUALIFIER_EXTRACTION = """

While you frame this plan — extract the task's qualifiers, do not just name its subject.

A task names a subject — the thing to compute, find, or build — and usually also
qualifies it: a point in time the answer is taken at, a version or revision, a
subset or region, a specific definition of the metric that decides "best". The
subject is load-bearing and hard to miss; the qualifier is easy to drop, because
the work runs to completion and yields a confident answer whether or not you
honored it. A dropped qualifier is dangerous precisely because nothing fails — you
simply answer a nearby question the task did not ask.

So, before the plan's shape is fixed: re-read the task statement and list every
qualifier as a first-class item, alongside the subject. Fold them into the plan
explicitly — as their own steps or as acceptance criteria — so that later, before
committing to a result, you check that the data you drew from and the answer you
return sit inside every one of them. A qualifier that is not in the plan is one the
execution silently skips.

Extracting a qualifier is only half the job — you must also read what boundary it
draws, in the direction the task means, not the direction your default preference
assumes. State each qualifier's meaning in your own words, and watch for the moment
you translate it into something more convenient. The common failure is bending the
constraint toward what is easiest to obtain: taking a boundary the task fixed and
quietly widening it to the larger, fresher, or more available thing on hand —
reading a bound on time as merely "recent enough", a specific named subset as the
broader whole, one definition of the metric as whatever number is easiest to
compute. A "newest / biggest / most available is best" instinct is exactly the bias
that overwrites a bounded qualifier: when the task fixes a boundary, the answer
inside it is right even if a fresher or larger candidate exists outside. A bound on
time is especially easy to invert — it can mean the state as it stood at a given
point, with anything that came later out of scope, or the state right now; these
select different answers, so let the task's words decide which, rather than
defaulting to the most up-to-date data.

And before you commit to a shape, answer two questions out loud — not "do I
understand this" (you can answer yes to that without it being true), but two
questions whose answers point at something outside your own confidence:

  1. CLASS — what known problem class is this an instance of? Name it (a search,
     a parse, a constraint-satisfaction, a scheduling, a diff-and-merge, a
     precision-alignment, ...). "It's its own thing" almost always means you have
     not placed it yet — keep looking until you can name the class.
  2. STANDARD METHOD — what is the established technique for that class, and what
     is the first concrete step it prescribes here?

If you cannot name the class, or cannot state its standard method, that inability
is the signal — you are not ready to fix the plan's shape yet, you are still
understanding. Do not paper over it by writing generic steps ("analyze",
"implement", "test"); those are what a plan looks like when the class was never
identified. Go find the class first. If you catch yourself about to plan an
enumeration / brute-force sweep, that is the tell you skipped this — the structure
the task is built on is still hidden from you.

One more thing to catch while the qualifiers are in front of you: some qualifiers
do not just narrow the answer, they PIN you to a concrete tool instance rather than
a category. The tell is a qualifier that carries a precise identifier a whole class
would not share — a specific version or build number, a revision/commit hash, a
release tag, a named model or dataset with a version suffix, a pinned dependency.
A qualifier of that shape is not naming a category (some embedding model, a
compiler, a library), it is naming ONE artifact, and one artifact often has
instance-specific usage requirements that differ from the generic API for its
category — the difference can take ANY form: the correct
call signature for this version, a required preprocessing or ordering step, a
default that changed, an expected input format or precision, an extra token or
prefix the invocation must carry. The trap is to recognize the category, reach for
the generic call you already know, and never discover the instance had its own
rules — the code runs, produces a confident output, and is silently wrong. So when
a qualifier pins you to a named instance, add a plan step (or acceptance criterion)
to CONSULT THAT INSTANCE'S OWN DOCUMENTATION before writing the call — its README /
model card / release notes / official example — and confirm how THIS instance is
meant to be invoked. But consulting is only the near half: the far half is APPLYING
what the doc says for the use the task puts you in. The task's VERB — what it asks
you to DO with the instance — selects which documented usage applies; when the doc
prescribes a specific invocation for that use, the documented way is the DEFAULT and
the plainer/barer call is the DEVIATION that needs justifying. Do not read the doc,
see the requirement, and then talk yourself out of it with "the task didn't
EXPLICITLY ask for that, so the minimal reading is the plain call" — a
literal-minimal reading is not more faithful, it silently swaps the task for a
nearby easier one the instance was not built to serve that way. "I read the docs and
there was genuinely no special requirement for this use" is a valid conclusion; "I
saw the requirement but skipped it because the task didn't spell it out" is not. The
judgment of whether a qualifier pins an instance is yours — this only asks you to
make it while framing the plan, not after the output comes out wrong."""


_COMPLETION_NO_PLAN = (
    "[Plan] Stop — you are about to signal task completion in an unsupervised "
    "(single-shot) run, but you never created a plan this run. A run that reached "
    "completion without ever framing a plan almost always means the task was "
    "under-analyzed: no structural decomposition, no acceptance criteria naming "
    "what THIS task required, no verification gate. Before completing:\n"
    "— If the task had real structure (multiple steps, a deliverable to verify, a "
    "correctness bar): call plan_create() now, work the steps, and verify each "
    "acceptance criterion before completing. The plan is the completion gate that "
    "no human is here to be.\n"
    "— If the task was genuinely trivial (a single lookup, a one-line answer with "
    "nothing to verify): override by re-emitting [TASK_COMPLETE] with an inline "
    "reason, like:\n"
    "  [TASK_COMPLETE]\n"
    "  _override_reason: <reason why no plan was warranted>\n"
    "The judgment is yours; this gate only forces the pause before an unplanned "
    "completion."
)


class PlanGuard(Guard):
    """Nudges (interactive) or requires (single-shot) an active plan.

    Interactive mode: a human supervises the loop, so a plan is optional.
    After REMIND_THRESHOLD tool calls without an active plan, injects a
    periodic reminder. Never blocks.

    Single-shot mode: no human is in the loop, so the plan (with its
    acceptance/verification) stands in as the structural supervisor and the
    completion gate. The agent is still allowed to observe the environment
    first, but once it has taken SINGLE_SHOT_BLOCK_THRESHOLD tool calls
    without a plan it is BLOCKED from further non-plan tools until it calls
    plan_create. Blocking is overridable for the rare genuinely-trivial task.
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
        # Whether plan_create was ever called in this run. The completion gate
        # (single-shot only) blocks [TASK_COMPLETE] when this is still False:
        # an unsupervised run that finished without ever framing a plan almost
        # certainly under-analyzed the task. Distinct from _task_plan.get_active()
        # — a plan may have been created then deactivated; what matters for the
        # completion gate is whether the agent ever planned at all.
        self._plan_ever_created = False
        # Qualifier extraction must reach the agent exactly once at plan-framing.
        # It rides the single-shot block when that fires first; but if the agent
        # is diligent and calls plan_create before the block threshold, the block
        # never fires, so we also inject it on the first plan_create. This flag
        # ensures it is delivered once by whichever path happens first.
        self._qualifier_reminded = False

    def set_single_shot(self, enabled: bool = True):
        """Enable single-shot enforcement at runtime (set once run mode known)."""
        self._single_shot = enabled

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            # Completion gate (single-shot only): block a [TASK_COMPLETE] signal
            # when no plan was ever created this run. Overridable for the rare
            # genuinely-trivial task via an inline `_override_reason: <reason>` in
            # the completion message (the text channel kernel extracts, since a
            # [TASK_COMPLETE] signal carries no tool_args). Blocks EVERY time until
            # either a plan is created or a valid override reason is supplied — a
            # bare re-emit of [TASK_COMPLETE] with neither stays blocked. Never
            # fires in interactive mode (a human supervises).
            if (self._single_shot
                    and not self._plan_ever_created
                    and ctx.assistant_text
                    and "[TASK_COMPLETE]" in ctx.assistant_text):
                return GuardVerdict.block(
                    message=_COMPLETION_NO_PLAN,
                    reason="single_shot_completion_without_plan",
                    category="plan_required",
                )
            return None

        # Plan-related tools don't count toward the no-plan budget. But the FIRST
        # plan_create is the plan-framing moment — if the agent got here without
        # having been blocked (diligent enough to plan before the threshold), the
        # qualifier-extraction demand has not yet been delivered, so inject it now.
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

        # Single-shot: after the observation budget, require a plan (block).
        if self._single_shot and self._calls_without_plan > self.SINGLE_SHOT_BLOCK_THRESHOLD:
            # The block carries the qualifier demand too, so an agent that only
            # sees the block still gets it. Mark it delivered so the subsequent
            # plan_create (which the agent makes to satisfy this block) does not
            # inject it a second time.
            self._qualifier_reminded = True
            return GuardVerdict.block(
                message=(
                    f"[Plan] Pause. You've made {self._calls_without_plan} tool calls "
                    f"in this unsupervised run without a plan. Stop and decide, deliberately: "
                    f"do you now understand the problem's structure well enough to plan it?\n"
                    f"— If yes: call plan_create() now, and hold it to the quality bar — "
                    f"steps whose boundaries fall on real checkpoints, acceptance criteria "
                    f"that name what THIS task specifically requires, not generic \"works correctly\".\n"
                    f"— If no — if you're still genuinely gathering the information needed to "
                    f"understand the structure: override this and keep investigating. A plan "
                    f"written before understanding is worse than no plan, because it locks in "
                    f"a shape you'll have to fight later.\n"
                    f"The judgment is yours. This gate only forces the pause, not the answer."
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
