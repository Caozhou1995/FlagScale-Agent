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

"""PlanUpdateGuard — detects being stuck on one step and prompts self-diagnosis.

Many iterations on the SAME step without progress is the runtime signature of a
stall. This guard fires on that timing to prompt diagnosis: escape DOWNWARD (smaller
experiment) or UPWARD (different method-class), never SIDEWAYS (another variant).

Two orthogonal signals, either fires the same reminder:
- COUNT: tool calls since last plan_update. First at FIRST_REMIND, then every
  REMIND_INTERVAL. Catches DENSE stalls (many quick thrashing calls).
- TIME: wall-clock since last plan_update, every TIME_REMIND_SECONDS. Catches
  SPARSE stalls (few calls but long thinks between them).

Escalation: every ESCALATE_AFTER-th reminder (across both signals) becomes a BLOCK
instead of inject, then the counter resets: inject, inject, block, repeat. A block
forces a substantive plan_update (or override) before continuing. Counters reset
on plan_update/plan_create. Meta tools (plan_status, evict, memory_read, ...) don't
count and don't tick.

Side-channel guard: on deliverable+threshold tasks, a fact only clears a block if
OBSERVED AT THE DELIVERABLE (edit + re-run + measure), not derived in scratch/prototype/
on paper. This wires PRINCIPLE 2 into the stall clearance bar.

Mechanism limit: guards run in check_post (AFTER a tool call). If the agent goes
silent with no tool calls, the guard never fires until the next call. In practice
thrash always emits tool calls, so the signal still fires.
"""

from __future__ import annotations

import time

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class PlanUpdateGuard(Guard):
    """Detects being stuck on one plan step and prompts self-diagnosis.

    Tracks iterations since the last plan_update while a step is active. Many
    iterations on the same step without a plan_update is treated as a stall
    signal, and the guard nudges the agent to stop and diagnose (escape
    downward/upward, not sideways) rather than merely to bookkeep the plan.
    """

    name = "plan_update"
    priority = 50

    # Fire the first reminder early — a stall caught early is cheap to escape.
    FIRST_REMIND = 10
    # After the first, remind periodically but spaced out, so a repeated nudge on
    # the same step does not dilute into noise the agent learns to skip.
    REMIND_INTERVAL = 20
    # Wall-clock stall signal: remind every this many seconds on the same step,
    # independent of tool-call count. Catches sparse-but-slow thrash.
    TIME_REMIND_SECONDS = 180
    # Every ESCALATE_AFTER-th stall reminder (counting across both signals)
    # escalates from advisory inject to a blocking verdict, then the counter
    # resets and the cycle repeats: inject, inject, block, inject, inject, block.
    # An advisory nudge can be read and ignored; a block cannot — it forces the
    # agent to actually stop and plan_update (or justify an override) before
    # continuing. We only escalate after ESCALATE_AFTER-1 ignored advisories,
    # so a block always means "gentle reminders demonstrably had no effect".
    ESCALATE_AFTER = 3

    # Tools that don't count toward threshold (meta-operations)
    _META_TOOLS = frozenset((
        "plan_status",  # Read-only
        "evict", "recall",
        "memory_read", "memory_list",
    ))

    # plan_update actions that are ALWAYS a substantive response: they mutate the
    # plan's real state (a step advanced/finished/abandoned, subtasks added,
    # acceptance refined, a batch of steps updated). Any of these earns a full
    # reset — including the escalation counter — because the agent genuinely moved
    # the plan forward, not merely pinged the guard.
    _PROGRESS_ACTIONS = frozenset((
        "step_done", "step_skip", "complete", "abandon",
        "add_steps", "update_acceptance", "batch",
    ))

    @classmethod
    def _is_substantive_update(cls, ctx) -> bool:
        """Distinguish a real plan response from an empty guard-clearing ping.

        The stall block is meant to be cleared by genuinely responding — marking
        a step done, or recording a concrete fact in notes. But the reset was
        unconditional, so a no-op ``plan_update(action='step_doing')`` with NO
        notes (same step, no info) also wiped the escalation counter, handing the
        agent a clean slate to resume the exact thrash the block was catching.

        A response is substantive iff it changes plan state or records info:
        - ``plan_create`` — a whole new plan.
        - a progress action (see ``_PROGRESS_ACTIONS``) — real state mutation.
        - any other action (step_doing/reactivate/deactivate) ONLY if it carries
          non-empty notes — i.e. it actually recorded something.

        This is domain-agnostic: it keys off the action type and notes presence,
        never off task content. It preserves the block's INTENDED clearance path
        (plan_update whose note names a concrete fact) while closing the empty-
        ping hole.
        """
        if ctx.tool_name == "plan_create":
            return True
        if ctx.tool_name != "plan_update":
            return False
        args = ctx.tool_args or {}
        action = args.get("action", "")
        if action in cls._PROGRESS_ACTIONS:
            return True
        notes = args.get("notes") or ""
        return bool(notes.strip())

    def __init__(self, task_plan, time_fn=time.monotonic):
        self._task_plan = task_plan
        self._iters_since_update = 0
        # Injected for testability; defaults to a monotonic wall clock.
        self._time_fn = time_fn
        # PERIOD anchor: decides WHEN the time signal fires. Re-anchored on every
        # fire so reminders are periodic (once per TIME_REMIND_SECONDS window).
        # Lazily initialized on first counting tool call (None = not yet anchored).
        self._time_anchor = None
        # STALL-START anchor: the wall-clock moment the current stall began (first
        # counting tool call after a plan_update). Unlike _time_anchor it is NOT
        # re-anchored on fire, so it measures the TOTAL time stuck on the step —
        # the quantity the reminder should display, cumulative like the count n.
        # Reset only on plan_update/plan_create. None = not yet anchored.
        self._stall_start = None
        # How many stall reminders have fired since the last plan_update. Shared
        # across BOTH signals (count and time) — the agent perceives "I've been
        # nudged N times", regardless of which signal produced each nudge. The
        # first ESCALATE_AFTER-1 fire as inject (advisory); the ESCALATE_AFTER-th
        # escalates to a block, then this resets to 0 (cycle repeats). A block is
        # only reached after the agent has IGNORED that many advisory nudges — so
        # it stays a backstop for "reminder had no effect", never the common case.
        self._stall_trigger_count = 0
        # Sticky-block latch. Set True the moment a stall reminder ESCALATES to a
        # block; cleared ONLY by a substantive plan response. While set, an empty
        # guard-clearing ping (plan_update with no state change and no notes)
        # cannot pass — it is re-blocked. This is what makes the block's clearance
        # bar real: the block says "name a concrete fact or mark the step done",
        # and this latch enforces that the NEXT plan_update actually does one of
        # those, instead of a no-op ping buying a clean slate.
        self._block_pending = False

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Check if plan needs updating after tool execution."""
        active_plan = self._task_plan.get_active()
        if not active_plan:
            return None

        steps = active_plan.get("steps", [])
        if not steps:
            return None

        # Plan touched — decide whether it was a SUBSTANTIVE response or an empty
        # guard-clearing ping. Both reset the count/time anchors (so the guard
        # does not immediately re-fire on the very next tool call — that would be
        # noise). But ONLY a substantive response resets the escalation counter:
        # a bare plan_update(step_doing) with no notes must NOT buy back a clean
        # slate, or the block is trivially gamed by pinging the guard and resuming
        # the same thrash. Preserving _stall_trigger_count across empty pings means
        # the escalation tier survives, so the next fire can still BLOCK — forcing
        # a real response (a step marked done, or a concrete fact in notes).
        if ctx.tool_name in ("plan_update", "plan_create"):
            substantive = self._is_substantive_update(ctx)
            # A block is pending clearance and this update is an empty ping (no
            # state change, no notes): re-block. The agent must actually respond
            # — mark the step done/skipped, or record a concrete fact in notes —
            # not merely ping the guard to reset the timers. Timers are left
            # untouched so the situation is unchanged until a real response comes.
            if self._block_pending and not substantive:
                return GuardVerdict.block(
                    message=(
                        "[PlanUpdate] This plan_update does not clear the pending "
                        "stall block: no step state changed and no notes recorded — "
                        "an empty guard-clearing ping. To continue: (a) mark the stalled "
                        "step done/skipped, or (b) plan_update with a note naming ONE "
                        "concrete, falsifiable fact (value measured, assumption "
                        "confirmed/refuted, candidate eliminated) plus the next "
                        "method-class. If the task requires an output that does not "
                        "exist on disk, write the crudest valid version to the path "
                        "now. If you cannot name a fact, that proves you are looping — "
                        "switch method-class, do not ping again."
                    ),
                    reason="empty_ping_does_not_clear_block",
                    category="plan_update",
                )
            self._iters_since_update = 0
            self._time_anchor = self._time_fn()
            self._stall_start = self._time_anchor
            if substantive:
                self._stall_trigger_count = 0
                self._block_pending = False
            return None

        # Meta tools don't count and don't tick the clock.
        if ctx.tool_name in self._META_TOOLS:
            return None

        self._iters_since_update += 1
        # Anchor the clock on the first counting tool call after a reset.
        now = self._time_fn()
        if self._time_anchor is None:
            self._time_anchor = now
        if self._stall_start is None:
            self._stall_start = now

        # COUNT signal: first at FIRST_REMIND, then every REMIND_INTERVAL after.
        n = self._iters_since_update
        fire_count = n == self.FIRST_REMIND or (
            n > self.FIRST_REMIND and (n - self.FIRST_REMIND) % self.REMIND_INTERVAL == 0
        )
        # TIME signal: fire once per TIME_REMIND_SECONDS window. Re-anchor on fire
        # so the next window starts fresh (periodic wall-clock reminders).
        elapsed = now - self._time_anchor
        fire_time = elapsed >= self.TIME_REMIND_SECONDS
        if fire_time:
            self._time_anchor = now

        if fire_count or fire_time:
            doing_steps = [s for s in steps if s.get("status") == "doing"]
            if doing_steps:
                step_id = doing_steps[0].get("id")
                # Describe whichever signal(s) fired, so the agent sees the actual
                # runtime evidence (many actions, or long wall-clock, or both).
                # Display the TOTAL time stuck on the step (cumulative from
                # _stall_start), NOT the periodic-window elapsed used to DECIDE
                # firing — the window resets on every fire, so it would under-report
                # the real stall duration and mismatch the cumulative count n.
                total_stuck = now - self._stall_start
                mins = int(total_stuck // 60)
                if fire_count and fire_time:
                    sig = f"{n} tool calls and ~{mins} min elapsed"
                elif fire_time:
                    sig = f"~{mins} min elapsed ({n} tool calls)"
                else:
                    sig = f"{n} iterations"

                # Count this reminder. Every ESCALATE_AFTER-th one escalates to a
                # block; then reset so the cycle repeats (inject, inject, block).
                self._stall_trigger_count += 1
                escalate = self._stall_trigger_count % self.ESCALATE_AFTER == 0

                body = (
                    f"[PlanUpdate] {sig} on step {step_id} with no plan update — "
                    f"the runtime signature of a stall. First, answer concretely: "
                    f"what did the last round tell you that you did not already know? "
                    f"Name the specific fact — a value measured, an assumption "
                    f"confirmed or refuted, a candidate eliminated. If your honest "
                    f"answer is \"nothing new\" / \"same as expected\" / \"still don't "
                    f"know why\", that is proof your information gain is zero and you "
                    f"are looping — not a phrasing problem. You cannot fix a stall by "
                    f"thinking harder in the same place; only a new fact moves you.\n\n"
                    f"Escape routes (pick one):\n"
                    f"— DOWNWARD: smallest experiment isolating which assumption is "
                    f"wrong.\n"
                    f"— UPWARD: switch to a different method-class.\n"
                    f"— NOT another variant of what failed (parallelizing, retuning, "
                    f"faster rewrite are all SIDEWAYS — same class).\n"
                    f"— THIRD escape: if rounds are oscillating — a metric wobbling "
                    f"around a plateau, each ending \"slightly worse, revert\" — you "
                    f"are burning budget on variance, not progressing. Stop tuning, "
                    f"deliver the best version you already measured, mark done.\n"
                    f"— FOURTH escape: if the task names a required output that does not "
                    f"exist on disk yet, STOP perfecting — BUDGET ORDER: write the "
                    f"crudest complete-but-valid version to the exact path now, confirm "
                    f"it exists, then resume improving. A rough answer that exists "
                    f"beats a perfect one that never got written.\n"
                    f"Record your shift in notes (\"tried X → failed because Y → now "
                    f"trying Z\") so you don't re-walk the dead path. If the step is "
                    f"genuinely finished or abandoned, mark it done/skipped."
                )

                if escalate:
                    # Advisory nudges were ignored this many times — force a stop.
                    self._block_pending = True
                    return GuardVerdict.block(
                        message=(
                            body
                            + f"\n\nThat is {self._stall_trigger_count} stall "
                            f"reminders with no plan_update — earlier advisories had no "
                            f"effect, so this one BLOCKS. A vague 'I am making progress' "
                            f"does NOT clear this — only a concrete, falsifiable fact does. "
                            f"To continue: (a) plan_update whose note names ONE assumption "
                            f"your last experiment tested and the result that REFUTED or "
                            f"confirmed it, plus the next method-class to try — or, if "
                            f"oscillating around a plateau, mark done after shipping best "
                            f"(\"gains stopped, further tuning adds variance\"); or "
                            f"(b) override with _override_reason stating the concrete new "
                            f"fact produced (value measured, hypothesis eliminated, bug "
                            f"found). For deliverable+threshold tasks: a fact only clears "
                            f"this block if OBSERVED AT THE DELIVERABLE — edited it, "
                            f"re-ran, measured score before/after. Derivation in a scratch "
                            f"script, prototype, or on paper is side-channel work, not "
                            f"progress — if the deliverable's score hasn't moved, you are "
                            f"optimizing the wrong medium. Land the next change in the "
                            f"deliverable and measure it."
                        ),
                        reason="repeated_stall_ignored",
                        category="plan_update",
                    )
                return GuardVerdict.inject(
                    message=body,
                    reason="possible_stall",
                    category="plan_update",
                )

        return None
