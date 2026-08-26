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

Framing: many iterations on the SAME step without progress is the runtime
signature of a stall — oscillating, re-deriving the same sub-problem, or tuning
variants of an approach that already failed. This guard is the runtime trigger
for the "stalling is a failure mode" discipline in the system prompt (escape
DOWNWARD to a smaller experiment or UPWARD to a different method-class, never
SIDEWAYS to another variant). It fires on that timing to prompt the agent to
stop and diagnose, not merely to remember to bookkeep the plan.

Two orthogonal stall signals, either one fires the same reminder:
- COUNT: tool calls (iterations) since last plan_update. First reminder at
  FIRST_REMIND (fire early — a stall caught early is cheap), then periodically
  every REMIND_INTERVAL after. Catches DENSE stalls (many quick tool calls
  thrashing variants).
- TIME: wall-clock elapsed since last plan_update. Reminder every
  TIME_REMIND_SECONDS. Catches SPARSE stalls (few tool calls but each preceded
  by a long think — the count signal is blind to these because it only counts
  actions, not how long each took). A sparse stall looks exactly like this:
  long per-call thinks (hundreds of seconds each) between tool calls, so many
  wall-clock minutes elapse before the count signal ever reaches its threshold.

The two are orthogonal coverage: count is sensitive to action frequency, time
is sensitive to per-action duration. Together they cover both thrash modes.

Escalation: a reminder can be read and ignored. To keep an ignored stall from
running unbounded (an ignored stall can otherwise burn every reminder and run
straight to the hard timeout), reminders escalate. Every ESCALATE_AFTER-th reminder (counted across BOTH signals,
since the agent just perceives "I've been nudged N times") becomes a BLOCK instead
of an advisory inject, then the counter resets: inject, inject, block, inject,
inject, block. A block forces the agent to stop and plan_update (or justify an
override) before continuing. The count resets on plan_update/plan_create, so only
CONSECUTIVE ignored reminders accumulate — responding earns a clean slate.

Side-channel guard on the block's clearance bar: the block is cleared by naming
a "concrete falsifiable fact", but a subtle stall can satisfy that bar while
never making real progress. On tasks with a real deliverable and a score/
similarity threshold, the agent can keep producing genuinely-new facts DERIVED
in a scratch script / prototype / on paper (path-tracing: re-derived the sphere-
tangent formula, then the shadow equation, then ...) — each clears the block,
resets the count, and sends it back into the same paper-derivation method-class
without ever editing+re-measuring the deliverable. So the block message
additionally requires: for deliverable+threshold tasks, a fact only counts if it
was OBSERVED AT THE DELIVERABLE (edited it, re-ran, measured the score move
before/after) — a derivation that never touched the graded artifact is side-
channel work, not progress. This wires PRINCIPLE 2 ("verify where the real
consumer observes the result") into the stall clearance bar.

Reset: counters/anchors AND the escalation count reset on plan_update/plan_create.
Meta tools (plan_status, evict, memory_read, ...) don't count and don't tick.

Mechanism limit: guards run in check_post (AFTER a tool call). If the agent goes
silent in pure thinking with NO tool calls at all, the guard is never invoked;
the time reminder lands on the NEXT tool call. In practice thrash always emits
tool calls (think -> write -> compile -> think), so the signal still fires.
"""

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
                        "stall block: it changed no step state and carries no "
                        "notes, so it is an empty guard-clearing ping — exactly the "
                        "move the block exists to stop. To continue you must make a "
                        "SUBSTANTIVE response: (a) mark the stalled step done/skipped "
                        "(if it is genuinely finished or abandoned), or (b) call "
                        "plan_update with a note naming ONE concrete, falsifiable "
                        "fact your last experiment produced (a value measured, an "
                        "assumption confirmed/refuted, a candidate eliminated) plus "
                        "the next method-class to try. If the task names a required "
                        "output that still does not exist on disk, the substantive "
                        "move is to write the crudest complete-but-valid version to "
                        "that exact path now, then note that you did. If you truly "
                        "cannot name a fact, that is the proof you are looping — "
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
                    f"[PlanUpdate] {sig} on step {step_id} with no plan "
                    f"update — the runtime signature of a stall. Before anything "
                    f"else, answer one question concretely: WHAT did the last round "
                    f"tell you that you did not already know one round ago? Name the "
                    f"specific fact — a value you measured, an assumption you "
                    f"confirmed or refuted, a candidate you eliminated. If your "
                    f"honest answer is \"nothing new\" / \"same as I expected\" / "
                    f"\"still don't know why\", that is not a phrasing problem — it "
                    f"is the proof your information gain is zero and you are looping, "
                    f"however much reasoning you produced. You cannot fix a stall by "
                    f"thinking harder in the same place; only a new fact moves you. "
                    f"The escape is NOT another "
                    f"variant of what just failed (parallelizing, retuning params, a "
                    f"faster rewrite are all SIDEWAYS — same method-class). Escape "
                    f"DOWNWARD: write the smallest experiment that isolates which "
                    f"assumption is wrong. Or UPWARD: switch to a different "
                    f"method-class for this problem. Then record the shift in notes "
                    f"(\"tried X → failed because Y → now trying Z\") so you don't "
                    f"re-walk the dead path. There is also a THIRD escape people miss "
                    f"on optimization/improvement steps: maybe nothing is wrong and you "
                    f"are simply DONE. If your recent rounds are oscillating — a metric "
                    f"wobbling around a plateau, a value nudged up then reverted, each "
                    f"round ending \"slightly worse, let me revert\" — that is not a "
                    f"stall to escape, it is the signal that distinct methods have "
                    f"stopped yielding gains and you are now burning budget on variance. "
                    f"Stop tuning, deliver the best version you already measured, and "
                    f"mark the step done — one more turn of the same dial is not "
                    f"progress. And if the step is genuinely finished or abandoned, "
                    f"mark it done/skipped. A FOURTH escape applies when the task "
                    f"names a required output (a file/artifact at a specific path) "
                    f"and it does not exist on disk yet: if these stalled rounds have "
                    f"been spent thinking, exploring, or perfecting one part while the "
                    f"required output still is not written anywhere, STOP — your "
                    f"finite budget can run out and whatever sits at that path is ALL "
                    f"that gets scored. The escape is BUDGET ORDER: write the crudest "
                    f"complete-but-valid version of the required output to the exact "
                    f"path RIGHT NOW (a stub, a naive result, a hardcoded-but-"
                    f"well-formed output), confirm it exists, THEN resume improving "
                    f"it. A rough answer that actually exists beats a perfect one that "
                    f"never got written."
                )

                if escalate:
                    # Advisory nudges were ignored this many times — force a stop.
                    # Latch the block so an empty guard-clearing ping cannot pass;
                    # only a substantive plan response clears it (see reset branch).
                    self._block_pending = True
                    return GuardVerdict.block(
                        message=(
                            body
                            + f"\n\nThat is {self._stall_trigger_count} stall "
                            f"reminders on this step with no plan_update in between — "
                            f"the earlier advisories had no effect, so this one BLOCKS. "
                            f"A vague 'I am making progress' does NOT clear this block — "
                            f"only a concrete, falsifiable fact does. You cannot continue "
                            f"until you either (a) call plan_update whose note names ONE "
                            f"specific assumption your last experiment tested and states "
                            f"the concrete result that confirmed or REFUTED it (e.g. "
                            f"'assumed F is linear → measured output, it is NOT, so the "
                            f"linear-approx method-class is dead'), plus the next "
                            f"method-class to try — or, on an optimization step where "
                            f"your attempts are oscillating around a plateau, mark it "
                            f"done after delivering the best version you already "
                            f"measured (\"gains stopped at 0.988, further tuning only "
                            f"adds variance, shipping best\"); or "
                            f"(b) override with _override_reason that states the concrete "
                            f"new fact the last round produced (a value measured, a "
                            f"hypothesis eliminated, a bug found) — not that you 'feel' "
                            f"you are progressing. If you cannot name a specific fact the "
                            f"last few rounds produced, that is itself the proof you are "
                            f"looping: stop and switch method-class. One more trap, "
                            f"specific to tasks with a real deliverable and a "
                            f"score/similarity threshold: a 'concrete fact' only clears "
                            f"this block if it was OBSERVED AT THE DELIVERABLE ITSELF — "
                            f"you edited the deliverable, re-ran it, and measured the "
                            f"score move (before/after). A fact DERIVED in a scratch "
                            f"script, a prototype, or on paper (e.g. 'I re-derived the "
                            f"sphere-tangent formula', 'the shadow equation should be X') "
                            f"is NOT progress however concrete it sounds — it is "
                            f"side-channel work that never touched the thing being graded. "
                            f"If your recent rounds each produced a new derivation but the "
                            f"deliverable's measured score has not moved (or you never "
                            f"re-measured it), THAT is the stall: you are optimizing the "
                            f"wrong medium. Escape by landing the next change in the "
                            f"deliverable and measuring it; if error persists, decompose "
                            f"the error by region/component to find the largest "
                            f"contributor before deriving anything further."
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
