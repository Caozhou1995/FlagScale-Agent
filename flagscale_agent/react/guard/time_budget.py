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

"""TimeBudgetGuard — periodic wall-clock awareness for the agent itself.

The resolved time budget is exported by the harness into
FLAGSCALE_AGENT_TIME_BUDGET_SEC and, until now, only reached the health judge
(which reasons about a *running* shell command). The agent's own ReAct loop was
blind to how much of its wall-clock allowance it had spent — so it would happily
compile single-threaded, block synchronously on long trainings, and tear down a
near-passing artifact to retry, only to be killed by the external timeout.

This guard closes that gap. It reads the SAME structured budget the health judge
sees (via a stats_fn injected at construction) and, as cumulative wall-clock
crosses escalating thresholds, injects a one-time advisory into the agent's
context so it can change strategy WHILE it still has budget left.

Key design decisions:
  • NO fabricated budget. stats_fn returns None whenever no concrete wall was
    injected (standalone / interactive runs). In that case this guard stays
    completely silent — it never invents a deadline, and never nags a session
    that has no real time pressure.
  • Percentage thresholds, not absolute seconds. The same 25/50/75/90/100% ladder
    works for a 20-minute wall and a 1-hour wall without any per-task tuning.
  • Two modes on the ladder:
      - 25/50/75%   → check_post inject (pacing / health-check advisories).
      - 90%         → check_pre BLOCK (overridable) — forces the agent to SHOW
                      (evidence, not claim) that a deliverable exists or that
                      THIS call produces it, before spending a near-final tool
                      call on exploration.
      - 100%        → check_post inject (WRAP-UP). The per-turn wall-clock is
                      spent: this is the agent's last stretch this turn. The
                      message tells it to finalize the current result and, if it
                      needs a human decision or more input to go further, hand
                      control back with NEED_USER_INPUT. This is deliberately an
                      inject, NOT a block: at/after timeout the agent must be free
                      to produce its closing response (and NEED_USER_INPUT) without
                      a gate standing in front of it. The 90% block is therefore
                      scoped to [90, 100) — once the wall is fully spent it steps
                      aside so the wrap-up can proceed. The agent is trusted to
                      understand "time is up, close out" without a hard stop.
  • Fire each threshold at most once per turn (a _fired set), so it does not
    spam every tool call once past 50%.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


def _fmt(sec: float) -> str:
    """Format seconds as compact Hh:MMm or Mm:SSs (mirrors agent._fmt)."""
    sec = int(max(0.0, sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


class TimeBudgetGuard(Guard):
    """Inject escalating wall-clock advisories as the task budget is consumed."""

    name = "time_budget"
    priority = 92  # Low priority — advisory only, near MemoryDiscipline.

    # Ordered high→low so the FIRST crossed-but-unfired threshold is the most
    # severe one still pending. Each maps to (label, builder).
    #
    # Why 25% is the FIRST rung (not 50%): the earliest advisory carries the
    # "front-load the expensive steps, background the long ones, re-check the
    # plan" guidance — but those are levers you can only pull EARLY, while the
    # decision window is still open. By 50% the costly download/compile/train has
    # either already started or already should have; the advice arrives after the
    # window it applies to has closed (observed: a run that only started its
    # dataset download at ~13% budget and gave up at ~40% would never even see a
    # 50% nudge). Firing at 25% puts the pacing guidance where the agent has
    # enough trajectory to gauge its burn rate AND enough budget left to correct.
    _THRESHOLDS = (100, 90, 75, 50, 25)

    def __init__(self, stats_fn):
        """stats_fn() -> dict|None with keys elapsed/budget/remaining/pct.

        None means no external wall is enforced; the guard stays silent.
        """
        self._stats_fn = stats_fn
        self._fired: set[int] = set()

    def reset_turn(self):
        # A new user message restarts the per-turn wall-clock accounting on the
        # agent side (self._turn_start is re-stamped), so clear which thresholds
        # we have already announced.
        self._fired = set()

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Block at 90%+ budget to force explicit deliverable reasoning.
        
        At the CRITICAL 90% threshold, switch from advisory (check_post inject) to
        blocking gate (check_pre block). The agent must SHOW, not merely claim:
          (1) A complete deliverable already exists at its required path — the
              override_reason must cite the output of a verification action just
              run (e.g. ls -la / cat on the delivery path), OR
          (2) This specific tool call will produce/finalize the deliverable —
              cite the delivery path it writes or this tool's concrete output.
        
        If neither is true, the agent should STOP optimizing and write a crude-but-
        complete answer instead. The override_reason becomes the forcing function.
        """
        if not ctx.tool_name:
            return None

        stats = None
        try:
            stats = self._stats_fn() if self._stats_fn else None
        except Exception:
            return None
        if not stats:
            return None

        pct = stats.get("pct", 0.0)
        # Block only in the [90, 100) window, once per turn (a checkpoint, not a
        # repeated wall). At/after 100% the wall-clock is fully spent and the guard
        # must NOT block: the agent needs a clear path to emit its wrap-up response
        # (and NEED_USER_INPUT) — that final rung is handled as a check_post inject.
        if 90 <= pct < 100 and 90 not in self._fired:
            # Mark ALL crossed thresholds as spent, preventing check_post from
            # re-injecting lower advisories and preventing this block from re-firing.
            for t in self._THRESHOLDS:
                if pct >= t:
                    self._fired.add(t)
            
            elapsed = _fmt(stats.get("elapsed", 0.0))
            remaining = _fmt(stats.get("remaining", 0.0))
            message = (
                f"[TimeBudget] CRITICAL CHECKPOINT — {pct:.0f}% of your enforced "
                f"wall-clock budget is spent ({elapsed} used, ~{remaining} left). "
                f"Before executing this tool, you must satisfy ONE of these:\n\n"
                f"  (1) A complete, valid deliverable ALREADY EXISTS at its required "
                f"path — SHOWN, not claimed: your _override_reason must cite the "
                f"output of a verification action you just ran (e.g. `ls -la "
                f"<delivery_path>` listing the file, or `cat <delivery_file>` "
                f"matching the expected content).\n"
                f"  (2) THIS specific tool call will produce or finalize the "
                f"deliverable — cite the delivery path it writes or this tool's "
                f"concrete output.\n\n"
                f"Re-classified so it cannot be misread: verifying a claim you are "
                f"about to bank is NOT refining. When a result you ALREADY hold is "
                f"suspect, the CHEAPEST trust-restoring action on that same "
                f"already-suspect value (re-run the one command that produced it, "
                f"re-check the file, re-execute the smallest verification) outranks "
                f"write-through of that value. Do not bank a value you hold an "
                f"unresolved doubt about: name the doubt, run the cheapest check "
                f"that settles it, or explicitly record why the current value is "
                f"still your strongest anchor (an audited degrade beats a silent "
                f"one).\n\n"
                f"If NEITHER (1) nor (2) is true — exploring, optimizing, or "
                f"polishing beyond the verification above — STOP. Write a "
                f"crude-but-complete answer to the delivery path RIGHT NOW instead. "
                f"Also, while you still hold the FULL context that is about to be "
                f"lost at timeout: memory_write() the reusable facts that live only "
                f"in this window (exact commands, paths, env state, pitfalls — the "
                f"survival-range test: cross-session truth -> memory, GLOBAL; "
                f"this-session progress -> plan notes; do not dump what one ls/grep "
                f"can cheaply re-derive).\n\n"
                f"Mechanism fact: guard injections are NOT a clock — their frequency "
                f"or presence says nothing about remaining time (silence between "
                f"them may just mean one long call is still running). The only "
                f"remaining-time source is the numbers in this message."
            )
            return GuardVerdict.block(
                message=message,
                reason="time_budget_90pct_block",
                category="time_budget",
                overridable=True,
            )
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Only react to an actually-executed tool call, matching how the other
        # cadence guards (memory_discipline) advance on real work rather than on
        # every check pass.
        if not ctx.tool_name:
            return None

        stats = None
        try:
            stats = self._stats_fn() if self._stats_fn else None
        except Exception:
            # A stats_fn hiccup must never break tool execution — fail silent.
            return None
        if not stats:
            return None

        pct = stats.get("pct", 0.0)
        # Find the most severe threshold that is crossed and not yet announced.
        # _THRESHOLDS is ordered high->low, so the first crossed-and-unfired one
        # is the most severe pending message.
        for thr in self._THRESHOLDS:
            if pct >= thr and thr not in self._fired:
                # Mark EVERY crossed threshold as spent, not just this one. If pct
                # jumped straight past several (e.g. 30 -> 95), we emit only the
                # most severe (90% CRITICAL) and must not dribble out the milder
                # 75%/50% advisories on later calls — that would walk urgency
                # BACKWARDS. Lower crossed thresholds are moot once a higher one
                # has fired.
                for t in self._THRESHOLDS:
                    if pct >= t:
                        self._fired.add(t)
                return GuardVerdict.inject(
                    self._message(thr, stats),
                    reason=f"time_budget_{thr}pct",
                    category="time_budget",
                )
        return None

    def _message(self, thr: int, stats: dict) -> str:
        elapsed = _fmt(stats.get("elapsed", 0.0))
        remaining = _fmt(stats.get("remaining", 0.0))
        pct = stats.get("pct", 0.0)
        if thr >= 100:
            # Per-turn wall-clock fully spent. This is a WRAP-UP nudge, not a gate:
            # the agent is free to run its closing tool calls, but should now be
            # finishing, not starting new work.
            return (
                f"[TimeBudget] TIME IS UP — {pct:.0f}% of your enforced wall-clock "
                f"budget for this turn is spent ({elapsed} used). Treat this as your "
                f"final stretch: this turn is ending. Your remaining slack buys "
                f"PROTECTIVE actions only, in this priority order:\n"
                f"  1. Make sure a COMPLETE, valid result is written through to its "
                f"required delivery path RIGHT NOW — a crude-but-complete answer that "
                f"is banked beats a perfect one that never gets saved.\n"
                f"  2. VERIFY what is banked (`ls`/`cat` the delivery path — verify, "
                f"do not recompute). NEVER start anything that could overwrite a "
                f"banked deliverable: a fresh full run that writes the delivery file "
                f"itself is NOT a wrap-up action, even if its 'payoff' looks better.\n"
                f"  3. If a doubt about the banked value stays unresolved, record it "
                f"explicitly next to the value (an audited degrade beats a silent "
                f"one).\n"
                f"  4. Only then housekeeping (proposal/memory tidy-ups come LAST, "
                f"not first) and close out. If finishing genuinely needs a human "
                f"decision, more input, or an action you cannot take, hand control "
                f"back with NEED_USER_INPUT stating exactly what you need and the "
                f"current state — rather than burning the overrun on more attempts.\n"
                f"Mechanism fact: guard injections are NOT a clock — 'the guard is "
                f"still giving me turns' is not evidence of slack. Silence between "
                f"injections may just mean one long call is still running; the only "
                f"remaining-time source is the numbers in this message."
            )
        head = (
            f"[TimeBudget] {pct:.0f}% of your enforced wall-clock budget is gone "
            f"({elapsed} used, ~{remaining} left before the harness terminates the "
            f"whole task)."
        )
        if thr >= 90:
            tail = (
                " CRITICAL — you are almost out of time. Do exactly ONE thing: make "
                "sure a COMPLETE, valid deliverable exists at its required path RIGHT "
                "NOW. If your best result so far is only in memory or a scratch file, "
                "write it through to the delivery path THIS STEP. A crude-but-complete "
                "answer that is banked beats a perfect one that never gets written. "
                "Stop refining; stop exploring new approaches."
            )
        elif thr >= 75:
            tail = (
                " Most of the budget is spent. If your current approach has not "
                "produced a passing result yet, it likely will not finish in time — "
                "switch to a faster method-class NOW rather than turning more knobs on "
                "the same one. Action checklist, in priority order:\n"
                "  1. Write-through the best valid result you already have to the "
                "delivery path, so a timeout cannot wipe it out.\n"
                "  2. List the doubts you already hold about that result. From now on "
                "your remaining slack is ONLY for: the cheapest trust-restoring "
                "action on a suspect result (verify/re-run the one command that "
                "produced it) — or the protected write-through above. No new lines "
                "of work. Verifying what you bank is not refining.\n"
                "  2b. Any doubt you cannot resolve now: record it explicitly next "
                "to the banked value — why the current value is still your strongest "
                "anchor. An audited degrade beats a silent one."
            )
        elif thr >= 50:
            tail = (
                " You are past the halfway mark — a HEALTH CHECK, not a pacing tip. "
                "Ask concretely: have the expensive steps (downloads, builds, "
                "trainings) actually STARTED, and are they on track to FINISH before "
                "the budget runs out? At the current rate, will a complete valid "
                "deliverable exist at its required path in time?\n"
                "Default actions if the honest answer is not a confident yes:\n"
                "  - Change course now while budget remains (a different, cheaper "
                "method-class — not more knobs on the same one).\n"
                "  - If your best result so far is not yet at its delivery path, "
                "write it there once NOW — before anything else.\n"
                "  - Name any doubt you already hold about your current best "
                "result, so it can be settled or recorded before it becomes the "
                "final answer."
            )
        else:  # 25
            tail = (
                " You are a quarter of the way in — the RIGHT moment to set your pace, "
                "while the decision window is still open. Re-check your plan against "
                "the time left: front-load the expensive steps (long downloads, "
                "builds, trainings) and START the longest/riskiest one NOW rather than "
                "later; run long operations in the background (background=true) and do "
                "OTHER real work while they run — never block idle on a long job. "
                "Order independent expensive steps to overlap instead of queueing them "
                "serially. Getting this pacing right now is far cheaper than "
                "discovering at 75% that a slow step should have started at 25%."
            )
        return head + tail
