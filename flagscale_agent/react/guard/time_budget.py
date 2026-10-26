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
  • Percentage thresholds, not absolute seconds. The same 25/50/65/75/80/90/100% ladder
    works for a 20-minute wall and a 1-hour wall without any per-task tuning.
  • Two modes on the ladder:
      - 25/50/75%   → check_post inject (pacing / health-check advisories).
      - 65%         → check_post inject (pre-completion falsification: prove
                      you are DONE before declaring it; see first_e2e_run.py).
      - 80%         → check_post inject (early WRAP-UP WINDOW — protective actions only).
      - 90%         → check_pre BLOCK (overridable) — forces the agent to SHOW
                      (evidence, not claim) that a deliverable exists, that
                      THIS call produces it, or that a background job already in
                      flight is producing it, before spending a near-final tool
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
from flagscale_agent.react.guard.utils import READ_ONLY_TOOLS


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
    # severe one still pending. Each maps to its wrap-up/advisory message via _message().
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
    _THRESHOLDS = (100, 90, 80, 75, 65, 50, 25)

    # A turn dies with the budget, not with a percentage: what matters is how
    # many more ACTIONS fit. This is the number of remaining action-rounds the
    # critical gate reserves before it refuses to spend one on an observation.
    # 6 rounds is the measured minimum for a real wrap-up (read the delivery path,
    # write the crude answer, run it, record the result, report) at the observed
    # per-call latency of this harness.
    _CRITICAL_ROUNDS = 6

    # Fallback when the wall reports no per-call latency (a stats_fn that only
    # knows elapsed/budget): fall back to the fraction of budget at which the
    # measured runs still had those rounds left.
    _CRITICAL_PCT_FALLBACK = 85.0

    def __init__(self, stats_fn):
        """stats_fn() -> dict|None with keys elapsed/budget/remaining/pct.

        None means no external wall is enforced; the guard stays silent.
        """
        self._stats_fn = stats_fn
        self._fired: set[int] = set()
        # Thresholds crossed while building a block that the registry may NOT
        # surface (a harder block won the same resolve). Kept separately from
        # _fired so block_not_surfaced() can hand them back: a nag the agent never
        # saw has not been delivered, so spending it would silently disable the
        # ONLY hard convergence point of the turn. Observed failure: at 91.7% a
        # rank-1 VerificationGuard block outranked this one on every call, 90 was
        # marked fired anyway, and the agent ran 6m34s past the 80% wrap-up with
        # no 90% gate ever reaching it.
        self._pending_block_fired: set[int] = set()
        # Same protection for the INJECT channel: check_post consumes thresholds
        # into _fired when building an advisory, and that advisory can be
        # suppressed by a concurrent block in the same resolve (generic injects
        # are dropped when a block wins). Track what was consumed so
        # block_not_surfaced() can hand it back — otherwise a swallowed 50%/75%
        # nudge stays spent forever and the turn runs out unguarded.
        self._pending_inject_fired: set[int] = set()

    def reset_turn(self):
        # A new user message restarts the per-turn wall-clock accounting on the
        # agent side (self._turn_start is re-stamped), so clear which thresholds
        # we have already announced.
        self._fired = set()
        self._pending_block_fired = set()

    def block_not_surfaced(self):
        """Hand back thresholds consumed while building a verdict that never surfaced.

        Covers BOTH channels: check_pre blocks outranked by a harder block in the
        same resolve, and check_post injects suppressed by a concurrent block.
        Those thresholds were never delivered to the agent, so they must be
        re-armable — otherwise the 90% gate or the 50-80% nudges are dead for
        the remainder of the turn while the wall-clock keeps draining.
        """
        self._fired -= self._pending_block_fired
        self._pending_block_fired = set()
        self._fired -= self._pending_inject_fired
        self._pending_inject_fired = set()

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Block at 90%+ budget to force explicit deliverable reasoning.
        
        At the CRITICAL 90% threshold, switch from advisory (check_post inject) to
        blocking gate (check_pre block). The agent must SHOW, not merely claim:
          (1) A complete deliverable already exists at its required path — the
              override_reason must cite the output of a verification action just
              run (e.g. ls -la / cat on the delivery path), OR
          (2) This specific tool call will produce/finalize the deliverable —
              cite the delivery path it writes or this tool's concrete output, OR
          (3) A background job already in flight is producing/finalizing the
              deliverable — cite its job id and the delivery path it writes, then
              collect it. A job in flight counts: do not kill it to pass the gate.
        
        If none is true, the agent should STOP optimizing and write a crude-but-
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
        budget = float(stats.get("budget", 0.0) or 0.0)
        remaining = float(stats.get("remaining", 0.0) or 0.0)
        rounds = int(stats.get("rounds", 0) or 0)
        # When does the turn actually run out of ACTION rounds? A percentage is a
        # proxy that assumes a fixed cost per round; the measured harness varies
        # from 3s to 174s a call, and 10% of an 1800s wall is ~2 rounds — too few
        # to write, run and report a deliverable (observed: a task scored 1 with no
        # gate and 0 with it). Derive the trigger from ROUNDS: arm once the
        # remaining wall cannot cover _CRITICAL_ROUNDS more actions at THIS turn's
        # own measured cost per action (elapsed / rounds). No history yet (rounds==0)
        # -> fall back to the fraction of budget at which measured runs still had
        # those rounds left.
        if budget > 0 and rounds >= 3:
            per_round = float(stats.get("elapsed", 0.0) or 0.0) / rounds
            critical = per_round > 0 and remaining <= per_round * self._CRITICAL_ROUNDS
        else:
            critical = pct >= self._CRITICAL_PCT_FALLBACK
        # Block only once per turn (a checkpoint, not a repeated wall), and only
        # while rounds remain: at/after 100% the wall-clock is fully spent and the
        # guard must NOT block — the agent needs a clear path to emit its wrap-up
        # response (and NEED_USER_INPUT); that final rung is a check_post inject.
        # No percentage floor here: the round count IS the floor. A fixed pct gate
        # contradicts it (78% with 5.7 actions left is exactly the case the rounds
        # criterion is meant to catch, but a >=80 floor would stay silent).
        if critical and pct < 100 and 90 not in self._fired:
            # A gate whose only lever is "spend a round asking permission" must not
            # collect that round from the very call it is asking for. Measured
            # failure: at 91% with ~1m16s left (~2 calls) the gate blocked the
            # `write_file` that WAS the deliverable; the agent then spent both
            # remaining rounds satisfying the override prose and had no time left
            # to run it — a task that scored 1 without the gate scored 0 with it.
            # The gate's subject is UNPRODUCTIVE time use, so it may only stop an
            # OBSERVATION round (read_file / grep / plan_status / memory_*), never a
            # production round (write_file / edit_file / shell), which is the
            # wrap-up it exists to force. Classification is the harness's existing
            # read-only tool set — a stable property of the tool, not this subject.
            if ctx.tool_name not in READ_ONLY_TOOLS:
                return None
            self._pending_block_fired = {t for t in self._THRESHOLDS if pct >= t}
            self._fired |= self._pending_block_fired
            
            elapsed = _fmt(stats.get("elapsed", 0.0))
            msg_rounds = (
                f"~{int(remaining / per_round)} actions left at this turn's rate"
                if (budget > 0 and rounds >= 3 and per_round > 0) else "final actions"
            )
            remaining_s = _fmt(remaining)
            message = (
                f"[TimeBudget] CRITICAL CHECKPOINT — {pct:.0f}% of budget spent "
                f"({elapsed} used, ~{remaining_s} left, {msg_rounds}).\n\n"
                f"This gate stops OBSERVATION calls only (read_file, memory_read, "
                f"plan_status, web_fetch, ...). Every PRODUCTION call — "
                f"write_file, edit_file, shell — already passes, so nothing you "
                f"need in order to deliver is blocked.\n\n"
                f"Do this, in this order, with no other calls in between:\n"
                f"  1. If you do NOT know the exact delivery path, discover it with "
                f"ONE shell call (`ls -la <dir>`); shell is a production tool and "
                f"is not gated.\n"
                f"  2. Write the crude-but-complete deliverable to that path NOW "
                f"(write_file). Partial and correct beats complete and unwritten.\n"
                f"  3. Run it and report what it printed (shell). Its output IS "
                f"your report — do not re-derive it.\n"
                f"  4. If a background job of yours is already producing the "
                f"deliverable, collect it instead of duplicating the work: poll "
                f"the job, then read the file it wrote; do not kill or re-run it.\n\n"
                f"Do not read, re-plan, or verify anything you are not required to "
                f"verify; verifying a claim is NOT refining, and a re-check only "
                f"wins when a doubt would otherwise make the deliverable wrong.\n\n"
                f"Also memory_write() any fact that lives only here (survival-range "
                f"test: cross-session truth -> memory, session progress -> plan "
                f"notes).\n\n"
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
                # has fired. Consumed state is recorded in _pending_inject_fired
                # so block_not_surfaced() can return it if this advisory is
                # suppressed by a concurrent block.
                for t in self._THRESHOLDS:
                    if pct >= t:
                        self._fired.add(t)
                        self._pending_inject_fired.add(t)
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
                f"budget for this turn is spent ({elapsed} used). Final stretch: this "
                f"turn is ending. Your remaining slack buys PROTECTIVE actions only, in "
                f"priority order:\n"
                f"  1. Make sure a COMPLETE, valid result is written through to its "
                f"required delivery path RIGHT NOW — a crude-but-complete answer that "
                f"is banked beats a perfect one that never gets saved.\n"
                f"  2. VERIFY what is banked (`ls`/`cat` the delivery path — verify, "
                f"do not recompute). NEVER start anything that could overwrite a "
                f"banked deliverable: a fresh full run writing the delivery file "
                f"itself is NOT a wrap-up action, even if its 'payoff' looks better.\n"
                f"  3. If a doubt about the banked value stays unresolved, record it "
                f"explicitly next to the value (an audited degrade beats a silent one).\n"
                f"  4. Only then housekeeping (proposal/memory tidy-ups come LAST) and "
                f"close out. If finishing needs a human decision, more input, or an "
                f"action you cannot take, hand control back with NEED_USER_INPUT "
                f"stating exactly what you need and the current state — rather than "
                f"burning the overrun on more attempts.\n"
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
        elif thr >= 80:
            # Wrap-up starts BEFORE the wall: the harness hard-stop at <=0 remaining
            # pre-empts the 100% nudge, so 80% is the earliest point where an
            # unattended run can still afford protective actions after reading it.
            tail = (
                " WRAP-UP WINDOW — this turn ends soon and the harness hard-stop will "
                "not wait for the 100% nudge. Your remaining slack buys PROTECTIVE "
                "actions only, in this priority order:\n"
                "  1. Make sure a COMPLETE, valid result is written through to its "
                "required delivery path RIGHT NOW — a crude-but-complete answer that "
                "is banked beats a perfect one that never gets saved.\n"
                "  2. VERIFY what is banked (`ls`/`cat` the delivery path — verify, "
                "do not recompute). NEVER start anything that could overwrite a "
                "banked deliverable.\n"
                "  3. If a doubt about the banked value stays unresolved, record it "
                "explicitly next to the value — an audited degrade beats a silent "
                "one.\n"
                "  4. No new lines of work after this; close out cleanly."
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
        elif thr >= 65:
            tail = (
                " Two-thirds in. If you believe you are DONE or nearly done, PROVE it "
                "before declaring completion: re-run the verifier's own decisive check "
                "(or the closest equivalent you can construct from the task's stated "
                "requirements — expected file, exact value, exact format) against the "
                "artifact at its delivery path, and READ the result. Do not count a "
                "self-check as proof: an artifact agrees with its own generator. If you "
                "are NOT nearly done, spend this checkpoint re-planning the tail: name "
                "the one step that still blocks completion and give it the next "
                "consecutive block of time."
            )
        elif thr >= 50:
            tail = (
                " You are past the halfway mark — a HEALTH CHECK, not a pacing tip. "
                "Ask concretely: have the expensive steps (downloads, builds, "
                "trainings) actually STARTED, and are they on track to FINISH before "
                "the budget runs out? At the current rate, will a complete valid "
                "deliverable exist at its required path in time?\n"
                "Default actions if the honest answer is not a confident yes:\n"
                "  - FIRST, this step: write a SKELETON deliverable to its required "
                "path — a crude-but-complete result built from what you ALREADY "
                "know, no new analysis. Do not write 'I will write it soon'; the "
                "file on disk is the only thing that scores. Refine it in place "
                "afterwards — never rewrite from scratch.\n"
                "  - Change course now while budget remains (a different, cheaper "
                "method-class — not more knobs on the same one).\n"
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
