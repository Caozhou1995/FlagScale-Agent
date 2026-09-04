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
  • Percentage thresholds, not absolute seconds. The same 50/75/90% ladder works
    for a 20-minute wall and a 1-hour wall without any per-task tuning.
  • Inject, never block. Time is a hard external constraint; blocking would only
    burn a round. The agent needs a nudge, not a gate.
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
    _THRESHOLDS = (90, 75, 50)

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
                "the same one. And immediately write-through the best valid result you "
                "have to the delivery path so a timeout cannot wipe it out."
            )
        else:  # 50
            tail = (
                " You are past the halfway mark. Re-check your plan against the time "
                "left: front-load the expensive steps, run long operations in the "
                "background (background=true) and do OTHER real work while they run — "
                "never block idle on a long job. If a step is overrunning its share, "
                "shrink it or pick a faster method now."
            )
        return head + tail
