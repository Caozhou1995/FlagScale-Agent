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

"""PollSpacingGuard — post-inject on back-to-back no-progress poll rounds.

Anti-pattern (observed 15+ rounds in one session): the agent polls a running
worker/dispatch, gets "not ready", and immediately polls again with NOTHING in
between. Each round is an LLM iteration (~74-84s startup + a full-size prompt)
that re-reads the same 43 tokens. A consecutive empty-poll streak means the
agent is blocked wearing a costume: it should spend the wait on real work
(prep the post-join steps, update the plan, write memory, audit premises) or
use ONE bounded sleep before the next look — not burn iterations on bare
polling.

Design:
- post-only + inject-only: fires AFTER a poll_tasks/dispatch_many round that
  ended not-ready; never blocks. The advisory rides along with the tool result
  (append_advisory), so it degrades with it and does not pollute history.
- streak counter, persisted in the guard instance across rounds, advanced in
  check_post ONLY for actually-executed poll rounds that ended not-ready
  (a blocked call never advances — mirrors KnowledgeSkillGuard).
- the message ESCALATES with the streak: 2+ = bounded-wait suggestion, 3+ =
  hard "your last rounds had zero information gain" nudge. Reset when a poll
  actually settles (not ready no longer true / dispatch complete) or any
  non-poll tool intervenes.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

# Rounds that must pass before the nudge starts (the FIRST wait is legitimate
# — a worker started seconds ago needs time; nudging on round 1 is noise).
INJECT_THRESHOLD = 2

_NOT_READY_MARKS = ("not ready", "state=running")


def _poll_round_not_ready(tool_name: str, tool_result: str | None) -> bool:
    """True when THIS executed tool call was a poll round that made no progress."""
    if tool_name not in ("poll_tasks", "dispatch_many"):
        return False
    text = (tool_result or "").strip()
    if not text:
        return False
    # Both the not-ready marker and the settled verdicts are judged on the
    # FIRST line only. Real formats put them there (reunite.py returns
    # "task <id>: not ready ..." / "...: PASSED|FAILED ..."; dispatch.py emits
    # "dispatch <id>: state=running|complete ..."), while LATER lines are
    # free-form payload: a settled-FAILED round's stdout_tail routinely
    # contains "not ready" / "state=running" (service probes, log dumps) and
    # a whole-text scan would misclassify that settled round as not-ready.
    first = text.splitlines()[0].strip()
    if first.startswith(("PASSED", "FAILED", "state=complete")):
        return False
    if first.startswith("dispatch ") and "state=complete" in first:
        return False
    # dispatch_many START (action='dispatch') output: not a poll round at all.
    if first.startswith("dispatched ") or first.startswith("dispatch_id:"):
        return False
    return any(mark in first for mark in _NOT_READY_MARKS)


class PollSpacingGuard(Guard):
    """Post-tool guard: nudge when consecutive poll rounds return not-ready."""

    name = "poll_spacing"
    priority = 70  # Advisory tier — never blocks

    def __init__(self):
        self._streak = 0
        self._last_marker: int | None = None

    def reset_turn(self):
        # A fresh user message ends any wait context; a new instruction may
        # legitimately start with one poll round.
        self._streak = 0
        self._last_marker = None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None  # post-only

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        name = ctx.tool_name or ""
        if name in ("poll_tasks", "dispatch_many"):
            if _poll_round_not_ready(name, ctx.tool_result):
                # The streak measures LLM ITERATIONS spent waiting, but
                # check_post runs once PER TOOL CALL and one assistant message
                # may emit several poll calls in parallel. Within one iteration
                # the history length is unchanged (tool results are appended
                # AFTER the per-call guard loop); it always grows between
                # iterations. Use that as the iteration marker: advance at most
                # once per distinct history length.
                marker = len(ctx.messages)
                if marker != self._last_marker:
                    self._streak += 1
                    self._last_marker = marker
                if self._streak >= INJECT_THRESHOLD:
                    return self._advisory(self._streak)
                return None
            # Settled poll round (PASSED/FAILED/state=complete) or a blocked
            # call (tool_result carries the [BLOCKED BY GUARD] marker and
            # matches neither branch) — the wait context is over.
            self._streak = 0
            self._last_marker = None
            return None
        # Any intervening non-poll tool ends the wait context: the agent went
        # and did something else, which is exactly what the nudge asks for.
        self._streak = 0
        self._last_marker = None
        return None

    def _advisory(self, streak: int) -> GuardVerdict:
        if streak >= 3:
            body = (
                f"[PollSpacing] {streak} consecutive not-ready polls with nothing "
                "in between — each round re-read the same output at full iteration "
                "cost. You are spinning. STOP polling: do real work now (prep the "
                "post-join steps, update the plan, write memory, audit premises) "
                "and only then look again; if there is genuinely nothing to do, "
                "take ONE bounded `sleep <=30s` between looks instead of bare "
                "back-to-back polls."
            )
        else:
            body = (
                f"[PollSpacing] {streak} not-ready polls in a row. Don't chain bare "
                "polls — each costs a full LLM iteration. Between looks, do real "
                "work (prep next steps, update plan/memory, audit the output "
                "contract) or take ONE bounded `sleep <=30s`."
            )
        return GuardVerdict.inject(
            body,
            reason="consecutive_not_ready_polls",
            category="poll_spacing",
        )
