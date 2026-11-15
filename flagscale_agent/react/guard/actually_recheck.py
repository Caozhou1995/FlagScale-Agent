# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License near the top of this file.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ActuallyRecheckGuard — post-inject when reasoning says "Actually, ...".

Anti-pattern (requested by the user, backed by measured traces): a reasoning
block contains the word "Actually", which almost always marks the agent
abandoning or flipping a conclusion it had just reached. Some of those turns
are genuine corrections; some are the agent talking itself OUT of a correct
conclusion and into a wrong one. In the dna-assembly traces the exact turning
points where a correct sub-requirement was dropped read as e.g.

    "Actually, the task doesn't require it. I'll keep it simple."

Those are one-way doors taken on reasoning alone, with no new evidence.

Design:
- post-only + inject-only: fires AFTER a tool round; never blocks and never
  asks the agent to change its answer.
- DETECT the bare token "actual" case-insensitively? No — the literal word
  "actually" in the reasoning of the current iteration. This is a LITERAL
  string test on purpose: no regex and no semantic classification of whether
  the flip was right or wrong. Deciding that is the agent's job, not the
  harness's; a harness that pretends to know which conclusion is correct is
  both wrong and unwelcome in its context.
- dedup per LLM iteration via len(ctx.messages) — the same mechanism and the
  same rationale used by SleepSpacingGuard/PollSpacingGuard: check_post runs
  once per tool call, but a single assistant message may emit several calls,
  so without the marker a burst would inject several copies for one thought.
- cost is therefore bounded by (number of iterations whose reasoning contains
  the word), never by token count or tool-call count.
- the message is generic and non-directive: it asks for the evidence behind
  the flip and invites keeping the earlier conclusion if no new evidence
  appeared. It does NOT name a subject, a fix, or which side to prefer.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

# The literal token to look for, matched case-insensitively.
TRIGGER_WORD = "actually"

_ADVISORY = (
    "[ActuallyRecheck] Your reasoning this iteration contains the word "
    "\"Actually\" — a point where you abandoned or reversed a conclusion you "
    "had just reached. Sometimes that is a real correction; sometimes it is "
    "talking yourself out of a correct reading with no new evidence, and the "
    "flip becomes a one-way door. Before you act on the reversal, check it "
    "the way you would check any other claim:\n"
    "  · Did NEW evidence arrive (a command output, a test result, a file "
    "you actually read), or is this only a new line of reasoning?\n"
    "  · If there is no new evidence, the honest position is that BOTH "
    "readings are still open — say which one you are choosing and on what "
    "grounds, rather than presenting the flip as settled.\n"
    "  · If the reversed conclusion was supported by something you observed "
    "earlier, re-read that observation before discarding it.\n"
    "You decide which conclusion is right — this note only asks that the "
    "reversal rest on evidence you can point to."
)


class ActuallyRecheckGuard(Guard):
    """Post-tool guard: one generic re-check nudge per Reasoning flip."""

    name = "actually_recheck"
    priority = 72  # advisory tier, beside SleepSpacingGuard (71)

    def __init__(self):
        # Last len(ctx.messages) we injected for; guards re-run per tool call.
        self._last_marker: int | None = None

    def reset_turn(self):
        self._last_marker = None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None  # post-only

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        thinking = ctx.thinking_text or ""
        if not thinking:
            # No reasoning published this iteration (provider emitted none, or
            # this is a pre/turn-start context). Nothing to inspect.
            return None
        if TRIGGER_WORD not in thinking.lower():
            return None
        marker = len(ctx.messages)
        if marker == self._last_marker:
            # Already nudged for this iteration's reasoning.
            return None
        self._last_marker = marker
        return GuardVerdict.inject(
            _ADVISORY,
            reason="reasoning_reversal",
            category="actually_recheck",
        )
