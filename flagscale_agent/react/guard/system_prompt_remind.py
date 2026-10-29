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

"""SystemPromptRemindGuard — the step_done self-interrogation beat.

Fires when the agent DECLARES a step complete (plan_update step_done, and any
batch row that declares one). That is the single moment where self-confirmation
is most likely and least checked: the agent has just convinced ITSELF, the
acceptance form was just filled in, and the next action feels like progress.

The guard injects a short POINTER list back to the doctrine already in the
system prompt — it deliberately does not restate the doctrine. Restating it
would (a) burn context every single step, and (b) let the agent satisfy the
reminder by reading a paraphrase instead of re-reading the rule. Four
questions, no content: did you run the protocol, did you look at the delivered
bytes, is your recorded observation literally verbatim, and are you about to
cross one of your own negative constraints.

No cooldown by design: the beat is per declaration, not per session. An agent
that declares ten steps gets ten beats, because a stale "I checked last time"
is exactly the rationalisation this exists to interrupt.
"""

from __future__ import annotations

from . import Guard, GuardContext, GuardVerdict

_REMIND_MESSAGE = """
[SystemPromptRemind] You just declared a step complete. Before the next action,
answer these four to yourself — in one line each, not in prose:
  1. Did you run the four-step protocol (S1 locate the producer -> S2 enumerate
     constructions with numbers -> S3 merge on arithmetic -> S4 enumerate once
     more after green) for the quantity this step decided? Which step did you skip?
  2. Have you READ the actual bytes at the delivery path since your last write —
     or are you reporting your model of them? Open the file, measure it, quote it.
  3. Is the `observed` you recorded a VERBATIM substring of a real tool output
     from this session, or a paraphrase/summary of what you believe happened?
  4. Does your next intended action cross one of your own negative constraints?
     Re-read the forbidden list before acting, not after being blocked.
If any answer is "no" or "I think so", the step is not done — fix it now, while
the fix still costs one edit instead of one run.
"""


class SystemPromptRemindGuard(Guard):
    """Inject-only self-check beat on every completion declaration."""

    name = "system_prompt_remind"
    # Sits just after PlanUpdateGuard (50) so it observes the declaration the
    # plan guard just acted on, and before the generic advisories (72/90).
    priority = 58

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None  # Inject-only

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "plan_update":
            return None
        args = ctx.tool_args or {}
        if not _declares_completion(args):
            return None
        result = (ctx.tool_result or "").lstrip()
        # Only when the declaration actually LANDED: a blocked or errored
        # plan_update changed no state, so there is no completion to interrogate.
        if not result or result.startswith("ERROR") or result.startswith(
            "[BLOCKED BY GUARD]"
        ):
            return None
        return GuardVerdict.inject(
            _REMIND_MESSAGE,
            reason="system_prompt_remind_beat",
            category="system_prompt_remind",
        )


def _declares_completion(args: dict) -> bool:
    """True when this plan_update declares at least one step DONE.

    Three shapes carry a declaration: the single-step `action=step_done`, an
    `action=batch` whose rows include `status: done`, and `action=complete`
    (the whole plan claimed finished — the strongest declaration of all).
    """
    action = str(args.get("action") or "").strip().lower()
    if action in ("step_done", "complete"):
        return True
    if action != "batch":
        return False
    for row in args.get("updates") or ():
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").strip().lower() == "done":
            return True
    return False
