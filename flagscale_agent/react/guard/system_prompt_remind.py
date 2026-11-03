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

Fires when the agent DECLARES a step complete (plan_update step_done or
complete). That is the single moment where self-confirmation
is most likely and least checked: the agent has just convinced ITSELF, the
mechanical acceptance form was just filled in, and the next action feels like
progress.

The guard BLOCKS the declaration exactly ONCE per step (a beat, not a barrier):
the agent must then do its scientific self-review — W1 a failure receipt
(you have settled-too-early at this kind of step before), W2 an item-by-item
TRAP INVENTORY walk over this step's deliverable, W3 one fresh probe never
run in this step with its real output, and a RISK SWEEP of logged-but-
unresolved risks — and re-issue the SAME plan_update.
The re-issued declaration is not blocked again: the review is the agent's own
judgment, the harness does not grade it. The harness side does exactly one
thing — one block per step, nothing else — and never blocks any other agent
behaviour.
"""

from __future__ import annotations

from . import Guard, GuardContext, GuardVerdict

_REMIND_MESSAGE = """
[Remind] This step-done declaration is blocked ONCE — the beat, not a refusal.
Re-issue the SAME plan_update WITH the evidence lines below — the re-issue
passes when they are present. The grader is INVISIBLE to you (if you could
read it you would just copy it) — so these lines are DERIVED, from the task
text and environment signals, never from a grader file.

  W0 UNVERIFIED LAYER. Name the ONE layer of this deliverable you have NOT
  checked yet — the consumer's own parse/interface contract, the boundary you
  assumed, the branch you never ran. If your last rounds re-checked layers you
  already proved green, you spent them in the wrong place: go verify the
  unproven layer FIRST, then re-issue. Answering "nothing unverified" requires
  naming every layer and the check that covered it.

  W1 FAILURE RECEIPT. You have failed at this kind of step — repeatedly. The
  record's pattern is the same: the correct check was reachable and never
  run. Your first pass is treated as settled-too-early, NOT as wrong — the
  right answer was usually within reach of one more probe.

  W2 TRAP INVENTORY WALK. Walk the Section-5 TRAP INVENTORY item by item
  against THIS step's deliverable and choices. For each item write one line:
  APPLIES -> what the choice is and how you checked it, or N/A -> why it
  cannot bite here. Enumerate FIRST, judge later — the missed item is the
  failure mode this beat exists to catch.

  W3 FRESH PROBE. Paste ONE command you have NOT run in this step, with its
  real output, reading the delivered bytes. If you cannot produce a fresh
  probe, say what you tried to run and why nothing new was runnable —
  confidence without a fresh probe is exactly what this beat blocks.

  RISK SWEEP: list every risk you LOGGED during this step and left
  unresolved (the suspended "Hmm." moments — a risk named but never settled).
  For each, write how it was resolved: the evidence that decided it and which
  side won. A logged-but-unresolved risk blocks this declaration — settle it
  now or name the probe you refuse to run and why.

The review is yours; the harness does not grade it. It only guarantees the beat
happened. Re-issue the same declaration WITH the evidence lines and continue.
"""


class SystemPromptRemindGuard(Guard):
    """Block a completion declaration ONCE per step; never block anything else.

    First submission of a step_done / complete declaration is blocked with
    the self-review pointer. The agent does its scientific self-review and
    re-issues the SAME declaration; the re-issue passes (the review is the
    agent's own judgment — the harness does not grade it, it only guarantees
    the beat happened). Every other tool call and every other plan_update is
    untouched.
    """

    name = "system_prompt_remind"
    # Sits just after PlanUpdateGuard (50) so it sees the declaration before
    # the plan guard acts on it, and before the generic advisories (72/90).
    priority = 58

    def __init__(self) -> None:
        # Signatures already beaten once this session: one beat per step.
        self._beaten: set[str] = set()

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "plan_update":
            return None
        args = ctx.tool_args or {}
        if not _declares_completion(args):
            return None
        sig = _declaration_sig(args)
        if sig in self._beaten:
            return None  # already beaten once this step — let it through
        self._beaten.add(sig)
        return GuardVerdict.block(
            _REMIND_MESSAGE,
            reason="system_prompt_remind_beat",
            category="system_prompt_remind",
        )

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None  # the beat is pre-side; nothing to inject afterwards


def _declares_completion(args: dict) -> bool:
    """True when this plan_update declares at least one step DONE.

    Two shapes carry a declaration: the single-step `action=step_done`,
    `action=complete` (the whole plan claimed finished — the strongest
    declaration of all), and the `batch` form carrying any row with
    status DONE (defense in depth: the tool dropped `batch`, but a
    hand-rolled caller can still route one through).
    """
    action = str(args.get("action") or "").strip().lower()
    if action == "batch":
        updates = args.get("updates") or []
        return any(
            isinstance(u, dict) and str(u.get("status") or "").strip().lower() == "done"
            for u in updates
        )
    return action in ("step_done", "complete")


def _declaration_sig(args: dict) -> str:
    """Identity of the declaration — one beat per step, not one per retry.

    step_done: the step number is the identity. batch: the set of DONE row
    ids (any change to which rows are declared done is a NEW declaration);
    rows not carrying DONE — notes-only edits — never key it. complete: one
    beat for the whole plan, keyed on the action alone. Fields the agent
    may legitimately change between the block and the re-issue — the
    evidence text, notes, and any other prose — do NOT change the
    signature: the re-issued declaration must pass without a second block.
    """
    action = str(args.get("action") or "").strip().lower()
    if action == "step_done":
        return f"{action}:{args.get('step_id')}"
    if action == "batch":
        ids = sorted(
            str(u.get("step_id"))
            for u in (args.get("updates") or [])
            if isinstance(u, dict)
            and str(u.get("status") or "").strip().lower() == "done"
        )
        return f"batch:{','.join(ids)}"
    return action
