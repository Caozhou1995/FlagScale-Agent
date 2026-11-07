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
the agent must then do its scientific self-review — W1 a failure receipt (the
failure this moment invites: a correct check left unrun), W2 an item-by-item
TRAP INVENTORY walk over this step's deliverable, W3 one fresh probe never
run in this step with its real output, chosen by which case the task exposes
(grader file, worked example, or prose only), and a RISK SWEEP of logged-but-
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
Re-issue the SAME plan_update WITH the evidence lines below; the re-issue
passes when it is re-issued. Answers take the bracketed form — a filled slot,
not a paragraph.

  W0 UNVERIFIED LAYER — <the one layer of this deliverable you have NOT checked,
  and the read you skipped>; or "all layers covered" plus the check that covered
  each one.

  W1 FAILURE RECEIPT — <the failure this moment invites: a correct check you left
  unrun>. Treat your first pass as settled-too-early until a probe says
  otherwise.

  W2 TRAP INVENTORY WALK — one line per layer, in this order: ENUMERATE /
  PARSE-SCHEMA / RANK / RUN-TIME. Each: <what it produced for THIS deliverable,
  or N/A and why it cannot bite>.

  W3 FRESH PROBE — one command never run in this step, and it must carry a
  reading, not a ritual:
  <command> → <its real output: the number it printed on YOUR artifact>;
  bar=<the limit the task or a grader in the filesystem states>;
  VERDICT=PASS|FAIL. A number on the wrong side of its bar is a verdict, not a
  caveat: repair it, or declare the goal unreachable and give the reading that
  shows why.

  RISK SWEEP — per risk logged during this step: <risk> → <the evidence that
  decided it> → <which side won>. A logged-but-unresolved risk blocks this
  declaration: settle it now or name the probe you refuse to run and why.

Re-issue the same declaration WITH these lines and continue.
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
