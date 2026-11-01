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
mechanical acceptance form was just filled in, and the next action feels like
progress.

The guard BLOCKS the declaration exactly ONCE per step (a beat, not a barrier):
the agent must then do its scientific self-review — did I run the protocol,
did I read the delivered bytes, is my recorded observation verbatim, does my
next action cross a negative constraint — and re-issue the SAME plan_update.
The re-issued declaration is not blocked again: the review is the agent's own
judgment, the harness does not grade it. The harness side does exactly one
thing — one block per step, nothing else — and never blocks any other agent
behaviour.
"""

from __future__ import annotations

from . import Guard, GuardContext, GuardVerdict

_REMIND_MESSAGE = """
[Remind] This step-done declaration is blocked ONCE — the beat, not a refusal.
Do your scientific self-review NOW, then re-issue the SAME plan_update (it will
pass). Answer to yourself, one line each:
  1. OBSERVED vs INFERRED — separate what you actually SAW printed (a command's
     real output) from what you reasoned to. Label each claim you wrote; never
     present an inference in observed clothing.
  2. COUNTERFACTUAL — every "X causes Y / X would fix it" claim you recorded is
     UNVERIFIED until you run the smallest probe that toggles X and read the
     result. Name the probe you ran, or downgrade the claim to a hypothesis.
  3. VERIFIER MUST BE ABLE TO FAIL — if this step trusted a self-built checker,
     has it ever gone red on a seeded defect? If not, corrupt one thing it
     claims to check and confirm it flags that; a checker that cannot fail
     checks nothing.
  4. STRIKE COUNT — is this attempt the same shape as a failure you already had?
     Two same-shape strikes: stop and rebuild the model (state what observation
     would change your prediction) instead of trying a third variant.
  5. TRAP — does your next intended action cross one of your own negative
     constraints? Re-read the forbidden list before acting, not after being
     blocked.
The review is yours; the harness does not grade it. It only guarantees the beat
happened. Re-issue the same declaration and continue.
"""


class SystemPromptRemindGuard(Guard):
    """Block a completion declaration ONCE per step; never block anything else.

    First submission of a step_done / batch-with-done / complete is blocked with
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


def _declaration_sig(args: dict) -> str:
    """Identity of the declaration — one beat per step, not one per retry.

    step_done: the step number is the identity. batch: the SET of done row ids
    is the identity (a re-issued batch with the same rows is the same beat;
    new rows added later are a different beat). complete: one beat for the
    whole plan, keyed on the action alone. Fields the agent may legitimately
    change between the block and the re-issue — the evidence text, notes, and
    any other prose — do NOT change the signature: the re-issued
    declaration must pass without a second block.
    """
    action = str(args.get("action") or "").strip().lower()
    if action in ("step_done", "complete"):
        return f"{action}:{args.get('step_id')}" if action == "step_done" else action
    if action == "batch":
        ids = []
        for i, r in enumerate(args.get("updates") or []):
            if not isinstance(r, dict):
                continue
            if str(r.get("status") or "").strip().lower() != "done":
                continue
            sid = r.get("step_id")
            # A done row with no usable id must still be distinct: fall back
            # to its position so two id-less batches cannot share one beat.
            ids.append(str(sid) if sid is not None else f"idx{i}")
        return "batch:" + ",".join(sorted(ids))
    return action
