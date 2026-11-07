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
passes when it is re-issued — the lines exist to force the review, not to be
graded by the harness.

  W0 UNVERIFIED LAYER. Name the ONE layer of this deliverable you have NOT
  checked yet — the consumer's own parse/interface contract, the boundary you
  assumed, the branch you never ran. If your last rounds re-checked layers you
  already proved green, you spent them in the wrong place: go verify the
  unproven layer FIRST, then re-issue. Answering "nothing unverified" requires
  naming every layer and the check that covered it.

  W1 FAILURE RECEIPT. This is the moment self-confirmation is strongest and
  external checking weakest — that asymmetry, not any prediction about you, is
  why this beat exists. So the failure worth guarding here is less often a
  wrong answer than a correct check left unrun. Treat your first pass as
  settled-too-early, not as wrong, until one more probe says otherwise.

  W2 TRAP INVENTORY WALK. Walk the Section-5 TRAP INVENTORY LAYER by LAYER
  against THIS step's deliverable — ENUMERATE, PARSE-SCHEMA, RANK, RUN-TIME,
  one line each, not one line per trap you happen to think of. For each layer
  write: what this layer produced for THIS deliverable, or N/A and why it
  cannot bite here. PARSE-SCHEMA is its own layer and is asked FIRST: name the
  prefix / marker / payload / tail a downstream parser slices, and the minimum
  it asserts that the task text never states. A walk whose list is
  word-for-word identical to your previous one proves no new layer was found —
  that repetition is itself the signal to go look at the layer you skipped.

  A FRESH PROBE THAT CARRIES A READING, NOT A RITUAL. Paste ONE command you
  have NOT run in this step, with its real output, reading the delivered bytes,
  and quote the number it printed on your OWN artifact saying whether that
  number clears the bar the task states. What you can read the requirement OFF
  differs by task, so name which case you are in before choosing the probe: a
  grader present in the filesystem — read it and treat each assertion as a
  clause, as a LOWER BOUND (it never asserts what it forgot); else an example
  carrying its expected output — push it through and diff, but it fixes FORM
  for the case it shows only, so author the cases it omits; else only the task
  text (or only a vague goal) — then the probe is the assertion you derived
  from that text, or a second independent path whose numbers you diff against
  the first, both run on your own artifact with the same instrument. If no new
  command is runnable, name the command you tried and the exact reason it could
  not run; confidence without a fresh probe is exactly what this beat blocks.
  Knowing a miss is not a caveat to attach to a finished step; it is a verdict:
  repair it, or declare the goal unreachable and give the reading that shows
  why. Two generic cases: a quantity legitimately expressible in two units,
  index bases or orientations — computing both and shipping one unranked is a
  coin flip, so rank them and say what decided it; and a self-measured score
  just short of a stated floor — labelling it "slightly below" does not clear it.

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
