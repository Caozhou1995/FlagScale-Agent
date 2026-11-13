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

import re

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
  PARSING RULE — labels start each slot; W3 runs to the next labelled line.
  Accepted: `cmd → output` on one line, OR cmd on the W3 line, output next line.

  W4 TRAP SHAPES — five shapes recur across unrelated tasks; a step can be green
  on W0-W3 and still miss one. For EACH, name YOUR instance on THIS deliverable
  or write "open bet" and why you cannot exclude it: (1) the check judges your
  own artifact — a generator agrees with itself; (2) the last format item — one
  char, one exit token, one header, one exact hash; (3) one clean sample
  standing in for the population — name the boundary inputs you did NOT run;
  (4) the fix landed in a scratch copy, memory or plan note, not the DELIVERED
  bytes — re-read the file, not your intent; (5) the visible example fixes FORM
  but not scale, offset, unit or index base. A trap never enumerated is a trap
  you are standing in.

  RISK SWEEP — per risk logged during this step: <risk> → <the evidence that
  decided it> → <which side won>. A logged-but-unresolved risk blocks this
  declaration: settle it now or name the probe you refuse to run and why.

Re-issue the same declaration WITH these lines and continue.
"""

_REMIND_MESSAGE_MISSING = """
[Remind] The re-issued declaration still does not carry the review slots: {missing}.
This is a STRUCTURAL check, not a judgement of your answers — the harness does
not read whether they are right. It only requires each slot to be addressed in
this declaration's own `verification` / `notes` text.

Fill them in VERBATIM (the literal labels, in this order):
  W0 UNVERIFIED LAYER — ...
  W1 FAILURE RECEIPT — ...
  W2 TRAP INVENTORY WALK — ...
  W3 FRESH PROBE — <command> → <its real output>; bar=<...>; VERDICT=PASS|FAIL
  W4 TRAP SHAPES — ...
  RISK SWEEP — ...

W3 needs EITHER an arrow (`command → output`) or the output on the NEXT line of
the same W3 slot (the slot ends at the next line starting with a label). After a
few re-issues the declaration is released anyway and the gap is recorded, since
re-issuing the same text again cannot change the outcome.

`_override_reason` does NOT release this gate — only filling the slots does.
"""


_REMIND_MESSAGE_RELEASED = """
[Remind] Structural release: this declaration re-issued the review slots {n} times
without matching the accepted shape — {missing}. It is LET THROUGH now, because
refusing it again cannot improve the text (the override channel is closed by
design, so an unshaped re-issue had no way forward). Record the gap in your plan
note: the step is completed with an INCOMPLETE self-review, and the missing slot
is what a later reader should treat as unverified.
"""


class _ReviewCoverage:
    """Result of the STRUCTURAL review check (never a semantic judgement)."""

    __slots__ = ("missing", "has_w3_two_slots")

    def __init__(self, missing, has_w3_two_slots):
        self.missing = list(missing)
        self.has_w3_two_slots = has_w3_two_slots

    def __bool__(self):
        # Truthy only when every label is present. The W3 two-slot check is
        # reported separately because it is a FORM requirement on one label,
        # not a seventh label.
        return not self.missing

    def __len__(self):
        return len(self.missing)


# The six literal labels the beat's message asks for. Matching is
# case-insensitive and word-bounded: "w3" inside a word does not count, so a
# stray "sw3" cannot fake coverage.
_REVIEW_LABELS = ("W0", "W1", "W2", "W3", "W4", "RISK")


def _review_labels_present(args: dict) -> _ReviewCoverage:
    """Do this declaration's own evidence/notes carry the six review labels?

    STRUCTURE ONLY. This never reads the content of the answers, never scores
    them, and never decides whether the review was any good — a filled-in wrong
    answer passes and an empty right one does not. That boundary is deliberate:
    the guard's job is to make the slots exist, not to grade them.

    Scanned fields: `verification` (the evidence array for step_done) and
    `notes`. Both are the agent's own text on the declaration being made, so
    the check is anchored to the same object the block asked the agent to fill.
    """
    parts = []
    ver = args.get("verification")
    if isinstance(ver, (list, tuple)):
        parts.extend(str(v) for v in ver)
    elif isinstance(ver, str):
        parts.append(ver)
    notes = args.get("notes")
    if isinstance(notes, str):
        parts.append(notes)
    blob = "\n".join(parts)

    missing = []
    for label in _REVIEW_LABELS:
        if not re.search(r"(?<![A-Za-z0-9_])" + label + r"(?![A-Za-z0-9_])", blob, re.I):
            missing.append(label)

    # W3 has its own FORM requirement: the message asks for
    # "<command> → <its real output>", i.e. two slots. A W3 label with no
    # reading means the agent wrote the heading and skipped the probe. This is
    # still structural — it checks the two slots EXIST, not what they say.
    #
    # SCOPE = the W3 SPAN, not one line. Measured (job 2026-11-10__10-44-55,
    # 22 trials, 102 beat blocks): the previous form looked for the arrow on
    # the FIRST line carrying the W3 label only, so the natural two-line write
    #     W3 FRESH PROBE — python3 -m pytest tests/x.py -q
    #     4 passed in 0.61s; bar=all green; VERDICT=PASS
    # was refused (probe: `_review_labels_present` -> False, missing=
    # ["W3(need ...)"]; the same text joined onto one line -> True). The model
    # then had to guess the accepted shape — its own words in that batch:
    # "Maybe the backticks are the issue", "Let me try with plain ASCII arrow",
    # i.e. four re-issues spent guessing a FORMAT while the reading was already
    # there. So the span runs from the first W3-bearing line up to the next
    # line carrying ANY label (W0/W1/W2/W3/W4/RISK).
    #
    # BOUNDARY (kept deliberately): the span does NOT extend past the next
    # label, so RISK SWEEP's "risk → evidence → side" arrows — which every
    # filled review has — still cannot satisfy W3 by accident, and a duplicated
    # W3 whose SECOND copy carries the arrow is still refused. Both are pinned
    # by tests; widening to the whole blob turns them red.
    span_re = re.compile(
        r"^[\s>*_#-]*(?:" + "|".join(_REVIEW_LABELS) + r")(?![A-Za-z0-9_])",
        re.I,
    )
    w3_head = re.compile(r"^[\s>*_#-]*W3(?![A-Za-z0-9_])", re.I)
    w3_span: list[str] = []
    for line in blob.splitlines():
        if not w3_span:
            if w3_head.search(line):
                w3_span.append(line)
            continue
        if span_re.search(line):   # next label HEAD line closes the slot
            break
        w3_span.append(line)
    if not w3_span:
        # FALLBACK — no line STARTS with W3, yet a label exists somewhere. The
        # shape that produces this is a declaration with every slot crammed onto
        # one line ("W0 ... W1 ... W2 ... W3 cmd -> out ... RISK ..."), where
        # there is no line boundary to segment on. Then the whole line carrying
        # W3 IS the slot (the pre-revision behaviour for a one-line write), so
        # take it verbatim and stop — no continuation is possible.
        for line in blob.splitlines():
            if re.search(r"(?<![A-Za-z0-9_])W3(?![A-Za-z0-9_])", line, re.I):
                w3_span.append(line)
                break

    span_text = "\n".join(w3_span)
    # FORM A: the arrow separator the message asks for, anywhere inside the span.
    has_arrow = bool(re.search(r"(?:→|->|=>)", span_text))
    # FORM B: the same information split over two LINES with no arrow written —
    # command on the W3 line, its reading on a following line. This is the form
    # that actually appeared in the trajectories above. Accepted only when the
    # span has a second non-empty line carrying a DIGIT (a reading is a number,
    # per the slot's own wording), so a bare heading, a prose sentence, or a
    # one-word "n/a" still fails.
    rest = w3_span[1:]
    has_reading_line = any(
        line.strip() and re.search(r"\d", line) for line in rest
    )
    w3_two_slots = has_arrow or has_reading_line
    if "W3" not in missing and not w3_two_slots:
        missing.append("W3(need '<command> → <output>')")

    return _ReviewCoverage(missing, w3_two_slots)


class SystemPromptRemindGuard(Guard):
    """Block a completion declaration until the self-review is STRUCTURALLY present.

    Two changes over the original "block once, then let anything through" form:

    1. NON-OVERRIDABLE. Measured (job 2026-11-10__00-11-22): of the 72 trials
       the beat fired on, 72 (100%) carried `OVERRIDE REQUIRED: Add
       "_override_reason"` and released the block that way — the beat averaged
       3.2 fires per trial (227 total) while no review was ever produced. An
       overridable block is an advisory, not a gate.

    2. STRUCTURAL SELF-CHECK, NOT A SEMANTIC ONE. The re-issued declaration is
       accepted only when its own evidence/notes text carries the six literal
       labels W0/W1/W2/W3/W4/RISK. The guard does NOT judge whether an answer is
       good, true, or complete — it only checks that the slots were addressed,
       exactly as a form checks that fields are non-empty. A wrong-but-filled
       answer passes; an empty one does not.

    Everything else is untouched: every other tool call and every other
    plan_update is never blocked by this guard.
    """

    name = "system_prompt_remind"
    # Sits just after PlanUpdateGuard (50) so it sees the declaration before
    # the plan guard acts on it, and before the generic advisories (72/90).
    priority = 58

    def __init__(self) -> None:
        # Signatures already beaten once this session: one beat per step.
        self._beaten: set[str] = set()
        # Per-signature count of REFUSALS delivered (the first beat plus every
        # rejected re-issue). Measured (job 2026-11-10__10-44-55, 22 trials):
        # the beat fired 102 times across 16 trials, and 3 trials hit the
        # "re-issued still empty" path — a model that had filled the slots but
        # not in the exact shape the checker wants had NO way through: the
        # override channel is closed by design (72/72 escaped that way before),
        # so its only remaining move was `step_skip` past the step. A gate with
        # no exit is not a gate, it is a wall. So: after _MAX_REFUSALS the
        # declaration is let through and the missing slots are written into the
        # plan note, i.e. the review is recorded as INCOMPLETE rather than
        # silently dropped. 3 is deliberately > 1: the first two refusals still
        # teach (measured shape errors are a 1-2 round fix), and 0/1 would
        # re-open the 72/72 override escape from the other side.
        self._refusals: dict[str, int] = {}

    _MAX_REFUSALS = 3

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "plan_update":
            return None
        args = ctx.tool_args or {}
        if not _declares_completion(args):
            return None
        sig = _declaration_sig(args)
        covers = _review_labels_present(args)
        if sig in self._beaten:
            # Beaten once. It may now pass ONLY if the review slots are filled.
            # A second empty re-issue is still refused (this is what makes the
            # beat real instead of a one-shot that any prose clears).
            if not covers:
                seen = self._refusals.get(sig, 1) + 1
                self._refusals[sig] = seen
                if seen <= self._MAX_REFUSALS:
                    return GuardVerdict.block(
                        _REMIND_MESSAGE_MISSING.format(missing=", ".join(covers.missing)),
                        reason="system_prompt_remind_still_empty",
                        category="system_prompt_remind",
                        overridable=False,
                    )
                # Exit valve: the slot text is still not in the accepted shape,
                # but blocking again cannot improve it (the agent has already
                # re-issued this same declaration _MAX_REFUSALS times). Let it
                # through as an INJECTION so the gap is on the record instead
                # of being swallowed.
                return GuardVerdict.inject(
                    _REMIND_MESSAGE_RELEASED.format(
                        n=seen - 1, missing=", ".join(covers.missing)
                    ),
                    reason="system_prompt_remind_released_incomplete",
                    category="system_prompt_remind",
                )
            return None  # slots addressed — let it through
        self._beaten.add(sig)
        self._refusals[sig] = 1
        return GuardVerdict.block(
            _REMIND_MESSAGE,
            reason="system_prompt_remind_beat",
            category="system_prompt_remind",
            overridable=False,
        )

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None  # the beat is pre-side; nothing to inject afterwards

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Refuse every override: the beat is released by filling the slots.

        A prose reason cannot substitute for the structural review, because the
        whole failure this guard exists for is an agent that BELIEVES it
        reviewed. Measured: 72/72 trials supplied a 5+ char `_override_reason`
        and moved on. So the only release path is `_review_labels_present`.
        """
        return False


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
