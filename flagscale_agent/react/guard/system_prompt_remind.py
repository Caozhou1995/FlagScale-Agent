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
[Remind] Blocked ONCE — a beat, not a refusal. Re-issue the SAME plan_update
WITH the lines below; it passes when re-issued. Answers: a slot, not prose.

  W0 UNVERIFIED LAYER — <layer NOT checked + skipped read>, or all covered.

  W1 FAILURE RECEIPT — <failure this moment invites: a check left unrun>; a first
  pass is settled only by a probe.

  W2 TRAP INVENTORY WALK — where this kind of work goes wrong. Read each
  against what you actually did; name the live one, or say N/A why. This is a
  list of FAILURE MODES, not a form to fill:
  ENUMERATE / PARSE-SCHEMA / RANK / RUN-TIME / UNSET-FIELDS / UNIT-FORMULA /
  LIST-FLOOR / READBACK-STATE / ABSENCE-AS-EVIDENCE / DEFINING-FIELD /
  OBSERVED-VS-CONSUMED / STATED-VS-SHIPPED.
  UNIT-FORMULA: a value that reads right in one unit and wrong in another —
  the conversion is where the error hides, not the arithmetic.
  LIST-FLOOR: a stated enumeration read as a ceiling. Having none of the
  forbidden items is not the same as having each required one.
  READBACK-STATE: your own probe wrote the thing the checker later reads, so
  it passes on residue or on a stale copy. Prose 'I cleaned it' is no receipt.
  ABSENCE-AS-EVIDENCE: the artifact is silent about a behaviour and you
  inferred the behaviour anyway. Silence is not an observation.
  DEFINING-FIELD: the field that DEFINES a category is not the one you
  proxied for it; check that field on a positive and a negative example.
  OBSERVED-VS-CONSUMED: the quantity that made your check go green is often
  not the quantity the consumer reads off the same object.
  STATED-VS-SHIPPED: the failure this list exists for. EVERY constraint you
  stated counts; one left off leaves this slot unfilled, so never reconcile a
  self-chosen subset. Everything you said can be TRUE while the bytes are
  wrong: a design written A+B+C ships as A+C by one omission, and a
  true-sounding sentence hides exactly that. A constraint living only in your
  description, not in characters you can point at, is not shipped.

  W3 FRESH PROBE — one command not run in this step:
  <command> → <its real output: the number printed on YOUR artifact>;
  bar=<the limit the task states>; VERDICT=PASS|FAIL.
  Wrong side of its bar: repair, or declare unreachable.
  PARSING RULE — labels start slots; W3 ends at next label; `cmd → output` on
  one line, or output on the next.

  W4 TRAP SHAPES — five recur; green on W0-W3 can still miss one. Name YOUR
  instance, or open bet why: (1) the check judges your own artifact — a
  generator agrees with itself; (2) the last format item — one char, exit token, header, hash; (3) one clean sample for the whole population — name boundary inputs you did NOT run; (4) the fix landed in a scratch copy, not the DELIVERED bytes; (5) the
  visible example fixes FORM, not scale, offset, unit or index base. A trap never enumerated is one you stand in.
  RISK SWEEP — per logged risk: <risk> → <evidence deciding it> → <side that won>.
  Unresolved risk blocks this declaration: settle it now, or name the probe.

Re-issue it WITH these lines and continue.
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
the same W3 slot (the slot ends at the next line starting with a label).

`_override_reason` does NOT release this gate — only filling the slots does.
"""


def _missing_labels(covers) -> list:
    """Human-readable list of what the re-issue still lacks (form-level too)."""
    out = list(covers.missing)
    if not covers.has_unset_subslot:
        out.append("W2(need UNSET-FIELDS: <field> -> <shipped> -> <object outside "
                   "your artifact that decided it> -> <rival reading's number>)")
    if not covers.has_unit_formula_subslot:
        out.append("W2(need UNIT-FORMULA: <field> -> <the conversion you actually "
                   "used, written as x_B = f(x_A)> -> <both readings' numbers>, or "
                   "an explicit 'no unit conversion' when no unit was in play)")
    if not covers.has_list_floor_subslot:
        out.append("W2(need LIST-FLOOR: <each item the statement enumerates> -> "
                   "<whether an exact-match probe of the DELIVERED bytes found it> "
                   "-> <the reading you are shipping, floor or ceiling>, or an "
                   "explicit 'no allowlist in the statement' when none was given)")
    if not covers.has_readback_state_subslot:
        out.append("W2(need READBACK-STATE: <each path/port/process the checker "
                   "will read> -> <whether MY OWN probe wrote it> -> <the state "
                   "you leave it in, and whether that residue can change what the "
                   "checker observes, in content or in timing>, or an explicit "
                   "'no shared read-back object' when your probes touch none)")
    if not covers.has_absence_evidence_subslot:
        out.append("W2(need ABSENCE-AS-EVIDENCE: <the artifact that was SILENT "
                   "about the behaviour> -> <the component whose behaviour you "
                   "inferred from that silence> -> <an observation of that "
                   "component's OWN behaviour>, or an explicit 'no inference from "
                   "silence' when every reading you weighed rests on a component "
                   "you actually exercised)")
    if not covers.has_defining_field_subslot:
        out.append("W2(need DEFINING-FIELD: <the field that DEFINES the category "
                   "a consumer assigns to my artifact> -> <the proxy evidence I "
                   "based the assignment on> -> <that field's value on a POSITIVE "
                   "example AND on a NEGATIVE one>, or an explicit 'no category "
                   "assignment in this task' when no consumer classifies it)")
    if not covers.has_stated_shipped_subslot:
        out.append("W2(need STATED-VS-SHIPPED: <EVERY constraint YOU STATED this "
                   "task, one line each; a stated one left off leaves this "
                   "unfilled> -> <what the DELIVERED bytes carry for each, "
                   "measured on the file> -> <agree, or the edit you make now>. "
                   "QUOTE the characters you read off the SHIPPED artifact for "
                   "each one — a design written A+B+C ships as A+C by one "
                   "omission, and a true-sounding sentence hides it; no "
                   "characters in the file embodying a constraint = unfilled), "
                   "or an explicit 'no constraint I stated' when you stated none")
    if not covers.has_observed_consumed_subslot:
        out.append("W2(need OBSERVED-VS-CONSUMED: <the quantity I used to call my "
                   "own check PASSED> -> <the quantity the consumer reads off the "
                   "same object> -> <whether they are the same quantity, and if "
                   "not, the probe that tied them together>, or an explicit 'my "
                   "check and the consumer read the same quantity' when they do)")
    return out


class _ReviewCoverage:
    """Result of the STRUCTURAL review check (never a semantic judgement)."""

    __slots__ = ("missing", "has_w3_two_slots", "has_unset_subslot",
                 "has_unit_formula_subslot", "has_list_floor_subslot",
                 "has_readback_state_subslot", "has_absence_evidence_subslot",
                 "has_defining_field_subslot",
                 "has_observed_consumed_subslot",
                 "has_stated_shipped_subslot")

    def __init__(self, missing, has_w3_two_slots, has_unset_subslot=True,
                 has_unit_formula_subslot=True, has_list_floor_subslot=True,
                 has_readback_state_subslot=True,
                 has_absence_evidence_subslot=True,
                 has_defining_field_subslot=True,
                 has_observed_consumed_subslot=True,
                 has_stated_shipped_subslot=True):
        self.missing = list(missing)
        self.has_w3_two_slots = has_w3_two_slots
        self.has_unset_subslot = has_unset_subslot
        self.has_unit_formula_subslot = has_unit_formula_subslot
        self.has_list_floor_subslot = has_list_floor_subslot
        self.has_readback_state_subslot = has_readback_state_subslot
        self.has_absence_evidence_subslot = has_absence_evidence_subslot
        self.has_defining_field_subslot = has_defining_field_subslot
        self.has_observed_consumed_subslot = has_observed_consumed_subslot
        self.has_stated_shipped_subslot = has_stated_shipped_subslot

    def __bool__(self):
        # Truthy only when every label is present AND the FORM sub-slots
        # are addressed. The W3 two-slot, W2 UNSET-FIELDS, W2 UNIT-FORMULA,
        # W2 LIST-FLOOR, W2 READBACK-STATE and W2 ABSENCE-AS-EVIDENCE checks
        # are form requirements on an existing label, not extra labels.
        return (not self.missing and self.has_w3_two_slots
                and self.has_unset_subslot and self.has_unit_formula_subslot
                and self.has_list_floor_subslot
                and self.has_readback_state_subslot
                and self.has_absence_evidence_subslot
                and self.has_defining_field_subslot
                and self.has_observed_consumed_subslot
                and self.has_stated_shipped_subslot)

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

    # UNSET-FIELDS sub-slot (2026-11-16 user ruling: the beat must force the
    # agent to THINK about an unset convention, not merely carry a W2 heading).
    # STRUCTURAL, same boundary as W3: it asks whether the sub-label is
    # addressed inside the W2 span, never whether the answer is any good.
    # Two accepted forms, because "every field was pinned" is a legitimate
    # answer and must not be punished:
    #   (a) the literal sub-label `UNSET-FIELDS`, or
    #   (b) an explicit none-declaration: "no unset" / "UNSET: none" / ...
    # A W2 that walks ENUMERATE/PARSE-SCHEMA/RANK/RUN-TIME and stops is
    # refused, which is exactly the failure this exists to catch.
    has_unset = bool(
        re.search(r"(?<![A-Za-z0-9_])UNSET[- _]?FIELDS(?![A-Za-z0-9_])", blob, re.I)
        or re.search(r"\bunset\b[^\n]{0,40}\bnone\b", blob, re.I)
    )

    # UNIT-FORMULA sub-slot (2026-11-16 user ruling, after raman-fitting went
    # 3-for-3 wrong the same way). The reading block had already reported
    # `unit or scale` as UNSET and listed the affected field — and the agent
    # still shipped the raw axis value. Reporting an unset convention does not
    # discharge it; the ARITHMETIC is what has to exist. So the same way
    # UNSET-FIELDS forces the field to be named, UNIT-FORMULA forces the
    # conversion to be written out with both readings' numbers.
    # STRUCTURAL, same boundary as W3/UNSET-FIELDS: it asks whether the
    # sub-label was addressed, never whether the conversion was the right one.
    # A no-op declaration is legitimate and must not be punished, so three
    # accepted forms: the literal sub-label, or an explicit none-declaration
    # ("no unit conversion" / "unit: none").
    has_unit_formula = bool(
        re.search(r"(?<![A-Za-z0-9_])UNIT[- _]?FORMULA(?![A-Za-z0-9_])", blob, re.I)
        or re.search(r"\bno\b[^\n]{0,40}\bunit\b[^\n]{0,40}\bconversion", blob, re.I)
        or re.search(r"\bunit\b[^\n]{0,20}\bnone\b", blob, re.I)
    )

    # LIST-FLOOR sub-slot (2026-11-16, same shape as UNIT-FORMULA one task later).
    # Measured on `large-scale-text-editing`: the statement enumerates exactly
    # which commands a deliverable may contain; the agent read that enumeration
    # as a CEILING ("nothing off-list"), shipped a file that never contained one
    # of the listed items, and then ran a scan that *confirmed only the
    # ceiling* -- so its self-test was green while the checker's set-membership
    # test failed. This is raman's "reported the UNSET, did not discharge it"
    # one level over: it reported the list and never probed each item against
    # the delivered bytes. UNIT-FORMULA forces the arithmetic; LIST-FLOOR forces
    # the per-item exact-match probe.
    # STRUCTURAL, same boundary: it asks whether the sub-label was addressed,
    # never whether the chosen reading was right. "no allowlist in the
    # statement" is a legitimate no-op and must not be punished.
    has_list_floor = bool(
        re.search(r"(?<![A-Za-z0-9_])LIST[- _]?FLOOR(?![A-Za-z0-9_])", blob, re.I)
        or re.search(r"\bno\b[^\n]{0,30}\ballow[- ]?list", blob, re.I)
        or re.search(r"\bno\b[^\n]{0,40}\benumerat", blob, re.I)
    )

    # READBACK-STATE sub-slot (2026-11-16, third axis, after make-doom-for-mips
    # shipped a frame object its OWN probe had created).
    # Measured: the agent ran the deliverable itself to check it, which wrote the
    # very file the checker waits on. The checker's wait loop is
    # `while not <path>.exists()`, so a residue makes it fall through instantly,
    # kill the process before the expected line is printed, and then compare the
    # checker's expected image against a frame produced by a LONG-RUNNING demo
    # rather than the start-up screen. One leftover file failed two of three
    # assertions: one on content, one on TIMING. The agent's own self-test was
    # green because it only checked that the file was valid.
    # This is the third instance of the same shape: raman reported an unset
    # convention without discharging it; the list task reported an enumeration
    # without probing each item; this one probed the artifact without asking
    # whether the probe had CHANGED the object the checker later reads.
    # STRUCTURAL, same boundary: it asks whether the sub-label was addressed,
    # never whether the answer is right. "no shared read-back object" is a
    # legitimate no-op and must not be punished.
    has_readback_state = bool(
        re.search(r"(?<![A-Za-z0-9_])READBACK[- _]?STATE(?![A-Za-z0-9_])",
                  blob, re.I)
        or re.search(r"\bno\b[^\n]{0,40}\bread[- ]?back\b", blob, re.I)
        or re.search(r"\bno\b[^\n]{0,40}\bprobe\b[^\n]{0,30}\bwrote\b", blob, re.I)
        # the exact no-op phrase the slot's own instructions offer, which the
        # first alternative above would otherwise miss ("read-back" vs
        # "readback" is already covered, but "shared ... object" is not).
        or re.search(r"\bno\b[^\n]{0,40}\bshared\b[^\n]{0,30}\bobject\b",
                     blob, re.I)
        # 2026-11-16 (sixth axis): the checker may reach a PRIVATE endpoint the
        # statement never names. Measured: 11 losses on one task, the newest
        # trial passing 3 of 4 assertions and failing only where the checker
        # dials an address the statement left unset. This alternative accepts
        # the explicit COVERAGE declaration ("every address it could read") or
        # the explicit leave-empty declaration, so the axis can be discharged
        # rather than merely named.
        or re.search(r"\bevery\b[^\n]{0,40}\baddress\b", blob, re.I)
        or re.search(r"\bno\b[^\n]{0,40}\b(private|rival|other)\b[^\n]{0,30}\baddress",
                     blob, re.I)
    )

    # ABSENCE-AS-EVIDENCE sub-slot (2026-11-16, fourth axis, after mteb-retrieve
    # shipped the wrong document while its own note had already computed the
    # rival reading).
    # Measured: the statement names a component whose behaviour is supplied by a
    # registry shipped with the named library -- the reference invocation passes
    # a usage-typed argument that selects it. The agent inspected a DIFFERENT
    # artifact (the model repo's own config), found it silent about that
    # behaviour, and concluded the behaviour did not exist -- even though its own
    # note recorded that applying it changed the answer. One file's silence was
    # used as a negative observation about another component.
    # This is the fourth instance of the same shape: raman reported an unset
    # convention without discharging it; the list task read an enumeration
    # without probing each item; the frame task probed the artifact without
    # asking what state it left; this one INFERRED A NEGATIVE from an artifact
    # that merely did not mention the subject.
    # STRUCTURAL, same boundary: it asks whether the sub-label was addressed,
    # never whether the inference was right. "no inference from silence" is a
    # legitimate no-op and must not be punished.
    has_absence_evidence = bool(
        re.search(r"(?<![A-Za-z0-9_])ABSENCE[- _]?AS[- _]?EVIDENCE(?![A-Za-z0-9_])",
                  blob, re.I)
        or re.search(r"\bno\b[^\n]{0,40}\binference\b[^\n]{0,30}\bsilence\b",
                     blob, re.I)
        or re.search(r"\bsilence\b[^\n]{0,40}\bnot\b[^\n]{0,20}\bobservation\b",
                     blob, re.I)
        or re.search(r"\bno\b[^\n]{0,40}\babsent\b[^\n]{0,20}\bevidence\b",
                     blob, re.I)
    )

    # DEFINING-FIELD sub-slot (2026-11-16, fifth axis, after a document-routing
    # task filed one document under the wrong category).
    # Measured: the consumer assigns each document a category by comparing it
    # against a label set. The agent instead classified by a PROXY that correlates
    # with the category on most inputs -- the document carried a monetary total
    # and its line items summed to that total -- and that proxy happened to fire
    # on a decoy the label set calls the OTHER category. Nothing in the artifact
    # was mis-extracted; one boundary was decided from the wrong field, and the
    # error cascaded into three more assertions (a set difference, a second set
    # difference, and a row count).
    # This is the fifth instance of the same shape: raman reported an unset
    # convention without discharging it; the list task read an enumeration
    # without probing each item; the frame task probed the artifact without
    # asking what state it left; the retrieval task inferred a negative from
    # silence; this one assigned a consumer-visible CATEGORY from a proxy that
    # agrees on the easy majority and fails exactly at the boundary.
    # STRUCTURAL, same boundary: it asks whether the sub-label was addressed,
    # never whether the field chosen was the right one.
    has_defining_field = bool(
        re.search(r"(?<![A-Za-z0-9_])DEFINING[- _]?FIELD(?![A-Za-z0-9_])",
                  blob, re.I)
        or re.search(r"\bno\b[^\n]{0,40}\bcategory\b[^\n]{0,30}\bassignment\b",
                     blob, re.I)
        or re.search(r"\bno\b[^\n]{0,40}\bclassif\w*[^\n]{0,30}\bconsumer\b",
                     blob, re.I)
        or re.search(r"\bno\b[^\n]{0,40}\bdefin\w*[^\n]{0,20}\bfield\b",
                     blob, re.I)
    )

    # The failure this sub-slot exists for: the agent's own check reported on
    # one quantity (a command was accepted, a process was alive, a file was
    # written) while the consumer reads a DIFFERENT quantity off the same
    # object (whether the observable STATE actually changed). Both can be true
    # at once and the check still scores zero -- so the slot asks which
    # quantity the consumer reads, not whether the agent's own check was green.
    # STRUCTURAL, same boundary as its siblings: it asks whether the two
    # quantities were named and compared, never whether that naming was right.
    has_observed_consumed = bool(
        re.search(r"(?<![A-Za-z0-9_])OBSERVED[- _]?VS[- _]?CONSUMED(?![A-Za-z0-9_])",
                  blob, re.I)
        or re.search(r"\bno\b[^\n]{0,50}\bmy check\b[^\n]{0,40}\bsame quantity\b",
                     blob, re.I)
        or re.search(r"\bno\b[^\n]{0,50}\bobserved\b[^\n]{0,40}\bconsumed\b",
                     blob, re.I)
    )

    # STATED-VS-SHIPPED sub-slot (eighth axis). Measured on a task that went 0
    # the same way across five generations of prompt clauses: the agent's own
    # log STATED the missing constraint verbatim AND named that its own
    # deliverable violated it -- then shipped the violating bytes anyway.
    # Nothing in the review asked for that constraint's value ON THE DELIVERED
    # FILE, and its self-check measured quantities the statement never
    # constrained, so it went green while the deliverable was wrong.
    # STRUCTURAL, same boundary as its siblings: it asks whether the statement
    # and the shipped value were both named and compared, never whether the
    # comparison was right. "no constraint I stated" is a legitimate no-op.
    has_stated_shipped = bool(
        re.search(r"(?<![A-Za-z0-9_])STATED[- _]?VS[- _]?SHIPPED(?![A-Za-z0-9_])",
                  blob, re.I)
        or re.search(r"\bno\b[^\n]{0,40}\bconstraint\b[^\n]{0,40}\bstate",
                     blob, re.I)
        or re.search(r"\bno\b[^\n]{0,40}\bstated\b[^\n]{0,30}\bshipped\b",
                     blob, re.I)
    )

    return _ReviewCoverage(missing, w3_two_slots, has_unset, has_unit_formula,
                           has_list_floor, has_readback_state,
                           has_absence_evidence, has_defining_field,
                           has_observed_consumed, has_stated_shipped)


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
       labels W0/W1/W2/W3/W4/RISK, PLUS three FORM sub-slots that ride on labels
       already in that list (they are not extra labels): the W3 two-slot form
       (`<command> → <output>`, or the reading on the next line), the W2
       UNSET-FIELDS sub-label, and the W2 UNIT-FORMULA sub-label (the
       conversion actually used, `x_B = f(x_A)`, with both readings' numbers).
       The guard does NOT judge whether an answer is
       good, true, or complete — it only checks that the slots were addressed,
       exactly as a form checks that fields are non-empty. A wrong-but-filled
       answer passes; an empty one does not.

    Everything else is untouched: every other tool call and every other
    plan_update is never blocked by this guard.

    THERE IS NO EXIT VALVE, BY USER RULING (2026-11-13). An earlier revision
    released the declaration after 3 refusals and recorded the gap as an
    INCOMPLETE self-review. That was the wrong repair: it makes the gate fail
    exactly on the runs that most need it — the ones whose author could not
    produce the slots. The deadlock it was meant to solve was a JUDGEMENT bug,
    not a missing exit: the W3 checker demanded the arrow on the FIRST W3 line
    only, so a correct two-line probe ("cmd" then "4 passed in 0.61s") was
    refused, and the model spent four re-issues guessing a format. That bug is
    fixed by the SPAN rule (see _review_labels_present); with it, a model that
    actually ran a probe has a shape that passes. If a declaration still cannot
    fill the slots, blocking again is the correct response — the step is not
    ready, and step_skip is a visible, recorded choice rather than a silent pass.
    """

    name = "system_prompt_remind"
    # Sits just after PlanUpdateGuard (50) so it sees the declaration before
    # the plan guard acts on it, and before the generic advisories (72/90).
    priority = 58

    def __init__(self) -> None:
        # Signatures already beaten once this session: one beat per step.
        self._beaten: set[str] = set()

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name == "plan_update":
            args = ctx.tool_args or {}
            if not _declares_completion(args):
                return None
            sig = _declaration_sig(args)
            covers = _review_labels_present(args)
            if sig in self._beaten:
                # Beaten once. It may now pass ONLY if the review slots are
                # filled. A second empty re-issue is still refused (this is
                # what makes the beat real instead of a one-shot any prose
                # clears).
                if not covers:
                    return GuardVerdict.block(
                        _REMIND_MESSAGE_MISSING.format(
                            missing=", ".join(_missing_labels(covers))),
                        reason="system_prompt_remind_still_empty",
                        category="system_prompt_remind",
                        overridable=False,
                    )
                return None  # slots addressed — let it through
            self._beaten.add(sig)
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
