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

"""VerificationGuard — requires verification evidence when marking steps complete.

Design principles:
- Block plan_update(action="step_done") if:
  * Step has acceptance criteria AND no verification provided
  * Step is complex (no acceptance defined) AND no _override_reason provided
- Other tool calls (read_file/shell/grep) are completely unaffected
- LLM can freely perform verification operations after being blocked
- Once verified, LLM calls with verification=["..."] or _override_reason to pass
- Does not check verification content — any non-empty list passes

Two verification modes:
1. Structured: step has acceptance → must provide verification=["proof1", "proof2"]
2. Override: no acceptance (simple step) → must provide _override_reason="checked X"

Why this works:
- Acceptance criteria define WHAT to verify
- Verification list records HOW it was verified
- Override_reason for simple steps maintains backward compatibility

Execution flow (structured):
1. LLM: plan_update(action="step_done", step_id=3)
2. Guard: BLOCK - step has acceptance, verification required

3. LLM: OK, let me verify acceptance criteria
4. LLM: shell("pytest tests/")  ← executes normally
5. LLM: read_file("output.log")  ← executes normally

6. LLM: plan_update(action="step_done", step_id=3, verification=["all tests passed", "log shows no errors"])
7. Guard: Has verification, allow ✓
"""

from __future__ import annotations

import os
import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.multi_agent.wiring import CONTRACT_PATH_ENV, is_worker
from flagscale_agent.react.display import has_override_completion, trailing_sentinel

# Mirrors plan_update._parse_step_id: the handler accepts "1", "step_1",
# "step 2", etc. and normalizes to an int before mutating the plan. The guard
# MUST normalize identically, or a string id ("step_1") misses the plan lookup
# (step.get("id") == 1 fails), the lookup falls back to permissive, and the
# gate is bypassed while the handler still marks the step done.
_STEP_ID_RE = re.compile(r"(?:step[_\s]?)?#?(\d+)", re.IGNORECASE)


def _normalize_step_id(raw) -> int | None:
    """Normalize a step id the SAME way the plan_update tool does.

    Returns an int for any accepted form (1, "1", "step_1", "step 2", "Step #3"),
    else None. The guard's lookups use this so what the guard checks is exactly
    the step the handler will mutate.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip()
        try:
            return int(s)
        except ValueError:
            pass
        m = _STEP_ID_RE.search(s)
        if m:
            return int(m.group(1))
    return None


def _contract_constraints() -> dict:
    """The spawned worker's `constraints` block, read from the structured
    contract JSON — NOT the rendered contract.prompt prose.

    The rendered prompt is prose: a contract that merely QUOTES a phrase (a
    reviewer contract quoting "## Reviewer discipline", or any body mentioning
    "framings of the task") would falsely suppress the demand. The sibling
    contract.json carries the authoritative `constraints` mapping, so gating on
    constraints.reviewer is exact. Errors fail silent-{} (callers fall through
    to the demand as usual).
    """
    try:
        path = os.environ.get(CONTRACT_PATH_ENV, "")
        if not path:
            return {}
        import json
        with open(os.path.join(os.path.dirname(path), "contract.json"),
                  "r", encoding="utf-8") as f:
            data = json.load(f)
        cons = data.get("constraints") or {}
        return cons if isinstance(cons, dict) else {}
    except Exception:
        return {}


_VERIFICATION_REQUIRED_WITH_ACCEPTANCE = """[VerificationGuard] Before this step is done — answer three questions per criterion.

Do NOT describe what you did ("ran the tests") — that is your activity, not
evidence the criterion holds. For EACH criterion below answer:
  1. OBSERVED — the concrete value/output you saw (a number, count, exit code,
     exact string — not "it worked").
  2. EXPECTED — what the criterion required.
  3. GAP — observed vs expected: match, or off by how much?
No concrete OBSERVED value means the criterion is NOT verified yet — go get the
value, do not invent one.

Retry with one verification entry per criterion, each carrying its observed value:
  plan_update(action="step_done", step_id=N, verification=[
    "criterion 1 — observed: <value>; expected: <value>; gap: <match/diff>"])

Acceptance criteria for this step:
{acceptance}.
"""

_VERIFICATION_REQUIRED_NO_ACCEPTANCE = """[VerificationGuard] Before this step is done — answer, do not assert.

No acceptance criteria here, so answer in _override_reason: what did you
OBSERVE that shows the goal is met — a concrete value/output you saw (count,
exit code, exact string), not "it worked". No observed value means not done — go
get one.

  plan_update(action="step_done", step_id=N, _override_reason="observed <value/output> which shows <goal> is met")
"""

_ACCEPTANCE_FROZEN = """[VerificationGuard] Acceptance change blocked — criteria are frozen.

Step {step_id} is already "{status}" — implementation has begun. Changing its
acceptance criteria now lets the standard be shaped by what you produced.

Define acceptance BEFORE implementing. If the original criteria were genuinely
wrong, retry with _override_reason explaining what was wrong with them and why
the new criteria are still independent of your implementation choices.
"""

_ACCEPTANCE_GUIDANCE = """[VerificationGuard] Writing acceptance criteria — a note on what makes them useful.

Acceptance describes an externally observable end state — a file at a path, a
command exiting 0, an artifact with an expected property. The check must be
independent of your implementation, or it cannot fail.

Re-read the task statement and list every operation it explicitly asks for; each
maps to a criterion. "I know how to do X, so X is done" is not a criterion — it is
the signal to execute X.

Traps:
  • Cover every input the task gives you, not just the one you developed against.
  • Verify against the judge's measurement, not your proxy.
  • A failure inside your declared scope is yours to fix, regardless of origin.

Delivery-time traps (fresh reload, backup+rollback, exact-contents,
byte-immutability, noisy margin) are enforced at completion — do not carry them into acceptance.
"""

# NOTE: The framing-time reminders (qualifier extraction, pre-mortem, wall-aware
# first cut, divergence check) lived in plan.py as inject blocks, then were
# consolidated into the fixed FIRST plan step's acceptance criteria
# (_FRAME_STEP_ACCEPTANCE in tools/plan_create.py). Reason: injecting on
# plan_create caused guard override_reason cross-talk — PlanGuard's single-shot
# block trained the agent to pass _override_reason, which then silently
# satisfied other guards' blocks too, so they never actually fired.

_BATCH_STEP_DONE_NO_EVIDENCE_BLOCK = """[VerificationGuard] Batch marks step {step_id} done, but that step has acceptance criteria and the batch carried no evidence for it.

A batch update that flips a step to "done" commits the SAME claim as a single
step_done — so it must clear the SAME bar. For a step that HAS acceptance
criteria, that bar is a non-empty `verification` list (Mode 1). A batch item
that omits it is the silent hole this check closes: the step reaches "done"
with nothing recording that it met its criteria.

This step's criteria:
{acceptance}

For EACH done item whose step has acceptance, pass the per-step evidence in the
batch item itself:

  plan_update(action="batch", updates=[
      {{"step_id": 1, "status": "done", "verification": ["<criterion> — observed: ...; expected: ...; gap: ..."]}},
      {{"step_id": 2, "status": "done", "verification": [...]}},
  ])

Or mark such a step individually:
  plan_update(action="step_done", step_id=1, verification=[...])

Non-done items (doing/skipped) need no evidence."""

_BATCH_STEP_DONE_BLOCK = """[VerificationGuard] Batch update marks a step done — same bar as a single step_done.

Marking done in bulk must not skip the verification each step would otherwise require.
Re-issue the batch with "_override_reason" affirming that for EACH step you are marking
done, you re-checked the result against what that step was supposed to satisfy.

If a step deserves real evidence, mark it via plan_update(step_done, verification=[...])
individually rather than folding it into a batch."""

_COMPLETION_NO_GRADED_READING = """[VerificationGuard] You are declaring the task complete, but no GRADED READING of your deliverable was ever declared.

The `deliverable_reading` field is a FIXED declaration on every step_done — and
`{"deliverable": "none"}` is a claim that there is NO graded artifact. If this
task DOES grade an artifact (its text is available to ground a reading), that
claim is unsupported until you have actually enumerated the readings the checker
may apply and run the acceptance test on the output you are shipping.

Before completing: on a step_done, declare `deliverable_reading` as a real object
naming the graded artifact, its `readings` (each with the exact `source` quote,
the `acceptance_test`, and the `normalized_side` the checker compares on), and the
`hedge` you ran. This block is NON-overridable: the declaration IS the exit —
even a task whose deliverable seems trivial closes it by declaring that
deliverable and its readings explicitly, never by skipping the declaration.
"""

_COMPLETION_NO_ACCEPTANCE_OBSERVED = """[VerificationGuard] You are declaring the task complete, but you never recorded a real ACCEPTANCE OBSERVATION.

Declaring the readings a checker may apply is not the same as observing that your
deliverable actually satisfies them. Before completion you must face that gap
explicitly, on a step_done, with an `acceptance_act` object carrying ONE of:

  \u2022 An OBSERVED act: {"command": "<the exact command you ran>", "observed":
    "<its ACTUAL output text>"}. The `observed` text must be output the harness
    really recorded for this session — a claimed-but-never-run command is refused.
  \u2022 An UNOBSERVABLE attempt: {"unobservable": "<the acceptance dimension you
    genuinely cannot observe, and why>", "command": "<the probe you ran to try>",
    "observed": "<what that probe actually returned>"} — for a hidden corpus or an
    oracle you have no means to run. This does NOT claim PASS; it carries the
    dimension as an unverified premise (INCONCLUSIVE, not PASS). But declaring a
    dimension invisible is not enough on its own: you must show the probe you RAN
    and that probe's real recorded output, so "I could not observe it" costs an
    ATTEMPT just as the OBSERVED form costs an OBSERVATION. A bare
    {"unobservable": "..."} with no attempt is refused.

This block is NON-overridable: the recorded attempt IS the exit — if no
acceptance can be observed, use the UNOBSERVABLE form above (with the probe you
ran and its real recorded output), then complete again.
"""

_COMPLETION_ORACLE_NOT_EXERCISED = """[VerificationGuard] You are declaring the task complete, but the independent oracle you bound at framing time was never EXERCISED.

At framing you named an oracle and wrote its FIRST runnable exercise -- an
intended ACTION, not a label. Naming an oracle is a DECLARATION; only running it
makes it a LOOK. Nothing in this run shows the exercise you declared was ever
executed: the completion boundary can find no recorded exercise of the bound
oracle in your session.

Before completing: actually RUN a command aimed at the oracle you named and
record it on a step_done as `oracle_act`:
{"command": "<the exact command you ran>", "observed": "<its ACTUAL output text>"}
The `observed` text must be output the harness really recorded for this session
-- a claimed-but-never-run exercise is refused. This block is NON-overridable:
the recorded exercise IS the exit — if the oracle you named turned out unrunnable,
run the closest probe you can and paste its real output as the attempt. A
framing-time declaration you never put into action is exactly the escape this arm
converts into a look.
"""

_COMPLETION_ORACLE_NOT_BOUND = """[VerificationGuard] You are declaring the task complete, but you never BOUND an independent oracle.

The frame step tells you to name the oracle that will decide this task and classify
its PREMISE as coming from a source you did NOT author -- 'differential' (two
independent implementations), 'provided' (a reference/golden artifact the
environment supplies), or 'consumer' (the exact parse/serialize path the consumer
replays). Naming it in prose is a DECLARATION; the harness records an oracle only
when you actually pass it to plan_create(oracle={...}). No oracle was ever bound in
this run -- so the one external check that could catch a premise you hold wrongly
was never established, and your closing verification rests only on your own reasoning.

Before completing, do ONE of these on a step_done:
  \u2022 BIND the oracle: call plan_create(oracle={"premise_source": "provided" |
    "differential" | "consumer", "oracle": "<what decides it>", "first_exercise":
    "<the exact command/check that runs it>"}) and then RUN that first exercise,
    recording it as `oracle_act` = {"command": "<the command>", "observed": "<its
    ACTUAL output text>"}. The `observed` text must be output the harness really
    recorded for this session -- a claimed-but-never-run exercise is refused.
This block is NON-overridable: BINDING the oracle IS the exit. Even a weak,
partial oracle (a format check against the task's own stated contract, a
sample-level differential comparison) is an external anchor -- bind the closest
one you can actually run. An axis with genuinely no external anchor surfaces as
[NEED_USER_INPUT], never as an unanchored completion.
"""

_COMPLETION_WITH_UNFINISHED_STEPS = """[VerificationGuard] You are declaring the task complete while plan step(s) {steps} are still open.

A plan marks the checkpoints where your results were to be verified. Completing with steps
unfinished skips the record that shows each result was checked against what it was meant to
satisfy — the same obligation plan_update(action="complete") already refuses to skip.

Do ONE of these, then complete again:
  • Close the steps: plan_update(action="step_done", step_id=<n>, verification=[...]) for each.
    That call carries the fixed convention / deliverable_reading declarations — the record of
    the decisive reading you shipped under. Mark a genuinely-obsolete step skipped instead.
  • If the remaining steps truly do not apply, re-issue this completion with an
    "_override_reason" naming which steps you are abandoning and why that is safe.
"""

_STEP_DONE_RECHECK_REMINDER = """[VerificationGuard] Before this step is marked done — check your premises once.

The last point where a wrong turn is cheap to catch. Verification is not
restating what you believed while working — it is testing that belief against
what you got.

For each criterion the step had to satisfy, read it as written and ask whether
your result lands inside it. Then examine the premises you leaned on: any reading
of a term, any "this is close enough", any value taken as given rather than
confirmed. A premise adopted for convenience is the one most likely wrong — treat
it as the thing under test, not settled ground.

Finally, check the target: re-read what the user actually asked for and confirm
this step moves toward THAT, not a nearby goal your framing swapped in. Verifying
the wrong question thoroughly is still off-target. When your reading and the
task's plain wording pull apart, the wording wins.

If the step's requirement was UNDER-SPECIFIED — the wording admitted two
defensible readings and the checker is invisible — do not settle it by feel. At
this boundary, name the reading you produced against and, if a candidate output
CLAIMED to satisfy BOTH readings, the probe output that demonstrated it — an
unprobed hedge is not evidence — and neither is an introspected "no hedge
possible": bet only once the intersection is CONSTRUCTIVELY shown empty. If the
readings are mutually exclusive you are betting, so name the tie you broke (the
wording the task EXPLICITLY grants first; otherwise the branch strongest on the
axes the task states WITHOUT ambiguity). A reading chosen silently is the one that fails.

To proceed, re-issue plan_update(step_done) with "_override_reason" naming a
CONCRETE ANCHOR, not a summary of diligence. "I re-checked and it's correct" is
the empty answer this replaces. Point at one of: the value you OBSERVED vs the
value the criterion required (report both); the exact premise you re-tested and
what retesting showed; the reading you chose for an under-specified clause and
the evidence for it; or — if a premise cannot be tested here — say so in those
words. An anchor you can point at, or an explicit "no evidence for X"; not an
adjective. This is not a content check — any override_reason lets the step
through; a reason with no anchor is itself the signal you did not look."""

_POST_RECOVERY_REMINDER = """[VerificationGuard] Context was just recovered via hard_reset.

Before continuing work:
1. Read key files to confirm current state
2. Check recent changes (git status, grep for markers, file checksums)
3. Verify assumptions from pre-recovery context still hold

The goal: avoid propagating stale assumptions into new work."""


# Second-stage task-complete gate. The first complete-recheck (Timing 0b) fires
# regardless of content and any override_reason releases it — it only forces a
# pause. But the pause is defeated by the exact failure the prompt's
# observation-vs-argument litmus warns about: the agent writes a reason that
# ARGUES its method is sound ("thresholds are relative / conservative / should
# generalize") instead of REPORTING an observation it made (ran something, read
# a value, compared output to a known answer). A confident rationale for a wrong
# answer is still a wrong answer. So when the release reason is pure argument with
# zero observation signal, we bite ONCE more and demand a concrete observation.
# Fires at most once (independent of the first gate) so it stays a checkpoint.

# Argument markers: method-defence vocabulary the litmus names explicitly. Their

# ---- Convention-check: a SHAPE-only gate on a derived-convention declaration ----
# The harness cannot judge whether a value is SEMANTICALLY right (it has no
# oracle), but it CAN refuse to accept a step SILENTLY.
#
# The declaration is a FIXED, REQUIRED structured field on every step_done —
# NOT a text tag that must appear in acceptance. The previous design was
# opt-in: it fired only when a step's acceptance literally contained the
# marker `convention-check`, and nothing ever added that marker automatically,
# so an agent that failed to RECOGNISE it rested on a derived convention never
# triggered the gate — exactly the agent that most needed it.
# The design after that replaced the marker with a required field, but kept a
# self-declared `derived` boolean as the trigger: it fired only when the agent
# declared `derived=true`, so an agent whose load-bearing convention WAS the
# uncertain thing could declare the step `{"derived": false}` and skip the
# falsifier — the exact retry11 escape (agent log L1554: "Actually the mapping
# is derived, but step5's criterion is format validation. derived:false is
# fine."). A self-classifying trigger is dodgeable by construction.
#
# The corrected design removes the boolean. The `convention` object is a FIXED
# shape whose ONLY discriminator is the convention's NAME — the thing the agent
# must state, and that it cannot classify away:
#   • `{"name": "none"}` (or the bare string "none") is the explicit
#     not-applicable declaration — a recorded, owned act, not a silent skip.
#   • `{"name": "<a real convention name>", ...}` is a NAMED convention and
#     MUST carry a falsifier (a real probe OR a proved hedge).
# There is no `derived` key: the old shape `{"derived": false}` has no `name`
# and therefore BLOCKS, so the retry11 reasoning has no expressible form — the
# agent must either NAME the convention (and is then forced to falsify it) or
# write `"none"` (a visible claim on the record). "Not applicable" remains an
# explicit act; naming remains the un-dodgeable trigger.
#
# Honest ceiling: this does NOT verify the values are true. A dishonest agent
# can still fill the fields wrongly. It only downgrades "silently ignores the
# rule" to "must declare on the record, and must lie deliberately to defeat it".
# Spellings an agent might use to mean "no value". `None` is the ONLY legal
# no-value token; any of these in probe/probe_output under conflict=yes means
# the "probe must be real" rule is being evaded (a deliberate lie, which the
# guard cannot and does not prevent — see honest ceiling above).
_CONVENTION_NO_VALUE = frozenset(
    {"none", "n/a", "na", "n.a.", "null", "nil", "-", "--", ""}
)


def _is_real_str(v):
    """A REAL evidence string: a non-empty ``str`` that is not a no-value token.

    Non-strings (list / dict / int / bool) are NOT real — a probe must be an
    actual COMMAND string, its output an actual TEXT, a hedge_output an actual
    result. ``str(v)`` coercion would let a container/int masquerade as evidence
    (a falsifier that was never an observed command), so the type is checked
    BEFORE the emptiness test. ``""`` (and whitespace) is in the no-value set.
    """
    return isinstance(v, str) and v.strip().lower() not in _CONVENTION_NO_VALUE

_CONVENTION_MISSING = """[VerificationGuard] Every step_done must carry the fixed `convention` field.

Declare the convention THIS step's output rests on — a base, a stride, a key
space, a normalisation, a serialisation, a default — that you DERIVED rather
than one the task text states. This is mandatory on EVERY step_done. The field
is ONE fixed shape; the only thing you choose is the `name`:

  # no derived convention governs this step's output:
  convention={"name": "none"}

  # a derived convention DOES govern the output — you MUST then carry a
  # FALSIFIER (a real probe OR a proved hedge), because a bare single reading
  # is exactly how an ambiguous convention gets committed wrongly:
  convention={
    "name": "address key space",
    "example": "<the task's own example value, or None>",
    "delivered": "<the value you actually delivered, or None>",
    "conflict": "yes|no|None",
    "probe": "<command whose output DISCRIMINATES the reading, or None>",
    "probe_output": "<raw output of that command, or None>",
    "hedge": "intersection|union|None",
    "hedge_output": "<the single output that satisfies EVERY reading, or None>",
    "extras_direction": "ignored|penalized|None",
    "rejected": "<a candidate reading/implementation you DELETED or rejected, or None>",
    "rejected_evidence": "<independent artifact that rejected it, or None>"
  }

There is no `derived` key. Either you write `{"name": "none"}` (an explicit,
owned claim that no derived convention governs this step) or you NAME the
convention — and a named convention is admissible only with one of:
  (a) a REAL probe AND probe_output (a command + output that could have refuted
      the reading and did not); or
  (b) a proved HEDGE — hedge one of intersection/union, a REAL hedge_output (the
      single output that lies in every reading's acceptance set), and
      extras_direction naming how the checker treats output it did not ask for
      (ignored|penalized) — because the hedge SHAPE is valid only under the
      matching direction.
A bare name with neither a probe nor a hedge is BLOCKED. When conflict="yes",
probe and probe_output must be REAL (not None). When `rejected` names a
candidate, `rejected_evidence` must be an INDEPENDENT artifact (a reference
output, the task-spec text, a differential-probe result) — removing a candidate
on suspicion alone is forbidden, because when extras are unpenalised INCLUSION
is the conservative branch and omission is the only losing direction.
This is a shape check — it does not judge whether your value is right, only that
the convention is named and its falsifier is on the record instead of skipped."""

_CONVENTION_MALFORMED = """[VerificationGuard] The `convention` declaration is malformed.

`convention` must be either the explicit not-applicable form
(`{"name": "none"}`, or the string "none"), or a structured object with a real
`"name"` you DERIVED plus the required context fields example/delivered/conflict
(all keys must be present; write `None` for a field with genuinely no value):
  • name a non-empty string other than "none";
  • conflict one of yes / no / None;
  • probe and probe_output REAL (not None) when conflict is "yes";
  • rejected_evidence REAL (not None) whenever `rejected` names a candidate;
  • a FALSIFIER for a NAMED convention: EITHER probe+probe_output REAL, OR a
    complete hedge (hedge one of intersection/union, hedge_output REAL,
    extras_direction one of ignored/penalized).
There is no `derived` key. Write `{"name": "none"}` if this step genuinely
rests on no derived convention.
Re-issue the step_done with a corrected `convention` field."""

_CONVENTION_NO_FALSIFIER = """[VerificationGuard] A convention was NAMED with NO falsifier — a bare single
reading with neither a real probe nor a proved hedge.

Naming a derived convention is the claim that an ambiguous choice was made; the
claim is admissible only when you carry evidence that could have REFUTED the
reading and did not. Add ONE of:

  1. A real discriminating PROBE — a command whose output distinguishes your
     reading from its rival, and its raw output:
       "probe": "readelf -S new.o | grep -A1 LOAD",  "probe_output": "<raw>"
     (`conflict` is "yes" when that probe shows the example value and your
     delivered value differ — keep it honest.)

  2. A PROVED HEDGE — the single output that lies in EVERY candidate reading's
     acceptance set, plus how the checker treats output it did not ask for:
       "hedge": "union",            (or "intersection")
       "hedge_output": "<the one output satisfying all readings, with its result>",
       "extras_direction": "ignored"  (or "penalized")
     Shape↔direction must match: "union"/superset is valid only when extras are
     IGNORED; "intersection" is valid when extras are PENALIZED. An introspected
     "no hedge possible" is a hypothesis — a hedge is empty only when shown so
     under BOTH shapes.

A probe or hedge whose value is None/"None" does not count — that is the silent
skip this gate exists to stop. Re-issue the step_done with a real falsifier, or
write `{"name": "none"}` if this step genuinely rests on no derived convention."""


def _check_convention_field(conv):
    """Unconditional structured gate for the fixed `convention` declaration.

    `conv` is the raw value of ``tool_args['convention']`` — the harness now
    requires it on EVERY step_done, so there is no marker/tag and no regex.

    Returns a GuardVerdict.block when the declaration is missing or malformed,
    else None. The explicit not-applicable form ({"name": "none"} or the string
    "none") passes: that is a recorded decision, not a silent skip. There is NO
    `derived` boolean — the trigger is the NAME itself, so a step whose output
    rests on a derived convention cannot be classified away from the gate.
    """
    # Missing entirely — the whole point of the fixed field is that "nothing"
    # must be an EXPLICIT declaration, never a silent omission.
    if conv is None:
        return GuardVerdict.block(
            message=_CONVENTION_MISSING,
            reason="convention_declaration_missing",
            category="verification_required",
        )
    # The string form is the explicit not-applicable token.
    if isinstance(conv, str):
        if conv.strip().lower() in _CONVENTION_NO_VALUE:
            return None
        return GuardVerdict.block(
            message=_CONVENTION_MALFORMED,
            reason="convention_string_not_none",
            category="verification_required",
        )
    if not isinstance(conv, dict):
        return GuardVerdict.block(
            message=_CONVENTION_MALFORMED,
            reason="convention_not_dict",
            category="verification_required",
        )
    name = conv.get("name")
    # Explicit not-applicable — the ONE owned claim that lets a step skip the gate.
    # ONLY the literal token "none" (any case) is accepted here. Every OTHER
    # no-value spelling ("", "  ", "n/a", "null", "nil", "-", ...) is NOT a
    # declaration of not-applicable — it is a forgotten/blank name, which must
    # NAME the convention (or the step is malformed). Using the wide
    # _CONVENTION_NO_VALUE set here would silently downgrade such a name to
    # "not applicable" and re-open the very escape this gate closes.
    if isinstance(name, str) and name.strip().lower() == "none":
        return None
    # A NAMED convention must be a real string. This is the trigger: there is no
    # `derived` boolean to classify the step away from the gate — either the
    # agent writes the name of the convention its output rests on, or "none".
    if not _is_real_str(name):
        return GuardVerdict.block(
            message=_CONVENTION_MALFORMED,
            reason="convention_name_missing",
            category="verification_required",
        )
    for f in ("example", "delivered", "conflict"):
        if f not in conv:
            return GuardVerdict.block(
                message=_CONVENTION_MALFORMED,
                reason="convention_field_missing",
                category="verification_required",
            )
    conflict = str(conv.get("conflict")).strip().lower()
    if conflict not in ("yes", "no", "none"):
        return GuardVerdict.block(
            message=_CONVENTION_MALFORMED,
            reason="convention_bad_conflict",
            category="verification_required",
        )
    if conflict == "yes":
        if not (_is_real_str(conv.get("probe")) and _is_real_str(conv.get("probe_output"))):
            return GuardVerdict.block(
                message=_CONVENTION_MALFORMED,
                reason="convention_conflict_without_probe",
                category="verification_required",
            )
    # A NAMED convention must carry a FALSIFIER: a real probe OR a proved hedge.
    # A bare single reading with neither is exactly how an ambiguous convention
    # (base locked to an example, serialisation guessed under an invisible grader)
    # gets committed wrongly — so it is blocked, not merely discouraged. This is
    # the STRUCTURAL teeth: the agent must CONSTRUCT a falsifier the guard checks,
    # not state a sentence it may drop.
    has_probe = _is_real_str(conv.get("probe")) and _is_real_str(conv.get("probe_output"))
    hedge = str(conv.get("hedge") or "").strip().lower()
    extras = str(conv.get("extras_direction") or "").strip().lower()
    has_hedge = (
        hedge in ("intersection", "union")
        and _is_real_str(conv.get("hedge_output"))
        and extras in ("ignored", "penalized")
    )
    if not (has_probe or has_hedge):
        return GuardVerdict.block(
            message=_CONVENTION_NO_FALSIFIER,
            reason="convention_no_falsifier",
            category="verification_required",
        )
    # A hedge's SHAPE must match the direction it declares. A superset ("union")
    # is a hedge ONLY when the checker IGNORES extras — a union declared under
    # "penalized" is self-contradictory (exactly the retry9 reasoning: "extras
    # count" -> reject the union). Reject the contradiction rather than accept a
    # hedge that cannot hold.
    if hedge == "union" and extras == "penalized":
        return GuardVerdict.block(
            message=_CONVENTION_NO_FALSIFIER,
            reason="convention_hedge_shape_mismatch",
            category="verification_required",
        )
    # Removing a candidate reading is the exact action that cost retry9: it must
    # be backed by an INDEPENDENT artifact, never by an invented penalty rule.
    # `False`/empty-list/None all mean "no candidate named" — a bool is NOT a
    # candidate reading, so only a real string triggers the evidence demand.
    rejected = conv.get("rejected")
    if not isinstance(rejected, bool) and _is_real_str(rejected):
        if not _is_real_str(conv.get("rejected_evidence")):
            return GuardVerdict.block(
                message=_CONVENTION_MALFORMED,
                reason="convention_rejected_without_evidence",
                category="verification_required",
            )
    return None


# ---- Deliverable reading-analysis: forces a CONSTRUCTED reading of the graded
# output at the decision point (the step that emits the graded artifact) ----
# The failure this closes: under an INVISIBLE grader the agent picks ONE reading
# of the graded output (e.g. "pack by file offset" vs "pack by segment VA"),
# self-verifies against its OWN oracle, and ships — because nothing forced it to
# ENUMERATE the readings the task admits and CONSTRUCT an output safe under all
# of them (or prove none exists with independent evidence). The convention gate
# above covers the step's technical convention; it does NOT force the
# grader-acceptance READING.
#
# The field is FIXED and REQUIRED on every step_done (no marker, no regex, no
# opt-in — a silent omission is impossible; "none" is an explicit owned claim):
#   {"deliverable": "none"}                            # no graded artifact
#   {"deliverable": "<path>", "grader_visible": true}  # visible -> just run it
#   {"deliverable": "<path>", "grader_visible": false,
#    "readings": [{"reading": ..., "source": "<verbatim task quote>",
#                  "acceptance_test": "<cmd deciding acceptance under it>",
#                  "normalized_side": "reference|candidate|both|neither"}],
#    "hedge": {"shape": union|intersection|none, "output": ..., "evidence": ...,
#              "extras_direction": ignored|penalized},
#    "if_none": {"why_intersection_empty": ...,
#                "readings_rejected": [{"reading":..., "evidence":...}],
#                "bet_on": ..., "bet_loses_when": ..., "why_safe": ...}}
# Each reading's `source` must appear VERBATIM in the task text (grounding: the
# reading must come from the task, not invention). A union hedge is valid only
# under extras=ignored; intersection only under extras=penalized. If no hedge
# exists the agent must name the rejected readings WITH independent evidence.
_DELIVERABLE_READING_MISSING = """[VerificationGuard] Every step_done must carry the fixed `deliverable_reading` field.

State, on the record, whether THIS step emits a graded artifact and — when the
grader is NOT visible — the readings you produced against. This is mandatory on
EVERY step_done (a silent omission is impossible; `{"deliverable": "none"}` is
an explicit owned claim). One fixed shape:

  # this step emits no graded artifact:
  deliverable_reading={"deliverable": "none"}

  # this step emits a graded artifact and the grader IS visible — you can run it:
  deliverable_reading={"deliverable": "<path>", "grader_visible": true}

  # this step emits a graded artifact and the grader is NOT visible — you MUST
  # enumerate the readings the task admits, ground each in the task text, and
  # CONSTRUCT a hedge (or prove none exists):
  deliverable_reading={
    "deliverable": "<path>",
    "grader_visible": false,
    "readings": [{"reading": "<one defensible reading>",
                  "source": "<verbatim quote from the task text>",
                  "acceptance_test": "<command that decides acceptance under it>",
                  "normalized_side": "reference|candidate|both|neither"}],
    "hedge": {"shape": "union|intersection|none", "output": "<for union/intersection: the REAL output safe under EVERY reading; for none: None>",
              "evidence": "<for union/intersection: the REAL command + raw output proving it; for none: None>",
              "extras_direction": "ignored|penalized",
              "direction_evidence": "<for union/intersection, when any reading's normalized_side is not 'neither': {\"reference\": {\"command\": \"<normalize the REFERENCE, compare the RAW candidate>\", \"observed\": \"<the RAW output it produced>\"}, \"candidate\": {\"command\": \"<normalize the CANDIDATE, compare the RAW reference>\", \"observed\": \"<the RAW output it produced>\"}} — a DISTINCT observed one-sided comparison per direction; for none: omit>"},
    "if_none": {"why_intersection_empty": "<why no single output can satisfy all>",
                "readings_rejected": [{"reading": "<a reading you dropped>", "evidence": "<independent artifact that rejected it>"}],
                "bet_on": "<the reading you bet on>", "bet_loses_when": "<world-state where it loses>",
                "why_safe": "<why betting is safe>"}
  }

Missing field = malformed = BLOCKED. See the spec for the sub-rules."""

_DELIVERABLE_READING_MALFORMED = """[VerificationGuard] The `deliverable_reading` declaration is malformed.

`deliverable_reading` must be either `{"deliverable": "none"}` (explicit
not-applicable), or a structured object naming the graded artifact:
  • "deliverable": a real path/name, or the literal "none";
  • "grader_visible": required and must be a real boolean (true|false);
       - true  -> the analysis is skipped; run the grader instead.
       - false -> "readings" AND "hedge" become mandatory (see below).
  • "readings": non-empty list; every entry needs a real "reading", a real
       "source" (a VERBATIM quote from the task text — the reading must come
       from the task, not invention), a real "acceptance_test" (a command
       that decides whether an output is accepted under that reading), and a
       real "normalized_side" (reference|candidate|both|neither — WHICH SIDE
       the checker normalizes before comparing). Any set that asserts a
       normalization must include at least one ONE-SIDED reading
       (reference|candidate — normalize ONE side, compare the other RAW); a set
       that is all "neither" declares a pure byte/value-equality task and is
       admissible. A set mixing "neither" with "both" but omitting the one-sided
       case is AGNOSTIC to the output's form and cannot discriminate the
       candidate forms the checker accepts. The
       SAME reading (same "reading"+"source") must not appear twice with
       CONFLICTING normalized_side — one reading has one direction.
  • "hedge": object with "shape" one of union|intersection|none. For
       union/intersection: a real "output" + "extras_direction" (union valid only
       under "ignored"; intersection only under "penalized") + an OBSERVED
       "evidence" = {"command": "<what you ran>", "observed": "<the RAW output it
       produced>"} whose "observed" text is FOUND in this session's tool results.
       For "none": "if_none" is required.
  • "if_none": object with a real "why_intersection_empty", an OBSERVED
       "union_attempt" = {"command": "<the union you actually built/ran>",
       "observed": "<its RAW output>"} (you may NOT declare the union empty from
       introspection — you must have CONSTRUCTED the union output and RUN it), a
       NON-EMPTY "readings_rejected" list (each a real "reading" + a "source"
       quoting the task + an OBSERVED "evidence" {command, observed}), and real
       bet_on/bet_loses_when/why_safe.
Re-issue the step_done with a corrected `deliverable_reading` field."""

_READING_NOT_GROUNDED = """[VerificationGuard] A reading is NOT grounded in the task text.

The `source` of each reading must appear VERBATIM (whitespace/case-insensitive)
in the task statement. A reading whose source you cannot quote from the task is
not one of the task's readings — it is invention, and an invented reading cannot
be the basis for a hedge or a bet.

Re-issue with "source" set to an exact quote from the task text that supports the
reading (or drop the reading)."""

_HEDGE_UNPROVEN = """[VerificationGuard] The hedge is not PROVEN.

Under an invisible grader you must either CONSTRUCT a single output safe under
every reading, or prove none exists — by introspection you cannot.

  • shape "union"/"intersection": "output" must be REAL, "extras_direction" a
    real ignored|penalized (shape↔direction must match: union only under
    IGNORED, intersection under PENALIZED), and "evidence" must be an OBSERVED
    {command, observed} pair whose "observed" text the harness can FIND in this
    session's tool results — a hedge you ASSERTED but never ran is not a hedge.
  • shape "none": "if_none" must carry a real why_intersection_empty, an
    OBSERVED "union_attempt" (the union you CONSTRUCTED and RAN — an introspected
    "no hedge possible" is a hypothesis, and is refused until the union was
    actually built and its result seen), a NON-EMPTY readings_rejected list
    (each rejected reading carrying a "source" quote AND an OBSERVED evidence
    {command, observed}), and real bet_on/bet_loses_when/why_safe.

  • DIRECTION-OF-LOSS (why the union was rejected): "if_none" must declare a real
    "extras_direction" (ignored|penalized) — how the checker treats output it did
    NOT ask for — and, when the task text is readable, an
    "extras_direction_source" that quotes the clause VERBATIM. Discarding the
    union is only valid if the task penalizes EXTRA output: a clause that
    penalizes only INCORRECT VALUES does NOT penalize extra PRESENCE — a union
    whose values are all correct survives it, so it is NOT a reason to drop the
    union. Do not cite a value clause as if it were a presence clause; quote the
    clause that actually penalizes extra addresses, or the union stands.

Re-issue with a PROVEN hedge, or a complete "if_none" proving emptiness."""

_READING_DIRECTION_BLIND = """[VerificationGuard] The reading set is DIRECTION-BLIND.

Each reading declares `normalized_side` — WHICH SIDE the checker normalizes
before comparing (one of: reference, candidate, both, neither). A set that asserts
a normalization (any side other than "neither") but carries NO one-sided
direction cannot distinguish the candidate forms the checker actually accepts:
"both" normalizes BOTH sides (idempotent) and "neither" normalizes NEITHER
(trivial), so a byte-exact output and a canonical-form output pass them equally,
while the real checker may normalize only ONE side and compare it to the other
side RAW, rejecting the other form. This is exactly how a self-oracle
green-washes the wrong serialization: a set {neither, both} looks asymmetric but
contains NO comparison that can reject a form difference.

You MUST include at least one reading whose direction is ONE-SIDED — "reference"
(normalize the reference, compare the candidate RAW) or "candidate" (normalize
the candidate, compare the reference RAW) — UNLESS every reading is "neither"
(a pure byte/value-equality task with no normalization dimension). A one-sided
comparison is the only kind that DISCRIMINATES the output's serialization:
normalization is idempotent, so an output that passes a one-sided predicate is
the conservative branch, and one that fails it is the form your agnostic oracle
hid.

Re-issue with `normalized_side` on EVERY reading (real value in
reference|candidate|both|neither) and at least one reading using reference or
candidate (or make every reading "neither" if no normalization is involved).

This block is NOT released by an `_override_reason`: declaring the direction is
the ONLY exit. A prose reason does not substitute for the declaration — the
requirement is structural precisely so awareness of the right output form is not
waivable by an unactioned sentence."""

_HEDGE_DIRECTION_EVIDENCE = """[VerificationGuard] The hedge evidence is not DIRECTION-ALIGNED.

A hedge claims ONE output is safe under EVERY reading. But a single, un-tagged
probe cannot show that for a reading that asserts a NORMALIZATION: only a
ONE-SIDED comparison can DISCRIMINATE an output's form — normalize ONE side and
compare the OTHER side RAW. "both" (normalize both) is idempotent and "neither"
(raw == raw) is trivial, so a byte-preserving output passes them while the real
checker, normalizing only one side, rejects it. That is exactly how a
form-difference gets green-washed: the one comparison that can REJECT the shipped
form was never run.

You do not know which side the invisible checker normalizes. So, whenever you
SHIP a hedge (union/intersection) and your readings assert a normalization (any
reading whose normalized_side is not "neither"), the hedge must carry DISTINCT
OBSERVED evidence for EACH one-sided comparison — one probe reused for both
directions does not count:

  hedge={"shape": "union|intersection", "output": "...", "extras_direction": "...",
         "direction_evidence": {
           "reference": {"command": "<normalize the REFERENCE, compare the RAW candidate>",
                         "observed": "<the RAW output it produced>"},
           "candidate": {"command": "<normalize the CANDIDATE, compare the RAW reference>",
                         "observed": "<the RAW output it produced>"}}}

Each `observed` string must be text the harness can FIND in this session's tool
results (an asserted-but-never-run probe does not count), and the two must
differ. A reading set that is ALL "neither" declares a pure value-equality task
with no normalization dimension and is exempt. (A shape="none" bet ships no
hedge; its own rule — an OBSERVED union_attempt + a named bet_loses_when —
already forces the comparison, so it is not governed here.)

Re-issue the step_done with `direction_evidence` covering both one-sided
directions — run the comparison you have not run."""


def _normalize_for_grounding(s: str) -> str:
    """Whitespace-collapsed, lowercased form for verbatim grounding checks."""
    return " ".join(str(s).split()).lower()


def _task_text_groundable(task_text: str) -> bool:
    """True when the task text can ground a verbatim quote.

    Fail-open when the harness could not recover the task text ("(unavailable…")
    or when it was truncated at the cap ("…[truncated]"): the guard must not
    block a reading it simply cannot check against. Structural checks still run.
    """
    t = (task_text or "").strip()
    if not t:
        return False
    # The guard receives a COMPOSITE: the original task + "\n\n" + the last
    # user turn (agent.py wiring). Each marker therefore sits at a SEGMENT edge,
    # not necessarily at t's ends. Ground quotes only against a COMPLETE corpus:
    # an unavailable OR truncated SEGMENT means real text is missing, so fail
    # open (skip the grounding check) rather than emit a spurious
    # reading_not_grounded block against a partial corpus. A mere substring
    # occurrence inside real task text must NOT disable the check.
    segments = [s.strip() for s in t.split("\n\n") if s.strip()]
    if not segments:
        return False
    if any(s.startswith("(unavailable") for s in segments):
        return False
    if any(s.endswith("...[truncated]") for s in segments):
        return False
    return True


def _observed_corpus(messages, full_log=None) -> str:
    """Normalized concatenation of every tool-result text the session produced.

    This is the NON-FORGEABLE evidence source: a probe/hedge claim must paste an
    observation the harness can find here, so an asserted-but-never-run falsifier
    is rejected. `full_log` (the append-only audit log) is preferred when given —
    it survives eviction; `messages` (the live window) is the fallback. Returns
    None when NO source was supplied (unavailable -> caller skips the check);
    returns "" when a source existed but held no tool-result text.
    """
    src = full_log if full_log else messages
    if not src:
        return ""
    parts = []
    for m in src:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "tool" and isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    parts.append(str(b.get("content", "")))
        elif role == "user" and isinstance(content, str) and content.startswith("tool_results"):
            parts.append(content)
    return _normalize_for_grounding(" \n ".join(parts))


def _evidence_observed(ev, corpus) -> bool:
    """True when `ev` carries an observation actually present in the session.

    `ev` is a dict {"command": ..., "observed": ...}; the `observed` text must
    appear (whitespace/case-insensitive) inside `corpus` — the tool-result text
    the harness recorded. STRUCTURAL validity is checked first and always: a
    missing/mis-shaped evidence is False regardless of corpus. Only the
    corpus-MEMBERSHIP test fails open when `corpus is None` (the source was
    unavailable, so we cannot check — never block on what we cannot see).
    `corpus == ""` means the source existed but held nothing, so nothing can be
    observed and the claim fails. Returns False for any non-dict/missing shape.
    """
    if not isinstance(ev, dict):
        return False
    obs = ev.get("observed")
    if not _is_real_str(obs):
        return False
    n = _normalize_for_grounding(obs)
    if not n:
        return False
    if corpus is None:
        return True  # shape OK but we cannot verify membership -> fail-open
    return n in corpus


def _declares_graded_reading(dr) -> bool:
    """True when `dr` is a REAL (non-none) deliverable_reading declaration.

    The not-applicable opt-out (`{"deliverable": "none"}` or the string
    "none") is an explicit claim that there is NO graded artifact; it returns
    False here. Any other dict that named a deliverable is a real declaration.
    """
    if not isinstance(dr, dict):
        return False
    deliverable = dr.get("deliverable")
    if isinstance(deliverable, str) and deliverable.strip().lower() == "none":
        return False
    # A `grader_visible: True` claim short-circuits the readings requirement
    # upstream (_check_deliverable_reading_field returns before checking
    # readings/hedge), so it enumerates NO graded reading. It therefore must NOT
    # satisfy the completion arm -- otherwise the arm is opt-outable again via a
    # self-asserted visibility. A genuinely visible grader still closes through
    # the arm's override, with the reason on record.
    if dr.get("grader_visible") is True:
        return False
    return _is_real_str(deliverable)


def _acceptance_act_recorded(act, observed) -> bool:
    """True when a step_done's `acceptance_act` carries a REAL acceptance record.

    Two admissible shapes, BOTH requiring a real corpus-backed observation
    (structural first, then corpus-membership):
      * {"command": <str>, "observed": <str>} — an OBSERVED act: the acceptance
        was exercised and PASSED. The `observed` text must be found in the
        session's tool-result corpus (`_evidence_observed`), so a
        claimed-but-never-run probe does not count.
      * {"unobservable": <str>, "command": <str>, "observed": <str>} — the
        acceptance dimension could NOT be observed (hidden corpus / un-runnable
        oracle). This records the gap as INCONCLUSIVE rather than PASS. But the
        INCONCLUSIVE form costs an ATTEMPT just as the PASS form costs an
        OBSERVATION: the agent must show the probe it actually RAN and that
        probe's real recorded output. "I could not observe it" is admissible only
        when the agent can show it TRIED and how.

    A bare `{"unobservable": ...}` with no attempt is a DECLARATION, not a
    record, and is refused — otherwise declaring a dimension invisible is a free
    pass that reaches completion without ever confronting the acceptance
    dimension the self-check exists to force.

    Either form fails open ONLY when the corpus itself is unavailable
    (observed is None). Any other shape (missing/mis-typed/blank) is False —
    silence is not a record.
    """
    if not isinstance(act, dict):
        return False
    if not (_is_real_str(act.get("command")) and _is_real_str(act.get("observed"))):
        return False
    return _evidence_observed(act, observed)


def _oracle_exercised(act, observed, first_exercise=None) -> bool:
    """True when a recorded act shows the bound oracle's exercise was RUN.

    The framing-time oracle declaration names an oracle and its ``first_exercise``
    -- an intended ACTION. This predicate is the ACTION half: an ``oracle_act``
    row ({"command": C, "observed": O}) counts only when BOTH legs are real
    strings AND the ``observed`` text is found in the session's non-forgeable
    tool-result corpus (``_evidence_observed``). A declared-but-never-run exercise
    therefore does not count -- the same bar the OBSERVED ``acceptance_act`` form
    meets.

    When ``first_exercise`` (the declared command/check) is supplied, the recorded
    ``command`` must TOPICALLY RELATE to it -- share at least one significant
    token -- else any prior tool output ("ok", a plan-create banner) would latch
    the exercise without any oracle ever being run, making the label meaningless.
    This is a class-level relation (identifier/path-like token overlap), not a
    per-task rule. It fails open when either side carries no significant token
    (a very short command), so a legitimate exercise is not spuriously refused.

    STRUCTURAL validity is checked first and always. The corpus-membership test
    fails open ONLY when the corpus itself is unavailable (``observed is None``),
    so a task whose session log cannot be read is not spuriously blocked.
    """
    if not isinstance(act, dict):
        return False
    if not (_is_real_str(act.get("command")) and _is_real_str(act.get("observed"))):
        return False
    if _is_real_str(first_exercise):
        _cmd_toks = _significant_tokens(act["command"])
        _ex_toks = _significant_tokens(first_exercise)
        if _cmd_toks and _ex_toks and not (_cmd_toks & _ex_toks):
            return False
    return _evidence_observed(act, observed)


# A "significant" token for the oracle-exercise topical relation: an
# identifier/path-like run of >=4 characters (so `torch`, `pytest`, `tests/foo.py`,
# `import`), case-folded. Short glue ("in", "the", "-c") is ignored so it cannot
# manufacture a spurious match. Used ONLY to require the recorded command to
# relate to the declared exercise -- generic, not task-specific.
_TOPK_RE = re.compile(r"[a-z0-9_./-]{4,}")


def _significant_tokens(s: str) -> set:
    """Identifier/path-like tokens (>=4 chars, case-folded) in ``s``."""
    return set(_TOPK_RE.findall((s or "").lower()))
# what it TESTED and what it merely ASSERTED ----
# The failure this closes: a step can rest on premises that live only in the
# reasoning trace ("this is close enough", "assume the field is 4 bytes", "the
# counter can't overflow") and were adopted or vetoed without ever being tested
# against the world. Reasoning is not evidence; an untested premise treated as a
# conclusion is the silent assumption this gate surfaces. It is a class-level
# obligation (every step reasons under unproven premises), fixed and REQUIRED on
# every step_done, no marker/tag/regex. The declaration reuses the SAME
# falsifier machinery as the convention gate: a premise that AFFECTS the result
# must carry a real observation the harness can find; a premise declared
# non-affecting must say why.
_UNVERIFIED_ASSUMPTIONS_MISSING = """[VerificationGuard] Every step_done must carry the fixed `assumptions` field.

Reasoning is not evidence. You can see your whole context — your own reasoning
trace AND the tool results — so re-read it before this step closes, and split
what you TESTED against the world from what you merely REASONED and treated as
settled: a reading of a term you never checked, a "this is close enough", a
value taken as given, a candidate you dropped or adopted on suspicion, a
mechanism you assumed rather than observed. Those untested premises are exactly
the ones that sink a step — and they are invisible in your final answer, because
reasoning reads identically whether or not it was true. Surface them here.

This is mandatory on EVERY step_done; the field is ONE fixed shape:
  • the explicit not-applicable form, when NO load-bearing premise was left untested:
      assumptions={"unverified": "none"}
  • otherwise, one entry per untested premise that could move the result:
      assumptions={"unverified": [
        {"claim": "<the premise you leaned on without testing it>",
         "affects_result": "yes|no",
         "probe": "<command that tests it>",  "probe_output": "<raw output>",
         #   ^ REQUIRED when affects_result == "yes": an affecting premise must be
         #     TESTED, and the observed output must be found in your session log.
         "reason": "<why it cannot affect the result>"
         #   ^ REQUIRED when affects_result == "no": a non-affecting premise must
         #     say WHY, not just be waved off.
        }]}
An empty list is also a real declaration. A silent omission is not — re-issue
the step_done with this field."""

_UNVERIFIED_ASSUMPTIONS_MALFORMED = """[VerificationGuard] The `assumptions` declaration is malformed.

Write either the explicit not-applicable form (`{"unverified": "none"}` or the
string "none" or an empty/unverified-free list), or a list under "unverified"
of objects, each with a real string "claim" and "affects_result" in yes/no:
  • affects_result == "yes": you MUST supply a real "probe" AND "probe_output" —
    the premise moved the result, so it cannot ship untested;
  • affects_result == "no": you MUST supply a real "reason" — say WHY it cannot
    move the result.
A claim with neither is exactly the reasoning-treated-as-evidence this gate
exists to surface. Re-issue the step_done with a corrected `assumptions` field."""

_UNVERIFIED_ASSUMPTIONS_AFFECTS = """[VerificationGuard] An unverified premise was declared to AFFECT the result
but carries no real observation.

If a premise can move the step's result, it may not ship as reasoning alone —
test it and paste the raw output. `probe_output` must be text the harness can
find in your session's tool results (a claimed-but-never-run probe does not
count). Supply a real "probe" + "probe_output" — or, if the premise is
observable in principle but unobservable here, say so in the `reason` and mark
`affects_result` honestly; what is forbidden is asserting an affecting premise
with no test at all."""


def _check_unverified_assumptions_field(field, observed: str = None):
    """Unconditional structured gate for the fixed `assumptions` field.

    `field` is the raw value of ``tool_args['assumptions']``. Returns a
    GuardVerdict.block when the declaration is missing/malformed, when an
    affecting premise carries no observed test, or when a non-affecting premise
    carries no reason — else None. The explicit not-applicable forms ("none",
    {"unverified": "none"}, or an empty list) pass: that is a recorded decision,
    not a silent skip. This is a shape + corpus check; it does not judge whether
    the premise is truly non-affecting, only that the split is on the record.
    """
    if field is None:
        return GuardVerdict.block(
            message=_UNVERIFIED_ASSUMPTIONS_MISSING,
            reason="assumptions_declaration_missing",
            category="verification_required",
        )
    if isinstance(field, str):
        if field.strip().lower() in _CONVENTION_NO_VALUE:
            return None
        return GuardVerdict.block(
            message=_UNVERIFIED_ASSUMPTIONS_MALFORMED,
            reason="assumptions_string_not_none",
            category="verification_required",
        )
    if isinstance(field, list):
        items = field
    elif isinstance(field, dict):
        if "unverified" not in field:
            return GuardVerdict.block(
                message=_UNVERIFIED_ASSUMPTIONS_MALFORMED,
                reason="assumptions_unverified_key_missing",
                category="verification_required",
            )
        items = field.get("unverified")
        if items is None or (isinstance(items, str)
                             and items.strip().lower() in _CONVENTION_NO_VALUE):
            return None
    else:
        return GuardVerdict.block(
            message=_UNVERIFIED_ASSUMPTIONS_MALFORMED,
            reason="assumptions_not_dict_or_list",
            category="verification_required",
        )
    if not isinstance(items, list):
        return GuardVerdict.block(
            message=_UNVERIFIED_ASSUMPTIONS_MALFORMED,
            reason="assumptions_unverified_not_list",
            category="verification_required",
        )
    for it in items:
        if not isinstance(it, dict) or not _is_real_str(it.get("claim")):
            return GuardVerdict.block(
                message=_UNVERIFIED_ASSUMPTIONS_MALFORMED,
                reason="assumptions_item_malformed",
                category="verification_required",
            )
        affects = str(it.get("affects_result") or "").strip().lower()
        if affects not in ("yes", "no"):
            return GuardVerdict.block(
                message=_UNVERIFIED_ASSUMPTIONS_MALFORMED,
                reason="assumptions_affects_result_missing",
                category="verification_required",
            )
        if affects == "yes":
            if not _evidence_observed(
                    {"command": it.get("probe"), "observed": it.get("probe_output")},
                    observed):
                return GuardVerdict.block(
                    message=_UNVERIFIED_ASSUMPTIONS_AFFECTS,
                    reason="assumptions_affecting_without_test",
                    category="verification_required",
                )
        else:
            if not _is_real_str(it.get("reason")):
                return GuardVerdict.block(
                    message=_UNVERIFIED_ASSUMPTIONS_MALFORMED,
                    reason="assumptions_nonaffecting_without_reason",
                    category="verification_required",
                )
    return None


def _check_deliverable_reading_field(dr, task_text: str = "", observed: str = None):
    """Unconditional structured gate for the fixed `deliverable_reading` field.

    `dr` is the raw value of ``tool_args['deliverable_reading']``. Returns a
    GuardVerdict.block when missing/malformed, else None. Mechanical only: no
    regex, no marker; every step_done carries the field. The grading READING is
    the agent's to CONSTRUCT — this checks the construction exists and is
    grounded, not that it is semantically right (honest ceiling).
    """
    if dr is None:
        return GuardVerdict.block(
            message=_DELIVERABLE_READING_MISSING,
            reason="deliverable_reading_missing",
            category="verification_required",
        overridable=False,
        )
    if isinstance(dr, str):
        # The string form is the explicit not-applicable token — ONLY the
        # literal "none" (any case). A blank/"n/a"/"null" string is NOT an
        # owned declaration; it must NAME the deliverable or be malformed.
        if dr.strip().lower() == "none":
            return None
        return GuardVerdict.block(
            message=_DELIVERABLE_READING_MALFORMED,
            reason="dr_string_not_none",
            category="verification_required",
        overridable=False,
        )
    if not isinstance(dr, dict):
        return GuardVerdict.block(
            message=_DELIVERABLE_READING_MALFORMED,
            reason="dr_not_dict",
            category="verification_required",
        overridable=False,
        )
    deliverable = dr.get("deliverable")
    if isinstance(deliverable, str) and deliverable.strip().lower() == "none":
        return None
    if not _is_real_str(deliverable):
        return GuardVerdict.block(
            message=_DELIVERABLE_READING_MALFORMED,
            reason="dr_deliverable_missing",
            category="verification_required",
        overridable=False,
        )
    gv = dr.get("grader_visible")
    if not isinstance(gv, bool):
        return GuardVerdict.block(
            message=_DELIVERABLE_READING_MALFORMED,
            reason="dr_grader_visible_missing",
            category="verification_required",
        overridable=False,
        )
    if gv:
        return None
    readings = dr.get("readings")
    if not isinstance(readings, list) or not readings:
        return GuardVerdict.block(
            message=_DELIVERABLE_READING_MALFORMED,
            reason="dr_readings_empty",
            category="verification_required",
        overridable=False,
        )
    groundable = _task_text_groundable(task_text)
    has_onesided_direction = False
    all_neither = True
    seen_direction = {}
    for r in readings:
        if not isinstance(r, dict):
            return GuardVerdict.block(
                message=_DELIVERABLE_READING_MALFORMED,
                reason="dr_reading_not_dict",
                category="verification_required",
            overridable=False,
            )
        if not (_is_real_str(r.get("reading"))
                and _is_real_str(r.get("source"))
                and _is_real_str(r.get("acceptance_test"))):
            return GuardVerdict.block(
                message=_DELIVERABLE_READING_MALFORMED,
                reason="dr_reading_field_missing",
                category="verification_required",
            overridable=False,
            )
        if groundable and (
            _normalize_for_grounding(r.get("source"))
            not in _normalize_for_grounding(task_text)
        ):
            return GuardVerdict.block(
                message=_READING_NOT_GROUNDED,
                reason="reading_source_not_grounded",
                category="verification_required",
            overridable=False,
            )
        # Direction-of-comparison: WHICH SIDE the checker normalizes. A set that
        # asserts a normalization (any side other than "neither") must carry at
        # least one ONE-SIDED direction (reference|candidate) -- a comparison
        # that normalizes ONE side and compares it to the OTHER side RAW. Only a
        # one-sided comparison can discriminate a candidate-form (serialization)
        # difference: "both" (normalize == normalize) is idempotent and "neither"
        # (raw == raw) is trivial, so a byte-preserving output passes both while
        # the real checker may compare normalize(reference) against the RAW
        # candidate and reject it. A set pairing "neither" with "both" and
        # omitting the one-sided case is the direction-blind set that green-washes
        # the serialization choice. A set that is ALL "neither" declares no
        # normalization dimension and is left admissible.
        raw_side = r.get("normalized_side")
        if raw_side is None:
            # A missing direction on ANY reading is the decisive awareness gap:
            # released only by the concrete corrective action (declare a real
            # normalized_side on every reading), never by a prose override. See
            # the class-level note on the direction blocks below.
            return GuardVerdict.block(
                message=_READING_DIRECTION_BLIND,
                reason="reading_direction_not_str",
                category="verification_required",
                overridable=False,
            )
        if not _is_real_str(raw_side):
            return GuardVerdict.block(
                message=_DELIVERABLE_READING_MALFORMED,
                reason="dr_normalized_side_missing",
                category="verification_required",
            overridable=False,
            )
        side = raw_side.strip().lower()
        if side not in ("reference", "candidate", "both", "neither"):
            return GuardVerdict.block(
                message=_READING_DIRECTION_BLIND,
                reason="reading_direction_invalid",
                category="verification_required",
                overridable=False,
            )
        # One reading has ONE direction: the same (reading, source) may not be
        # re-declared with a different normalized_side. Without this, an agent
        # could pad an all-"both" set with a spurious duplicate carrying
        # "reference" to satisfy the asymmetry requirement while the real
        # readings stay symmetric — the deception this gate exists to stop.
        key = (
            _normalize_for_grounding(r.get("reading")),
            _normalize_for_grounding(r.get("source")),
        )
        prev = seen_direction.get(key)
        if prev is not None and prev != side:
            return GuardVerdict.block(
                message=_DELIVERABLE_READING_MALFORMED,
                reason="reading_direction_conflict",
                category="verification_required",
            overridable=False,
            )
        seen_direction[key] = side
        if side in ("reference", "candidate"):
            has_onesided_direction = True
        if side != "neither":
            all_neither = False
    # A set that is ALL "neither" declares a pure byte/value-equality task with
    # no normalization dimension -- legitimate (e.g. an exact-JSON grader). Any
    # OTHER set asserts a normalization is in play, so it MUST enumerate at
    # least one ONE-SIDED direction (reference|candidate): "both" (normalize both
    # sides) is IDEMPOTENT and cannot reject a wrong output form by itself, and a
    # set that pairs "neither" with "both" while omitting the one-sided case is
    # direction-blind -- exactly how a serialization choice is green-washed.
    if not has_onesided_direction and not all_neither:
        return GuardVerdict.block(
            message=_READING_DIRECTION_BLIND,
            reason="reading_direction_uniform",
            category="verification_required",
            overridable=False,
        )
    hedge = dr.get("hedge")
    if not isinstance(hedge, dict):
        return GuardVerdict.block(
            message=_DELIVERABLE_READING_MALFORMED,
            reason="dr_hedge_missing",
            category="verification_required",
        overridable=False,
        )
    shape = str(hedge.get("shape") or "").strip().lower()
    if shape in ("union", "intersection"):
        if not _is_real_str(hedge.get("output")):
            return GuardVerdict.block(
                message=_HEDGE_UNPROVEN,
                reason="hedge_output_missing",
                category="verification_required",
            overridable=False,
            )
        if not _evidence_observed(hedge.get("evidence"), observed):
            return GuardVerdict.block(
                message=_HEDGE_UNPROVEN,
                reason="hedge_evidence_unobserved",
                category="verification_required",
            overridable=False,
            )
        extras = str(hedge.get("extras_direction") or "").strip().lower()
        if extras not in ("ignored", "penalized"):
            return GuardVerdict.block(
                message=_HEDGE_UNPROVEN,
                reason="hedge_extras_direction_invalid",
                category="verification_required",
            overridable=False,
            )
        if shape == "union" and extras == "penalized":
            return GuardVerdict.block(
                message=_HEDGE_UNPROVEN,
                reason="hedge_shape_mismatch",
                category="verification_required",
            overridable=False,
            )
        # Direction-aligned evidence. A hedge for a normalization-bearing reading
        # set (some reading not "neither") must carry an OBSERVED one-sided
        # comparison for EACH direction, so the side the invisible checker
        # normalizes is RUN, not asserted. An un-tagged probe cannot discriminate
        # a form difference: "both" is idempotent and "neither" trivial, so both
        # green-wash a byte-preserving output. See _HEDGE_DIRECTION_EVIDENCE.
        if not all_neither:
            de = hedge.get("direction_evidence")
            if not isinstance(de, dict):
                return GuardVerdict.block(
                    message=_HEDGE_DIRECTION_EVIDENCE,
                    reason="hedge_direction_evidence_not_dict",
                    category="verification_required",
                overridable=False,
                )
            seen_obs = set()
            for dir_name in ("reference", "candidate"):
                ev = de.get(dir_name)
                # Each direction must be a real {command, observed} pair whose
                # observed is present in the session — AND the two directions must
                # be DISTINCT. One probe reused for both is the agnostic tag the
                # gate exists to refuse: a single un-tagged comparison cannot show
                # the SHIPPED form survives the side the checker normalizes.
                if (not isinstance(ev, dict)
                        or not _is_real_str(ev.get("command"))
                        or not _evidence_observed(ev, observed)):
                    return GuardVerdict.block(
                        message=_HEDGE_DIRECTION_EVIDENCE,
                        reason="hedge_direction_evidence_malformed",
                        category="verification_required",
                    overridable=False,
                    )
                key = _normalize_for_grounding(ev.get("observed"))
                if key in seen_obs:
                    return GuardVerdict.block(
                        message=_HEDGE_DIRECTION_EVIDENCE,
                        reason="hedge_direction_evidence_not_distinct",
                        category="verification_required",
                    overridable=False,
                    )
                seen_obs.add(key)
    elif shape == "none":
        if_none = dr.get("if_none")
        if not isinstance(if_none, dict):
            return GuardVerdict.block(
                message=_HEDGE_UNPROVEN,
                reason="hedge_if_none_missing",
                category="verification_required",
            overridable=False,
            )
        if not _is_real_str(if_none.get("why_intersection_empty")):
            return GuardVerdict.block(
                message=_HEDGE_UNPROVEN,
                reason="hedge_why_intersection_empty_missing",
                category="verification_required",
            overridable=False,
            )
        # An introspected "no hedge possible" is a HYPOTHESIS, not a finding:
        # to claim the union is EMPTY the agent must have CONSTRUCTED the union
        # output and RUN it — `union_attempt` must carry an observation present
        # in the session's real tool-result stream. This refuses the retry13
        # reasoning ("can't hedge; I'll take the principled reading") until the
        # hedge was really tried and its result seen.
        if not _evidence_observed(if_none.get("union_attempt"), observed):
            return GuardVerdict.block(
                message=_HEDGE_UNPROVEN,
                reason="hedge_union_unattempted",
                category="verification_required",
            overridable=False,
            )
        # Direction-of-loss: rejecting the built union is valid only if the task
        # penalizes EXTRA output. A clause that penalizes only INCORRECT VALUES
        # does not penalize extra PRESENCE — a union whose values are all correct
        # is safe under it, so discarding that union on a value-clause throws away
        # the only conservative branch. The direction is a claim about the checker
        # and must be declared AND, when the task text is available, grounded in a
        # verbatim quote — an ungrounded direction is how a fabricated "extras are
        # penalized" silently kills the union. (The message states the rule; this
        # is the structural obligation: name the direction, quote its clause.)
        xdir = str(if_none.get("extras_direction") or "").strip().lower()
        if xdir not in ("ignored", "penalized"):
            return GuardVerdict.block(
                message=_HEDGE_UNPROVEN,
                reason="hedge_extras_direction_undeclared",
                category="verification_required",
            overridable=False,
            )
        if groundable:
            if not _is_real_str(if_none.get("extras_direction_source")):
                return GuardVerdict.block(
                    message=_HEDGE_UNPROVEN,
                    reason="hedge_extras_direction_unquoted",
                    category="verification_required",
                overridable=False,
                )
            if (
                _normalize_for_grounding(if_none.get("extras_direction_source"))
                not in _normalize_for_grounding(task_text)
            ):
                return GuardVerdict.block(
                    message=_READING_NOT_GROUNDED,
                    reason="extras_direction_source_not_grounded",
                    category="verification_required",
                overridable=False,
                )
        rej = if_none.get("readings_rejected")
        if not isinstance(rej, list) or not rej:
            return GuardVerdict.block(
                message=_HEDGE_UNPROVEN,
                reason="hedge_readings_rejected_empty",
                category="verification_required",
            overridable=False,
            )
        for e in rej:
            if not (isinstance(e, dict)
                    and _is_real_str(e.get("reading"))
                    and _is_real_str(e.get("source"))):
                return GuardVerdict.block(
                    message=_HEDGE_UNPROVEN,
                    reason="hedge_rejection_entry_malformed",
                    category="verification_required",
                overridable=False,
                )
            # A rejected candidate's "evidence" must be an OBSERVED artifact,
            # not a sentence: a fabricated rejection cannot manufacture an empty
            # intersection if the rejection itself had to be run.
            if not _evidence_observed(e.get("evidence"), observed):
                return GuardVerdict.block(
                    message=_HEDGE_UNPROVEN,
                    reason="hedge_rejection_unobserved",
                    category="verification_required",
                overridable=False,
                )
            # A rejected candidate carries the SAME provenance requirement as
            # an accepted reading: its source must quote the task verbatim, so a
            # fabricated rejection cannot manufacture an empty intersection.
            if groundable and (
                _normalize_for_grounding(e.get("source"))
                not in _normalize_for_grounding(task_text)
            ):
                return GuardVerdict.block(
                    message=_READING_NOT_GROUNDED,
                    reason="rejected_source_not_grounded",
                    category="verification_required",
                overridable=False,
                )
        for k in ("bet_on", "bet_loses_when", "why_safe"):
            if not _is_real_str(if_none.get(k)):
                return GuardVerdict.block(
                    message=_HEDGE_UNPROVEN,
                    reason="hedge_bet_forfeit_missing",
                    category="verification_required",
                overridable=False,
                )
    else:
        return GuardVerdict.block(
            message=_DELIVERABLE_READING_MALFORMED,
            reason="dr_hedge_shape_invalid",
            category="verification_required",
        overridable=False,
        )
    return None


class VerificationGuard(Guard):
    """Requires verification evidence when marking steps complete.
    
    Key design:
    - Only blocks plan_update(action="step_done"), other tool calls unaffected
    - Two modes:
      * Step has acceptance → must provide verification=["..."]
      * Step has no acceptance → must provide _override_reason="..."
    - LLM can freely execute verification operations after being blocked
    - Does not check verification/override_reason content
    
    Also injects a reminder after hard_reset recovery.
    """
    
    name = "verification"
    priority = 55
    
    def __init__(self, plan=None, proposals=None, task_text_fn=None,
                 full_log_fn=None):
        self._plan = plan
        self._proposals = proposals
        # Callable returning the append-only full message log (the eviction-proof
        # source of real tool-result text for observed-evidence checks). None =>
        # fall back to the live window (ctx.messages).
        self._full_log_fn = full_log_fn
        # Callable returning the session's original task text verbatim (the
        # grounding source for a deliverable_reading's `source` quotes). None or
        # an unavailable/truncated result => grounding is skipped (fail-open).
        self._task_text_fn = task_text_fn
        self._post_recovery = False
        self._recovery_reminded = False
        self._acceptance_guidance_given = False
        self._step_done_recheck_reminded = False
        # Per-RUN: True once a REAL (non-none) deliverable_reading was declared
        # and passed the gate. The completion-boundary arm reads it to force a
        # graded-reading declaration before [TASK_COMPLETE] when none was ever
        # made — closing the opt-out (`{"deliverable": "none"}`) bypass.
        self._graded_reading_seen = False
        # Per-RUN: True once a step_done recorded a REAL acceptance act for the
        # completion boundary — EITHER an observed command+output (found in this
        # session's tool-result corpus) OR an explicit `unobservable` declaration.
        # The completion arm reads it to force the agent to confront whether it
        # OBSERVED its deliverable's acceptance before [TASK_COMPLETE] — the gap
        # R7 exposed (a reading declared on the wrong dimension still shipped).
        self._acceptance_observed = False
        # Per-RUN: True once a step_done recorded a REAL exercise of the oracle
        # bound at framing time (`oracle_act` with a corpus-backed command+output).
        # The completion arm reads it: a bound oracle that was named but never RUN
        # is the declaration->action hole this closes -- framing forces the agent
        # to NAME an oracle, this forces it to LOOK through it at least once.
        self._oracle_exercised_seen = False
        # Per-RUN: the oracle declaration OBSERVED at plan_create time, latched
        # independently of the active plan. Why a latch, not just the persisted
        # plan: the tool completion route `plan_update(action="complete")` clears
        # the active pointer BEFORE the text `[TASK_COMPLETE]` is consulted, so a
        # read of get_active() at the boundary sees NO oracle and the obligation
        # silently disarms. A re-frame (a second plan_create with no oracle) or a
        # deactivate() has the same effect. Capturing the declaration here keeps
        # the obligation alive for the whole run: once an oracle was BOUND, it must
        # be EXERCISED before the task may close, however the plan is later
        # mutated. Empty dict == no oracle bound this run.
        self._oracle_bound = {}

    def reset_turn(self):
        """Reset per-turn state on a new user message — clears the step_done
        re-check latch so a fresh task starts with the premise re-check armed.
        """
        self._step_done_recheck_reminded = False


    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # The step_done pre-mortem and the independent-reviewer demand now
        # live in the plan's fixed review step (see
        # tools/plan_create.py _REVIEW_STEP_ACCEPTANCE). VerificationGuard
        # keeps only the plan step_done evidence gate in check_pre below.
        return None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Capture the oracle declaration the MOMENT a plan binds it, into a
        # per-RUN latch. The completion boundary must be able to see an oracle
        # bound at framing time even after the plan is closed or replaced: the
        # tool route `plan_update(action="complete")` clears the active pointer
        # (plan.py completes -> _clear_active), and a re-frame replaces it, so a
        # read of get_active() at the boundary alone would MISS the oracle and
        # silently disarm the obligation. Latched here (idempotent; first binding
        # wins for the run), the obligation survives any later plan mutation.
        if ctx.tool_name == "plan_create":
            _decl = (ctx.tool_args or {}).get("oracle")
            if isinstance(_decl, dict) and _decl:
                self._oracle_bound = _decl

        # The pure-text [TASK_COMPLETE] finish-path gate stack and the
        # plan_update(action="complete") completion-gate cascade previously
        # lived here. They are SUPERSEDED by the plan's fixed review step plus
        # the retained plan step_done evidence gate (Timing 1 below), so the
        # terminal completion gates are removed.

        # Completion-boundary arm: a text-only [TASK_COMPLETE] must obey the same
        # closure rule plan_update(action="complete") already enforces
        # (plan.py: "Cannot complete plan: N step(s) not finished"). Without this
        # the two completion routes are inconsistent: an agent under deadline
        # pressure can skip every plan_update entirely and declare done, so the
        # fixed convention / deliverable_reading record — the declaration of the
        # decisive reading it shipped under — is never written. The gate is
        # reachable two ways, matching the existing plan_update route: either
        # close the steps (each step_done then runs Timing 1's fixed-field check)
        # or override with a claim; both make the reason explicit.
        #
        # Fired ONLY on the completion-path consultation: tool_name=="" AND the
        # sentinel came from THIS iteration's fresh LLM output
        # (llm_responded=True). The top-of-loop check_pre also runs with
        # tool_name=="" every iteration to catch stale completions out of
        # history; requiring llm_responded keeps THIS arm off that stale scan.
        if ctx.tool_name == "" and ctx.llm_responded:
            # Only a [TASK_COMPLETE] claims the work is DONE; [NEED_USER_INPUT]
            # is the opposite (the agent is stuck and asking) and must never be
            # blocked by a closure rule. Mirror the kernel's EXACT completion
            # predicate (kernel.py:378: `trailing == "[TASK_COMPLETE]" or
            # has_override_completion(text)`) — a text-only completion has two
            # shapes, and testing only the trailing one would leak the override
            # form (marker + inline `_override_reason:` line, where text follows
            # the marker so trailing_sentinel returns None).
            _text = (ctx.assistant_text or "").rstrip()
            _is_task_complete = (
                trailing_sentinel(_text) == "[TASK_COMPLETE]"
                or has_override_completion(_text)
            )
            if _is_task_complete:
                unfinished = self._unfinished_steps()
                # Release through the SAME override bar the tool route uses
                # (registry -> owner.accept_override; base default = reason
                # longer than 5 chars), so the text and tool completion routes
                # enforce an identically strong bar rather than any non-empty
                # string.
                if unfinished and not self.accept_override(ctx.override_reason, ctx):
                    return GuardVerdict.block(
                        message=_COMPLETION_WITH_UNFINISHED_STEPS.format(
                            steps=", ".join(str(s) for s in unfinished),
                        ),
                        reason="completion_with_unfinished_steps",
                        category="verification_required",
                    )
                # Graded-reading obligation at the boundary: the deliverable_read-
                # ing gate is otherwise OPT-OUTABLE — `{"deliverable": "none"}`
                # satisfies it without any reading enumerated, so an agent can
                # close every step on an ungrounded opt-out and never confront
                # the reading the checker applies. When the task text is retriev-
                # able (groundable) and no REAL reading was declared this run,
                # require one before the task may be declared done. NON-overridable
                # (M2): the exit is the ACTION of declaring a real
                # deliverable_reading on a step_done, always available.
                if (
                    not self._graded_reading_seen
                    and _task_text_groundable(self._task_text())
                ):
                    return GuardVerdict.block(
                        message=_COMPLETION_NO_GRADED_READING,
                        reason="completion_without_graded_reading",
                        category="verification_required",
                        overridable=False,
                    )
                # Acceptance-observation obligation: a declared reading is not an
                # OBSERVATION of the acceptance. When the task text is groundable
                # and the agent never recorded an observed acceptance act (a real
                # command+output, corpus-checked) or an explicit `unobservable`
                # declaration, it must confront that gap before declaring done.
                # NON-overridable (M2): the exit is the ACTION of recording a
                # real `acceptance_act` (a run command + its observed output),
                # always available. This is the R8 lever — semantic, no case list.
                if (
                    not self._acceptance_observed
                    and _task_text_groundable(self._task_text())
                ):
                    return GuardVerdict.block(
                        message=_COMPLETION_NO_ACCEPTANCE_OBSERVED,
                        reason="completion_without_acceptance_observation",
                        category="verification_required",
                        overridable=False,
                    )
                # Oracle-exercise obligation: at framing the agent BOUND an
                # independent oracle and wrote its FIRST runnable exercise — an
                # intended ACTION. A declaration is not a look. When an oracle was
                # bound this run and no corpus-backed `oracle_act` ever recorded
                # that the exercise was RUN, the agent must exercise it before
                # declaring done. NON-overridable (M2): the exit is the ACTION of
                # running the bound oracle's first exercise (the agent authors it,
                # so a runnable one is always available) and recording `oracle_act`.
                # Fires ONLY when an oracle was actually bound.
                if (
                    self._bound_oracle()
                    and not self._oracle_exercised_seen
                ):
                    return GuardVerdict.block(
                        message=_COMPLETION_ORACLE_NOT_EXERCISED,
                        reason="completion_without_oracle_exercise",
                        category="verification_required",
                        overridable=False,
                    )
                # Oracle-BIND obligation: the frame step mandates naming the oracle
                # that will decide the task and classifying its EXTERNAL premise.
                # Observed failure mode (0/76 plans in a full TB2.1 run ever bind):
                # agents satisfy that mandate in PROSE on the frame step's
                # verification note and never pass plan_create(oracle=...), so the
                # bind->exercise->completion chain above is dead by construction and
                # the agent closes on a purely self-authored verification. When the
                # task text is groundable and NO oracle was ever bound this run, the
                # agent must confront that gap before declaring done: bind an
                # external-premise oracle (and run its first exercise) OR record an
                # explicit record of the gap. NON-overridable (M2): the exit is the
                # ACTION of binding an oracle via plan_create(oracle={...}) (any
                # non-empty dict latches; plan_create pauses the old plan and
                # creates a new one) and then running its first exercise.
                if (
                    not self._bound_oracle()
                    and _task_text_groundable(self._task_text())
                ):
                    return GuardVerdict.block(
                        message=_COMPLETION_ORACLE_NOT_BOUND,
                        reason="completion_without_oracle_bound",
                        category="verification_required",
                        overridable=False,
                    )
            return None

        # Timing 0: acceptance is frozen once a step leaves "pending".
        # Once implementation has begun, letting the criteria change means the
        # standard can be contaminated by the outcome — verification would no
        # longer be independent of the work. Overridable: if the original
        # criteria were genuinely wrong, the agent must say so explicitly.
        if ctx.tool_name == "plan_update":
            if ctx.tool_args.get("action") == "update_acceptance":
                step_id = ctx.tool_args.get("step_id")
                status = self._step_status(step_id)
                if status is not None and status != "pending":
                    return GuardVerdict.block(
                        message=_ACCEPTANCE_FROZEN.format(step_id=step_id, status=status),
                        reason="acceptance_frozen_after_implementation_started",
                        category="acceptance_frozen",
                    )
                # Still pending — criteria may be edited freely.
                return None

        # Timing 1: step_done requires verification evidence
        if ctx.tool_name == "plan_update":
            action = ctx.tool_args.get("action")
            
            # Only check on step_done, other actions (step_doing/add_steps) pass through
            if action == "step_done":
                step_id = ctx.tool_args.get("step_id")
                verification = ctx.tool_args.get("verification", [])
                override_reason = ctx.override_reason.strip()  # Use ctx.override_reason, not tool_args

                # Timing 1a: premise re-check, forced on the FIRST step_done of the
                # run. Symmetric with the plan_create qualifier block (check_pre):
                # BLOCKS once, any override_reason releases it, content is not
                # checked. Its purpose is different from the verification/override
                # checks below — those ensure evidence *exists*; this forces the
                # agent to stop and re-examine the premises it operated under before
                # any evidence is accepted. The plan-time qualifier block cannot
                # reach this failure: a qualifier can be captured correctly into the
                # plan yet be quietly re-interpreted during execution ("close
                # enough", "this value is current", a convenient reading of a term),
                # and that re-interpretation only surfaces at finalize time. Firing
                # once (not every step) keeps it a genuine checkpoint, not noise.
                # Fires ahead of the Mode 1/2 evidence checks so the agent re-reads
                # its premises before writing the verification it will be judged on.
                if not self._step_done_recheck_reminded:
                    if not override_reason:
                        return GuardVerdict.block(
                            message=_STEP_DONE_RECHECK_REMINDER,
                            reason="step_done_premise_recheck",
                            category="verification_required",
                        )
                    # Override provided → agent paused to re-check. Release for good,
                    # then fall through to the Mode 1/2 evidence checks below: the
                    # re-check does not exempt the step from still carrying evidence.
                    self._step_done_recheck_reminded = True
                
                # Get step's acceptance criteria if plan is available
                acceptance = self._step_acceptance(step_id)
                
                # Mode 1: Step has acceptance → require verification list
                if acceptance:
                    if not verification:
                        msg = _VERIFICATION_REQUIRED_WITH_ACCEPTANCE.format(
                            acceptance="\n".join(f"  • {a}" for a in acceptance)
                        )
                        return GuardVerdict.block(
                            message=msg,
                            reason="step_done_with_acceptance_no_verification",
                            category="verification_required"
                        )
                # Mode 2: No acceptance (simple step) → require override_reason
                else:
                    if not override_reason:
                        return GuardVerdict.block(
                            message=_VERIFICATION_REQUIRED_NO_ACCEPTANCE,
                            reason="step_done_no_verification",
                            category="verification_required"
                        )

                # The convention declaration is a FIXED, REQUIRED field on EVERY
                # step_done — no tag, no regex, no opt-in. "Not applicable" is
                # itself an explicit declaration ({"name": "none"} or "none"),
                # so a silent omission is impossible. Runs AFTER the mode checks
                # so the agent first supplies its evidence, then the declaration.
                bad = _check_convention_field(ctx.tool_args.get("convention"))
                if bad is not None:
                    return bad
                dr = ctx.tool_args.get("deliverable_reading")
                bad = _check_deliverable_reading_field(
                    dr, self._task_text(), self._observed(ctx))
                if bad is not None:
                    return bad
                # The unverified-assumption audit is a FIXED, REQUIRED field on
                # EVERY step_done, parallel to convention/deliverable_reading:
                # reasoning is not evidence, so any premise left untested must be
                # surfaced — an affecting one tested, a non-affecting one
                # justified. "none" is an explicit owned claim.
                bad = _check_unverified_assumptions_field(
                    ctx.tool_args.get("assumptions"), self._observed(ctx))
                if bad is not None:
                    return bad
                if _declares_graded_reading(dr):
                    self._graded_reading_seen = True
                if _acceptance_act_recorded(
                        ctx.tool_args.get("acceptance_act"), self._observed(ctx)):
                    self._acceptance_observed = True
                if _oracle_exercised(
                        ctx.tool_args.get("oracle_act"), self._observed(ctx),
                        self._bound_oracle().get("first_exercise")):
                    self._oracle_exercised_seen = True
                return None

            # Timing 1b: batch action marking any step done bypasses the per-step
            # step_done gate above (action=="batch", not "step_done"). A batch done
            # commits the same claim; without this it is a silent hole through which
            # every verification check is skipped. Require _override_reason when any
            # update sets status=="done". Non-done batches (doing/skipped) pass freely.
            if action == "batch":
                updates = ctx.tool_args.get("updates") or []
                has_done = any(
                    isinstance(u, dict) and u.get("status") == "done"
                    for u in updates
                )
                if has_done and not ctx.override_reason.strip():
                    return GuardVerdict.block(
                        message=_BATCH_STEP_DONE_BLOCK,
                        reason="batch_step_done_no_verification",
                        category="verification_required",
                    )
                # Per-item evidence (Mode 1 parity): a batch item that flips a
                # step to "done" commits the SAME claim as a single step_done, so
                # if that step HAS acceptance criteria it must carry the same
                # non-empty `verification` evidence. Without this, the batch path
                # skips Mode 1 entirely — the hole the evidence check exists to
                # close. Runs BEFORE the convention check so the agent first
                # supplies evidence, then the declaration (mirrors step_done).
                for u in updates:
                    if not (isinstance(u, dict) and u.get("status") == "done"):
                        continue
                    sid = u.get("step_id")
                    acceptance = self._step_acceptance(sid)
                    item_verification = u.get("verification") or []
                    if acceptance and not item_verification:
                        return GuardVerdict.block(
                            message=_BATCH_STEP_DONE_NO_EVIDENCE_BLOCK.format(
                                step_id=sid,
                                acceptance="\n".join(f"  • {a}" for a in acceptance),
                            ),
                            reason="batch_step_done_with_acceptance_no_verification",
                            category="verification_required",
                        )
                # The convention declaration is a FIXED, REQUIRED field on ANY
                # claim of "done" — including a batch that marks a step done. A
                # batch done commits the same claim as a single step_done, so the
                # same declaration is mandatory here; otherwise the batch path is
                # the silent hole the convention field exists to close.
                if has_done:
                    # A batch item that flips a step to "done" commits the SAME
                    # claim as a single step_done — and the natural place to write
                    # that claim is ON the item. Read each done item's OWN
                    # declaration first, falling back to the top-level one, so a
                    # declaration nested in the item cannot EVADE the fixed-field
                    # checks (the hole this closes: the agent wrote a real
                    # convention/deliverable_reading inside updates[i], the guard
                    # read only the top level, saw None, and silently skipped the
                    # direction gate — the one lever that forces the agent to face
                    # its grader's oracle).
                    for u in updates:
                        if not (isinstance(u, dict) and u.get("status") == "done"):
                            continue
                        conv = u.get("convention")
                        if conv is None:
                            conv = ctx.tool_args.get("convention")
                        bad = _check_convention_field(conv)
                        if bad is not None:
                            return bad
                        dr = u.get("deliverable_reading")
                        if dr is None:
                            dr = ctx.tool_args.get("deliverable_reading")
                        bad = _check_deliverable_reading_field(
                            dr, self._task_text(), self._observed(ctx))
                        if bad is not None:
                            return bad
                        asm = u.get("assumptions")
                        if asm is None:
                            asm = ctx.tool_args.get("assumptions")
                        bad = _check_unverified_assumptions_field(
                            asm, self._observed(ctx))
                        if bad is not None:
                            return bad
                        if _declares_graded_reading(dr):
                            self._graded_reading_seen = True
                        act = u.get("acceptance_act")
                        if act is None:
                            act = ctx.tool_args.get("acceptance_act")
                        if _acceptance_act_recorded(act, self._observed(ctx)):
                            self._acceptance_observed = True
                        oact = u.get("oracle_act")
                        if oact is None:
                            oact = ctx.tool_args.get("oracle_act")
                        if _oracle_exercised(oact, self._observed(ctx),
                                             self._bound_oracle().get("first_exercise")):
                            self._oracle_exercised_seen = True

        # Timing 2: post-recovery, inject reminder on first step_doing
        if self._post_recovery and not self._recovery_reminded:
            if ctx.tool_name == "plan_update":
                action = ctx.tool_args.get("action")
                if action == "step_doing":
                    self._recovery_reminded = True
                    return GuardVerdict.inject(
                        message=_POST_RECOVERY_REMINDER,
                        reason="post_recovery_reminder",
                        category="post_recovery"
                    )
        
        # Timing 3: soft guidance when acceptance criteria are first defined.
        # Fires once — a nudge toward externally observable criteria, never blocks.
        if not self._acceptance_guidance_given and self._defines_acceptance(ctx):
            self._acceptance_guidance_given = True
            return GuardVerdict.inject(
                message=_ACCEPTANCE_GUIDANCE,
                reason="acceptance_quality_guidance",
                category="acceptance_guidance",
            )

        # All other tools (read_file/shell/edit_file) completely unaffected
        return None

    @staticmethod
    def _defines_acceptance(ctx: GuardContext) -> bool:
        """True if this tool call is defining acceptance criteria on any step.

        Covers plan_create/add_steps (steps list carrying "acceptance") and
        plan_update(action="update_acceptance", acceptance=[...]).
        """
        if ctx.tool_name == "plan_update":
            if ctx.tool_args.get("action") == "update_acceptance":
                return bool(ctx.tool_args.get("acceptance"))
            return False
        if ctx.tool_name in ("plan_create", "add_steps"):
            steps = ctx.tool_args.get("steps") or ctx.tool_args.get("new_steps") or []
            if isinstance(steps, list):
                for s in steps:
                    if isinstance(s, dict) and s.get("acceptance"):
                        return True
        return False

    def _task_text(self) -> str:
        """Original task text for grounding a deliverable_reading's `source`.

        Reads through the caller-supplied callable (agent.py binds
        _mech_original_task). Fail-open: any error, or a None/absent callable,
        yields "" and the grounding check is skipped — the guard must not block
        a reading it cannot check.
        """
        if self._task_text_fn is None:
            return ""
        try:
            return str(self._task_text_fn() or "")
        except Exception:
            return ""

    def _observed(self, ctx) -> "str | None":
        """Normalized corpus of this session's REAL tool-result text, or None.

        This is the non-forgeable evidence source for probe/hedge claims: a claim
        whose `observed` paste is not found here was never actually produced.
        Prefers the append-only full log (survives eviction); falls back to the
        live message window. Returns None only when BOTH sources are unavailable,
        which makes the evidence check fail-open (never block on what we cannot
        see).
        """
        full_log = None
        try:
            if self._full_log_fn is not None:
                full_log = self._full_log_fn()
        except Exception:
            full_log = None
        msgs = getattr(ctx, "messages", None)
        if not msgs and not full_log:
            return None
        return _observed_corpus(msgs or [], full_log)

    def _unfinished_steps(self) -> list:
        """Return the ids of steps in the active plan NOT yet done/skipped.

        Mirrors _step_status's permissive contract: no plan or a failed lookup
        yields [] so the completion arm does not block when state is unknown.
        A "pending" step that was never started COUNTS as unfinished — the same
        rule plan_update(action="complete") already enforces (plan.py refuses on
        any status not in done/skipped). Blocking here is therefore consistent
        with the tool route, not a new obligation.
        """
        if not self._plan:
            return []
        try:
            plan_data = self._plan.get_active()
            if not plan_data:
                return []
            return [
                s.get("id", i) for i, s in enumerate(plan_data.get("steps", []))
                if s.get("status") not in ("done", "skipped")
            ]
        except Exception:
            return []

    def _bound_oracle(self) -> dict:
        """Return the oracle declaration bound this run, or {} if none.

        Reads the per-RUN latch captured at plan_create time (``_oracle_bound``),
        so the obligation survives a later plan close/replace -- the tool route
        ``plan_update(action="complete")`` clears the active pointer and a
        re-frame swaps it, either of which would make a get_active()-only read
        MISS a bound oracle and silently disarm the completion arm. Falling back
        to the persisted active plan keeps the arm live for a guard constructed
        AFTER the binding (e.g. a resumed session whose plan already carries the
        oracle). Permissive contract (mirrors _step_status): no plan, no
        declaration, or a failed lookup yields {} so the completion arm does not
        block when state is unknown -- it fires ONLY when an oracle was bound.
        """
        if self._oracle_bound:
            return self._oracle_bound
        if not self._plan:
            return {}
        try:
            plan_data = self._plan.get_active()
            if not plan_data:
                return {}
            oracle = plan_data.get("oracle")
            return oracle if isinstance(oracle, dict) else {}
        except Exception:
            return {}

    def _step_status(self, step_id) -> str | None:
        """Return the status of a step in the active plan, or None if unknown.

        None means we cannot determine the status (no plan, no step_id, or
        lookup failed) — in that case the caller does not block, staying
        permissive when state is uncertain.
        """
        if not self._plan or not step_id:
            return None
        norm = _normalize_step_id(step_id)
        if norm is None:
            return None
        try:
            plan_data = self._plan.get_active()
            if not plan_data:
                return None
            for step in plan_data.get("steps", []):
                if step.get("id") == norm:
                    return step.get("status")
        except Exception:
            return None
        return None

    def _step_acceptance(self, step_id) -> list:
        """Return a step's acceptance criteria, or [] if none/unknown.

        Mirrors _step_status's permissive contract: no plan, no step_id, or a
        failed lookup yields [] so the caller does not block on unknown state.
        Used by the batch arm to apply Mode-1 (evidence required when a step has
        acceptance) to each done item, exactly as the single-step path does.
        """
        if not self._plan or not step_id:
            return []
        norm = _normalize_step_id(step_id)
        if norm is None:
            return []
        try:
            plan_data = self._plan.get_active()
            if not plan_data:
                return []
            for step in plan_data.get("steps", []):
                if step.get("id") == norm:
                    return step.get("acceptance", []) or []
        except Exception:
            return []
        return []

    def notify_recovery(self):
        """Called by hard_reset logic to signal recovery."""
        self._post_recovery = True
        self._recovery_reminded = False
