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
from flagscale_agent.react.oracle import (
    arbiter_observation_matches,
    named_arbiter_obligation,
    rival_reading_distinct,
    rival_reading_obligation,
)
from flagscale_agent.react.instruments import (
    check_bound_margin,
    check_differential,
    check_instrument_input,
    check_negative_control,
    calibration_needs_negative_control,
    differential_satisfied,
    UNSETTLED_PREDICTIONS_MSG,
)


def _normalize_step_id(raw) -> int | None:
    """Normalize a step id EXACTLY the way the plan_update tool does.

    Delegates to the handler's own parser rather than keeping a second copy of
    the rule. The guard's lookups must agree with what the handler will mutate;
    two implementations of "which step does this id name" drift apart silently —
    the handler accepts a form the guard does not, the guard's lookup misses, and
    the gate is bypassed while the step still gets marked done. One parser, used
    by both, makes that impossible. Falls back to a local scalar normalize (no
    text scanning) if the handler is unavailable.
    """
    try:
        from flagscale_agent.react.tools.plan_update import _parse_step_id

        return _parse_step_id(raw)
    except Exception:
        pass
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
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

WHERE THE FOUR DECLARATIONS GO (this is the part that costs rounds if guessed):
  · EVERY done item inside `updates[]` must carry its OWN four declaration
    fields, nested INSIDE that item's object:
      {"step_id": 1, "status": "done",
       "convention": {"name": "none"},
       "deliverable_reading": {"deliverable": "none"},
       "assumptions": {"unverified": "none"},
       "acceptance_act": {"command": "...", "observed": "<real output>"},
       "verification": ["<what you ran + what it printed>"]}
  · Writing them at the TOP LEVEL of the plan_update call is NOT enough and is
    exactly how this gate keeps firing — the per-item check reads updates[i]
    first and only falls back to the top level. Put them in the item; do not
    rely on the fallback.
  · `acceptance_act.observed` must be text a tool really printed this session
    (copy it from a real tool result; a paraphrase is refused). If the step has
    no runnable check use the unobservable form:
    {"unobservable": "<what and why>", "command": "...", "observed": "..."}
  · If a step deserves real evidence, mark it via plan_update(step_done,
    verification=[...]) individually rather than folding it into a batch."""

_STEP_DONE_NO_ACCEPTANCE_ACT = """[VerificationGuard] This step reports done with no `{field}`, so the requirement-coverage sweep cannot run.

Measured failure this prevents: a trial reported 21 step_done calls and submitted
`acceptance_act` ZERO times, so the coverage gate never executed once; it later
delivered an artifact that violated a requirement the agent had itself written
down earlier in its own log. The requirement was known and unmeasured — that is
exactly the gap this field closes, one step at a time, instead of only at
plan-close.

COPY THIS SHAPE — filled in, not placeholders. Every string below is replaced by
something you actually ran and actually read; nothing about the FORM is yours to
choose:

  acceptance_act = {
    "command": "python3 audit.py",
    "observed": "3/3 clauses PASS",
    "constraints": [
      {"quote": "each artifact must be at least the minimum stated length",
       "command": "python3 audit.py",
       "observed": "min_length=21 PASS"},
      {"quote": "the measured reading must stay within the stated interval",
       "command": "python3 measure.py --input artifact.bin",
       "observed": "61.83"},
      {"quote": "no disallowed site may appear inside the artifact",
       "command": "python3 audit.py",
       "observed": "disallowed_sites=none"}
    ]
  }

HOW EXACT IS "EXACT" — this is the part that wastes turns guessing, so here it is:
  · `command` and `observed` are FREE TEXT. Nothing parses them. Any length, any
    wording, several commands joined with `;` — all fine.
  · The ONE rule: the `observed` string must be text that a tool really printed
    in this session. Copy it, do not retype it from memory.
  · Whitespace and letter case are ignored in that comparison. An ellipsis, a
    summary, a paraphrase, or a number you recalled instead of saw is NOT ignored
    and will not match.
  · `constraints` is optional as a list, but for a task that enumerates its
    requirements it is how the coverage sweep runs: one row per requirement, each
    row's `quote` copied from the requirement's own words.
  · If you do not have the text handy, print it and copy that:
    `python3 audit.py | head -c 400` then paste those characters verbatim.

If the acceptance has no runnable check, use the UNOBSERVABLE form instead —
same rule, an attempt still costs a real probe:
  {"unobservable": "grader corpus is hidden so I cannot observe the score",
   "command": "ls /grader",
   "observed": "No such file or directory"}

Cover the requirements the task STATES and the ones it DEFERS to an external
standard (a clause of the form "the design must satisfy <named standard>") — a
deferred requirement is still a requirement, and it is the one most easily
dropped. When the wall is nearly spent the N-requirement sweep degrades to
advisory rather than blocking delivery; the field itself is still owed."""

_STEP_DONE_UNGROUNDED_ACT = """[VerificationGuard] `{field}` WAS received — its shape is fine. What failed is ONE thing: the `observed` string is not text this session's tools actually printed.

Do not re-send the field and do not change its shape. Only the text of `observed`
needs to become real output. You do not have to guess which part is wrong: the
diagnosis below names the longest piece of YOUR string that actually does occur
in your recorded tool results, when such a piece exists."""


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

CHECK THE TASK'S STATED RELATION, NOT A CONVENIENT PROXY. Re-read the wording
that defines what "correct" means and make your probe exercise THAT relation in
THAT form. Two traps this closes:
  \u2022 COVERAGE. If the task states an equivalence over a COLLECTION -- "equal up
    to permuting the neurons", "matches the reference set", "the same N items",
    "found all K" -- then a probe that samples a subset or scores only ONE
    direction is not the relation. Compare CARDINALITIES (your count vs the
    reference's) AND match each reference element to a DISTINCT element of yours
    (a bijection). A result that reproduces part of the target but omits items is
    only caught by this; a "distinct N" over your OWN items is not coverage of
    the reference.
  \u2022 METRIC. Use the comparison the task's own words denote. If it says "up to
    scaling", test that each matched pair is an exact scalar multiple (an
    elementwise ratio constant to tolerance) -- a NORMALIZED/cosine similarity is
    a weaker proxy that accepts near-parallel items the stated relation REJECTS.
    If it says "equal"/"identical", test equality at the stated tolerance. Match
    the metric to the wording; do not substitute a metric you happen to have.
A probe that passes only because it asked a weaker question than the task does
is the failure this observation exists to catch.

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

EXERCISE THE ORACLE AT THE STRENGTH THE TASK STATES, NOT A WEAKER PROXY. When
you run the exercise, the probe must test the relation the task's own wording
denotes, over its FULL domain: if the task says "equal up to permuting/scaling"
or otherwise compares two COLLECTIONS, your probe must compare CARDINALITIES
(yours vs the reference's) and match each reference element to a DISTINCT element
of yours (a bijection), using the stated metric (for "up to scaling": an exact
scalar multiple / elementwise-ratio-constant, NOT a normalized cosine, which
accepts near-parallel items the relation rejects). A probe that covers only part
of the target, or scores one direction, or swaps in a weaker similarity measure,
will PASS while the deliverable still fails the stated relation — that is the
exact gap this arm exists to force you to see. If the oracle's output is weaker
than the task's stated relation, the exercise is not complete: strengthen the
probe until it tests what the task literally requires.
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

_PLAN_COMPLETE_NO_NAMED_ARBITER = """[VerificationGuard] You are closing the plan, but the DECIDER named by the task text was never pointed at the DELIVERED bytes.

A task's own wording often names who decides a disputed quantity ("consider X's
output the ground truth", "must match the reference implementation", "verified
with tool Y"). That sentence is in the task text you can read — it is not the
hidden checker. It is the one external anchor you are allowed to use, and it is
the only one that cannot be made to agree with your own definition of the
quantity.

Three distinct ways this is currently unmet, each with its own exit:

  (a) NOT DECLARED. The plan's oracle never said which tool the task text names.
      Exit: plan_create(oracle={..., "named_arbiter": "<the tool/library the task
      text names, quoted from the task text>", "deliverable_path": "<the file you
      deliver>"}). If the task text names NO deciding tool, declare that
      explicitly as "named_arbiter": "none" — the point is that you LOOKED.

  (b) DECLARED, NEVER RUN ON THE DELIVERABLE. An observation must exist whose
      command runs the named arbiter AND names the delivered artifact's path.
      Exit: run it, on the artifact, and record it on a step_done as
      arbiter_act={"command": "<the exact command>", "observed": "<its ACTUAL
      output>"}.

  (c) RUN, BUT ON SOMETHING YOU CONSTRUCTED. Pointing the arbiter at a hand-typed
      sample, a re-derived substring, or your own scratch instance measures the
      arbiter's arithmetic, not your shipped bytes. The definition of WHICH
      bytes get measured is exactly where a self-authored check goes wrong
      silently — so the command must name the delivered path.

WHEN YOUR OWN READING AND THE ARBITER DISAGREE, THE ARBITER WINS.
A disagreement between your verifier and the task's named decider is the single
most informative observation in the run — it means one of the two DEFINITIONS of
the quantity is wrong, and it is not the arbiter's job to be wrong. Do not
explain the gap away as a bug in your own script and re-run until your number is
clean: that converts the one measurement that could have saved the task into
noise you deleted. Resolve it by asking WHICH DEFINITION the arbiter's own
default path applies (the boundary IT uses, the prompt IT adds, the unit IT
reports), then compute both and diff them. If you still cannot reconcile them,
say so explicitly — a stated residual disagreement is an honest close; a
silently re-derived number is not.

Nothing here reads or guesses a hidden checker: the arbiter name must come from
the task text, and the obligation is satisfied by a command you run yourself.
"""


_PLAN_COMPLETE_ONE_READING = """[VerificationGuard] You are closing the plan with ONE reading of a quantity whose definition you chose yourself.

Pointing the task's named decider at your delivered file is necessary but not
sufficient. The decider measures whatever YOUR command feeds it — and the rule
that CUTS the measured substring out of those bytes is still yours to pick. A
single definition always agrees with itself, so one reading cannot detect that
you cut in a different place than the consumer does. Measured shape: a command
that named the delivered file, that stripped a fixed-length prefix before
measuring (an overhang, a header, a tag), and whose number landed just inside
the stated bound — while the consumer's construction of the SAME quantity
included those characters and landed just outside it.

Record a SECOND, genuinely different reading over the SAME delivered bytes:

  arbiter_act={"command": "<reading A: your extraction>",
               "observed": "<its actual output>",
               "rival_command": "<reading B: the OTHER defensible extraction — a
                                 different cut, boundary, unit or prompt>",
               "rival_observed": "<its actual output>"}

Both commands must name the delivered path. The two commands must genuinely
DIFFER: pasting the same command twice is one reading, not two.

Two outcomes to compare, and only one of them is comfortable:
  * the two readings AGREE and the shared number is inside the bound with room
    to spare — the definition does not decide the answer; close with that
    number and say so;
  * the two readings DISAGREE — then the definition decides the answer, and the
    consumer's construction wins. Work out which cut the consumer's own default
    path applies (the boundary IT uses, the characters IT includes), and ship a
    value that survives BOTH readings. A value that is only correct under your
    own cut is not correct.

If the quantity truly admits exactly one defensible definition, declare that
explicitly as "rival_command": "none" — an auditable declaration, the complement
of having actually looked. An omitted rival_command reads as never having asked
the question, which is the loss this check exists for.
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



def _found_report(path, value, want):
    """One line stating the field, the value actually received, and what was
    wanted. This is the whole point of a mechanical check: it can name the
    rejected value, so the reader never has to guess it.

    Kept format-stable and short: the value is repr'd (so an empty string, a
    dict, a None and a typo'd key are all visually distinct) and truncated so a
    huge nested object cannot flood the context it is trying to help.
    """
    try:
        shown = repr(value)
    except Exception:
        shown = "<unrepresentable %s>" % type(value).__name__
    if len(shown) > 240:
        shown = shown[:240] + "...(truncated)"
    return ("\n[WHERE IT FAILED] %s: found %s (type %s); wanted %s.\n"
            "[FIX] Keep every other field exactly as it is and correct only "
            "this one, then re-issue the same call.\n"
            % (path, shown, type(value).__name__, want))


def _collect_fixed_field_failures(checks):
    """Run every fixed-field check and merge ALL failures into ONE block.

    Why merge: the checks are independent and the model can only fix what it is
    told about. Returning the first failure makes three malformed fixed fields
    cost three rounds (measured: a single round reveals 1 of 3), and each round
    is a full model call — so short-circuiting is exactly what turns a
    mechanical shape error into the "guess, get blocked, guess again" loop. The
    three checks are pure shape+copies, so evaluating them all is free and
    cannot produce a wrong accept.

    ``checks`` is a list of (verdict_or_None). Returns one merged verdict or None.
    """
    bad = [v for v in checks if v is not None]
    if not bad:
        return None
    if len(bad) == 1:
        return bad[0]
    head = bad[0]
    parts = [head.message]
    for extra in bad[1:]:
        parts.append("\n" + "-" * 68 + "\n" + extra.message)
    head.message = (
        "[VerificationGuard] %d required fields are wrong in this ONE call. "
        "Fix ALL of them together and re-issue once — they are independent, so "
        "fixing one at a time costs one round each.\n" % len(bad)
        + "WHERE they go: this call is a plan_update batch — each missing field "
        "belongs INSIDE the done item it names (updates[i][field]), NOT at the "
        "top level of the call; the [RECEIVED] lines below already show the "
        "dotted path each field must occupy.\n"
        + "MINIMAL LEGAL FORMS (copy the one matching each missing field; use "
          "each only when its claim is true for THIS step):\n"
        "  convention={\"name\": \"none\"}\n"
        "  deliverable_reading={\"deliverable\": \"none\"}\n"
        "  assumptions={\"unverified\": \"none\"}\n"
        + "\n".join(parts)
        + "\n[REJECTED FIELDS] " + ", ".join(v.reason or "?" for v in bad) + "\n"
    )
    return head


def _echo_received(bad, name, raw):
    """Append the RAW value the validator actually received to any block.

    This is the one place that covers every malformed-field branch at once: the
    checker already holds ``raw`` — echoing it converts "the declaration is
    malformed" (guess which part) into "here is exactly what you sent" (copy and
    fix). Applied at the six dispatch sites rather than in 59 individual
    branches, so a future branch inherits the behaviour for free.
    """
    if bad is None:
        return None
    try:
        shown = repr(raw)
    except Exception:
        shown = "<unrepresentable %s>" % type(raw).__name__
    if len(shown) > 400:
        shown = shown[:400] + "...(truncated, %d chars total)" % len(repr(raw))
    bad.message = (bad.message
                   + "\n[RECEIVED] %s = %s\n" % (name, shown)
                   + "[FIX] compare the value above with the shape the message "
                     "asks for, change only the part that differs, and re-issue "
                     "the same call.\n")
    return bad


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

COPY-PASTE MINIMAL LEGAL FORM (use this when no derived convention governs
this step's output — it is an owned declaration, not a skip):
  convention={"name": "none"}

Then read WHY the field exists and when the full form is required:

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

COPY-PASTE MINIMAL LEGAL FORM (use this when THIS step emits no graded artifact
— it is an owned declaration, not a skip):
  deliverable_reading={"deliverable": "none"}

Then read WHY the field exists and the other two shapes:

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

Missing field = malformed = BLOCKED. See the spec for the sub-rules.

COPY-PASTE STARTERS — the three shapes, complete and valid. Replace the
angle-bracket values; keep every key. Copying one of these and editing the
values is faster and safer than composing the object from the description.

  # A. this step emits no graded artifact
  deliverable_reading={"deliverable": "none"}

  # B. grader invisible, ONE reading, union hedge over an output both accept
  deliverable_reading={
    "deliverable": "/app/out.txt",
    "grader_visible": false,
    "readings": [{"reading": "the checker compares the raw file bytes",
                  "source": "<paste the task sentence that says this>",
                  "acceptance_test": "cmp /app/out.txt /app/expected.txt",
                  "normalized_side": "neither"}],
    "hedge": {"shape": "union",
              "output": "<the exact bytes you will ship>",
              "evidence": {"command": "cmp /app/out.txt /app/expected.txt",
                           "observed": "<the command's RAW output>"},
              "extras_direction": "ignored"}}

  # C. grader invisible, two readings that DISAGREE -> intersection only
  deliverable_reading={
    "deliverable": "/app/out.npy",
    "grader_visible": false,
    "readings": [{"reading": "checker rescales both sides",
                  "source": "<paste the task sentence that says this>",
                  "acceptance_test": "python check.py --rescale",
                  "normalized_side": "both"},
                 {"reading": "checker compares raw values, no rescale",
                  "source": "<paste the second task sentence>",
                  "acceptance_test": "python check.py --raw",
                  "normalized_side": "candidate"}],
    "hedge": {"shape": "intersection",
              "output": "<the exact artifact you will ship>",
              "evidence": {"command": "python check.py --rescale && python check.py --raw",
                           "observed": "<the command's RAW output>"},
              "extras_direction": "penalized",
              "direction_evidence": {
                "reference": {"command": "<normalize the REFERENCE, compare the RAW candidate>",
                              "observed": "<that command's RAW output>"},
                "candidate": {"command": "<normalize the CANDIDATE, compare the RAW reference>",
                              "observed": "<that command's RAW output>"}}}}"""

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


# Per-normalized-source failure tally, so a REPEATED identical miss can be
# reported as such ("you have now failed this same quote N times — rewording is
# not the fix"). Module-level and intentionally not persisted: its only purpose
# is to break a live reword-and-retry loop within one session.
_grounding_miss_counts: dict = {}


def _task_segments(task_text: str) -> list:
    """Split the composite task text into its real segments (task / last turn)."""
    t = (task_text or "").strip()
    if not t:
        return []
    return [s.strip() for s in t.split("\n\n") if s.strip()]


def _note_grounding_miss(source) -> int:
    """Increment and return the failure count for this normalized `source`."""
    if not _is_real_str(source):
        return 0
    key = _normalize_for_grounding(source)
    _grounding_miss_counts[key] = _grounding_miss_counts.get(key, 0) + 1
    return _grounding_miss_counts[key]


def _reading_source_diagnosis(source, task_segments, full_task_text) -> str:
    """Human-readable WHY a `source` failed the verbatim-grounding check, plus the fix.

    Grounding is a membership test: normalize(source) must occur in
    normalize(task_text). When it fails, the message used to say only "not
    grounded", so the agent had to guess WHICH corpus was compared and HOW its
    quote differed — and guessed wrong repeatedly (observed live: the agent
    pasted its own plan title as the "task quote" six times, each time only
    rewording, never discovering that the corpus was the user's original
    message and that a different string was needed). Everything below is derived
    mechanically from the agent's own declaration and the harness's own corpus —
    no task content is invented, and no per-case text is hardcoded:

      1. Echo what was checked: the normalized needle and its length.
      2. If a substantial slice of it DOES occur in the corpus, show that real
         slice and say: quote THIS exactly.
      3. Otherwise say it does not occur at all and point at the two real
         corpora (original task message / last user turn) so the agent looks in
         the right place instead of paraphrasing again.
      4. When the failure repeats, say so explicitly — repeating the same shape
         of edit is the signal that the WRONG THING is being edited (rewording a
         quote never changes membership).
    """
    if not _is_real_str(source):
        return (
            "\n\n[how to fix] `source` must be a VERBATIM quote from the user's "
            "own task text. You passed a non-string or an empty value."
        )
    needle = _normalize_for_grounding(source)
    corpus = _normalize_for_grounding(full_task_text or "")
    out = [
        f"\n\n[what was checked] `source` normalized to {len(needle)} chars and "
        "was searched as a SUBSTRING of the task text (whitespace-collapsed, "
        "case-insensitive). Nothing else is compared — not semantics, not intent."
    ]
    snippet = _longest_common_snippet(needle, corpus, min_len=8)
    if snippet:
        out.append(
            "\n[how to fix — this is the whole problem] The longest part of your "
            f"`source` that really occurs in the task text is the slice below. "
            "Quote THAT text byte-for-byte (whitespace and case are ignored, "
            "nothing else is):\n"
            f"  ---8<--- real text from the task ---\n  {snippet}\n"
            "  --->8---\n"
            "Rewording the rest of your quote will never help: only a substring "
            "of the task text can pass."
        )
    else:
        out.append(
            "\n[how to fix — this is the whole problem] NONE of your `source` "
            f"({len(needle)} chars) occurs in the task text. That means it was "
            "written from memory, from your own plan/notes, or paraphrased — not "
            "quoted from the user. The corpus compared here is the user's own "
            "message(s), not your plan title and not your summary of the task."
        )
    if task_segments:
        out.append(
            "\n[where the corpus comes from] the task text is a composite of "
            f"{len(task_segments)} segment(s):"
        )
        for i, seg in enumerate(task_segments[:3], 1):
            flat = " ".join(seg.split())
            head = flat[:160] + ("…" if len(flat) > 160 else "")
            out.append(f"  {i}. ({len(flat)} chars) {head}")
        if len(task_segments) > 3:
            out.append(f"  … plus {len(task_segments) - 3} more segment(s)")
        out.append(
            "  Quote from one of these verbatim (a short distinctive phrase is "
            "enough — the whole sentence is not required)."
        )
    misses = globals().get("_grounding_miss_counts")
    if isinstance(misses, dict):
        n = misses.get(needle, 0)
        if n >= 2:
            out.append(
                f"\n[repeated failure ×{n}] This exact `source` has now failed "
                f"{n} times. Changing only the wording will not fix it — the check "
                "is substring membership, so a reworded quote is still absent. "
                "Either (a) copy a phrase from the segment list above, or (b) if "
                "no phrase of the task supports this reading, the reading is not "
                "one of the task's readings: drop it rather than re-deriving it."
            )
    return "".join(out)


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


def _longest_common_snippet(needle_norm: str, corpus_norm: str, min_len: int = 12) -> str:
    """Longest run of `needle_norm` that actually occurs in `corpus_norm`.

    Used ONLY to explain a failed grounding match back to the agent. It is a
    deterministic boundary search (coarse-to-fine over the needle, both already
    normalized), not a similarity heuristic: the returned text is guaranteed to
    be a real substring of the corpus, so the agent can paste it verbatim and
    pass on the next attempt.

    `min_len` is deliberately small (12): a real paraphrase of a real tool output
    usually still shares 12+ contiguous characters with it (an identifier, a
    number with its label, a file path), and that shared run is exactly the
    anchor the agent needs. When even that does not line up, the observation did
    not come from any tool result — which is itself the signal, and the caller
    then falls back to showing the corpus itself.
    """
    if not needle_norm or not corpus_norm:
        return ""
    if needle_norm in corpus_norm:
        return needle_norm
    # Sliding window over the needle: shrink from both ends until a substring of
    # the needle is found in the corpus. Deterministic, O(len^2) worst case but
    # the needle is a short pasted observation, and this runs only on a FAILED
    # check (rare, and the cost is dwarfed by the round trip it saves).
    best = ""
    n = len(needle_norm)
    step = max(1, n // 64)  # coarse-to-fine: bounded work on long needles
    for start in range(0, n, step):
        for end in range(n, start + len(best), -step):
            cand = needle_norm[start:end]
            if len(cand) > len(best) and cand in corpus_norm:
                best = cand
                break
    if len(best) < min_len:
        return ""
    return best


def _evidence_observed_diagnosis(ev, corpus, corpus_raw=None) -> str:
    """Human/LLM-readable WHY a failed grounding check failed, plus the fix.

    Appended to the block message so the agent does not have to guess how strict
    "must be findable in your session's tool results" is (observed in a live run:
    the agent re-pasted paraphrase after paraphrase, each time guessing at the
    required exactness, burning its wall-clock budget). The guidance is derived
    from the agent's OWN declaration and the harness's OWN corpus — no task
    content, no per-case text:

      1. Echo what was checked (the normalized needle, its length).
      2. If a substantial prefix/slice of it DOES occur in the corpus, show the
         largest real matching slice and say: paste THIS exactly.
      3. Otherwise say the text does not occur at all, and give the concrete
         command shape that dumps the true tool output so it can be copied.
    """
    obs = ev.get("observed") if isinstance(ev, dict) else None
    if not _is_real_str(obs):
        return (
            "\n\n[how to fix] `observed` must be the RAW text a tool produced in "
            "this session. You passed a non-string or an empty value. Copy the "
            "exact characters out of a tool result, e.g. the line `ls` printed."
        )
    needle = _normalize_for_grounding(obs)
    if corpus is None:
        return ""
    if not corpus:
        return (
            "\n\n[how to fix] No tool-result text has been recorded in this "
            "session yet, so nothing can be observed. RUN the probe first, then "
            "paste its output."
        )
    snippet = _longest_common_snippet(needle, corpus)
    if snippet:
        return (
            "\n\n[how to fix — this is the whole problem] Your `observed` is "
            f"{len(needle)} chars; the longest part of it that really occurs in "
            "your recorded tool results is the slice below. Paste THAT text "
            "byte-for-byte (whitespace and case are ignored, nothing else is):\n"
            f"  ---8<--- real text from your tool results ---\n  {snippet}\n"
            "  --->8---\n"
            "Do not paraphrase, summarize, or reformat an observation: the check "
            "is membership in the recorded output, not semantic similarity."
        )
    return (
        f"\n\n[how to fix — this is the whole problem] None of your `observed` "
        f"({len(needle)} chars) occurs in any tool result recorded this session. "
        "That means it was written from memory or paraphrased. Produce the text "
        "with a real command and copy its raw output, e.g.:\n"
        "  · re-run the probe and paste stdout verbatim, or\n"
        "  · `cat <the file you wrote>` / `head -c 400 <file>` and paste exactly "
        "what it prints.\n"
        "Whitespace and letter case are ignored; nothing else is — no ellipses, "
        "no `...`, no re-wording."
        + _corpus_tail_hint(corpus)
    )


def _corpus_tail_hint(corpus_norm: str, tail_chars: int = 600) -> str:
    """Show the END of the recorded tool-result text so the agent can copy from it.

    When nothing of the agent's `observed` matches at all, the most useful thing
    the harness can hand back is what the recorded output ACTUALLY looks like —
    the agent then sees the real wording/format instead of guessing again. This
    is bounded (last `tail_chars` of the normalized corpus) and derived only from
    the harness's own record, so it carries no task content.
    """
    if not corpus_norm:
        return ""
    tail = corpus_norm[-tail_chars:]
    head = "…" if len(corpus_norm) > tail_chars else ""
    return (
        "\n\nFor reference, the END of what your tool results actually recorded "
        "(normalized: whitespace collapsed, lowercased). Copy a contiguous run "
        f"from here — it is guaranteed to match:\n  ---8<---\n  {head}{tail}\n"
        "  --->8---"
    )


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


#: Minimum number of DISTINCT checkable requirements a task statement must
#: enumerate before the per-constraint coverage gate applies. Below this the
#: statement is prose-shaped and a coverage demand would be noise; at or above
#: it the statement hands the agent a checkable list, and checking ONE of them
#: is a proxy for the task, not the task.
_CONSTRAINT_COVERAGE_MIN = 3


def _stated_constraints(task_text: str, judge=None) -> list:
    """The DISTINCT checkable requirements a task statement enumerates.

    READING the statement is a language task: requirements are stated in free
    prose ("the difference must stay under a stated bound", "no more than N
    bytes", "exactly four shards"), with no fixed shape to key off and no
    vocabulary that is not
    task-specific. So the reading is delegated to the judge (an LLM), which
    returns each requirement VERBATIM. Nothing here pattern-matches the text:
    a differently-worded statement of the same requirement yields the same
    answer, which a shape rule cannot promise.

    DEFERRED REQUIREMENTS ARE INCLUDED. A statement can point AT a standard
    instead of stating the requirement ("check that X satisfies Y's
    requirements", "per the manufacturer's manual"). Those requirements are
    checkable and the deliverable must satisfy them, but they appear nowhere in
    the statement's own words -- so reading only the literal sentences leaves
    them out of the obligation set entirely. Measured loss: a task deferred to a
    synthesis kit's requirements; a real one (a protective base 5' of the
    recognition site) never entered the agent's checklist, so a self-check that
    passed EVERY stated clause still failed the grader. The agent had read that
    requirement earlier and then dropped it on a guess about whether it would be
    checked -- exactly the silent evaporation an obligation prevents. The judge
    extracts these too (a second, separately-prompted reading), and they join the
    same coverage demand, so each one needs a corpus-backed observation.

    Returning [] when no judge is supplied (or the judge reads nothing) leaves
    the gate inert rather than inventing obligations -- a dead provider must not
    manufacture a block.
    """
    if not _is_real_str(task_text):
        return []
    if judge is None:
        return []
    out: list = []
    try:
        out = list(judge.constraints(task_text) or [])
    except Exception:
        out = []
    try:
        deferred = judge.deferred_requirements(task_text) or []
    except Exception:
        deferred = []
    seen = {" ".join(s.split()).lower() for s in out}
    for d in deferred:
        if not isinstance(d, dict):
            continue
        req = d.get("requirement")
        if not _is_real_str(req):
            continue
        phrase = " ".join(req.split())
        if phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        out.append(phrase)
    return out



def _constraint_coverage_missing(act, task_text: str, observed, judge=None) -> list:
    """Constraint phrases the task states that the acceptance record did NOT cover.

    Returns [] when the gate does not apply (statement enumerates fewer than
    `_CONSTRAINT_COVERAGE_MIN` checkable constraints, or the coverage question
    cannot be answered). Otherwise returns the list of stated-constraint
    phrases for which the `acceptance_act` carries no corpus-backed observation.

    WHY THIS EXISTS: `_acceptance_act_recorded` accepts ONE observed act as the
    acceptance record. When the statement enumerates N independent checkable
    constraints, one observation is a PROXY for the task, not the task -- an
    agent can observe a cheap constraint, bank the deliverable, and only then
    discover a different constraint it violated (measured: an agent wrote its
    deliverable, then 10s later self-found that the paired quantity had to be
    computed on a narrower derived part of it, then hit the wall). The gate
    closes that by demanding
    one covered entry per stated constraint BEFORE completion.

    SPLIT OF WORK -- both halves are load-bearing:
      * READING prose is semantic (which claims address which requirement), so
        the judge answers it. Nothing here pattern-matches wording; a
        differently-worded claim for the same measurement still counts.
      * ADMISSIBILITY is mechanical and stays here: a claim is only eligible if
        it carries a command and an `observed` value that really occurs in the
        session's tool-output corpus. So the entries cannot be fabricated -- a
        claim about a measurement never actually made is dropped before the
        judge ever sees it.
    """
    if not _is_real_str(task_text):
        return []
    stated = _stated_constraints(task_text, judge)
    if len(stated) < _CONSTRAINT_COVERAGE_MIN:
        return []
    if observed is None:
        return []  # fail open: no corpus, coverage is unjudgeable
    if judge is None:
        return []  # fail open: no reader for the prose, nothing to judge
    rows = act.get("constraints") if isinstance(act, dict) else None
    if not isinstance(rows, list):
        return stated
    quotes = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        quote = row.get("quote")
        if not _is_real_str(quote):
            continue
        if not (_is_real_str(row.get("command")) and _is_real_str(row.get("observed"))):
            continue
        if not _evidence_observed(row, observed):
            continue  # observed text not in the real corpus -> not a coverage
        quotes.append(quote)
    if not quotes:
        return stated
    try:
        covered = set(judge.constraint_coverage(quotes, stated, task_text))
    except Exception:
        return []  # fail open: a dead judge must not manufacture a block
    return [s for i, s in enumerate(stated) if i not in covered]


def _completion_constraint_coverage_missing(missing: list) -> str:
    """Block message: which stated constraints the acceptance record missed.

    Names the ACTUAL uncovered constraints (not just "coverage missing") and
    shows the exact row shape to add, so the agent does not have to guess.
    """
    items = "\n".join("  \u2022 %s" % m for m in missing)
    example = (missing[0] if missing else "<one of the constraints above>")
    return f"""[VerificationGuard] Your acceptance record does not cover every constraint the task states.

This task's text enumerates {len(missing)} checkable constraint(s) for which your
`acceptance_act` carries NO corpus-backed observation:
{items}

One observed act is a PROXY for such a task, not the task itself. Observing one
cheap constraint and banking the deliverable is exactly the failure this gate
exists to stop: the deliverable is written, the remaining constraint is violated,
and there is no time left to fix it.

Add a `constraints` list to `acceptance_act` -- ONE row per constraint above, each
carrying the constraint's own wording, the command you actually ran, and that
command's REAL recorded output (output you never produced is refused):

  acceptance_act={{
    "command": "<the command you already ran>",
    "observed": "<its real recorded output>",
    "constraints": [
      {{"quote": "{example}",
        "command": "<command that measures THIS constraint>",
        "observed": "<that command's real output, e.g. the measured value>"}}
      ... one row for every constraint listed above ...
    ]
  }}

If a constraint cannot be measured in this environment, say so in that row
explicitly rather than omitting it -- an omitted constraint and an unmeasurable
one are different states, and only the second is admissible."""


def _oracle_exercised(act, observed, first_exercise=None, judge=None) -> bool:
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

    The topical-relation test between the recorded command and the declared
    first_exercise is SEMANTIC ("is this the same check on the same subject?"),
    so it is delegated to the judge — an LLM reads both and decides. Token
    overlap was the previous implementation and it drew the conclusion from a
    bag of shared words: a `cat` of a file whose name happened to match counted
    as "the oracle was exercised", while a genuine re-invocation through a
    different runner (`python -m pytest -q tests/a.py::K` vs `pytest tests/a.py`)
    could be rejected. Re-wording is not a different exercise. With no judge we
    fail OPEN on this axis (we cannot read the relation, so we do not block),
    while the corpus-membership half below stays fully mechanical.
    """
    if not isinstance(act, dict):
        return False
    if not (_is_real_str(act.get("command")) and _is_real_str(act.get("observed"))):
        return False
    if _is_real_str(first_exercise):
        if judge is None:
            # FAIL CLOSED: with no reader for the prose relation we cannot tell
            # whether the recorded command is the declared exercise, so we do
            # NOT claim it was exercised. A verification predicate must not
            # produce a false PASS; refusing keeps the label meaningful. Every
            # production call site passes ctx.judge (agent.py builds one
            # unconditionally), so this arm only fires on the test/degraded path.
            return False
        try:
            same = judge.classify(
                "acts_on_same_subject",
                {"first_exercise": first_exercise, "action": act["command"]},
                default=True,
            )
        except Exception:
            # A dead judge must not manufacture a block: keep this arm open,
            # because the failure mode here is the JUDGE, not the agent.
            same = True
        if same is False:
            return False
    return _evidence_observed(act, observed)


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

COPY-PASTE MINIMAL LEGAL FORM (use this when NO load-bearing premise was left
untested — it is an owned declaration, not a skip):
  assumptions={"unverified": "none"}

Then read WHY the field exists and when the full form is required:

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
                message=_UNVERIFIED_ASSUMPTIONS_MALFORMED
                + _found_report("step_done.assumptions", field,
                                "a dict whose only required key is \"unverified\""),
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
    for _i, it in enumerate(items):
        if not isinstance(it, dict) or not _is_real_str(it.get("claim")):
            return GuardVerdict.block(
                message=_UNVERIFIED_ASSUMPTIONS_MALFORMED
                + _found_report("step_done.assumptions[%d]" % _i, it,
                                "a dict with a non-empty string \"claim\""),
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
            _ev_assume = {"command": it.get("probe"),
                          "observed": it.get("probe_output")}
            if not _evidence_observed(_ev_assume, observed):
                return GuardVerdict.block(
                    message=(_UNVERIFIED_ASSUMPTIONS_AFFECTS
                             + _evidence_observed_diagnosis(_ev_assume, observed)),
                    reason="assumptions_affecting_without_test",
                    category="verification_required",
                )
        else:
            if not _is_real_str(it.get("reason")):
                return GuardVerdict.block(
                    message=_UNVERIFIED_ASSUMPTIONS_MALFORMED
                    + _found_report("step_done.assumptions[%d].reason" % _i,
                                    it.get("reason"),
                                    "a non-empty sentence: why it cannot change the result"),
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
            _note_grounding_miss(r.get("source"))
            return GuardVerdict.block(
                message=(_READING_NOT_GROUNDED
                         + _reading_source_diagnosis(
                             r.get("source"),
                             _task_segments(task_text),
                             task_text)),
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
                message=(_HEDGE_UNPROVEN
                         + _evidence_observed_diagnosis(hedge.get("evidence"), observed)),
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
                        message=(_HEDGE_DIRECTION_EVIDENCE
                                 + _evidence_observed_diagnosis(ev, observed)),
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
                message=(_HEDGE_UNPROVEN
                         + _evidence_observed_diagnosis(
                             if_none.get("union_attempt"), observed)),
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
                _note_grounding_miss(if_none.get("extras_direction_source"))
                return GuardVerdict.block(
                    message=(_READING_NOT_GROUNDED
                             + _reading_source_diagnosis(
                                 if_none.get("extras_direction_source"),
                                 _task_segments(task_text),
                                 task_text)),
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
                    message=(_HEDGE_UNPROVEN
                             + _evidence_observed_diagnosis(e.get("evidence"), observed)),
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
                _note_grounding_miss(e.get("source"))
                return GuardVerdict.block(
                    message=(_READING_NOT_GROUNDED
                             + _reading_source_diagnosis(
                                 e.get("source"),
                                 _task_segments(task_text),
                                 task_text)),
                    reason="rejected_source_not_grounded",
                    category="verification_required",
                overridable=False,
                )
        for k in ("bet_on", "bet_loses_when", "why_safe"):
            if not _is_real_str(if_none.get(k)):
                # Say WHICH field and WHAT was found. A mechanical checker knows
                # both facts; withholding them makes the reader guess the field
                # name from a 600-word spec, and it will guess wrong (observed:
                # 42 identical rejections in one trial, the reader cycling
                # "maybe it wants X... or maybe Y..." on every round).
                _v = if_none.get(k, "<missing>")
                _shown = repr(_v) if not isinstance(_v, str) else (
                    "an empty/placeholder string " + repr(_v) if not _v.strip()
                    else "a placeholder token " + repr(_v))
                return GuardVerdict.block(
                    message=_HEDGE_UNPROVEN + (
                        "\n[WHERE IT FAILED] if_none.%s is not a real string: "
                        "found %s (type %s). Keep every other field exactly as "
                        "you have it and replace ONLY this one with a sentence "
                        "of plain prose.\n"
                        "  bet_on           = the one reading you are betting on\n"
                        "  bet_loses_when   = the world-state in which that bet is wrong\n"
                        "  why_safe         = why that world-state cannot happen here\n"
                        % (k, _shown, type(_v).__name__)
                    ),
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

    # Completion gate master switch.
    #   False (default, user instruction 2026-10-22): [TASK_COMPLETE] is not
    #     gated at all -- the obligations below are skipped.
    #   True: the full obligation set is enforced (original behavior).
    # The gate bodies are kept intact and stay unit-tested; tests that exercise
    # them monkeypatch this flag on, so disabling the gate never means deleting
    # the logic, and re-enabling it is a one-line flip here.
    COMPLETION_GATE_ENABLED = False
    
    def __init__(self, plan=None, proposals=None, task_text_fn=None,
                 full_log_fn=None, stats_fn=None):
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
        # Callable returning {elapsed,budget,remaining,pct} or None. Used ONLY to
        # downgrade the constraint-coverage gate when the wall is nearly spent:
        # demanding a per-constraint measurement sweep at 3% remaining trades a
        # banked deliverable for metadata (measured: model-extraction went 1 -> 0
        # because a coverage sweep consumed the whole 900 s turn).
        self._stats_fn = stats_fn
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
        # Per-RUN: the last corpus-verified `arbiter_act` — a command that ran the
        # task's NAMED decider, with output the harness really recorded. Latched
        # like the oracle act (survives plan mutation and the active-pointer
        # clear at complete()); the TARGET check (does the command name the
        # arbiter and the delivered path) is evaluated at the plan-complete
        # decision point, NOT here, so this pass can never return a verdict.
        self._arbiter_act_seen = None

    def reset_turn(self):
        """Reset per-turn state on a new user message — clears the step_done
        re-check latch so a fresh task starts with the premise re-check armed.
        """
        self._step_done_recheck_reminded = False


    # Fraction of the per-turn wall past which the constraint-coverage sweep stops
    # being mandatory. The sweep costs one tool call per stated constraint; once
    # the wrap-up window opens those calls are better spent producing/verifying
    # the deliverable itself. Below this fraction the gate is fully enforced.
    _COVERAGE_GATE_MIN_REMAINING = 0.25

    def _latch_evidence_pass(self, ctx) -> None:
        """Record corpus-verified evidence BEFORE any route-specific block.

        Runs once per plan_update call and NEVER returns a verdict: it can only
        raise `_acceptance_observed` / `_oracle_exercised_seen` to True. That
        asymmetry is the point — blocking a call for an unrelated formal reason
        must not answer "no observation happened" for a fact the harness has
        already checked against the real tool-result corpus.

        Reads the act from the top level AND from every `updates[i]` item, so
        the batch route (whose declarations live on the item) latches too.
        """
        args = ctx.tool_args or {}
        candidates = []
        updates = args.get("updates")
        if isinstance(updates, list):
            for u in updates:
                if isinstance(u, dict):
                    candidates.append(u)
        candidates.append(args)
        observed = None
        for src in candidates:
            act = src.get("acceptance_act")
            if act is not None:
                if observed is None:
                    observed = self._observed(ctx)
                if (not self._acceptance_observed
                        and _acceptance_act_recorded(act, observed)):
                    miss = _constraint_coverage_missing(
                        act, self._task_text(), observed,
                        judge=getattr(ctx, "judge", None))
                    # A coverage miss simply means "not latched yet"; the
                    # per-route gate below still blocks and still names the
                    # uncovered constraints, so nothing is weakened here.
                    if not miss or not self._coverage_gate_affordable():
                        self._acceptance_observed = True
            oact = src.get("oracle_act")
            if oact is not None and not self._oracle_exercised_seen:
                if observed is None:
                    observed = self._observed(ctx)
                if _oracle_exercised(
                        oact, observed,
                        self._bound_oracle().get("first_exercise"), ctx.judge):
                    self._oracle_exercised_seen = True
            # The plan-complete arbiter obligation reads its OWN act field, on the
            # same corpus-verified observation source: a command whose `observed`
            # text the harness never recorded is not an act. The target check
            # (arbiter name + delivered path in the command) is deferred to the
            # decision point so this pass stays a pure latch — it must never
            # return a verdict and never un-latch.
            aact = src.get("arbiter_act")
            if aact is not None and not self._arbiter_act_seen:
                if observed is None:
                    observed = self._observed(ctx)
                if _evidence_observed(aact, observed):
                    self._arbiter_act_seen = aact

    def _acceptance_act_obligation(self, act, ctx, field_label: str = "acceptance_act"):
        """Per-step acceptance-observation obligation (FIXED required field).

        `act` is passed EXPLICITLY rather than read off `ctx.tool_args`: a plan
        update reaches "done" through more than one route (a single `step_done`,
        a `batch` whose `updates[i]` items carry their own declarations), and
        reading the top level from inside this method means any route that nests
        the field escapes the obligation entirely. Measured: a trial used the
        batch route on every step (10 `plan_update` calls, one batch), the batch
        branch's fixed-field list carried only the three earlier declarations,
        and `acceptance_act` appeared ZERO times in 2664 log lines -- the
        coverage sweep never ran. Keeping the reader outside makes "which routes
        are covered" a visible list in one place instead of a hidden default.

        Returns None when satisfied or when the obligation does not apply, a
        GuardVerdict.block otherwise.

        Does NOT apply when the task text is not groundable -- with no stated
        requirement there is nothing to observe against.

        The budget valve deliberately does NOT live here. Two different costs are
        in play and only one of them is expensive:
          * recording ONE observed act (a command plus its real output) is a
            single probe the agent was going to run anyway, and
          * sweeping EVERY stated constraint is N probes, the shape that
            measured reward 1 -> 0 (a coverage sweep ate an entire 900 s turn
            and no deliverable was ever written).
        Degrading the cheap obligation would keep the sweep unreachable (that is
        precisely the hole this fixes: 21 step_done calls, ZERO acts, so the
        sweep body never executed). The wall-pressure valve therefore stays where
        the cost is -- inside the coverage sweep, unchanged.
        """
        if not _task_text_groundable(self._task_text()):
            return None
        if _acceptance_act_recorded(act, self._observed(ctx)):
            return None
        if isinstance(act, dict) and _is_real_str(act.get("observed")):
            # The field EXISTS and is well-shaped; what failed is corpus
            # membership of `observed`. Measured (trial 19-21-11): the agent
            # submitted a real act, was told the field was MISSING, read the
            # received-echo, and spent its final minutes guessing -- "the guard
            # says acceptance_act missing but I did pass it". The lie is worse
            # than the gate: it sends a correct submission into a blind loop.
            # The diagnosis names the longest real slice of the agent's own
            # text that DOES occur in the corpus, so the fix is a paste, not a
            # guess. Same reason the message must name the field it checked
            # rather than a generic "required fields are wrong".
            return GuardVerdict.block(
                _STEP_DONE_UNGROUNDED_ACT.replace("{field}", field_label)
                + _evidence_observed_diagnosis(act, self._observed(ctx)),
                reason="step_done_acceptance_act_ungrounded",
                category="verification_required",
            )
        return GuardVerdict.block(
            _STEP_DONE_NO_ACCEPTANCE_ACT.replace("{field}", field_label),
            reason="step_done_without_acceptance_act",
            category="verification_required",
        )

    def _coverage_gate_affordable(self) -> bool:
        """Is there enough wall left to pay for a per-constraint measurement sweep?

        True (gate stays mandatory) when
          * no concrete per-turn budget exists  -> never degrade a standalone run,
          * the remaining fraction is at least `_COVERAGE_GATE_MIN_REMAINING`,
          * or the budget reading is unusable    -> fail CLOSED on the safe side,
            i.e. keep enforcing (an unreadable clock is not evidence of pressure).

        False only on a real reading of a nearly-spent wall -- the measured case
        this exists for (model-extraction 1 -> 0: a coverage sweep consumed the
        entire 900 s turn, so no deliverable was ever written).
        """
        fn = self._stats_fn
        if fn is None:
            return True
        try:
            stats = fn()
        except Exception:
            return True  # a dead clock must not silently drop a gate
        if not isinstance(stats, dict):
            return True
        pct = stats.get("pct")
        if not isinstance(pct, (int, float)):
            return True
        return (1.0 - float(pct) / 100.0) >= self._COVERAGE_GATE_MIN_REMAINING

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # The step_done pre-mortem and the independent-reviewer demand now
        # live in the plan's fixed review step (see
        # tools/plan_create.py _REVIEW_STEP_ACCEPTANCE). VerificationGuard
        # keeps only the plan step_done evidence gate in check_pre below.
        return None

    def _at_task_complete(self, ctx: GuardContext) -> bool:
        """Is this the completion-path consultation of a [TASK_COMPLETE] claim?

        Fired ONLY on the completion-path consultation: tool_name=="" AND the
        sentinel came from THIS iteration's fresh LLM output (llm_responded=True).
        The top-of-loop check_pre also runs with tool_name=="" every iteration to
        catch stale completions out of history; requiring llm_responded keeps THIS
        arm off that stale scan. A text-only completion has two shapes, and
        testing only the trailing one would leak the override form (marker plus an
        inline `_override_reason:` line, where text follows the marker so
        trailing_sentinel returns None) — mirror the kernel's EXACT completion
        predicate (kernel.py:378: `trailing == "[TASK_COMPLETE]" or
        has_override_completion(text)`).
        """
        if ctx.tool_name != "" or not ctx.llm_responded:
            return False
        _text = (ctx.assistant_text or "").rstrip()
        return (
            trailing_sentinel(_text) == "[TASK_COMPLETE]"
            or has_override_completion(_text)
        )

    def _completion_obligations(self, ctx: GuardContext) -> list[GuardVerdict]:
        """EVERY unmet obligation behind one [TASK_COMPLETE], not just the first.

        The completion boundary evaluates several INDEPENDENT obligations at ONE
        decision point: the plan's steps must be finished, the acceptance must
        have been OBSERVED, and an oracle bound at framing must have been both
        BOUND and EXERCISED. Each demand reads a different piece of state, so
        returning the first (the old shape) left the agent to discover the rest
        one full round trip at a time — the same cascade the registry-level merge
        removed ACROSS guards, here inside a single guard.

        Safe to evaluate the whole set in one pass: every gate below only READS
        state (no latch is written in this method; the latches `_graded_reading_
        seen` / `_acceptance_observed` / `_oracle_exercised_seen` are written on
        the step_done route, not here), so a full pass has no side effect to
        preserve in order. The list keeps the original severity order, so a
        caller that presents only the leader behaves exactly as before.
        """
        # Completion gate is DISABLED by default (user instruction 2026-10-22).
        # The gate bodies below are preserved and still TESTED: a test that
        # exercises them opts in via `VerificationGuard.COMPLETION_GATE_ENABLED
        # = True` (monkeypatched), so the obligations remain verified logic
        # rather than deleted code, and re-enabling is a one-line flip.
        if not self.COMPLETION_GATE_ENABLED:
            return []
        pending: list[GuardVerdict] = []
        unfinished = self._unfinished_steps()
        # Release through the SAME override bar the tool route uses (registry ->
        # owner.accept_override; base default = reason longer than 5 chars), so
        # the text and tool completion routes enforce an identically strong bar.
        if unfinished and not self.accept_override(ctx.override_reason, ctx):
            pending.append(GuardVerdict.block(
                message=_COMPLETION_WITH_UNFINISHED_STEPS.format(
                    steps=", ".join(str(s) for s in unfinished),
                ),
                reason="completion_with_unfinished_steps",
                category="verification_required",
            ))
        # Acceptance-observation obligation (M4): a declared reading is not an
        # OBSERVATION of the acceptance. NON-overridable: the exit is the ACTION
        # of recording a real `acceptance_act`. The graded-reading arm is
        # deliberately absent — the acceptance act strictly subsumes it at this
        # boundary (see the step_done route for the per-step gate).
        if (
            not self._acceptance_observed
            and _task_text_groundable(self._task_text())
        ):
            pending.append(GuardVerdict.block(
                message=_COMPLETION_NO_ACCEPTANCE_OBSERVED,
                reason="completion_without_acceptance_observation",
                category="verification_required",
                overridable=False,
            ))
        # Oracle-exercise obligation: binding an oracle at framing wrote an
        # intended ACTION. NON-overridable: the exit is running that exercise.
        if self._bound_oracle() and not self._oracle_exercised_seen:
            pending.append(GuardVerdict.block(
                message=_COMPLETION_ORACLE_NOT_EXERCISED,
                reason="completion_without_oracle_exercise",
                category="verification_required",
                overridable=False,
            ))
        # Oracle-BIND obligation: 0/76 plans in a full TB2.1 run ever bound an
        # oracle, closing on a purely self-authored verification. NON-overridable:
        # the exit is binding one via plan_create(oracle={...}) and running it.
        if (
            not self._bound_oracle()
            and _task_text_groundable(self._task_text())
        ):
            pending.append(GuardVerdict.block(
                message=_COMPLETION_ORACLE_NOT_BOUND,
                reason="completion_without_oracle_bound",
                category="verification_required",
                overridable=False,
            ))
        return pending

    def _plan_complete_obligations(self, ctx: GuardContext) -> list[GuardVerdict]:
        """Obligations at the moment the agent CLOSES THE PLAN — not at [TASK_COMPLETE].

        WHY THIS ANCHOR AND NOT THE COMPLETION SENTINEL. An obligation parked on
        the completion sentinel puts the agent in a trap with no next action: it
        has decided it is done, the harness says it is not, and the only way
        forward is to argue — which is exactly the 20+ round livelock this gate
        was switched off for on 2026-10-22. ``plan_update(action="complete")`` is
        a DIFFERENT shape: it is a voluntary act with an obvious refusal path
        (finish the step, then close again). The tool route already refuses and
        teaches the next move ("Cannot complete plan: 1 step(s) not finished …
        Mark them as done/skipped first"), and the agent demonstrably recovers
        from it inside one round. So the same substantive obligation costs a
        round here instead of a livelock there.

        WHAT IS CHECKED. Only the independently-decidable half of "did you point
        the task's own named decider at the delivered bytes":

          * the plan's oracle declared a ``named_arbiter`` (or explicitly
            declared that the task names none), and
          * a recorded ``arbiter_act`` runs that arbiter against the delivered
            path, with ``observed`` text the harness really recorded.

        Nothing here can verify that the named tool IS the right decider — a
        name quoted from the task text is a declaration, auditable by a reader.
        That limit is stated rather than papered over: the check catches "never
        looked, or looked and pointed it at a scratch sample", which is the
        measured loss, not every possible mislabel.

        Returns ALL unmet obligations (not just the first) so one round-trip
        surfaces the whole set, matching ``_completion_obligations``' rationale.
        """
        pending: list[GuardVerdict] = []

        # INSTRUMENT 3 (PRE-REGISTRATION), checked FIRST and INDEPENDENTLY of the
        # oracle: a prediction recorded before a run and never settled is an open
        # obligation no matter what else was declared. Placed above the
        # oracle-early-return below, which is the trap this placement avoids --
        # a plan with no oracle declaration would otherwise close with dangling
        # predictions. Measured need: a long-running prediction on this harness
        # was left unsettled after its run was stopped, and nothing surfaced it.
        unsettled = []
        if self._plan:
            try:
                unsettled = self._plan.unsettled_predictions() or []
            except Exception:
                unsettled = []
        if unsettled:
            items = "\n".join(
                f"  - {p.get('id')}: {p.get('statement', '')}"
                + (f" -> expected {p['expected']}" if p.get("expected") else "")
                for p in unsettled
            )
            pending.append(GuardVerdict.block(
                message=UNSETTLED_PREDICTIONS_MSG.format(items=items),
                reason="plan_complete_unsettled_predictions",
                category="verification_required",
            ))

        decl = self._bound_oracle()
        if not decl:
            # No oracle declaration at all. The plan-complete path does NOT
            # require one (that obligation stays where it was, and the framing
            # acceptance already asks for it) — requiring it here would add a
            # round to every plan close without catching the measured loss.
            return pending

        arbiter = named_arbiter_obligation(decl)
        if arbiter == "":
            pending.append(GuardVerdict.block(
                message=_PLAN_COMPLETE_NO_NAMED_ARBITER,
                reason="plan_complete_arbiter_malformed",
                category="verification_required",
            ))
            return pending
        if arbiter is None:
            # Declared "none" (task text names no deciding tool) or the key is
            # absent. Only the explicit declaration is an exit; an absent key is
            # not, because "did not look" must not read the same as "nothing
            # there". The message's (a) branch teaches the declaration.
            if "named_arbiter" not in decl:
                pending.append(GuardVerdict.block(
                    message=_PLAN_COMPLETE_NO_NAMED_ARBITER,
                    reason="plan_complete_arbiter_undeclared",
                    category="verification_required",
                ))
            return pending

        act = self._arbiter_act_seen
        if not act:
            pending.append(GuardVerdict.block(
                message=_PLAN_COMPLETE_NO_NAMED_ARBITER,
                reason="plan_complete_arbiter_not_exercised",
                category="verification_required",
            ))
            return pending

        path = decl.get("deliverable_path")
        path = path.strip() if isinstance(path, str) else ""
        if not arbiter_observation_matches(act, arbiter, path):
            pending.append(GuardVerdict.block(
                message=_PLAN_COMPLETE_NO_NAMED_ARBITER,
                reason="plan_complete_arbiter_wrong_target",
                category="verification_required",
            ))
            return pending

        # THE SECOND READING. Passing the check above proves the arbiter was
        # pointed at the delivered bytes -- it does NOT prove the bytes fed to it
        # were cut out of that file the way the consumer cuts them. Measured
        # shape: the command named the delivered file (so the target check
        # passed) while the shell rule that extracted the measured substring was
        # the agent's own (`${s:12}` -- dropping a fixed-length overhang), and the
        # consumer's construction of the same quantity included those characters
        # and landed just outside the stated bound. One reading cannot reveal
        # that: a single definition always agrees with itself. So a second,
        # DIFFERENT extraction over the SAME file is required, and its number is
        # a re-check of the delivered value rather than a formality.
        rival = rival_reading_obligation(act)
        if rival == "":
            pending.append(GuardVerdict.block(
                message=_PLAN_COMPLETE_ONE_READING,
                reason="plan_complete_single_reading",
                category="verification_required",
            ))
        elif rival is None:
            pass  # declared "there is no defensible second reading" -- auditable
        elif not rival_reading_distinct(act, rival) or not _evidence_observed(
                rival, self._observed(ctx)):
            pending.append(GuardVerdict.block(
                message=_PLAN_COMPLETE_ONE_READING,
                reason="plan_complete_rival_not_distinct",
                category="verification_required",
            ))

        # INSTRUMENT 1b backstop: the input-construction declaration is demanded
        # at creation time, but plans created BEFORE this mechanism (resumed
        # sessions) persisted a calibration without it. The backstop is the one
        # boundary a mixed tree cannot outflank: whenever the plan closes, its
        # calibration must either carry the construction declaration (or an
        # explicit not_applicable), or the close names the gap. A refusal here
        # has an obvious next action: re-frame the plan with the field filled,
        # or re-issue plan_create with the declaration added -- the same
        # recoverable shape the plan-close gates above use.
        _cal = {}
        if self._plan:
            try:
                _active = self._plan.get_active()
                if isinstance(_active, dict):
                    _cal = _active.get("calibration") or {}
            except Exception:
                _cal = {}
        _inp = check_instrument_input(_cal)
        if _inp is not None:
            pending.append(GuardVerdict.block(
                message=_inp[1],
                reason=_inp[0],
                category="verification_required",
            ))
        return pending

    def check_all_pre(self, ctx: GuardContext) -> list[GuardVerdict]:
        """Every unmet requirement at this decision point, not just the first.

        Only the two COMPLETION boundaries evaluate several INDEPENDENT
        obligations at one decision point (see _completion_obligations and
        _plan_complete_obligations). Every other route this guard covers is a
        single sequential pipeline over ONE declaration — the step_done route
        already merges its own fixed fields via _collect_fixed_field_failures —
        so those keep check_pre's first-failure shape and this method delegates.

        Both completion arms are collected in full so one round trip surfaces the
        whole set; leaving either on the base-class default would report only the
        first violation and cost a round trip per remaining obligation.
        """
        if self._at_task_complete(ctx):
            return self._completion_obligations(ctx)
        if (ctx.tool_name == "plan_update"
                and (ctx.tool_args or {}).get("action") == "complete"):
            # Same latch-before-judge order as check_pre's plan-close arm: the
            # registry enters through THIS method, so the act supplied on the
            # complete call itself must latch here too, or every retry blocks.
            self._latch_evidence_pass(ctx)
            return self._plan_complete_obligations(ctx)
        verdict = self.check_pre(ctx)
        return [] if verdict is None else [verdict]

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
        if self._at_task_complete(ctx):
            _pending = self._completion_obligations(ctx)
            if _pending:
                return _pending[0]
            return None

        # Plan-close arm: the voluntary act of closing the plan is where the
        # "point the task's own named decider at the delivered bytes" obligation
        # is checked. Deliberately NOT on [TASK_COMPLETE]: a refusal here leaves
        # an obvious next action (finish the step, close again), whereas a
        # refusal on the completion sentinel leaves only an argument — the exact
        # livelock that got the old gate switched off. Evaluated before the
        # step_done route so the two never interleave.
        if ctx.tool_name == "plan_update":
            if ctx.tool_args.get("action") == "complete":
                # LATCH BEFORE JUDGING. The agent may supply `arbiter_act` on the
                # very call that closes the plan — that is a natural thing to do
                # ("here is the reading, done"). If the obligation were evaluated
                # first, the act would never reach the latch pass (whose normal
                # call site sits below, past this early return), so the guard
                # would block every retry forever: a livelock with the fix
                # already in hand. The pass is verdict-free and idempotent, so
                # running it here can only let the SAME submission satisfy the
                # check it is being judged against.
                self._latch_evidence_pass(ctx)
                _pending = self._plan_complete_obligations(ctx)
                if _pending:
                    return _pending[0]

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

            # EVIDENCE LATCHING PRECEDES ROUTE DISPATCH.
            #
            # Every plan_update route (single step_done, batch, and any future
            # route) has its own formal-field checks that RETURN EARLY. Those
            # returns answer a different question from the one this pass
            # answers:
            #   * did an observation really happen?  -> corpus-checked, durable
            #   * is this claim formally complete?   -> the per-route checks
            # Measured failure this prevents (10 consecutive rounds): a call
            # carried an acceptance_act whose `observed` text had ALREADY passed
            # the non-forgeable corpus-membership check, but the same call also
            # missed a cheap formal field, so an early return discarded the
            # verified evidence. The agent re-ran the probe, re-submitted, missed
            # another field, and the latch never set — many correct probe runs
            # and a completion boundary that never releases.
            #
            # Nothing is weakened: every route still blocks exactly as before.
            # Only the EVIDENCE SURVIVES the block, so the next attempt (which
            # still must fix the formal fields) does not re-establish facts the
            # harness has already verified. The pass is idempotent and never
            # returns a verdict — it can only ever raise a latch to True.
            self._latch_evidence_pass(ctx)
            
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
                #
                # MERGED REPORTING (M13): this check no longer returns on its own.
                # It is evaluated together with the Mode 1/2 evidence check and the
                # three fixed-field declarations, and ALL unmet ones are reported in
                # ONE message. Rationale is the measured M8 lesson: the model can
                # only fix what it is told about, and every round trip is a full
                # model call, so short-circuiting turns a purely mechanical shape
                # error into a guess→block→guess loop (measured: four sequential
                # blocks on a single step_done). Nothing is weakened — every check
                # still blocks exactly as before, and any unmet one still blocks the
                # whole call; only the reporting is merged.
                if not self._step_done_recheck_reminded and override_reason:
                    # Override provided → agent paused to re-check. Release for
                    # good (content is never checked), and do not exempt the step
                    # from still carrying evidence — the Mode 1/2 and fixed-field
                    # checks below still apply.
                    self._step_done_recheck_reminded = True

                # Get step's acceptance criteria if plan is available
                acceptance = self._step_acceptance(step_id)

                # Mode 1: Step has acceptance → require verification list
                # Mode 2: No acceptance (simple step) → require override_reason
                _mode_bad = None
                if acceptance:
                    if not verification:
                        _mode_bad = GuardVerdict.block(
                            message=_VERIFICATION_REQUIRED_WITH_ACCEPTANCE.format(
                                acceptance="\n".join(f"  • {a}" for a in acceptance)
                            ),
                            reason="step_done_with_acceptance_no_verification",
                            category="verification_required"
                        )
                else:
                    if not override_reason:
                        _mode_bad = GuardVerdict.block(
                            message=_VERIFICATION_REQUIRED_NO_ACCEPTANCE,
                            reason="step_done_no_verification",
                            category="verification_required"
                        )

                # The convention declaration is a FIXED, REQUIRED field on EVERY
                # step_done — no tag, no regex, no opt-in. "Not applicable" is
                # itself an explicit declaration ({"name": "none"} or "none"),
                # so a silent omission is impossible. Runs AFTER the mode checks
                # so the agent first supplies its evidence, then the declaration.
                dr = ctx.tool_args.get("deliverable_reading")
                # The unverified-assumption audit is a FIXED, REQUIRED field on
                # EVERY step_done, parallel to convention/deliverable_reading:
                # reasoning is not evidence, so any premise left untested must be
                # surfaced — an affecting one tested, a non-affecting one
                # justified. "none" is an explicit owned claim.
                bad = _collect_fixed_field_failures([
                    _echo_received(
                        None if (self._step_done_recheck_reminded or override_reason)
                        else GuardVerdict.block(
                            message=_STEP_DONE_RECHECK_REMINDER,
                            reason="step_done_premise_recheck",
                            category="verification_required",
                        ),
                        "_override_reason", override_reason),
                    _mode_bad,
                    _echo_received(
                        _check_convention_field(ctx.tool_args.get("convention")),
                        "convention", ctx.tool_args.get("convention")),
                    _echo_received(
                        _check_deliverable_reading_field(
                            dr, self._task_text(), self._observed(ctx)),
                        "deliverable_reading", dr),
                    _echo_received(
                        _check_unverified_assumptions_field(
                            ctx.tool_args.get("assumptions"), self._observed(ctx)),
                        "assumptions", ctx.tool_args.get("assumptions")),
                    _echo_received(
                        self._instrument_obligation(
                            ctx.tool_args.get("convention"),
                            ctx.tool_args.get("differential"),
                            ctx.tool_args.get("negative_control"),
                            ctx.tool_args.get("bound_margin")),
                        "differential/negative_control/bound_margin",
                        {"differential": ctx.tool_args.get("differential"),
                         "negative_control": ctx.tool_args.get("negative_control"),
                         "bound_margin": ctx.tool_args.get("bound_margin")}),
                    # The acceptance observation is a FIXED, REQUIRED field on
                    # EVERY step_done once the task text is groundable -- the same
                    # bar convention/deliverable_reading/assumptions carry.
                    #
                    # WHY: the requirement-coverage sweep used to live ONLY
                    # inside `if _acceptance_act_recorded(...)`, and a measured
                    # trial made 21 step_done calls with ZERO acceptance_act
                    # submissions, so the sweep never executed once. It then
                    # delivered an artifact violating a requirement it had itself
                    # written down. The obligation at plan-close did not help,
                    # because it fires only at the end -- after every step has
                    # already been waved through. Making the act a per-step
                    # obligation is what puts the sweep on the actual path.
                    #
                    # BUDGET SAFETY: the expensive part of this gate is the
                    # N-constraint SWEEP, and that is where the wall-pressure
                    # valve lives (`_coverage_gate_affordable`, unchanged). The
                    # obligation to record ONE observed act is a single probe the
                    # agent should be running anyway, so it is not degraded --
                    # degrading it would re-open the exact hole being closed.
                    _echo_received(
                        self._acceptance_act_obligation(
                            ctx.tool_args.get("acceptance_act"), ctx, "acceptance_act"),
                        "acceptance_act", ctx.tool_args.get("acceptance_act")),
                ])
                # EVIDENCE LATCHING PRECEDES FORMAL BLOCKING.
                #
                # The two questions below are INDEPENDENT:
                #   * did an observation really happen?   -> corpus-checked, durable
                #   * is this claim formally complete?     -> the declarations above
                # Blocking on the second must NOT destroy the answer to the first.
                #
                # Measured failure this prevents (10 consecutive rounds): a
                # step_done carried an acceptance_act whose `observed` text had
                # ALREADY passed the non-forgeable corpus-membership check, but
                # the same call also missed a cheap formal field, so the old
                # `return bad` below ran BEFORE this block and discarded the
                # verified evidence. The agent re-ran the probe, re-submitted,
                # missed another field, and the latch never set — many correct
                # probe runs and a completion boundary that never releases.
                #
                # Nothing is weakened: `bad` still blocks this call exactly as
                # before. Only the EVIDENCE SURVIVES it, so the next attempt
                # (which still must fix the formal fields) does not have to
                # re-establish facts the harness has already verified.
                _act = ctx.tool_args.get("acceptance_act")
                if _acceptance_act_recorded(_act, self._observed(ctx)):
                    # Coverage gate (M10): ONE observed act is a proxy when the
                    # statement enumerates several checkable constraints. The
                    # LATCH itself already happened in `_latch_evidence_pass`
                    # (which skips the latch on an uncovered act); what remains
                    # here is the verdict — name the uncovered constraints and
                    # block, unless the wall is nearly spent.
                    _miss = _constraint_coverage_missing(
                        _act, self._task_text(), self._observed(ctx),
                        judge=getattr(ctx, "judge", None))
                    if _miss and self._coverage_gate_affordable():
                        _coverage_bad = GuardVerdict.block(
                            _completion_constraint_coverage_missing(_miss),
                            reason="acceptance_constraint_coverage_missing",
                            category="verification_required",
                        )
                        if bad is None:
                            return _coverage_bad
                if _oracle_exercised(
                        ctx.tool_args.get("oracle_act"), self._observed(ctx),
                        self._bound_oracle().get("first_exercise"), ctx.judge):
                    self._oracle_exercised_seen = True

                if bad is not None:
                    # The premise re-check is a ONE-SHOT checkpoint: it must fire
                    # again on the next step_done if the agent has still not
                    # paused, so latching stays tied to an accepted release
                    # (override_reason present), never to merely having been
                    # told about it in a merged message. NOTE: the evidence
                    # latches above are ALREADY set — blocking here does not
                    # discard a verified observation (see the comment there).
                    return bad
                if _declares_graded_reading(dr):
                    self._graded_reading_seen = True
                return None
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
                        dr = u.get("deliverable_reading")
                        if dr is None:
                            dr = ctx.tool_args.get("deliverable_reading")
                        asm = u.get("assumptions")
                        if asm is None:
                            asm = ctx.tool_args.get("assumptions")
                        # Same bar for the acceptance observation: the item's OWN
                        # act first, top level as fallback -- exactly the shape of
                        # the conv/dr/asm fallback above. Without this the batch
                        # route is the escape hatch: measured trial used the batch
                        # route on every step, the fixed-field list below carried
                        # only the three earlier declarations, and
                        # `acceptance_act` appeared ZERO times in 2664 log lines.
                        act = u.get("acceptance_act")
                        if act is None:
                            act = ctx.tool_args.get("acceptance_act")
                        bad = _collect_fixed_field_failures([
                            _echo_received(
                                _check_convention_field(conv),
                                "updates[].convention", conv),
                            _echo_received(
                                _check_deliverable_reading_field(
                                    dr, self._task_text(), self._observed(ctx)),
                                "updates[].deliverable_reading", dr),
                            _echo_received(
                                _check_unverified_assumptions_field(
                                    asm, self._observed(ctx)),
                                "updates[].assumptions", asm),
                            _echo_received(
                                self._acceptance_act_obligation(
                                    act, ctx, "updates[].acceptance_act"),
                                "updates[].acceptance_act", act),
                        ])
                        if _declares_graded_reading(dr):
                            self._graded_reading_seen = True
                        if bad is not None:
                            return bad

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

    def _instrument_obligation(self, conv, differential, negative_control,
                              bound_margin=None):
        """Instruments 2 & 4: the obligations that follow from what was DECLARED.

        Both are conditional by design — the point of the measured lesson is that
        a gate applied everywhere gets routed around, so each fires only on the
        declaration that creates the risk:

        * DIFFERENTIAL fires when the step's `convention` NAMES a convention (i.e.
          the output rests on a quantity the agent DEFINED) while declaring that
          definition non-conflicting. The shape it targets: a quantity measured
          under the agent's own scope, self-scored PASS, while the named tool's
          own scope for that same quantity lands outside the stated bound. The
          obligation asks for the SAME quantity down a second, mechanically
          different path, because the interesting outcome is not "my code has a
          bug" but "this number is decided by my definition".

        * NEGATIVE CONTROL fires when the plan's calibration said
          "no_known_case" — the instrument was never checked against a known
          answer, so the only remaining evidence that it can detect failure is
          seeing it go RED on an input that must fail it.

        Returns a GuardVerdict.block to MERGE with the other fixed-field failures,
        or None. With no active plan both obligations are skipped (unknown state
        must not block).
        """
        conv_name = ""
        conflict = ""
        if isinstance(conv, dict):
            conv_name = str(conv.get("name") or "").strip()
            conflict = str(conv.get("conflict") or "").strip().lower()
        named = bool(conv_name) and conv_name.lower() != "none"

        # Instrument 2: a NAMED convention whose conflict the agent resolved on
        # its own must be recomputed down an independent path -- BUT ONLY when
        # the task actually named a deciding tool. That scoping is not leniency,
        # it is what the instrument is FOR: a differential answers "does this
        # quantity depend on MY definition?", a question that only has teeth when
        # some OTHER definition is in play. With no named arbiter there is no
        # rival definition to differ from, so demanding two paths would be
        # ceremony -- and ceremony is what gets routed around.
        #
        # The measured loss this targets: the task text named a tool as the
        # ground truth; the agent measured the quantity over only the region it
        # chose (self-scored PASS) while that named tool's own scope put the same
        # quantity out of the stated bound. Two paths there would have surfaced
        # the choice.
        arbiter = ""
        decl = self._bound_oracle() if self._plan else {}
        if isinstance(decl, dict):
            arbiter = str(decl.get("named_arbiter") or "").strip()
        has_named_decider = bool(arbiter) and arbiter.lower() != "none"

        if (named and conflict == "no" and has_named_decider
                and not differential_satisfied(differential)):
            verdict = check_differential(differential)
            if verdict is not None:
                reason, msg = verdict
                return GuardVerdict.block(
                    # NOT .format(): the message body itself contains literal
                    # brace groups (e.g. "step_done(differential={...})") that
                    # str.format would try to interpret.
                    message=msg.replace("{name}", conv_name or "unspecified"),
                    reason=reason,
                    category="verification_required",
                )

        # Instrument 4: an uncalibrated instrument must be shown able to fail.
        cal = {}
        if self._plan:
            try:
                active = self._plan.get_active()
                if isinstance(active, dict):
                    cal = active.get("calibration") or {}
            except Exception:
                cal = {}
        if calibration_needs_negative_control(cal):
            verdict = check_negative_control(negative_control)
            if verdict is not None:
                reason, msg = verdict
                return GuardVerdict.block(
                    message=msg,
                    reason=reason,
                    category="verification_required",
                )

        # Instrument 5: a BOUND claim whose margin is smaller than the shift a
        # rival definition of the same quantity would cause.
        #
        # Fires only when the step DECLARES a bound_margin at all. It is not a
        # new required field on every step: a step making no bound claim has
        # nothing to decide this way, and demanding the declaration everywhere is
        # the ceremony that gets routed around. The declaration creates the
        # obligation, exactly as `convention` and `calibration` do.
        #
        # The measured loss this targets: the agent's own number sat one unit
        # inside a stated upper limit while the construction that the task's
        # named arbiter uses for that same quantity came out several units
        # larger -- outside the limit. Its self-check reported PASS, and the
        # margin was never compared against the size of that disagreement. The
        # harness now does that comparison instead of trusting a
        # `within_tolerance` claim.
        if bound_margin is not None:
            verdict = check_bound_margin(bound_margin)
            if verdict is not None:
                reason, msg = verdict
                return GuardVerdict.block(
                    message=msg,
                    reason=reason,
                    category="verification_required",
                )
        return None

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
