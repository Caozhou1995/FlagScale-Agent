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

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


_VERIFICATION_REQUIRED_WITH_ACCEPTANCE = """[VerificationGuard] Before this step is done — answer one question per criterion.

Do NOT describe what you did ("ran the tests", "checked the file"). That is an
account of your activity, not evidence the criterion holds. For EACH criterion
below, answer these three, and notice if you cannot:

  1. OBSERVED — what concrete value / output did you actually see? (a number, a
     count, an exit code, an exact string — not "it worked", not "looks correct")
  2. EXPECTED — what value did the criterion require?
  3. GAP — observed vs expected: match, or off by how much?

If for any criterion you have no concrete OBSERVED value to write — that is the
signal the criterion is NOT verified yet. Go get the value; do not invent one to
fill the slot. A verification entry with no observable value in it is the thing
this guard exists to catch.

Retry with one verification entry per criterion, each carrying its observed value:
  plan_update(action="step_done", step_id=N, verification=[
    "criterion 1 — observed: <value>; expected: <value>; gap: <match/diff>", ...])

Acceptance criteria for this step:
{acceptance}
"""

_VERIFICATION_REQUIRED_NO_ACCEPTANCE = """[VerificationGuard] Before this step is done — answer, do not assert.

This step has no acceptance criteria, so answer this in your _override_reason:
what did you OBSERVE that tells you the step's goal is met — a concrete value or
output you actually saw (a count, an exit code, an exact string), not "it works"
or "should be fine". If you have no observed value to point to, the step is not
done yet — go get one rather than asserting completion.

  plan_update(action="step_done", _override_reason="observed <value/output> which
    shows <goal> is met")
"""

_ACCEPTANCE_FROZEN = """[VerificationGuard] Acceptance change blocked — criteria are frozen.

Step {step_id} is already "{status}" — implementation has begun. Changing its
acceptance criteria now lets the standard be shaped by what you produced, so the
verification would no longer be independent of the work. That is exactly the
failure this guard prevents: a check that cannot fail carries no information.

Define acceptance BEFORE implementing. If the original criteria were genuinely
wrong (not merely inconvenient to meet), retry with _override_reason explaining
what was wrong with them and why the new criteria are still independent of your
implementation choices.
"""

_ACCEPTANCE_GUIDANCE = """[VerificationGuard] Writing acceptance criteria — a note on what makes them useful.

Re-read the task statement and confirm you executed every operation it explicitly asks for.
If the task names artifacts, verify they exist at the named paths. "I know how to
do X, so X is done" is the signal to stop and execute X.

Acceptance should describe an externally observable end state — a file exists at a
specific path, a command exits 0 / produces expected output, an artifact has an
expected property. Reproduce the judge's condition from scratch; a check that
shares its premise with the implementation cannot fail.

Traps — same root cause: letting implementation, not the task contract, define
verification. These four are cross-task; they apply no matter the domain:

  • Cover every input the task gives you, not just the one you developed against.
    Passing on your debug sample says nothing about the ones the judge scores. Run the solution on each
    input artifact. Ask: what inputs have I not yet run it on?
  • Verify generalization, not answer-recall. For a blind sample, verify structurally valid
    output (right shape, parses, no crash); you cannot verify a number you were
    never told. A green result on the only visible sample shows the code runs, not
    that it generalizes — it is an accident, not a promise. Only features the task
    contract names as fixed are guaranteed on the scored input.
  • Don't excuse a real failure by origin. The question is not whether you
    introduced it, but whether it falls within what the task asks you to deliver.
    "pre-existing / external / unrelated to my change" is a CLAIM that needs
    evidence. The task judges the result, not its origin — a failure inside your
    declared scope is yours to fix, regardless of who introduced it.
  • Before delivering, verify using the judge's measurement. Use the verifier
    script or reproduce its logic. A solution that passes your proxy but fails the
    judge's measure still scores zero.

The situational traps — reloading the delivered artifact fresh, backup+rollback on
overwrite-then-measure, exact-contents of the delivery dir, byte-immutability of
inputs, margin on noisy metrics, best-so-far write-through — apply only at DELIVERY,
not at plan time. They are enforced as a completion-time check, where they are
actionable; do not carry them here.

Check for a constraint or hint you skipped — designing a recovery strategy is not the same as executing the write.
If the same wall stops you twice, narrows the problem before the next attempt.
Hitting the limit of the method is different from hitting the limit of one tuning parameter — a problem made easy will keep failing no matter how you tune it.
Shape your method before you commit.
Approach from the givens, not from your implementation.
"know how to do X, so X is done" is assuming completion from intent.
Verify those exact artifacts exist at the named paths.
Let the data decide the path. Ask: what you have not yet built?
"""

# NOTE: The plan_create qualifier-extraction reminder (_PLAN_CONSTRAINT_REMINDER)
# was moved to plan.py (_QUALIFIER_EXTRACTION). Reason: it belongs at the
# plan-framing moment, which PlanGuard owns. Keeping it here as a separate
# Timing -1 block caused guard override_reason cross-talk — PlanGuard's
# single-shot block trains the agent to pass _override_reason, which then
# silently satisfied this guard's block too, so it never actually fired.
# See plan.py check_pre and pitfall/flagscale_agent/guard_override_crosstalk.

_BATCH_STEP_DONE_BLOCK = """[VerificationGuard] Batch update marks a step done — same bar as a single step_done.

You are marking one or more steps "done" via action="batch". A batch done is not a
lighter action than a single plan_update(step_done) — it commits the same claim that
the work landed, and whatever depends on those steps will trust the result without
re-examining it. Marking done in bulk must not become a way to skip the verification
each step would otherwise require.

To proceed, re-issue the batch with "_override_reason" affirming that for EACH step you
are marking done, you re-checked the result against what that step was supposed to
satisfy — not that the process felt right, but that the output, checked directly, meets
the words. This is a forcing checkpoint, not a content check; any override_reason lets
the batch through. If a step's completion deserves real evidence, prefer marking it via
plan_update(step_done, verification=[...]) individually rather than folding it into a batch."""

_STEP_DONE_RECHECK_REMINDER = """[VerificationGuard] Before this step is marked done — check your own premises, once.

You are about to finalize a step, and whatever depends on it will trust the result
without re-examining it. This is the last point at which a wrong turn is cheap to
catch, so spend it well.

Verification is not restating what you believed while doing the work — it is testing
that belief against what you actually got. The dangerous failure here is quiet: the
work ran to completion, produced a confident result, and the account you would write
of it simply repeats the assumptions you operated under. If one of those assumptions
was wrong, repeating it does not reveal the error, it launders it into a conclusion.

So separate the two. For each thing the step was supposed to satisfy, read it again
as written and ask whether the result you hold actually lands inside it — not whether
your process felt right, but whether the output, checked directly, meets the words.
Then look at the premises you leaned on to get there: any reading of what a term
means, any "this is close enough", any value you took as given rather than confirmed.
A premise you adopted for convenience is exactly the one most likely to be wrong, and
the one you are least likely to question because it never announced itself. Treat each
such premise as the thing under test, not as settled ground you build the check on.

And check the target, not just the workmanship: re-read what the user actually asked
for and confirm this step moves toward THAT, not toward a nearby goal your own framing
swapped in. Verifying a result thoroughly is worthless if it is the result to the wrong
question — competence aimed off-target still misses. When your reading of the task and
its plain wording pull apart, the wording wins.

To proceed, re-issue plan_update(step_done) with "_override_reason" stating that you
re-checked the result against each criterion directly and re-examined the premises you
relied on. This is a forcing checkpoint, not a content check — any override_reason
lets the step through; the point is that you stopped to look before committing."""

_TASK_COMPLETE_RECHECK_REMINDER = """[VerificationGuard] Before completing the whole task — one look back at the finish line.

After this, nothing re-examines whether what you delivered was the best you could do.
Spend this one checkpoint deliberately.

First, re-read the task as the USER stated it and confirm what you are about to
deliver answers THAT, not a nearby question you drifted into. The failure this
catches is confident competence aimed slightly off-target: you solved a problem
thoroughly but it was an adjacent problem your own framing substituted. The drift
enters through an early premise (a value parsed, a state reconstructed, an input
interpreted) that went unquestioned while all your effort piled onto steps
downstream. Deep downstream checking builds false confidence because it never
revisits that upstream premise. When your polished answer and the user's plainly
stated need diverge, the need wins.

Now trace where your answer came from. Did I OBSERVE this — read it from a tool
output, a command's stdout, a file I actually opened — or did I INFER it from what
I expected to be there? Prior knowledge tells you what is PLAUSIBLE, never what is
ACTUALLY there, and the task is graded on what is actually there. (Answers you
legitimately COMPUTED — a number you derived, a result your own code produced —
are observed.)

Then decide what kind of task this was, because that decides what "done" means:

  • pass/fail condition — a build, test suite, artifact existence: "done" means the
    condition is met. Hold confirmation to the real condition, not a signal arranged
    to look like it. If the task named a non-negotiable specific (version, tool,
    format) and you could not obtain it, "done" is NOT reachable by delivering a
    substitute: standing up directories or a wrapper so a sanity command exits 0
    dresses a substitute to pass the check, and the check was only ever a proxy for
    the genuine artifact. Disclosing the swap does not rescue it — honesty is not permission: a disclosed substitute is still a substitute. When the genuine
    article is unreachable, the only honest closes are to keep searching for a legal
    path or report BLOCKED with nothing counterfeit delivered — an empty honest BLOCKED beats a populated fake. Also check WHERE you ran it: a pass in your own
    working environment (installed packages, env vars, helper files, services) does
    not prove it works in the clean environment that will actually consume it. The
    deliverable must carry its dependencies inside itself — a shipped script must
    not import a library you pip-installed only locally. A distinct flavor: not
    what you ADDED but HOW the deliverable gets addressed — your interactive shell
    resolves it via PATH/rc/alias/cwd, while the consumer invokes it bare as a
    non-login non-interactive subprocess. "It runs when I TYPE it" is not "it runs
    when a PROGRAM calls it": an artifact meant to be found by name must live in
    the target's standard install location, not merely be reachable through your
    session's config. Re-run the check the bare way the consumer will, or ask
    "would this still pass without the setup I personally added AND without my
    shell having been primed to find it?"
    One more binary trap: if the task named a precise list of exceptions, that
    enumeration is a closed constraint. A failure OUTSIDE the list does not earn a
    place on it. Quietly widening the carve-out is rewriting the acceptance bar to
    match your result. And "this failure is pre-existing / external / unrelated to
    my change" is a claim that needs evidence, not a default exit. If an item
    outside the exemption list still fails, "done" is not reached.
    Finally, if you developed against ONE visible sample but the grade is on hidden
    inputs — passing on that sample is the floor, not the finish. The tell is a
    solution full of magic constants (absolute threshold, fixed cutoff) you tuned
    until the sample came out right. Before completing, audit every hardcoded number
    and prefer relative / normalized / structure-derived judgments. You cannot
    confirm generalization by testing on one sample — the burden is to make the
    method principled, not sample-calibrated.

  • open-ended — optimize, minimize, maximize, make it faster: the first working
    result is NOT the finish line. Ask: is there another distinct method I have not
    tried? Did I actually measure alternatives, or just plan and ship the first?
    Judge whether your "distinct" attempts were actually distinct: if results all
    land in the same narrow band — the same order of magnitude — that near-tie
    means they were variants within one method-family, and you have hit the depth limit
    of that family. The response is to widen: reach for a structurally different family that attacks from another angle. You do not need to know the target's absolute best; the flat spread is enough signal. Only after trying a
    genuinely different family and it too fails is "reached the method's limit"
    honest — otherwise you have reached one family's limit and called it the task's.

Whichever kind it is, name which kind it is — then run these orthogonal self-checks
against the task's REAL purpose (the actual use your work serves), not against any
imagined check. These axes are independent — a result can be fully optimized yet
fail to generalize, or general yet not the best available. Answer each on its own:

  1. OPTIMIZED — is what you hold the best you reached, or just the first thing that
     worked? If a distinct method could plausibly do better and you have not ruled
     it out by measurement, not done.
  2. GENERAL — does your solution handle the whole CLASS, or did you special-case
     it to the one input you developed against? Would it produce a well-formed
     answer on a sibling input that differs in the ways the task does not fix?
  3. GENERALIZES — you verified under conditions you could OBSERVE. The real use
     imposes conditions you could NOT observe. Name that gap concretely, then say
     whether you REPRODUCED it and read the result, or merely argued it would be
     fine. The gap is never zero — manufacture it (perturb the input, run in a fresh
     context) and watch what happens first.

If an axis does not apply, say so. To proceed, re-issue plan_update(action="complete")
with "_override_reason" that answers all three against the real purpose. This is a
forcing checkpoint, not a content check — any override_reason lets it through; the
point is that you stopped and looked along every axis before deciding you are done."""


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
# presence means the agent is reasoning ABOUT the method rather than reporting a
# measurement.
# Observation markers: signs the reason reports something the agent actually DID
# and READ — ran a command/test, compared to a known/expected/ground-truth value,
# read a concrete output. Presence of any of these means it is not pure argument.
_OBSERVATION_MARKERS = (
    "ran ", "i ran", "output was", "printed", "returned", "measured",
    "observed", "compared", "matches", "matched", "ground truth", "ground-truth",
    "expected value", "known answer", "against the sample", "on the sample",
    "test passed", "tests passed", "test failed", "pytest", "assert",
    "reproduced", "in a clean", "fresh environment", "exit code", "stdout",
    "the result was", "got ", "equals", "==", "diff ",
)


def _has_observation(reason: str, classify_fn=None) -> bool:
    """True when the reason reports a positive OBSERVATION — something the agent
    ran and read: a command/test executed, an output/value read, a comparison to
    a known/expected/ground-truth answer.

    Primary path is an LLM judge (classify_fn): a wrong answer has no lexical
    fingerprint, and the keyword MARKERS below both false-positive on innocent
    phrasings and are gameable by sprinkling observation words into the override
    reason. When a classify_fn is available we ask it semantically. The regex
    MARKERS remain as a FALLBACK only when no provider is wired (tests, offline).

    The judge answers "reason_lacks_observation" = True when the reason is
    argument-only; this function returns the inverse (has_observation = not lacks).
    """
    if not reason:
        return False
    if classify_fn is not None:
        # default=False → on judge failure, treat as "lacks observation" is False,
        # i.e. do NOT fabricate a block; the fires-once gate already bounds nagging.
        lacks = classify_fn(
            "reason_lacks_observation", {"reason": reason}, default=False
        )
        return not lacks
    low = reason.lower()
    return any(m in low for m in _OBSERVATION_MARKERS)


# Third-stage markers. The second gate passes any reason that reports an
# observation. But an observation can be TRUE and still overfit: the agent ran
# something and read a value, yet only on the ONE sample it developed against —
# tuning constants until that visible case came out right — while the real grade
# lands on inputs it never saw. Passing the visible sample is the FLOOR, not the
# goal. These markers catch the runtime form of that: the reason reports work
# confined to the development sample (or admits fitting/tuning to it) but shows no
# sign the method was tested for GENERALIZATION beyond that one instance.

# Sample-local (overfit) markers: the method was FITTED or TUNED to the single
# sample the agent could see. The distinguishing tell is fitting, not location —
# merely observing "on the sample" or comparing output to a known/expected answer
# is legitimate verification (it is the second gate's own prescribed escape), so
# those are deliberately NOT markers here. Only explicit fitting/tuning language —
# the sign that constants were bent until the visible case came out right — counts.
_SAMPLE_LOCAL_MARKERS = (
    "tuned", "tuning", "adjusted until", "tweaked until", "tweaked it until",
    "calibrated", "fit to", "fitted to", "overfit", "hand-picked", "handpicked",
    "hardcoded", "hard-coded", "magic constant", "magic number",
    "until it matched", "until it worked", "until it came out",
    "got it right by", "picked so that", "chosen so that", "set so that",
    # Magic-ASSUMPTION markers: the non-numeric twin. These reveal the agent bound
    # a CONCEPT to the one concrete FORM it took in the visible sample — a fixed
    # prefix / exact label / assumed format — WITHOUT enumerating the field's real
    # value universe. Kept narrow (each phrase implies an assumption about a
    # categorical value's shape, not a generic mention of a string/filter) so the
    # gate's no-false-positive posture holds. The escape is a generalization marker
    # below, which now includes value-universe exploration (checking the field's
    # actual distinct values) — i.e. the agent who checked the real range passes.
    "assumes the format", "assumed the format", "assume the format",
    "assumes the value", "assumed the value", "assumes the role", "assumed the role",
    "always starts with", "always formatted", "always formatted as",
    "hardcoded prefix", "hard-coded prefix", "hardcoded string", "hard-coded string",
    "exact prefix", "exact-form", "exact form match", "only form", "single form",
    "starts-with filter", "starts with filter", "starts-with the string",
)

# Generalization markers: signs the agent went BEYOND the single sample — ran on a
# different / hidden / held-out input, or manufactured a perturbed stress input and
# read whether the output stayed stable. Presence of any means it is not
# sample-local-only.
_GENERALIZATION_MARKERS = (
    "hidden", "held-out", "held out", "unseen", "different input",
    "other input", "another input", "perturbed", "perturbation", "stress input",
    "stress test", "rescaled", "reframed", "reordered", "noised", "variant",
    "generaliz", "out-of-sample", "out of sample", "multiple inputs",
    "several inputs", "stayed stable", "remained stable", "did not swing",
    "each input", "across inputs",
    # Value-universe exploration: the escape for the magic-ASSUMPTION markers. An
    # agent who enumerated the field's real distinct values BEFORE committing a
    # filter has calibrated to the concept's true boundary, not to the sample's
    # accidental form — that IS generalization for categorical matching.
    "distinct values", "distinct query", "value universe", "all distinct",
    "enumerated the values", "enumerated the categories", "checked the range",
    "unique values", "full range of values", "set-membership", "set membership",
    "value list", "normalized comparison", "normalized match", "all the forms",
    "all forms", "every form", "range of forms",
)


def _is_sample_local_only(reason: str, classify_fn=None) -> bool:
    """True when the reason reports an observation confined to the ONE visible
    sample (or admits fitting/tuning to it) with NO sign of generalization.

    Runtime form of the prompt's single-sample-overfit warning: "it works on the
    example" is the beginning of the check, never the end. A reason that shows the
    method was exercised on a different / hidden / perturbed input passes; a reason
    that only reports sample-local work — and names fitting or tuning to that one
    case — is caught once. A reason with neither signal is left alone (no false
    positive).
    """
    if not reason:
        return False
    if classify_fn is not None:
        return classify_fn(
            "reason_overfits_sample", {"reason": reason}, default=False
        )
    low = reason.lower()
    has_local = any(m in low for m in _SAMPLE_LOCAL_MARKERS)
    has_generalization = any(m in low for m in _GENERALIZATION_MARKERS)
    return has_local and not has_generalization


# Fourth-stage task-complete gate — GIVEN-vs-RANGE substitution. Distinct from the
# observation/generalization gates: those catch a result that is unverified or
# overfit; this catches a result the agent KNOWS is not the thing the task named and
# is delivering anyway, wrapped in a disclosure. The prompt's CONSTRAINT LOYALTY rule
# says a GIVEN (a point value the task states exactly — a version, timestamp, size,
# filename, commit) has zero tolerance, and disclosing the substitution ("I could not
# get X so I used near-equivalent Y") does NOT legalize delivering Y. The runtime tell
# is the disclosure vocabulary itself: the agent narrates why a substitute should count
# instead of reporting it produced the named artifact or hit BLOCKED.

# Substitution-disclosure markers: language that admits the delivered thing is not the
# named one, or argues a near-equivalent should count in its place.
_SUBSTITUTION_MARKERS = (
    "could not obtain", "couldn't obtain", "could not get", "couldn't get",
    "could not find", "couldn't find", "was unavailable", "is unavailable",
    "not available", "instead used", "used instead", "in place of",
    "backward compatible", "backward-compatible", "backwards compatible",
    "successor", "drop-in replacement", "drop in replacement", "close enough",
    "essentially the same", "near-equivalent", "near equivalent", "closest available",
    "closest match", "nearest version", "different version", "newer version",
    "older version", "may differ", "slightly different", "approximation of",
    "substitute", "substituted", "替代", "近似", "差不多", "版本不同", "兼容",
)

# BLOCKED-report markers: signs the agent is NOT delivering the substitute but is
# instead reporting inability to meet the constraint — the honest outcome. Presence of
# any means the disclosure is a BLOCKED report, not a delivered counterfeit.
_BLOCKED_REPORT_MARKERS = (
    "blocked", "cannot proceed", "could not complete", "did not deliver",
    "delivered nothing", "no artifact", "reporting inability", "unable to complete",
    "abandon", "not the named", "not delivering", "refuse to substitute",
    "report blocked", "报告blocked", "无法完成", "未交付",
)


def _is_disclosed_substitution(reason: str, classify_fn=None) -> bool:
    """True when the reason DISCLOSES delivering a substitute for a GIVEN value
    without reporting BLOCKED.

    Runtime form of CONSTRAINT LOYALTY: disclosing a substitution does not legalize
    it. A reason that names substitution/near-equivalent vocabulary AND does not frame
    the outcome as BLOCKED (no counterfeit delivered) is caught once. A reason that
    reports inability (BLOCKED) passes — that is the honest path. A reason with neither
    signal is left alone (no false positive)."""
    if not reason:
        return False
    if classify_fn is not None:
        return classify_fn(
            "reason_discloses_substitution", {"reason": reason}, default=False
        )
    low = reason.lower()
    has_substitution = any(m in low for m in _SUBSTITUTION_MARKERS)
    has_blocked = any(m in low for m in _BLOCKED_REPORT_MARKERS)
    return has_substitution and not has_blocked


_TASK_COMPLETE_OBSERVATION_DEMAND = """[VerificationGuard] Your reason argues the method; it does not report an observation.

Read it back: it explains why your approach OUGHT to be right — "reasonable",
"conservative", "should generalize", "relative to the signal". That is reasoning
ABOUT the method. It never states what you RAN and what you READ. The consumer does
not grade your justification; it grades your output against ground truth, and a
confident rationale for a wrong answer is still a wrong answer.

The litmus: is your evidence an OBSERVATION (you ran something and read a result)
or an ARGUMENT (you explained why the result should be right)? An argument lives in
the same head that wants the task done, so it satisfies itself for free. Only an
observation can contradict you.

So produce one, then complete. If the task handed you a sample whose correct answer
you can derive or measure independently, RUN your solution on that sample and COMPARE
its output to the known answer — report the two values and whether they match. If the
real grade is on hidden inputs, MANUFACTURE a stress input: write a perturbed copy of
your sample to disk (rescaled, reframed, reordered, noised — perturb what the task does
NOT measure), feed it through your actual solution, and READ whether the output stays
stable or swings/crashes. Either way your override_reason must name a concrete result
you OBSERVED — a value, an output, a pass/fail — not another sentence about why the
method is sound.

To proceed, re-issue plan_update(action="complete") with "_override_reason" reporting
that observation. This gate fires once — if you genuinely cannot run any check (no
derivable sample, nothing to perturb), say so explicitly and it releases."""


_TASK_COMPLETE_GENERALIZATION_DEMAND = """[VerificationGuard] Your observation covers only the sample you developed against.

Read your reason back: it reports what you ran and read — but every result sits on
the ONE instance you could see, and it names fitting or tuning to that instance.
That is the FLOOR, not the goal. A score you can see is only a SAMPLE of a real need
that lands on inputs you cannot see. Passing the visible case proves your method fits
THAT case; it says nothing about whether it generalizes — and a method tuned until the
one sample came out right is exactly the one that breaks on the next input.

The gap between the conditions you developed under and the conditions your result is
graded under is never zero — if it were, there would be no unseen input. So do not
argue the method "should generalize": name that gap and REPRODUCE it. You are not
limited to the one sample — MANUFACTURE a stress input from it. A stress input is not
a sentence about what might break; it is a NEW input artifact you write to disk: take
your sample and perturb some property the task does NOT measure — its scale, length,
framing, ordering, resolution, noise — while leaving intact what the task DOES measure.
Feed that variant through your ACTUAL solution and READ the output. Either it stays
stable (evidence of generalization) or it swings wildly / crashes (the overfit,
surfaced now instead of by the hidden grade). A crash on your own variant is the most
valuable signal you can get: proof the method has a precondition the pristine sample
happened to satisfy.

This applies with full force to the quieter, NUMBER-FREE version of the overfit: the
MAGIC ASSUMPTION. If your reason keys a filter / match / parse on the one concrete FORM
a categorical value took in your sample — a fixed prefix, an exact label, "it always
starts with X", one spelling, one format — you bound a CONCEPT to an ACCIDENT. The
concept the task means (a category, a status, a membership class) is structural; the FORM
your sample rendered it in (one prefix, one casing, one word order, one spelling among
synonyms) is not. The hidden data is free to render the SAME concept differently — a
longer variant of the label, a synonym, an extra qualifier, a reordered compound — and
your form-keyed test then returns false and silently DROPS the very rows it should keep.
The fingerprint is a whole expected ROW missing from your output, not a present-but-wrong
attribute. The fix is not a stress input here (though that works too): it is to EXPLORE
THE VALUE UNIVERSE first — check what distinct values that field actually takes in the
data you can reach, using whatever tool fits your format — so your match is calibrated
to the concept's real boundary and every observed form, then prefer set-membership / a
normalized comparison / a broader pattern over an exact-form test. Keep the GIVEN/RANGE
line sharp: a value the TASK named (a stated count, a specific year) you reproduce
verbatim; only the forms YOU read off the sample are magic.

To proceed, re-issue plan_update(action="complete") with "_override_reason" naming a
concrete result you OBSERVED on an input OTHER than the original sample — a perturbed
variant, a held-out or hidden input, OR the enumerated distinct values of the field you
filtered on — a value, a pass/fail, stable-or-swung, the value set you saw. This gate
fires once; if there is genuinely nothing to perturb and no other input exists, say so
explicitly and it releases."""


_TASK_COMPLETE_SUBSTITUTION_DEMAND = """[VerificationGuard] Your reason discloses a substitute for a value the task named exactly.

Read it back: it says the required thing was unavailable, or that what you delivered is
a successor / backward-compatible / near-equivalent / different version — and you are
completing anyway. That is the exact move CONSTRAINT LOYALTY forbids. A value the task
states exactly (a version, a timestamp, a size, a filename, a commit) is a GIVEN: it has
ZERO tolerance. 3.0.10 is not 2.2; T+1 is not T; the successor release is not the release
named. The task never declared a tolerance band, so you do not get to widen its point
value into a range because a range would be easier to hit.

Disclosing the substitution does NOT legalize delivering it. Writing "I could not get X
so I used Y" is honest about the deviation, but honesty about a deviation is not
permission to ship it — the task still fails on its own terms, and a checker that
compares against the named artifact (a hash, a version string, an exact match) scores it
zero regardless of your footnote. "Backward compatible", "successor", "close enough",
"drop-in", "近似", "差不多" are arguments for why a substitute SHOULD count; none of them
turn Y into the X the task named.

There are exactly two honest outcomes, and neither is "complete with a disclosed
substitute": (1) keep searching for a legal path to the EXACT named thing — a different
mirror, an archive, a build from the precise source — or (2) report BLOCKED, delivering
NO counterfeit. An empty honest BLOCKED beats a populated fake.

To proceed, re-issue plan_update(action="complete") with "_override_reason" that either
reports you obtained the EXACT named artifact (name the value you verified — the version
string, the hash, the exact match), or states plainly that you are reporting BLOCKED and
did NOT deliver a substitute. This gate fires once."""


_TASK_COMPLETE_DELIVERY_HYGIENE = """[VerificationGuard] Before completing — check the DELIVERED artifact, not the process that made it.

Your reason spoke to whether the work is right. This is a different question: does what
sits at the delivery path, right now, actually carry that result — and nothing else? The
judge reads the path cold; it never sees the in-session object you configured or the
best candidate you held in memory. These traps bite at delivery, which is why they are
raised here and not at plan time. Walk the ones that apply:

  • Reloaded FRESH. Measure the artifact re-read from the delivery path in a clean state
    — not an in-memory object you configured programmatically. If settings did not
    serialize into the file, the cold-loaded artifact behaves nothing like what you
    measured. Only a measurement on the reloaded deliverable proves the round-trip.
  • Best-so-far is BANKED, not just found. If you searched, the best candidate must be
    written THROUGH to the delivery path the moment it measured better and passed
    validity — never held only in memory. A timeout or kill ships whatever is on disk;
    an unpersisted winner scores as if never found.
  • Backup + rollback on overwrite-then-measure. If measuring overwrites the delivery
    path, keep a side backup; after measuring, if the candidate is NOT strictly better,
    restore the backup BEFORE the next experiment. Otherwise a regression ships on
    timeout.
  • Exact contents. If the task names an EXACT set (often one file), the path holds that
    set and NOTHING MORE — a stray sibling fails an exact-contents check as hard as a
    missing file, and silently. A verify command the task SHOWS you is the consumer's
    action, not where your byproducts go; clean scratch/temp out of the delivery path.
  • Byte-immutability of inputs. If an input must stay UNCHANGED (hash / checksum /
    exact bytes), reverting a change is NOT never changing it — add-then-remove leaves
    the file grown and rewritten. Restore from a pre-touch backup, or operate on a copy
    and keep the original pristine.
  • Margin on a NOISY metric. If the grade is a measured number with run-to-run
    variance, passing by a hair on YOUR OWN measurement is not passing — your
    instrument and the judge's differ. Measure the deliverable rigorously (many
    repeats, trim outliers, central statistic) and demand margin proportional to the
    noise.

To proceed, re-issue plan_update(action="complete") with "_override_reason" reporting
which of these you checked on the DELIVERED artifact — or stating plainly that none
apply to this task (e.g. no file is delivered, no search, no noisy metric). This gate
fires once."""


# Pre-mortem, delivered AFTER a step_done goes through (check_post). The pre-side
# checks all ended in you asserting the step is done — confidence at its peak. This
# flips the direction: instead of "why am I right", ask "if I am WRONG, where". A
# confirming question ("did I do it right?") recruits confirmation bias and narrows
# the distribution toward "yes"; a dis-confirming question ("assume it's wrong —
# where?") forces enumeration of failure modes the confident account never sampled.
# Inject-only ON PURPOSE: whether the agent actually runs the falsifying input is
# unobservable, so this does not pretend to force it — it just puts the reversed
# question in front of the agent at the moment its guard is down.
_STEP_DONE_PREMORTEM = """[VerificationGuard] Step marked done — now flip the question, once.

Every check up to here asked whether you were right, and answered yes. That direction
recruits confirmation — you look for support and find it. So reverse it now: ASSUME the
result you just committed is WRONG, and ask where the error most likely hides.

Three moves, and the third is the one that counts:
  1. Name the single most likely failure point — a CONCRETE input condition, not
     "a parameter needs tuning". Where does your method rest on an assumption the one
     sample you saw happened to satisfy?
  2. Say what the failure would LOOK like — the observable symptom, the wrong output,
     the off-by-one, the crash.
  3. If you can construct an input that triggers it, RUN it and READ the result — a
     perturbation of the sample you have, not a thought about one. An answer you only
     argued ("it adapts", "it's principled", "it should generalize") is not an answer;
     an output you did not predict and then observed is.

This is a nudge, not a gate — it does not block and nothing checks whether you acted on
it. But the failure you are confident is not there is exactly the one this catches."""


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
    
    def __init__(self, plan=None):
        self._plan = plan
        self._post_recovery = False
        self._recovery_reminded = False
        self._acceptance_guidance_given = False
        self._step_done_recheck_reminded = False
        self._complete_recheck_reminded = False
        self._complete_observation_demanded = False
        self._complete_generalization_demanded = False
        self._complete_substitution_demanded = False
        self._complete_delivery_hygiene_demanded = False
        # Set by check_pre when a step_done is about to pass through, so the
        # paired check_post fires the pre-mortem right after that same call.
        self._premortem_pending = False

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Pre-mortem: fires immediately AFTER a step_done that passed the pre-side
        # checks. The reversal ("assume you're wrong") lands hardest right when the
        # agent has just asserted the step is complete. Inject-only, fires per
        # step_done (re-armed by check_pre each time).
        if self._premortem_pending:
            self._premortem_pending = False
            return GuardVerdict.inject(
                message=_STEP_DONE_PREMORTEM,
                reason="step_done_premortem",
                category="verification",
            )
        return None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # NOTE: qualifier extraction at plan-framing time used to live here as a
        # separate block on the first plan_create. It was moved into PlanGuard
        # (see plan.py, _QUALIFIER_EXTRACTION). Reason: PlanGuard is the guard that
        # actually forces the plan into existence, and having a second block fire
        # at the same plan-framing moment created cross-talk — an override_reason
        # the agent supplied to satisfy PlanGuard's plan requirement also released
        # this block, so the qualifier demand never actually bit. Consolidating
        # into one gate with one override channel fixes that. VerificationGuard now
        # owns only the finalize-time premise re-check (Timing 1a below), which is
        # genuinely a verification concern and fires at a different moment.

        # Timing 0b: task-completion re-check, forced on the FIRST plan_update(
        # action="complete") of the run. Symmetric with the step_done premise
        # re-check (Timing 1a): BLOCKS once, any override_reason releases it for
        # good, content is not checked. Its purpose is distinct from step_done —
        # step_done fires per-step on incremental work; this fires once at the
        # whole-task finish line, the one moment where the standard shifts from
        # "did this step's work land" to "is the delivered result actually the
        # best this task calls for". The message makes the agent classify the
        # task (binary pass/fail vs open-ended optimization) and hold itself to
        # the matching bar, rather than hardcoding "optimize more" — which would
        # be noise on a pass/fail task. Fires once so it stays a checkpoint, not
        # nagging.
        if ctx.tool_name == "plan_update" and ctx.tool_args.get("action") == "complete":
            if not self._complete_recheck_reminded:
                if not ctx.override_reason.strip():
                    return GuardVerdict.block(
                        message=_TASK_COMPLETE_RECHECK_REMINDER,
                        reason="task_complete_premise_recheck",
                        category="verification_required",
                    )
                # Override provided → agent paused to classify the task and judge
                # its result against the matching standard. Release the first gate.
                self._complete_recheck_reminded = True
                # Second gate (INCLUSION): the pause above releases on any non-empty
                # reason, but that is defeated by a confidently-wrong reason that
                # merely ASSERTS the result is correct — no observation of any run.
                # A wrong answer has no lexical fingerprint, so the primary path is
                # the LLM judge (classify_fn); regex is a no-provider fallback.
                # The default posture must be POSITIVE: unless
                # the reason reports something the agent RAN and READ (a run, an
                # output, a comparison to a known answer), bite ONCE more and demand a
                # concrete observation. Fires at most once so it stays a checkpoint;
                # the honest escape (nothing runnable, said on the retry) releases it.
                if not self._complete_observation_demanded and not _has_observation(
                    ctx.override_reason, ctx.classify_fn
                ):
                    self._complete_observation_demanded = True
                    return GuardVerdict.block(
                        message=_TASK_COMPLETE_OBSERVATION_DEMAND,
                        reason="task_complete_observation_demand",
                        category="verification_required",
                    )
                # Third gate: the second passes any reason reporting an observation,
                # but an observation can be TRUE and still overfit — taken only on the
                # ONE development sample, with constants tuned until it came out right,
                # while the grade lands on inputs never seen. If the reason is
                # sample-local-only (reports work confined to the sample, or admits
                # fitting to it, with no generalization signal), bite ONCE more and
                # demand an observation on an OTHER input — a perturbed stress variant
                # or a held-out/hidden input. Fires at most once so it stays a
                # checkpoint, not nagging.
                if not self._complete_generalization_demanded and _is_sample_local_only(
                    ctx.override_reason, ctx.classify_fn
                ):
                    self._complete_generalization_demanded = True
                    return GuardVerdict.block(
                        message=_TASK_COMPLETE_GENERALIZATION_DEMAND,
                        reason="task_complete_generalization_demand",
                        category="verification_required",
                    )
                # Fourth gate: the prior gates check whether the result is verified and
                # generalizes; this checks whether it is even the thing the task NAMED.
                # If the release reason discloses delivering a substitute for a GIVEN
                # (near-equivalent / successor / different version, without reporting
                # BLOCKED), bite ONCE more and demand either the exact named artifact or
                # an honest BLOCKED. Fires at most once so it stays a checkpoint.
                if not self._complete_substitution_demanded and _is_disclosed_substitution(
                    ctx.override_reason, ctx.classify_fn
                ):
                    self._complete_substitution_demanded = True
                    return GuardVerdict.block(
                        message=_TASK_COMPLETE_SUBSTITUTION_DEMAND,
                        reason="task_complete_substitution_demand",
                        category="verification_required",
                    )
                # Fifth gate: the prior gates check the RESULT (verified, generalizes,
                # is the named thing). This checks the DELIVERED ARTIFACT — the thing
                # the judge actually reads from disk, which can diverge from the result
                # the agent proved in-session (settings not serialized, best-so-far held
                # in memory, byproducts polluting the delivery dir, an input mutated,
                # a noisy metric passed by a hair). These traps bite only at delivery,
                # so they belong here rather than in the plan-time acceptance note. Bite
                # ONCE unconditionally: unlike gates 2-4 there is no lexical trigger,
                # because the failure is silent — the honest escape ("none apply: no
                # file delivered / no search / no noisy metric") releases it.
                if not self._complete_delivery_hygiene_demanded:
                    self._complete_delivery_hygiene_demanded = True
                    return GuardVerdict.block(
                        message=_TASK_COMPLETE_DELIVERY_HYGIENE,
                        reason="task_complete_delivery_hygiene",
                        category="verification_required",
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
                acceptance = []
                if self._plan and step_id:
                    try:
                        plan_data = self._plan.get_active()
                        if plan_data:
                            for step in plan_data.get("steps", []):
                                if step.get("id") == step_id:
                                    acceptance = step.get("acceptance", [])
                                    break
                    except Exception:
                        # If plan lookup fails, fall back to simple check
                        pass
                
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
                    # Has verification, allow — arm the pre-mortem for check_post.
                    self._premortem_pending = True
                    return None
                
                # Mode 2: No acceptance (simple step) → require override_reason
                else:
                    if not override_reason:
                        return GuardVerdict.block(
                            message=_VERIFICATION_REQUIRED_NO_ACCEPTANCE,
                            reason="step_done_no_verification",
                            category="verification_required"
                        )
                    # Has override_reason, allow — arm the pre-mortem for check_post.
                    self._premortem_pending = True
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

    def _step_status(self, step_id) -> str | None:
        """Return the status of a step in the active plan, or None if unknown.

        None means we cannot determine the status (no plan, no step_id, or
        lookup failed) — in that case the caller does not block, staying
        permissive when state is uncertain.
        """
        if not self._plan or not step_id:
            return None
        try:
            plan_data = self._plan.get_active()
            if not plan_data:
                return None
            for step in plan_data.get("steps", []):
                if step.get("id") == step_id:
                    return step.get("status")
        except Exception:
            return None
        return None

    def notify_recovery(self):
        """Called by hard_reset logic to signal recovery."""
        self._post_recovery = True
        self._recovery_reminded = False
