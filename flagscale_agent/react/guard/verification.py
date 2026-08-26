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

Before marking a step complete, re-read the task statement and confirm you have
executed every operation it explicitly asks for. A task that says "recover data
from X and write to Y" needs both;
designing a recovery strategy is not the same as executing the write.
If the task names artifacts to produce, verify those exact artifacts exist
at the named paths — you cannot verify what you have not yet built.
Reasoning "I know how to do X, so X is done" is the signal to stop and execute X.

Acceptance should describe an externally observable end state — something a
neutral third party could check without knowing how you implemented it:
  • a file exists at a specific absolute path
  • a command exits 0 / produces output matching an expected value
  • an artifact has an expected property

Avoid phrasing that only makes sense from inside your own implementation (e.g.
"the path I chose to write to"). When you verify, reproduce the judge's condition
from scratch — don't confirm using the same convenient handle you built with. A
check that shares its premise with the implementation cannot fail, so it proves
nothing.

More traps, same root cause — letting your implementation, not the task contract,
define what verification covers:
  • Cover every input the task gives you, not just the one you developed against.
    If the task ships multiple samples, verification must actually
    run the solution on each. Passing on your debug sample says nothing about the
    ones the judge scores. Before marking complete, ask concretely: which input
    files in the task directory have I NOT yet run my solution on? List them, then
    run the solution on each — the ones you skipped are usually the ones scored.
  • Verify generalization, not answer-recall. For a sample whose answer you know,
    checking the output matches is fine. For a blind sample whose answer you don't
    know, do NOT assert it falls in a range you reverse-engineered from the known
    one — that re-encodes your assumption. Verify the output is structurally valid
    (right shape, parses, no crash); you cannot verify a number you were never
    told. When the scored input is one you cannot see, be honest about what your
    check proves: passing on the sample in front of you shows the code runs, not
    that it generalizes. Only features the task contract names as fixed are
    guaranteed on the scored input; anything merely true of your one sample is an
    accident, not a promise.
  • Don't excuse a real failure by its origin. When a check fails and you're
    tempted to dismiss it as "pre-existing", "environmental", or "not caused by
    my change" — that is the wrong test. The question is not whether you
    introduced the failure, but whether the failing behavior falls within what
    the task asks you to deliver. If the failure is in functionality the task
    requires to work, you must make it pass regardless of who caused it, because
    acceptance judges the result, not its origin. Only dismiss a failure when the
    task's declared scope genuinely does not cover it — like skipping an unrelated
    test suite in a bug-fix task versus a task that says "make all tests pass".
  • Ground definitional choices in the authoritative source, not a convenient
    assumption. When the task names a source of truth — a README, a spec, a
    reference file, especially one it calls "critical" — any definitional choice
    (what counts as X, which field / subset / format is meant) must be grounded
    there. Watch for the moment you ask "what exactly counts as X?" and answer
    with a plausible guess instead of checking: that unresolved question is the
    crack where a wrong interpretation enters. A number computed from your own
    definition and checked against that same definition cannot fail — it only
    re-confirms the assumption, the same premise-sharing trap as above. Close
    every definitional question by grounding its answer in the named source.
  • Compute each compared quantity exactly the way the task defines it. When
    verification means deriving a value and checking it against a threshold or
    reference, the derivation must match the task's / judge's definition verbatim
    — for every quantity compared, not just the obvious one. If the task specifies
    a transform before measuring (reverse, complement, normalize, canonicalize,
    decode, round), applying your measure to the raw input skips that transform and
    measures a different quantity; the check can pass while the judge fails. Say
    what each side of the comparison is, then confirm your computation performs
    every named step — especially ones that look like harmless shorthand. Reusing
    the convenient formula from your implementation instead of re-deriving from the
    definition is the premise-sharing trap again.
  • Snapshot fragile or one-shot state before you touch it. When the task involves
    recovering, forensically analyzing, or reconstructing state that is delicate or
    non-reproducible — a file that may be deleted, corrupted, or transformed on
    access — copy the raw original to a safe location before using any tool that
    might open, modify, checkpoint, or normalize it. Inspect read-only or in hex
    first; when unsure whether a tool writes back, assume it does. A convenient
    high-level tool can silently mutate or destroy the evidence you were asked to
    recover, and that loss is usually irreversible. An untouched copy is what lets
    you check your result against the true starting point, not against state your
    own probing altered.
  • Close the gap when your own measurement already shows you fall short. When the
    task states a quantitative acceptance bound — a numeric range, error tolerance,
    similarity floor, maximum difference — and you can compute your metric the way
    the judge will, do not stop at a result that is close but outside the bound.
    Treat the distance to the threshold as an optimization target: while any
    measured metric misses its bound, adjust and recompute, until every metric
    satisfies its threshold — or until you reach the limit of the method, where
    satisfying one bound would break another. Make that stopping decision
    explicitly; don't invoke "good enough" to avoid another iteration. "I measured
    it and it passes" is a finish signal; "I measured it and it is just short" is a
    signal to keep optimizing, not to deliver.
  • Before delivering, verify using the judge's measurement. When the task is
    scored on some metric — a count, format check, exact string match, correctness
    oracle — run that exact measurement yourself first. If the judge's tool is
    accessible (a verifier script, reference output, validation command), use it
    directly; otherwise reproduce its logic from the task's description. A solution
    that passes your convenient proxy but fails the judge's actual measure still
    scores zero. The gap between "seems right" and "passes the judge's test" is
    where last-mile failures live.
  • Measure the delivered artifact itself, reloaded from where it ships — not an
    in-memory or in-session object you configured programmatically. When you tune by
    mutating a live object (setting fields on a loaded model, patching a config in
    memory, monkey-patching a running instance) and measure THAT, you have verified
    the proxy in your session, not the file the judge reads. The judge loads your
    artifact cold from disk; if your settings did not actually serialize into that
    file — or serialized differently — the reloaded artifact behaves nothing like
    the object you measured. The tell: your in-session metric looks great, yet the
    delivered file is byte-identical (or nearly) to the original, or the judge's
    number is wildly off. Close the gap the way the judge will: after writing the
    artifact, load it back FRESH (new process / re-read from the delivery path,
    discarding every in-memory object you touched) and re-run the metric on that
    reloaded copy. Only a measurement on the cold-loaded deliverable proves the
    settings survived the round-trip; a measurement on the object you edited proves
    only that your edit worked in RAM.
  • When the bar is pass/fail on a NOISY metric (speed, latency, memory, throughput,
    accuracy on a sample) and the judge's measurement procedure is hidden, passing by
    a hair on YOUR OWN measurement is not passing. Your number and the judge's number
    are two draws from instruments that differ — repeat count, warmup, outlier
    trimming, machine load, averaging. When your margin over the bar is smaller than
    that instrument disagreement, the "pass" is over-fit to your measuring stick and
    says nothing about where the judge's draw lands. Two fixes, both observations:
    (1) measure the deliverable the way a rigorous judge would — many repeats, discard
    extreme percentiles/outliers, take a trimmed central statistic — not a casual
    one-shot timing; (2) demand margin proportional to the noise — clearing the
    threshold by a fraction of a percent is NOT clearing it. If your best robust
    re-measurement sits right on the line, treat the task as NOT passing: push to a
    different method-class that opens real headroom, or keep improving until neither
    run-to-run variance nor a reasonable alternative measurement method could flip the
    verdict. "My measurement just cleared it" is the finish-signal trap; "I re-measured
    the artifact the judge's way, repeated and trimmed, and it clears with margin" is
    the real finish signal.
  • During open-ended search, persist your best-so-far to the delivery path the MOMENT
     you find it — never hold a winner only in memory. When you iterate over candidates
     (tuning params, trying methods, refining a solution) and each is built and measured
     as an in-process value — a variable, an in-memory object, a scratch string you plan
     to "write out at the end" — that winner is NOT banked. The judge reads the delivery
     path, not your process memory; if you are stopped before the end (timeout, kill,
     budget exhaustion), an unpersisted in-memory best scores exactly as if you never
     found it, and the delivery path ships whatever earlier or intermediate version
     happened to be there. The tell: you spent the whole budget improving candidates,
     each better than the last, but the file the judge reads was written early and never
     refreshed — so your measured gains evaporate with the process. The fix is a
     write-through discipline: the instant a candidate measures better than what is
     currently at the delivery path AND passes the validity checks, write it THROUGH to
     the delivery path right then, before continuing the search. Best-so-far lives on
     disk at the delivery path, never only in memory — so whenever you are stopped, the
     best version you actually measured is exactly what ships. Holding winners in memory
     to persist "later" is the same class of failure as never producing them: the reward
     is identical (zero for that improvement), and it is entirely avoidable.
  • When the delivery path IS what gets measured, protect it with backup + rollback.
     Some tasks force overwrite-then-measure: the judge (or your own measurement) reads
     the artifact at the delivery path directly, so you cannot measure a candidate WITHOUT
     first writing it to that path — "test before you overwrite" is physically impossible.
     Every experiment then temporarily lands an UNCONFIRMED candidate on the delivery
     path, and the naive loop destroys your best work: you write candidate N over your
     verified best, measure it, find it WORSE, and move on to N+1 without restoring, so
     the delivery path is left holding a regression — if you time out there, that
     regression ships even though you had a better version earlier. The invariant
     "delivery path always holds the current best VERIFIED version" still must hold; the
     mechanism that preserves it under overwrite-then-measure is explicit backup +
     rollback. The moment a version measures as your new best, copy it to a side backup
     (cp delivery.ext best.ext) so the known-good survives independently of the delivery
     path. Run each new experiment knowing the delivery path is now scratch; after
     measuring, if the new candidate is NOT strictly better, immediately restore the
     backup over the delivery path (cp best.ext delivery.ext) BEFORE the next experiment
     — never leave an unconfirmed-or-worse candidate at the delivery path across
     iterations. Only a strictly-better candidate becomes the new backup. Treat the
     restore as a mandatory step of the loop, not an afterthought — so at every instant
     between experiments the delivery path equals your best verified version, and a
     timeout at any point ships that best, not a half-tested regression.
  • Let the task's stated constraints shape your method before you commit to one.
    When the task supplies a fact that narrows the problem — a bound on the search
    space, a structural property of the input, a hint about which technique fits —
    that fact steers your approach, not decoration to skim past. Ask what the
    constraints make cheap that a naive method makes expensive; reaching for the
    strategy your intuition defaulted to while leaving a stated constraint unused
    lets your default approach, not the task contract, decide the path. The tell:
    you hit a wall the task told you how to avoid — an approach that exhausts
    memory, blows past a time budget, or scales to a size the task's own numbers
    say is unnecessary. When the same wall stops you twice, that is not a cue to
    keep tuning in place; it is a cue to re-read the task statement
    for a constraint or hint you skipped, because a method that ignores what the
    task made easy will keep failing no matter how you tune it.
    Re-derive the approach from the givens,
    then resume. A computation running with no visible progress is itself that wall
    — being blocked on it is the signal, not a reason to wait it out. If a run's
    progress indicator sits still while the clock burns, the method, not the
    runtime, is stuck: interrupt it and re-derive from the constraints.
  • Keep the deliverable directory clean and minimal. When the task specifies a
    delivery directory, it should contain only what you are asked to deliver —
    nothing more. Do not leave debugging scripts, temporary test files,
    intermediate outputs, backup copies, or exploration artifacts in the delivery
    path. Use a separate temporary directory outside it for scratch work, then copy
    only the final required files in. A deliverable contaminated with unrelated
    files fails the implicit contract: the judge expects exactly what the task
    named, not a workspace snapshot. Before completing, list the delivery
    directory's contents and confirm each file is required; remove everything else.
    Take the contract literally: if it names an EXACT set (often exactly one file),
    the location must hold that set and NOTHING MORE — one stray sibling (a compiled
    binary or object file, a build output, a generated or downloaded artifact, a log,
    a scratch copy) fails an exact-contents check just as hard as a missing
    deliverable, and it fails SILENTLY because your real artifact IS present and looks
    right. Do not assume "I only created the one file" — your OWN verification steps
    may have deposited others without you thinking of them as deliverables. A specific
    and easy-to-miss source of such strays: a command the TASK SHOWS you — the exact
    invocation the consumer/grader uses to run, build, compile, or test your artifact —
    describes THEIR action on your deliverable; it is NOT a spec for where YOUR
    intermediate products go. Copying that shown command verbatim to self-verify is the
    trap: if it emits a byproduct and the example directs that byproduct INTO the
    delivery location, running it as-is drops the byproduct exactly where the
    exact-contents check trips over it. Self-verify by redirecting byproducts to a
    scratch/temp path OUTSIDE the delivery location, or if you reproduce the shown
    command as-is, delete every byproduct it created from the delivery location before
    finishing.
  • If the task says an input resource must stay UNCHANGED (a "do not modify X"
    constraint, often checked by hash / checksum / exact bytes), remember that
    reverting your change is NOT the same as never changing it. An edit-then-undo
    on a byte-checked file almost never reproduces the original bytes — adding an
    internal structure and then removing it typically leaves the file grown and
    rewritten (the removal frees space but does not shrink or zero it), so its
    checksum no longer matches even though the structure is "gone". Before you finish, if you touched such a resource
    at all, restore it from a pre-touch backup (not an undo command), or better,
    make the deliverable never depend on mutating it — operate on a copy and keep
    the protected original pristine. "I added it then removed it" fails a
    byte-level immutability check.
"""

# NOTE: The plan_create qualifier-extraction reminder (_PLAN_CONSTRAINT_REMINDER)
# was moved to plan.py (_QUALIFIER_EXTRACTION). Reason: it belongs at the
# plan-framing moment, which PlanGuard owns. Keeping it here as a separate
# Timing -1 block caused guard override_reason cross-talk — PlanGuard's
# single-shot block trains the agent to pass _override_reason, which then
# silently satisfied this guard's block too, so it never actually fired.
# See plan.py check_pre and pitfall/flagscale_agent/guard_override_crosstalk.

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

You are about to close out the entire task, not just a step. After this, nothing
re-examines whether what you delivered was the best you could do — the result is
final. Spend this one checkpoint deliberately.

Before anything else, re-read the task as the USER stated it and confirm what you
are about to deliver answers THAT, not a nearby question you drifted into. The
failure this catches is not laziness — it is confident competence aimed slightly
off-target: you solved a problem thoroughly and verified it thoroughly, but it was
an adjacent problem your own framing substituted for the user's. The drift usually
enters through an early premise (a value you parsed, a state you reconstructed, an
input you interpreted) that then went unquestioned while all your effort — and all
your verification — piled onto the steps downstream of it. Deep, repeated checking
of the downstream work builds false confidence precisely because it never revisits
that upstream premise. So test the premise, not just the conclusion: does the input
you actually worked from match what the user gave? And read the user's phrasing
literally for what it asks you to PRODUCE — every explicit clause (the exact form,
"print them ALL", "one per line", the specific artifact and path), and every implicit
sanity check the task's nature implies (if the task shape says a solution should
exist, a result that says otherwise is a red flag on your premise, not a finding).
When your polished answer and the user's plainly stated need diverge, the need wins —
do not talk yourself into believing your version is what they "really" meant.

Now trace where your answer CAME FROM. Point at the content you are about to deliver
and ask: did I OBSERVE this — read it out of a tool output, a command's stdout, a file
I actually opened — or did I INFER it from what I expected to be there? If the task
handed you an artifact to read information out of, the answer must trace back to a tool
call that genuinely transformed or inspected that artifact and surfaced the content;
a value that never appeared in any output you saw, but that you "know" from prior
experience of what such things usually contain, is a guess wearing the clothes of an
answer. Prior knowledge tells you what is PLAUSIBLE, never what is ACTUALLY there — and
the task is graded on what is actually there. (Answers you legitimately COMPUTED — a
number you derived, a result your own code produced — are observed; this is about not
substituting a prior expectation for a reading you never took.)

Then decide what kind of task this was, because that decides what "done" means:

  • If the task has a definite pass/fail condition — a build that either succeeds,
    a test suite that either goes green, an artifact that either exists at the named
    path — then "done" means that condition is met, and there is nothing to optimize
    beyond it. Confirm the condition holds and complete. Do not manufacture doubt
    where the contract is binary. But hold that confirmation to the real condition,
    not to a signal you arranged to look like it. If the task named a non-negotiable
    specific — a version, a tool, a format, a named source or identity — and you could
    not actually obtain it, then "done" is NOT reachable by delivering a near-equivalent
    and making the surface read as satisfied: standing up the expected directory,
    filenames, or a wrapper so a sanity command exits 0 dresses a substitute to pass the
    check, and the check was only ever a proxy for the genuine artifact. Passing the
    proxy is not meeting the constraint. Disclosing the swap does not rescue it either —
    admitting "I used Y instead of the required X" in this very override_reason is honest
    about the deviation, but honesty is not permission: a disclosed substitute is still a
    substitute and still fails the task on its terms. When the genuine article is
    unreachable, the only honest closes are to keep searching for a legal path or to
    report BLOCKED with nothing counterfeit delivered — an empty honest BLOCKED beats a
    populated fake that scores the check. And when you DID confirm the condition by
    running something, check WHERE you ran it: a pass produced in your own working
    environment — one your prior actions have loaded up with installed packages, set
    env vars, helper files, or running services — does not prove the deliverable works
    in the clean environment that will actually consume it. If you had to install a
    dependency or set something up to make your check go green, that setup is part of
    what made it pass, and the grader / fresh machine / other user does not have it.
    So the deliverable must carry its dependencies inside itself or use only what the
    target is guaranteed to have (standard tools, the exact tools the task named) — a
    shipped script must not import a library you pip-installed only locally, an artifact
    must not read a file or env var or service that exists only in your session. A
    distinct flavor of the same trap is not what you ADDED but HOW the deliverable gets
    addressed: your own interactive shell resolves the artifact for you — via an
    interactive/login-shell PATH, a sourced rc file, an alias, or your cwd — while the
    consumer invokes it bare, as a non-login non-interactive subprocess running the plain
    command by name, with none of your shell's accumulated resolution. "It runs when I
    TYPE it" is not "it runs when a PROGRAM calls it": an artifact meant to be found by
    name must live where the target's own invocation looks (the standard install location
    it guarantees), not merely be reachable through your session's config. Before
    completing, re-run the check the way the consumer would: in a clean context, invoking
    it the bare way the grader will (fresh non-login shell / direct subprocess / the exact
    command), or at minimum ask "would this still pass without the setup I personally
    added AND without my shell having been primed to find it?"
    One more binary-contract trap: if the task named a precise list of exceptions —
    "all tests pass EXCEPT these files", "every case but X and Y", "these are the only
    known-skipped ones" — that enumeration is a closed constraint. Listing exactly those
    and only those means everything else must hold, and a failure OUTSIDE the list does
    not earn a place on it. Quietly widening the carve-out to swallow the thing you could
    not fix is rewriting the acceptance bar to match your result — the same violation as
    substituting a near-equivalent, just editing the exceptions instead of the artifact.
    And check the excuse you are leaning on: "this failure is pre-existing / an external
    library's bug / unrelated to my change" is a CLAIM that needs evidence, not a default
    exit when you are stuck. Before dropping a requirement on that basis, prove it (the
    failure is identical on an untouched reference, or the task explicitly excludes it);
    absent proof, "I could not figure out how to fix it" is not "it cannot or should not
    be fixed", and when the task says "fix any X issues" or "read the errors, they tell
    you what to fix", an unexplained failure is far more likely inside your mandate than
    outside it. If an item outside the exemption list still fails, "done" is not reached.
    Finally, if you developed against ONE visible sample but the real grade is on hidden
    inputs — one example input, one demo file, one reference case — passing on that sample
    is the floor, not the finish. It shows your method fits that instance, not that it
    generalizes. The tell is a solution full of magic constants (an absolute threshold, a
    fixed cutoff, a hardcoded "exceeds N") you tuned until the visible sample came out
    right; each is a place you fit the sample's incidental scale/length/noise rather than
    the phenomenon, and a hidden input that differs slightly slides outside the band. Before
    completing, audit every hardcoded number — "why THIS value, would it survive an input
    unlike my one sample?" — and prefer relative / normalized / structure-derived judgments
    (a monotonic turning point, a ratio, a change of direction) over absolute cutoffs you
    hand-picked. With only one labeled example you cannot confirm generalization by testing,
    so the burden is to make the method principled, not sample-calibrated.

  • If the task is open-ended — optimize, minimize, maximize, make it faster,
    improve some metric, produce the "best" X — then the first working result that
    beats a naive baseline is NOT automatically the finish line. A large improvement
    proves your first idea worked; it says nothing about whether a genuinely
    different approach does better still. Ask concretely: is there another distinct
    method for this problem class I have not tried? Did I actually implement and
    measure the alternatives against each other, or did I just plan them and ship
    the first one? If a different technique could plausibly beat what I have and I
    have not ruled it out by measurement, the task is not yet at its best.
    Now judge whether your "distinct" attempts were actually distinct. Read the
    spread of your measured results: if the alternatives you compared all land in
    the same narrow band — the same order of magnitude, differing by only a few
    percent — that near-tie is itself the signal. It usually means they were not
    different methods at all but variants within one method-family, and you have
    hit the DEPTH limit of that family, where further tweaks only trade noise. A
    clustered spread is NOT evidence you reached the global best; it is evidence
    you explored one family thoroughly and no other. The response is to widen, not
    to stop: name the method-family your attempts share (e.g. "I only rewrote the
    same computation three ways"), then reach for a STRUCTURALLY different family
    that attacks the cost from another angle entirely — precompute instead of
    recompute, index/materialize instead of scan, a different algorithm class, a
    different data structure. You do not need to know the target's absolute best
    to do this; the flat spread across same-family attempts is enough of a signal
    on its own. Only after you have tried at least one genuinely different family
    and it too fails to beat the current best is "reached the method's limit"
    honest — otherwise you have reached one family's limit and called it the task's.

Whichever kind it is, name which kind it is — then, before you close out, run these orthogonal
self-checks against the task's REAL purpose (the actual use your work serves), not
against any imagined check or what you guess is being looked at. Guessing what a
check inspects and satisfying THAT is its own overfitting; build the thing that
genuinely serves the purpose and any check passes as a side effect. These axes are
independent — a result can be fully optimized yet fail to generalize, or general in
shape yet not the best available. Answer each on its own, do not let a strong yes on
one stand in for the others:

  1. OPTIMIZED — if the task admits a "better" (faster, smaller, more accurate,
     cleaner), is what you hold actually the best you reached, or just the first
     thing that worked? Did you implement and measure a genuinely different approach
     against it, or only plan one and ship the first? If a distinct method could
     plausibly do better and you have not ruled it out by measurement, not done.

  2. GENERAL — does your solution handle the whole CLASS of situation the task
     describes, or did you special-case it to the one input / example / situation
     you developed against? A method studded with values or branches that only make
     sense for the specific case in front of you is fitted to that case, not to the
     problem. Would it still produce a well-formed answer on a sibling input that
     differs in the ways the task does not fix?

  3. GENERALIZES — you verified under the conditions you could OBSERVE (the input
     you had, the environment you built in, the scale you tried). The real use
     imposes conditions you could NOT observe — a different input, a cleaner
     environment, a larger or messier case. The trap is to treat "it worked on what
     I could see" as "it works": passing on the one sample in front of you shows the
     method fits that instance, not that it holds on the instances you never saw.
     Name that gap concretely, then say whether you REPRODUCED it and read the
     result, or merely argued it would be fine. An argument that "it should hold" is
     not an observation that it does; the gap is never zero, so manufacture it
     (perturb the input, run in a fresh context) and watch what happens first.

If an axis genuinely does not apply to this task, say so and why — do not skip it
silently. To proceed, re-issue plan_update(action="complete") with an
"_override_reason" that answers all three against the real purpose: optimized (best
measured, or the method's limit), general (handles the class, not one case),
generalizes (reproduced the gap between what you observed and what the real use
imposes, and observed it still holds). This is a forcing checkpoint, not a content
check — any override_reason lets it through; the point is that you stopped and
looked along every axis before deciding you are done."""


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
_ARGUMENT_MARKERS = (
    "should generalize", "will generalize", "should work", "would work",
    "reasonable", "principled", "makes sense", "is fine", "conservative",
    "relative to the signal", "relative to signal", "robust", "sound",
    "i believe", "i think", "logically", "in principle", "by design",
)

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


def _is_pure_argument(reason: str) -> bool:
    """True when the reason DEFENDS a method (argument) but reports NO observation.

    This is the runtime form of the prompt's litmus: is the evidence an
    OBSERVATION (ran something and read a result) or an ARGUMENT (explained why
    the result should be right)? Only pure-argument reasons — at least one
    method-defence marker AND no observation marker — are caught; a reason that
    reports any measurement passes, and a reason that is neither (a terse
    "re-checked each criterion") is left alone to avoid false positives.
    """
    if not reason:
        return False
    low = reason.lower()
    has_argument = any(m in low for m in _ARGUMENT_MARKERS)
    has_observation = any(m in low for m in _OBSERVATION_MARKERS)
    return has_argument and not has_observation


def _has_observation(reason: str) -> bool:
    """True when the reason reports a positive OBSERVATION — something the agent
    ran and read: a command/test executed, an output/value read, a comparison to
    a known/expected/ground-truth answer.

    This is the INCLUSION form of the observation gate. _is_pure_argument was an
    EXCLUSION filter — it only bit when the reason contained self-incriminating
    method-defence vocabulary AND no observation, so a confidently-wrong reason
    that merely ASSERTS correctness ("the query returns the right set") — no
    argument marker, no observation marker — fell in the gap between the marker
    sets and passed untouched. A wrong answer has no lexical fingerprint, so the
    default posture must be "require positive evidence of a run+read", not "trust
    unless the wording confesses a bad pattern". A reason with no observation
    marker is caught; any concrete run/read/compare signal releases it. The
    honest escape (nothing runnable to check, said explicitly) is handled by the
    fires-once flag on the gate, not by this predicate.
    """
    if not reason:
        return False
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


def _is_sample_local_only(reason: str) -> bool:
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


def _is_disclosed_substitution(reason: str) -> bool:
    """True when the reason DISCLOSES delivering a substitute for a GIVEN value
    without reporting BLOCKED.

    Runtime form of CONSTRAINT LOYALTY: disclosing a substitution does not legalize
    it. A reason that names substitution/near-equivalent vocabulary AND does not frame
    the outcome as BLOCKED (no counterfeit delivered) is caught once. A reason that
    reports inability (BLOCKED) passes — that is the honest path. A reason with neither
    signal is left alone (no false positive)."""
    if not reason:
        return False
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
                # merely ASSERTS the result is correct — no observation, no argument
                # marker either, so an EXCLUSION filter (_is_pure_argument) let it
                # slip through the gap between marker sets. A wrong answer has no
                # lexical fingerprint, so the default posture must be POSITIVE: unless
                # the reason reports something the agent RAN and READ (a run, an
                # output, a comparison to a known answer), bite ONCE more and demand a
                # concrete observation. Fires at most once so it stays a checkpoint;
                # the honest escape (nothing runnable, said on the retry) releases it.
                if not self._complete_observation_demanded and not _has_observation(
                    ctx.override_reason
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
                    ctx.override_reason
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
                    ctx.override_reason
                ):
                    self._complete_substitution_demanded = True
                    return GuardVerdict.block(
                        message=_TASK_COMPLETE_SUBSTITUTION_DEMAND,
                        reason="task_complete_substitution_demand",
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
                    # Has verification, allow
                    return None
                
                # Mode 2: No acceptance (simple step) → require override_reason
                else:
                    if not override_reason:
                        return GuardVerdict.block(
                            message=_VERIFICATION_REQUIRED_NO_ACCEPTANCE,
                            reason="step_done_no_verification",
                            category="verification_required"
                        )
                    # Has override_reason, allow
                    return None
        
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
