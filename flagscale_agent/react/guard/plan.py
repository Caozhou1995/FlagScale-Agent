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

"""PlanGuard — reminds agent to create a plan for long tasks."""

from __future__ import annotations

import os

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.multi_agent.wiring import CONTRACT_PATH_ENV, is_worker


from flagscale_agent.react.guard.verification import _contract_constraints


def _self_is_review_or_diverge_worker() -> bool:
    """True when THIS process is a spawned reviewer OR diverger worker.

    Both roles are exempt from the parent-side diverger-DISPATCH block: the
    demand ("spawn ONE diverger after plan_create, then rule") is a PARENT-side
    habit, and a worker — an independent reviewer or the diverger itself — must
    not nest it (that self-references the demand chain and contends for the
    shared concurrency gate). The two roles carry DIFFERENT structured flags in
    contract.json: a reviewer has `reviewer: True` (also `role: reviewer` /
    `re_reviewer`), a diverger is tagged `diverger: True` by the spawn layer
    (spawn.maybe_auto_inject_diverger) — a diverger is deliberately NOT
    reviewer-class (it must not enter the reviewer preset). Both are checked
    here so the one exemption covers both roles. Detected from the structured
    contract, never from prose in the rendered contract.prompt (a body that
    merely QUOTES "framings of the task" must not suppress the demand). Errors
    fail silent-False (the demand fires as usual).
    """
    try:
        if not is_worker():
            return False
        cons = _contract_constraints()
        return cons.get("reviewer") is True or cons.get("diverger") is True
    except Exception:
        return False


def _self_is_diverger_worker() -> bool:
    """True for a spawned diverger worker (historical name).

    A diverger carries the structured `constraints.diverger` flag (set by the
    spawn layer for a diverger goal); a reviewer carries `constraints.reviewer`.
    Both are exempt — see `_self_is_review_or_diverge_worker`.
    """
    try:
        if not is_worker():
            return False
        return _contract_constraints().get("diverger") is True
    except Exception:
        return False


def _dispatch_includes_diverger(tool_name: str, tool_args: dict) -> bool:
    """True when a spawn_worker/dispatch_many call actually dispatches a
    DIVERGER — goal reads like a diverger task (the harness's own diverger
    contract wording) or the caller tagged `constraints.diverger`. ONLY such a
    dispatch satisfies the per-framing divergence demand; any other spawn (e.g.
    a step-1 worker) does not. Imported lazily to avoid any import cycle."""
    if not isinstance(tool_args, dict):
        return False
    from flagscale_agent.react.multi_agent.spawn import _is_diverger_goal
    if tool_name == "spawn_worker":
        cons = tool_args.get("constraints")
        if isinstance(cons, dict) and cons.get("diverger") is True:
            return True
        return _is_diverger_goal(tool_args.get("goal", ""))
    for spec in (tool_args.get("specs") or []):
        if not isinstance(spec, dict):
            continue
        cons = spec.get("constraints")
        if isinstance(cons, dict) and cons.get("diverger") is True:
            return True
        if _is_diverger_goal(spec.get("goal", "")):
            return True
    return False


def _tool_result_indicates_failure(result) -> bool:
    """True when a tool_result shows the call did NOT really succeed: a kernel/
    executor exception wrapper ("Error executing tool: ..."), a guard block
    ("[BLOCKED BY GUARD] ..."), or the tool's own "ERROR: ..." return.

    `None` is treated as NOT-a-failure: callers that don't plumb a result (e.g.
    unit tests, or a path where the result was not captured) must keep their
    historical success semantics. Only a POSITIVE error signal flips this.
    """
    if result is None:
        return False
    text = str(result).lstrip()
    return (
        text.startswith("Error executing tool")
        or text.startswith("[BLOCKED BY GUARD]")
        or text.startswith("ERROR")
    )


# Folded into the plan-framing gate on purpose: qualifier extraction is a
# plan-framing concern, and PlanGuard forces the plan into existence. One gate,
# one override, no laundering — a separate block caused cross-talk where the
# override_reason for the plan also released the qualifier block.
_QUALIFIER_EXTRACTION = """

While framing this plan — extract the task's qualifiers, not just its subject.

A task names a subject and qualifies it: a point in time, a version, a subset, a metric definition. The subject is obvious; the qualifier is easy to drop — the work produces a confident answer whether or not you honored it, and a dropped qualifier silently answers a nearby question the task did not ask.

Treat every qualifier as machine-verified by a grader you cannot reach or fool: it re-parses the time boundary, diffs the exact version string, compares the output byte-for-byte against a reference, re-counts the entries. Reporting done does not feed that check — if a qualifier is not literally satisfied, the result is FAIL no matter how finished the work looks. A qualifier is a hard pass/fail condition to build the plan around.

Re-read the task statement; list every qualifier as a first-class item and fold each into the plan as a step or acceptance criterion — a qualifier not in the plan is one execution silently skips. Extraction is only half: you must read the boundary each qualifier draws in the direction the task means and state its meaning in your own words. The common failure bends a fixed boundary toward what is convenient (the larger, fresher, more available thing) — a "newest is best" instinct overwrites a bounded qualifier, yet the answer inside the boundary is right even if a better candidate exists outside. A bound on time is especially easy to invert: the state as it stood at a point (things that came later out of scope) vs. the state right now select different answers — read it in the direction the task means, letting the task's words decide, not the most up-to-date data.

Before committing to shape, apply P1's framework: name the problem CLASS, identify the STANDARD METHOD, and include a SMALL-SAMPLE step validating the method on the smallest meaningful input before scaling — the small run also estimates time, and each full-scale debugging iteration costs 10-100x more. Generic steps ("analyze", "implement", "test") mean the class was never identified; a brute-force sweep / enumeration is the tell.

Some qualifiers pin you to a concrete tool instance (specific version, revision, named model) whose usage may differ from the generic API in ANY form. When pinned, add a step to consult that instance's own documentation before writing the call — per P1's TOOL INSTANCE rule the task's VERB selects the documented usage, and the documented way is the DEFAULT. Consulting is the near half; APPLYING what the doc says is the far half.

FACET MANIFEST — extraction kept in your head cannot be reused. Write the task's qualifiers to a manifest file (e.g. analysis/facets.md; state the path in the plan) with three sections, each entry pinned to the task's OWN words — quote the line and cite where it says so. Do NOT read beyond the task statement or guess the grader's logic.
  1. FORM conventions left open (output format, units, field order, naming): list candidates and pin ONE supported by the task text, with its source line.
  2. Reference artifacts the task NAMES (required files, commands, example paths): quote each verbatim — this IS the grading surface.
  3. Example vs graded domain: state whether the task's example IS the graded instance; if not, write "the graded instance is another — do not extrapolate." Run the REAL graded input at least once when obtainable, else record the extrapolation basis.
Reuse the manifest at close-out: delivery-time reconciliation reads it from this file, not from re-scraping the conversation."""

_WRITE_FILE_NO_PLAN = (
    "[Plan] You are about to write or edit a file — the concrete signal that you have "
    "started producing a deliverable. A plan must exist before you produce output.\n"
    "\n"
    "Call plan_create() now: frame the task as ordered steps with acceptance "
    "criteria naming what THIS task requires. A budget-order first step ('land a "
    "crude but complete, scorable deliverable at the required path, then refine') "
    "keeps an unsupervised run from spending its whole budget with nothing on disk.\n"
    "\n"
    "If this write is a genuinely throwaway scratch/draft file (not the "
    "deliverable), override with a one-line reason and proceed."
)


# Divergence demand, delivered on EVERY plan_create — per framing
# (the premise-inherited-unexamined half: "我想窄了"). A plan written from a
# framing locks in that framing: every later step inherits premises that
# were never inspected against alternatives — and a LATER framing (re-plan)
# inherits its own unexamined premises just the same, so the demand re-arms
# per framing. This inject does NOT judge the plan —
# it demands a divergence check by a SEPARATE reasoning process: spawn ONE
# diverger worker (read-only, no verdict, no agreement) that proposes alternative
# framings of the task, then the parent issues a one-line ruling for EACH
# alternative (adopt → revise the plan, or reject → why). Rulings are the
# Delivered as an inject ON the plan_create call itself, then ENFORCED with a
# block (check_pre) on the FIRST non-plan tool call after plan_create when no
# diverger was dispatched. Firing the block on the tool AFTER plan_create (not
# on plan_create) keeps plan_create advisory, breaks the old cross-talk with the
# qualifier inject (a plan_create block released the qualifier too), and leaves
# the block overridable for a genuinely-trivial framing. Exempt for reviewer/
# diverger workers.
# Enforcement header prepended to the dispatch BLOCK (fired on the first
# non-plan tool call after plan_create, when no diverger was dispatched). The
# demand text (_DIVERGER_INJECT) is reused verbatim so the two never drift; this
# only states that the check is now enforced, not merely suggested.
_DIVERGER_ENFORCE = """[Plan] Diverger not dispatched.

The divergence check is part of framing the plan, not an optional afterthought — it
is where a wrong premise is cheapest to fix and hardest to see. plan_create delivered
the demand; the first non-plan tool call after it is blocked (once, overridable) until
a diverger is dispatched or you override with a reason.

The demand below is reused verbatim from the plan_create inject; its "advisory /
never blocking / skipping is legal" wording describes THAT channel (a spawn-and-
continue background worker, off your critical path). Here the check is ENFORCED.

The FACET MANIFEST requirement (the plan_create inject's facet section) also rides
this channel: your override reason must state BOTH — that the diverger was dispatched
(or why it is genuinely skippable) AND the facet manifest file path (or why the task
has no extractable facets). The manifest is the upstream source for close-out
reconciliation: without it, the final deliverable check re-scrapes the conversation
instead of reading the pinned qualifiers.
"""


_DIVERGER_INJECT = """

Divergence check — dispatch a diverger the moment this plan exists. It runs
IN PARALLEL with your plan and costs the time wall NOTHING.

PARALLEL, ZERO WALL-CLOCK: the diverger is a spawn-and-continue BACKGROUND worker — you fire it and keep working your own next step; never blocked on it, it never consumes your time window. So the DEFAULT is to DISPATCH: unless the task is genuinely trivial (a single known command, a rename, a version bump with no design choice), spawn it. "Skipping saves time" is a false economy — the worker is ~3-minute and off the critical path, so skipping saves zero wall-clock and forfeits the one independent check you cannot run on your own framing.

PURPOSE — three questions, any one of which can change your plan:
  1. A BETTER method — faster / cheaper / simpler than the one this plan commits to (a different algorithm, tool, or cut)?  [optimality]
  2. A MORE CORRECT method — one that avoids a correctness risk this plan carries (a wrong assumption, an overfit-to-one-sample, a silent-failure mode)?  [correctness]
  3. An UNCONSIDERED method — a framing this plan never put on the table (a different decomposition, or a different reading of an ambiguous term)?  ["did I think narrowly?"]
If it surfaces none of the three, that is itself the gain: "I checked what else it could be" replaces "I think this framing is right".

WHY: you framed this plan, so its premises are invisible to you — the same context that produced a framing cannot reliably judge it (self-review inherits the blind spot; Huang et al. 2024). A wrong framing is the most expensive defect class: every step, test and hour downstream inherits it, and it surfaces only as late-stage dead-ends when the budget is nearly gone. The diverger reads the task WITHOUT your plan, so it sees framings you did not.

Spawn ONE diverger worker (spawn_worker) RIGHT AFTER framing this plan — do not wait for step 2, and do not wait for the result before acting. Contract:
  - goal: "Propose 2-3 genuinely different framings of the task — is there a BETTER / MORE CORRECT / not-yet-considered method? Different method-class, different decomposition, or a different reading of an ambiguous term — NOT refinements of this plan."
  - constraints: read-only (writable: [one scratch dir containing output_ptr — an empty list is rejected by validation]; forbidden: modify any file), max_minutes: 3. Give it the task's ORIGINAL statement, not your plan.
  - The diverger proposes IDEAS ONLY: no verdicts, no ranking, no agreeing. It answers "what else could this problem be", not "is this plan good".
When it returns, issue a ONE-LINE ruling for EACH alternative — adopt (then actually revise the plan) or reject (name WHY, e.g. "incompatible with X constraint"). A ruling with no reason is not a ruling; an unruled alternative is an unexamined premise. This fires per framing — at every plan_create."""


# Wall-aware first plan: the FIRST framing must bank a
# minimal verifiable version inside the time wall before scaling up. The wall
# value is read from FLAGSCALE_AGENT_TIME_BUDGET_SEC — the same env the
# agent's budget-stats path reads — so the plan can order steps against the
# actual number. Write-through: later increments REPLACE the deliverable in
# place, never rewrite from scratch, so the best version so far is always on
# disk when the wall arrives. Folded into the plan-framing gate (one gate, one
# override — same pattern as the qualifier and diverger demands above).
_WALL_AWARE_FIRST_PLAN = """

Wall-aware first plan — a plan made without a time wall can spend the whole budget
framing with nothing settled. Order steps so a MINIMAL VERIFIABLE VERSION lands
within the wall: a crude but complete, scorable deliverable at the required path
first (write-through), then refine — later increments REPLACE it in place, never
rewrite from scratch. Price every expensive step against the wall; if the full
version would overrun, the minimal version IS the deliverable, not a checkpoint on
the way to one.

If SEVERAL plausible paths exist and you cannot tell which is right, do NOT make the
first plan a linear commit to path #1. Make it a PORTFOLIO: one step running a
DISCRIMINATING micro-experiment per candidate (2-3 candidates, ~10-15% of budget
each) whose every outcome rules at least one candidate in or out, then a step on the
survivor. Serial trial-and-error taxes each wrong path at full cost; a discriminating
probe buys the elimination at a fraction of it."""


def _wall_aware_block() -> str:
    """Wall-aware first-plan demand, parameterized by the actual wall.

    Reads FLAGSCALE_AGENT_TIME_BUDGET_SEC (the same env the agent's budget
    stats read, first config then env); a set wall is quoted as minutes so
    the agent can order steps against a number, not a vibe. No wall here
    still injects the generic demand — the consumer's run may have a wall
    this process cannot see (config-only or external harness limit).
    Malformed values degrade to the generic form, never crash the guard.
    """
    raw = os.environ.get("FLAGSCALE_AGENT_TIME_BUDGET_SEC")
    if raw:
        try:
            secs = float(raw)
            if secs > 0:
                return (
                    f"\n\nTime wall: ~{secs / 60.0:.0f} minutes from task start."
                    + _WALL_AWARE_FIRST_PLAN
                )
        except (TypeError, ValueError):
            pass
    return "\n\n" + _WALL_AWARE_FIRST_PLAN


def _premortem_block() -> str:
    """Feature-keyed premortem one-liners, injected at plan framing.

    Surfaces a PAID-FOR lesson at the decision moment — but only when the
    task's own workspace shows the matching OBSERVABLE feature, so a task that
    does not match the archetype gets no noise (no feature → no inject). The
    archetypes and their trigger features come from the 89-dossier fail-family
    analysis: grader-contract-invisible (§ the largest family), bypassed-judge-
    tool, single-path over-bet. Detection is a cheap NON-RECURSIVE scan of the
    agent process working directory (env ``FLAGSCALE_AGENT_TASK_DIR`` when the
    launcher exports it, else cwd; subdirectories are NOT scanned — artifacts
    placed under fixtures/tests are invisible to this probe by design), and
    every failure degrades to "" — a guard must never crash the framing path.
    Known calibration limit: when the agent process happens to run from a
    directory that itself contains harness-like files (e.g. a source checkout
    with pyproject.toml/test_*.py), the harness note fires on that cwd — the
    note is keyed to the process cwd, not a verified task workspace.
    """
    root = os.environ.get("FLAGSCALE_AGENT_TASK_DIR") or os.getcwd()
    try:
        names = set(os.listdir(root))
    except OSError:
        return ""

    notes: list[str] = []

    def _any(*preds) -> bool:
        return any(p(n) for n in names for p in preds)

    has_ref = _any(
        lambda n: n in ("expected_output.txt", "output.txt", "expected.txt",
                        "sample_output.txt", "reference.txt", "golden.txt"),
        lambda n: n.startswith("expected") or n.endswith(".expected"),
    )
    has_harness = _any(
        lambda n: n in ("verify.sh", "run_tests.sh", "test.sh", "Makefile",
                        "conftest.py", "pytest.ini", "pyproject.toml"),
        lambda n: n.startswith("test_") or n.endswith("_test.py"),
    )

    if has_ref:
        notes.append(
            "Reference artifacts are present in the working directory — the "
            "task's own acceptance surface is observable here. Before inventing "
            "your own convention (file format, field order, rounding), diff it "
            "against these reference artifacts; a self-built convention that was "
            "never checked back is the largest silent-failure family."
        )
    if has_harness:
        notes.append(
            "A test/verify harness is present in the working directory — USE IT "
            "as the grader, do not bypass it. Run it early and after every "
            "change; a hand-rolled check that shadows the provided harness "
            "doubles the cost and can still miss the real contract."
        )
    if not notes:
        return ""
    return "\n\nPremortem — this task shows the shape of a known loss archetype:\n" + "\n".join(
        f"- {n}" for n in notes
    )


class PlanGuard(Guard):
    """Nudges (interactive) or requires (single-shot) an active plan.

    Interactive: after REMIND_THRESHOLD tool calls without a plan, injects a
    periodic reminder. Never blocks.

    Single-shot: the plan stands in as structural supervisor. Enforcement is
    front-loaded to the "start producing" moment: the first write_file with no
    plan blocks (overridable) to force plan_create before deliverable output.
    After SINGLE_SHOT_BLOCK_THRESHOLD calls without a plan, blocks further
    non-plan tools until plan_create. Overridable for genuinely-trivial tasks.
    There is NO completion-time block: completing without a plan is never a hard
    failure (that punished weak models with a livelock kill); the plan is forced
    earlier, at write_file, where it is actionable rather than punitive.
    """

    name = "plan"
    priority = 35

    # Interactive: periodic nudge cadence. Single-shot uses the tighter
    # SINGLE_SHOT_REMIND cadence — an unsupervised run should be reminded sooner,
    # since the write_file gate (not a completion block) is the real enforcement.
    REMIND_THRESHOLD = 15
    SINGLE_SHOT_REMIND = 5
    # Single-shot: allow this many observation/exploration calls before requiring
    # a plan. Set generously — an unsupervised run legitimately needs to probe the
    # environment (paths, configs, GPU state, prior findings) before it can frame a
    # plan whose steps land on real checkpoints. Blocking too early forces a plan
    # written before understanding, which is worse than no plan.
    SINGLE_SHOT_BLOCK_THRESHOLD = 20

    # Tools whose COMPLETED call can settle the divergence demand. Named once:
    # check_pre's gate exempts them (a compliant dispatch must not be blocked
    # on the very call that complies) and check_post settles them.
    _DISPATCH_TOOLS = ("spawn_worker", "dispatch_many")

    def __init__(self, task_plan=None, single_shot: bool = False):
        self._task_plan = task_plan
        self._single_shot = single_shot
        # ── State ledger ─────────────────────────────────────────────────
        # Every field below is one obligation of the plan-framing contract.
        # Lifecycle is tagged per field: [per-turn] reset by reset_turn(),
        # [per-framing] armed on each plan_create, [per-resolve] transient
        # within one registry resolve, [per-run] sticky for the whole run.
        #
        # The ONLY flag check_post mutates is _diverger_pending: check_post is
        # the SETTLEMENT site — it observes the completed call (tool_result +
        # full tool_args), which check_pre cannot see, and discharges an
        # obligation only when the call actually DID the required thing, never
        # on the appearance of it (attempt != done). Silence is deliberate:
        # settlement state feeds check_pre's future gates; a verdict here would
        # be per-obligation chatter with zero new information.

        # No-plan nag budget. [per-turn]
        # Armed: incremented by every non-plan check_pre. Discharged: a
        # plan_create (check_post resets to 0 — the framing happened) or
        # reset_turn().
        self._calls_without_plan = 0

        # A plan was actually framed at least once. [per-run]
        # Distinct from task_plan.get_active(): a plan may later be completed/
        # abandoned, but the "you have not planned yet" obligation stays
        # discharged. Set ONLY by a plan_create whose result shows success —
        # a blocked/errored attempt leaves the obligation live.
        self._plan_ever_created = False

        # Qualifier-extraction demand delivered for the CURRENT framing.
        # [per-framing]
        # Armed: a plan_create consumes-and-delivers it (or a block carries it
        # first). reset_turn() re-arms for the next framing.
        self._qualifier_reminded = False

        # Divergence demand delivered for the CURRENT framing. [per-framing]
        # Same lifecycle as _qualifier_reminded; a later re-framing's premises
        # are as unexamined as the first's (user ruling 2026-09-23), so the
        # demand re-arms per framing. Anti-ritualization lives in the demand's
        # contract (a one-line ruling for EVERY alternative), not in a
        # once-per-run flag.
        self._diverger_reminded = False

        # The divergence demand was delivered and NOT yet settled by an actual
        # diverger dispatch. [per-framing gate]
        # Armed: delivery sites (plan_create / single-shot block). Settled:
        # check_post observing a real diverger dispatch; or consumed by the
        # one-shot block; or reset_turn(). Re-armed by block_not_surfaced()
        # when the block never reached the agent.
        self._diverger_pending = False

        # A one-shot marker within the resolve where OUR diverger block fired.
        # [per-resolve transient]
        # Lets block_not_surfaced() know a higher-ranked block outranked OURS
        # (the agent never saw the demand) so the demand must re-arm — without
        # spuriously re-arming when an unrelated block (write_file / single-shot)
        # lost the ranking. Cleared at the top of every check_pre and in
        # reset_turn().
        self._diverger_block_consumed = False

    def set_single_shot(self, enabled: bool = True):
        """Enable single-shot enforcement at runtime (set once run mode known)."""
        self._single_shot = enabled

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Fresh resolve: clear the transient "our diverger block fired this
        # resolve" marker (re-set below iff it fires), so a stale True from a
        # previous turn can never cause a spurious re-arm.
        self._diverger_block_consumed = False
        if not ctx.tool_name:
            # No completion-time gate. Completing without a plan is intentionally
            # NOT blocked: the old NON-OVERRIDABLE completion block livelocked weak
            # models (they neither reliably override nor plan_create, so the loop
            # burned MAX_CONSECUTIVE_COMPLETION_BLOCKS and auto-killed the task →
            # reward 0). Enforcement moved earlier to write_file, where forcing a
            # plan is actionable instead of punitive.
            return None

        # write_file gate (single-shot): the first attempt to write a file with
        # no plan is the "start producing a deliverable" moment. Block once
        # (overridable) to force plan_create before deliverable output. Only
        # write_file triggers — read_file/shell exploration never does, so the
        # investigation phase is not disturbed. Once a plan exists this never
        # fires.
        if ((ctx.tool_name == "write_file" or ctx.tool_name == "edit_file")
                and self._single_shot
                and not self._plan_ever_created
                and not (self._task_plan and self._task_plan.get_active())):
            self._qualifier_reminded = True
            return GuardVerdict.block(
                message=_WRITE_FILE_NO_PLAN + _QUALIFIER_EXTRACTION + _wall_aware_block() + _premortem_block(),
                reason="write_file_without_plan",
                category="plan_required",
                overridable=True,
            )

        # Plan-related tools don't count toward the no-plan budget. plan_create is
        # the plan-framing moment — deliver the qualifier-extraction demand on
        # every framing (re-armed by reset_turn), and fold the divergence demand
        # into EVERY framing as well (both ride the same one-verdict-per-check_pre
        # channel).
        if ctx.tool_name == "plan_create":
            # Both demands belong to EACH framed plan (user ruling 2026-09-23:
            # a re-framing's premises are as unexamined as the first's, so the
            # diverger re-arms per framing exactly like the qualifier). A block
            # earlier in THIS framing (write_file gate / single-shot) may
            # already have carried them — consume the flags unconditionally
            # either way, so the NEXT framing re-arms them.
            inject_qualifier = not self._qualifier_reminded
            print_diverger = (
                not self._diverger_reminded
                and not _self_is_review_or_diverge_worker()
            )
            # Arm the dispatcher gate: the NEXT non-plan tool call blocks once
            # if no diverger was dispatched. Only an actual spawn_worker /
            # dispatch_many clears it (check_post). The single-shot block below
            # carries the demand AND arms this gate itself, so a later
            # plan_create in that framing re-delivers nothing (print_diverger
            # False) yet the gate stays armed for its first non-plan tool call.
            if print_diverger:
                self._diverger_pending = True
            self._qualifier_reminded = False
            self._diverger_reminded = False
            if not inject_qualifier and not print_diverger:
                return None
            msg = "[Plan] Framing the plan."
            if inject_qualifier:
                msg += _QUALIFIER_EXTRACTION
            if print_diverger:
                msg += _DIVERGER_INJECT
            msg += _wall_aware_block()
            msg += _premortem_block()
            return GuardVerdict.inject(
                message=msg,
                reason="qualifier_extraction",
                category="plan_required",
            )
        if ctx.tool_name in ("plan_update", "plan_status"):
            return None

        # Diverger-dispatch gate: after a framing delivered the divergence
        # demand, the first subsequent tool call is blocked once (overridable)
        # if no diverger was dispatched. Three outcomes, in order:
        if self._diverger_pending:
            if ctx.tool_name in self._DISPATCH_TOOLS:
                # Compliance candidate — never block the very call that
                # complies. Do NOT clear here: only an ACTUAL diverger
                # dispatch settles the demand, and only check_post can judge
                # the args (F1: a plain spawn leaves the gate armed).
                pass
            elif _self_is_review_or_diverge_worker():
                # The demand is parent-side only; a worker context neither
                # enforces nor is bound by it. Clear so it does not linger.
                self._diverger_pending = False
            else:
                self._diverger_pending = False
                # Mark that THIS resolve consumed the demand via our block. If a
                # higher-ranked block outranks ours, the registry calls
                # block_not_surfaced() and we re-arm (the agent never saw us).
                self._diverger_block_consumed = True
                return GuardVerdict.block(
                    message=(
                        _DIVERGER_ENFORCE
                        + _DIVERGER_INJECT
                        + _wall_aware_block()
                        + _premortem_block()
                    ),
                    reason="diverger_not_dispatched",
                    category="plan_required",
                    overridable=True,
                )

        # If plan exists, nothing to do
        if self._task_plan and self._task_plan.get_active():
            return None

        # A plan that was created and then COMPLETED/ABANDONED clears the active
        # pointer, so get_active() returns None and the check above reads it as
        # "never planned" — re-incrementing the counter and re-nagging after the
        # plan is already done. In single-shot there is exactly one turn, so the
        # sticky _plan_ever_created flag is equivalent to per-turn: once a plan
        # has SUCCESSFULLY been framed, the "you have not planned yet" obligation
        # is discharged and the no-plan budget must not re-fire. (The flag covers
        # any later plan state — completed, abandoned, or paused — because the
        # nag's purpose is to get a FIRST plan framed, which has happened.)
        # Scoped to single_shot ONLY: an interactive multi-turn session must keep
        # nagging a fresh later turn that has not planned yet.
        if self._single_shot and self._plan_ever_created:
            return None

        self._calls_without_plan += 1

        # Single-shot: after observation budget, require a plan (block).
        if self._single_shot and self._calls_without_plan > self.SINGLE_SHOT_BLOCK_THRESHOLD:
            # Block carries BOTH demands (qualifier + diverger); mark both
            # delivered to prevent double-injection on the subsequent plan_create.
            self._qualifier_reminded = True
            self._diverger_reminded = True
            self._diverger_pending = True
            return GuardVerdict.block(
                message=(
                    f"[Plan] Pause. {self._calls_without_plan} tool calls in this "
                    f"unsupervised run without a plan. Do you understand the problem's "
                    f"structure well enough to plan it?\n"
                    f"— If yes: call plan_create() now. Steps must land on real "
                    f"checkpoints, acceptance criteria naming what THIS task requires, "
                    f"not generic \"works correctly\".\n"
                    f"— If no: override and keep investigating. A plan written before "
                    f"understanding is worse than no plan — it locks in a shape you'll "
                    f"fight later."
                    + _QUALIFIER_EXTRACTION
                    + _DIVERGER_INJECT
                    + _wall_aware_block()
                    + _premortem_block()
                ),
                reason="single_shot_plan_required",
                category="plan_required",
            )

        # Periodic reminder. Single-shot uses the tighter cadence (5) since the
        # write_file gate is the real enforcement and an unsupervised run should
        # be nudged toward a plan sooner.
        cadence = self.SINGLE_SHOT_REMIND if self._single_shot else self.REMIND_THRESHOLD
        if self._calls_without_plan % cadence == 0:
            return GuardVerdict.inject(
                message=(
                    f"[Plan] {self._calls_without_plan} tool calls without a plan. "
                    f"If environment exploration is done and you've begun producing "
                    f"output, you understand the structure — call plan_create() to "
                    f"freeze it into steps with acceptance criteria."
                ),
                reason="plan_reminder",
                category="plan_needed",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Settlement site for the plan-framing obligations.

        Nothing here is surfaced to the LLM: check_post runs AFTER the tool,
        and its verdicts are advisory-only (this method always returns None).
        Its job is to SETTLE an obligation by OBSERVING that the completed
        call actually did the required thing — never by the appearance of
        it. It is the only hook that can, because check_pre runs before the
        tool and cannot see the result or the dispatch content.

        Two settlements:
          * plan_create — reset the no-plan nag budget, and discharge the
            "you have not planned" obligation ONLY if the result shows the
            plan really was produced (a failed/blocked ATTEMPT does not).
          * a real diverger dispatch — settle the divergence demand the
            framing armed (a spawn for any other purpose does not).

        The settled state feeds a LATER check_pre gate; silence here is
        deliberate — a per-obligation verdict would be chatter with no new
        information.
        """
        if ctx.tool_name == "plan_create":
            self._calls_without_plan = 0
            # Only a plan_create that ACTUALLY produced a plan may discharge the
            # no-plan obligation. A failed/blocked call (exception wrapper, guard
            # block, or the tool's own "ERROR: ..." return) leaves `tool_result`
            # carrying a failure marker — do NOT set the sticky flag, or the
            # single-shot no-plan block would be silently disabled for the rest of
            # the run by a mere ATTEMPT. `None` (no result plumbed) keeps the
            # historical success semantics.
            if not _tool_result_indicates_failure(ctx.tool_result):
                self._plan_ever_created = True
        # A dispatch satisfies the divergence demand for THIS framing: the
        # agent actually spawned the independent proposer. Clear the gate so the
        # following tool call is not blocked.
        elif ctx.tool_name in self._DISPATCH_TOOLS:
            # Only an ACTUAL diverger dispatch satisfies the demand — a spawn
            # for any other purpose leaves the gate armed (F1).
            if _dispatch_includes_diverger(ctx.tool_name, ctx.tool_args):
                self._diverger_pending = False
        return None

    def reset_turn(self):
        """New user message resets counter.

        _qualifier_reminded and _diverger_reminded are both reset: both
        demands are per-framing, so a new turn's plan_create must re-inject
        them (a prior block may have carried them earlier in the same turn —
        that is why they are not cleared at the top).

        Anti-ritualization note: re-arming per turn does not ritualize the
        checks — the qualifier's value is its framing-time questions, and the
        diverger's is its one-line-ruling contract, both of which survive
        repetition. What was ritual-prone was firing the SAME plan shape
        again, which re-framing is not.

        Note: _plan_ever_created is intentionally NOT reset here. In single-shot
        mode there is only one turn, so it never matters; in interactive mode the
        completion gate never fires anyway. Keeping it sticky avoids a spurious
        block if reset_turn is ever called mid-single-shot-run.
        """
        self._calls_without_plan = 0
        self._qualifier_reminded = False
        self._diverger_reminded = False
        # The pending-dispatch gate is per framing/turn; a new turn re-arms it
        # only when its plan_create re-delivers the demand.
        self._diverger_pending = False
        self._diverger_block_consumed = False

    def block_not_surfaced(self):
        """Registry callback: OUR diverger block lost the severity ranking to a
        higher-ranked block and never reached the agent.

        The block branch clears `_diverger_pending` (one-shot) at the moment it
        fires. If a harder block (escalate / non-overridable) outranks ours in
        the SAME resolve, the agent never sees the diverger demand, yet the flag
        was already consumed — the demand silently vanishes. Re-arm it here so
        it re-fires on the next call once the winning block clears (F4).

        Scoped by `_diverger_block_consumed`, which is set ONLY in the diverger
        block branch (never by the write_file / single-shot blocks), so those
        blocks losing the ranking do NOT spuriously re-arm the dispatch gate.
        """
        if self._diverger_block_consumed:
            self._diverger_pending = True
            self._diverger_block_consumed = False
