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

"""Plan create tool — create a structured task plan."""

import json

from flagscale_agent.react.tools.base import Tool
from flagscale_agent.react.oracle import (
    ORACLE_ACCEPTANCE_ITEM,
    STATED_CONSTRAINT_ITEM,
    AMBIGUITY_ANCHOR_ITEM,
    PREMISE_SOURCES,
    build_oracle_binding_line,
    declares_external_premise,
)


# The fixed review step the harness appends to EVERY plan. It is the single
# place where the completion-time verification content lives now that the
# terminal completion gates are gone: instead of a stack of gates each biting
# once at the finish line, the checks are SCHEDULED into the plan as one
# ordinary step whose acceptance criteria the (retained) step_done evidence
# gate enforces. The agent works through it in order like any other step.
_REVIEW_STEP_TITLE = "Review (FlagScale-Agent Plan fixed template)"

_REVIEW_STEP_ACCEPTANCE = [
    "Observation over argument: every conclusion points to an output you "
    "actually RAN and READ (command + result); arguing that the method 'should "
    "be right' is not evidence. If something has no runnable check, state NOT "
    "VERIFIED explicitly.",
    "Requirements reconciliation: list each requirement of the task verbatim, "
    "give a command + real output for each (or NOT VERIFIED), and confirm the "
    "deliverable satisfies every one at its final path, including form "
    "constraints (format, naming, version, timing, counts).",
    "Generalization: verify on at least one perturbed or held-out input, not "
    "only on the single development sample; if there is no generalizable point, "
    "say why.",
    "Delivery hygiene: the delivery path contains only the artifacts the task "
    "requires (no .bak/scratch/build byproducts); confirm the real artifact is "
    "non-empty and well-formed, not a placeholder or truncated file.",
    "Unsettled premises: if a conclusion rests on a load-bearing premise you "
    "yourself described as unverifiable/assumed/no-reference, either give a "
    "probe output whose design intent was to falsify it, or state the "
    "world-state in which the premise is false and why it is a safe bet.",
    "Pre-mortem (assume you are wrong): name the single point most likely to "
    "fool you, and how you ruled it out.",
    "Reviewer — a deliberate trade-off, not a mandate. Know the reviewer's "
    "benefit (a fresh context that surfaces the defect your own tests cannot "
    "construct) and price it against your time budget (a worker's whole "
    "deadline). Dispatch when the benefit justifies the cost — most true where a "
    "change can fail SILENTLY (a boundary such as parser/permission/config/"
    "serialization/cross-call-site, or a multi-file behavior change); skip when "
    "it clearly does not (a trivial single-file constant/config/naming/version "
    "edit — self-verify). STATE the decision and WHY either way, tying the WHY to "
    "the predicate that applied (the boundary type, or the single-file/"
    "no-behavior-change status) — the reviewer "
    "demand lives in THIS fixed Review step (not at each step_done, which keeps "
    "only the evidence gate).",
]


# The fixed RSI step the harness appends to EVERY plan, AFTER the review step.
# RSI = Recursive Self-Improvement: it protects the HARNESS, not this
# deliverable. The point is NOT to log what happened — it is to make the
# HARNESS itself better, so that improving the harness gets easier next time
# (the recursion). Primary: turn this run's gap / failure mode into a MECHANISM
# change (a proposal, a prompt/guard/tool/knowledge edit) — the improvement of
# the improvement process. Secondary (only when no mechanism-level gap exists):
# land task knowledge to memory as fuel for FUTURE tasks. Either outcome is a
# legal close (skippable with a reasonable reason); silently skipping is not.
# This keeps the fixed plan tail = ... → review → RSI (RSI last).
_RSI_STEP_TITLE = "RSI (Recursive Self-Improvement — fixed template, final step)"

_RSI_STEP_ACCEPTANCE = [
    "Recursive Self-Improvement: this step improves the HARNESS, not this "
    "deliverable — so its highest-value output is a MECHANISM change: a "
    "proposal(), or a prompt/guard/tool/knowledge edit, that would make a class "
    "of FUTURE tasks (or FUTURE improvements) go better. Ask first: can this "
    "run's gap or failure mode become a durable change to the harness? If yes, "
    "LAND it and name the concrete target (proposal id, file, or the exact edit).",
    "Report ALL THREE RSI channels explicitly in the wrap-up (each under its own "
    "sub-heading, even if empty — then say why): (a) landed mechanism changes "
    "(file:line / proposal id / memory key); (b) proposals, BOTH newly raised "
    "this session AND historical OPEN/unadjudicated ones still awaiting a human "
    "decision (status proposed/approved) — surface historical open proposals by "
    "id (run proposal(action='list') if needed); never silently drop one; (c) "
    "memory updates landed this session (key + one-line what).",
    "Task knowledge is the FALLBACK, not the goal: only when there is genuinely "
    "no mechanism-level gap, land a memory_write (fact/pitfall/insight) as fuel "
    "for future tasks, and say WHY it does not rise to a harness change. Naming "
    "a memory key alone does not satisfy this step if a mechanism gap existed.",
    "If there is truly nothing at either level, state that explicitly and say "
    "WHY the task taught nothing reusable — a legal close, but it must be "
    "asserted, not left implicit.",
    "Keep it falsifiable and landing-capable: point to a concrete artifact "
    "(proposal id, file, or memory key), not a vague 'be more careful'. A lesson "
    "that never lands breaks the recursion — an unpersisted improvement is not "
    "an improvement.",
]


# The fixed FRAMING step the harness PREPENDS to EVERY plan. This is the single
# home for the plan-framing discipline that used to be delivered as a
# plan_create-time [Plan] inject: the divergence check (self + dispatched
# diverger + synthesis), qualifier/facet extraction, the wall-aware first cut,
# the premortem, and the name-the-standard-method rule. Because it is an
# ordinary step with acceptance criteria, the (retained) step_done evidence gate
# enforces it at the moment the agent marks it done — one systematic place, no
# separate guard, no chatty inject. Worked FIRST, before the agent's own steps.
_FRAME_STEP_TITLE = "Frame (FlagScale-Agent Plan fixed template)"

_FRAME_STEP_ACCEPTANCE = [
    "Negative constraints (不做清单) — write this BEFORE you start working, in the "
    "SAME plan_create call as your steps: `forbidden=[...]`, 2-5 entries, each "
    "{action, why, instead}. `steps` says what to do; `forbidden` says what NOT to "
    "do, and the harness re-renders it at the TOP of your plan context every turn "
    "beside the time ledger. This is not paperwork — measured over real losses, "
    "aggregate behaviour barely differs from wins, but two shapes separate sharply: "
    "(1) >=8 consecutive edits with no successful execution (65% of losses vs 22% of "
    "wins) and (2) ending the run while still churning — the last stretch spent on "
    "same-shape probing or polishing rather than verifying (20% of losses vs 6% of "
    "wins). The point of writing it NOW is that a prohibition decided at planning "
    "time fires at the decision point; one you would only think of at the deadline "
    "arrives after the mistake. Name ACTIONS specific to THIS task, not virtues: "
    "'do not re-run the same command with a bigger timeout — read the partial "
    "output and change the approach' is a prohibition; 'be careful' is not. You may "
    "append more later with plan_update(action='add_forbidden') as traps appear.",
    ORACLE_ACCEPTANCE_ITEM,
    STATED_CONSTRAINT_ITEM,
    AMBIGUITY_ANCHOR_ITEM,
    "Divergence check — the self-divergence half is ALWAYS done; the dispatched "
    "half is priced (per the next item, a deliberate trade-off). (1) YOUR OWN "
    "deeper divergence: put 2-3 genuinely different framings of the task's "
    "ORIGINAL statement on the table — a BETTER method (faster/cheaper/simpler), a "
    "MORE CORRECT method (avoids a risk this plan carries), or an UNCONSIDERED "
    "decomposition/reading. (2) A read-only diverger worker — dispatch it (per the "
    "trade-off in the next item) when the framing warrants a fresh set of eyes; "
    "spawn_worker "
    "with role=\"diverger\" and a read-only contract — the diverger tag is a LEAF "
    "role: it gets NO reviewer preset (it has no deliverable path + rev marker "
    "before framing) and cannot itself spawn — whose sole job "
    "is to debate this framing "
    "from a FRESH context — it does not inherit your premises, which is the entire "
    "point; pass it the task's ORIGINAL statement verbatim and let it name its own "
    "alternatives. Its findings are CLAIMS, not verdicts: reproduce-or-refute each "
    "yourself. (3) SYNTHESIZE both — issue a one-line ruling for EACH framing "
    "(yours and the worker's): adopt (then revise the plan) or reject (name WHY). "
    "You framed this plan, so its premises are invisible "
    "to you — a wrong framing is inherited by every step downstream, and a "
    "fresh-context diverger is the one check that does not share your blind spot.",
    "Diverger — a deliberate trade-off, not a mandate. Know the diverger's "
    "benefit (it probes the FRAMING before any artifact exists — the cheapest "
    "moment to fix a wrong premise, and the one you cannot self-audit from inside "
    "your own frame) and price it against your time budget (a worker's whole "
    "deadline). Dispatch when the framing is genuinely open (a novel or ambiguous "
    "task, several plausible decompositions, an irreversible direction); skip when "
    "it is settled and cheap to state (a small, well-specified edit — do the "
    "self-divergence half only). STATE the decision and WHY, in plan notes, tying "
    "the WHY to the predicate that applied (framing open vs settled/small). What "
    "THIS Frame step dispatches is the DIVERGER — a PRE-artifact framing probe; a "
    "reviewer needs a FROZEN artifact and belongs at the Review step, never here "
    "(the later reviewer decision is handled there).",
    "Qualifier extraction: re-read the task and list EVERY qualifier (a point in "
    "time, a version, a subset, a metric definition, a named tool/model instance) "
    "as a first-class item, folding each into a step or acceptance criterion. Read "
    "each boundary in the direction the task MEANS, not the convenient one (a "
    "time bound selects the state as it stood then, not the newest data; a version "
    "pin selects that instance's documented usage). A dropped qualifier silently "
    "answers a nearby question the task never asked.",
    "Facet manifest: write the qualifiers to a manifest file (e.g. "
    "analysis/facets.md; state the path in this plan), each entry pinned to the "
    "task's OWN words — quote the line. Three sections: (1) FORM conventions left "
    "open (format, units, field order, naming) — pin ONE supported by the task "
    "text; (2) reference artifacts the task NAMES — quote verbatim; (3) is the "
    "task's example the GRADED instance? Do not read beyond the task statement or "
    "guess the grader. Reuse this manifest at close-out instead of re-scraping the "
    "conversation.",
    "Acceptance-facet manifest: BEFORE you build, enumerate every FACE the "
    "consumer will actually drive — each surface the task text says is checked, "
    "each artifact it NAMES, each command it implies it runs. Write one row per "
    "verifiable clause of the task text: the row COUNT is fixed by the clause "
    "count, not by how many faces you happen to think of, so a short manifest is "
    "a defect, not a judgment call. Each row carries (a) the task clause quoted "
    "verbatim, (b) the exact command/input that exercises that face, (c) the "
    "observed result when you ran it. A row with no executed probe is a sentence, "
    "not a check; and a face you deliberately silence — a flag you set to quiet "
    "output, a stream you stop printing — is a face you stopped checking while "
    "the consumer still reads it. Segment on the task text's own sentences and "
    "bullets; treat each stated constraint inside a sentence (a numeric bound, a "
    "form rule, a named artifact, a named command) as its own clause, and when in "
    "doubt split rather than merge — never fold several clauses into one row, "
    "because under-splitting is the losing direction and a merged row hides the "
    "face it absorbed. Close-out reviews THIS manifest row by row, not your "
    "memory of it.",
    "Reading ledger: when the task text leaves the OUTPUT FORM genuinely open — "
    "two readings of the same requirement that would produce different bytes — do "
    "not pick one and build around it. First write the ledger: every defensible "
    "reading of that requirement, each pinned to the task text's own words, plus "
    "the concrete artifact each reading would produce. Then run a dominance test "
    "against the consumer's own material: does any reading make the other "
    "impossible, or does one reading's artifact satisfy the other's acceptance "
    "test as well? Choose the dominant reading when one exists. First determine "
    "which direction the comparison penalizes: cover both readings only when "
    "extra output is ignored; when extras are penalized, cover with a switch "
    "defaulted to the dominant reading instead of shipping both. When no reading "
    "dominates, cover the top-2 (or top-k) in the SAME run — build each reading's "
    "artifact and ship both variants or a switch — because an undominated "
    "ambiguity is a coin flip whose wrong side costs a whole run, and betting it "
    "in one run at a time is how a single ambiguous clause consumes run after "
    "run. This is "
    "PRINCIPLE 4's hedge/bet rule applied at BUILD time, not a separate doctrine: "
    "enumerate, test dominance, cover when undominated, and only bet a single "
    "reading once the top-2 coverage is shown impossible.",
    "Wall-aware first cut: order steps so a minimal but complete, scorable "
    "deliverable lands at the required path within the time wall BEFORE scaling "
    "up, then refine in place (write-through — never rewrite from scratch, so the "
    "best version so far is always on disk). Price every expensive step against the "
    "wall; if the full version overruns, the minimal version IS the deliverable. If "
    "several plausible paths exist and you cannot tell which is right, make this "
    "first plan a PORTFOLIO: a discriminating micro-experiment per candidate whose "
    "every outcome rules one in or out, then a step on the survivor.",
    "Premortem on the workspace: if reference artifacts are present (expected_*/"
    "golden/reference files), diff your own convention against them BEFORE "
    "inventing one; if a test/verify harness is present, USE it as the grader, run "
    "it early and after every change — a hand-rolled check that shadows it doubles "
    "the cost and can still miss the contract.",
    "Standard-method first: name the problem CLASS and its STANDARD METHOD, and "
    "include a small-sample step that validates the method on the smallest "
    "meaningful input before scaling (generic 'analyze/implement/test' steps mean "
    "the class was never identified; a brute-force sweep is the tell).",
    "Convention falsifier: EVERY step_done carries a `convention` — either "
    "{\"name\": \"none\"} (no derived convention governs the step) or the "
    "convention your output rests on (a base, a stride, a key space, a "
    "normalisation, a default, a serialisation) WITH a falsifier: a real probe "
    "whose output could have refuted it, or a proved hedge. A bare name is "
    "blocked. The guard message states the exact rule and accepted shape.",
    "Deliverable reading-analysis: EVERY step_done states its graded artifact — "
    "{\"deliverable\": \"none\"} if none, else the path and whether you can run "
    "the grader. When the grader is NOT visible, enumerate the readings the task "
    "admits (each grounded in a verbatim task quote) and the hedge covering them "
    "(a real output + observed evidence, not an assertion). The guard message "
    "states the exact rule.",
    "Constructive hedge-feasibility: when the grader is invisible and the task "
    "admits 2+ readings of the output, do not decide feasibility by introspection "
    "— construct the union of all readings' expected content and test whether one "
    "output satisfies every reading (union when extras are ignored; intersection "
    "when penalized). Only a hedge shown constructively empty licenses betting on "
    "one reading.",
    "A task's EXAMPLE value locks FORM — shape, units, stride — NOT a specific "
    "value or base.",
]


class PlanCreateTool(Tool):
    name = "plan_create"
    description = (
        "Create a task plan with ordered steps for multi-step work. "
        "A plan has TWO halves: `steps` (what to do) and `forbidden` (what NOT to "
        "do — the negative constraints the harness re-displays at the top of your "
        "plan context every turn). Write both: the negative half is what stops you "
        "from burning the last of the clock polishing something that already works. "
        "Use when about to produce a deliverable or act on a task. "
        "Only one plan can be active at a time."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short plan title, e.g. 'ESPnet LibriSpeech training reproduction'.",
            },
            "steps": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "acceptance": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "est_sec": {
                                    "type": "number",
                                    "description": (
                                        "Your PLANNING-TIME estimate of how many "
                                        "seconds this step will cost against the "
                                        "wall. Commit to a number here: an "
                                        "unestimated step is a step you have not "
                                        "actually planned, and the ledger cannot "
                                        "warn you about a budget you never "
                                        "allocated. Estimate from the work's own "
                                        "shape (how many probes/builds/reads it "
                                        "needs times what each costs here), not "
                                        "from what you hope it takes."
                                    ),
                                },
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": (
                                        "1-based step ids this step must wait "
                                        "for. DEFAULT: the previous step, i.e. a "
                                        "single serial chain. Override it when "
                                        "work is genuinely parallel: give two "
                                        "steps that do not consume each other's "
                                        "output the SAME preceding dependency, "
                                        "and the ledger will stop adding them "
                                        "end to end and report the real critical "
                                        "path instead. Expressing parallelism "
                                        "here is what lets the ledger (and you) "
                                        "see that a plan that looks over budget "
                                        "serially may fit once independent "
                                        "branches run at the same time. Use [] "
                                        "for a step that can start immediately."
                                    ),
                                },
                            },
                            "required": ["title"],
                        },
                    ]
                },
                "description": "Ordered list of step descriptions (strings) or structured steps (objects with title and optional acceptance).",
            },
            "thinking": {
                "type": "string",
                "description": (
                    "Optional but recommended for hard/open-ended tasks: your CURRENT "
                    "model of the problem — the bottleneck, the load-bearing hypothesis, "
                    "the evidence for it, and a FALSIFIABLE prediction for the next move "
                    "('if I change X, metric should move to ~Y'). This is the HOME for "
                    "deliberate/deep reasoning, distinct from fast action. Leave empty when "
                    "the task is mechanical and fast action is clearly right; fill it when "
                    "the task needs a theory before you act. You can update it later via "
                    "plan_update(thinking=...). "
                    "For any task whose deliverable is judged by a grader/verifier "
                    "separate from your own reasoning, answer these three here "
                    "before committing to an approach: "
                    "(1) SUCCESS PREDICATE — state the exact condition the checker "
                    "applies to the deliverable, in the checker's own shape (a value, "
                    "a set, a count, a range, a file, a behavior — whatever the "
                    "task's contract literally says; quote it verbatim). Do not "
                    "re-express it in your own preferred format. "
                    "(2) INDEPENDENT ORACLE — name the tool or method that decides "
                    "that predicate WITHOUT sharing your assumptions: it must be "
                    "capable of rejecting your answer (a reference engine, a "
                    "differential check, a property test — never your own pipeline "
                    "re-run on your own parse). Confirm it is actually "
                    "installable/reachable BEFORE planning around it. "
                    "(3) SELF-CORRECTION LOOP — if the oracle rejects your premises "
                    "or your answer, state the concrete retry path (fix the parse, "
                    "switch method, re-read the contract). If any of the three has "
                    "no answer, that is a wrong-approach signature — change the "
                    "approach, not the answer. "
                    "Bind the oracle at framing time: write its FIRST exercise "
                    "(the exact command/check that runs it) into the acceptance "
                    "of the FIRST load-bearing step, so the acceptance→verification "
                    "channel forces the oracle's evidence to exist early, while the "
                    "premises are still cheap to revise."
                ),
            },
            "oracle": {
                "type": "object",
                "description": (
                    "Optional but strongly encouraged at framing time: the oracle "
                    "that will decide this task, with a premise EXTERNAL to your own "
                    "implementation. Object with keys: `premise_source` (one of "
                    "'differential' | 'provided' | 'consumer'), `oracle` (what "
                    "decides it), and `first_exercise` (the exact command/check that "
                    "runs it). When supplied, its first exercise is bound into the "
                    "acceptance of the FIRST load-bearing step you list, so the "
                    "acceptance->verification channel forces the oracle's evidence "
                    "to exist early. A self-authored premise_source is refused."
                ),
                "properties": {
                    "premise_source": {"type": "string"},
                    "oracle": {"type": "string"},
                    "first_exercise": {"type": "string"},
                },
            },
            "oracle_step_id": {
                "type": "integer",
                "description": (
                    "Optional : the 1-based number of the agent step that is the "
                    "FIRST load-bearing step, so the oracle's first exercise binds "
                    "to THAT step's acceptance (default: the first listed step)."
                ),
            },
            "forbidden": {
                "type": "array",
                "description": (
                    "NEGATIVE constraints — the actions you must NOT take in this "
                    "task. `steps` says what to do; `forbidden` says what not to "
                    "do, and the harness renders it at the TOP of your plan context "
                    "every turn, right beside the time ledger. Write it NOW, at "
                    "planning time, while you can still think clearly about the "
                    "failure modes: a prohibition never written is a prohibition "
                    "you cannot honour when the clock gets tight. Each item is "
                    "either a string or an object "
                    "{action, why, instead}: `action` = the thing you must not do "
                    "(name the ACTION, not a vague virtue); `why` = the concrete "
                    "reason in THIS task; `instead` = what to do when you feel the "
                    "temptation. Good items are specific and falsifiable "
                    "('do not hand-tune constants to chase a metric — run the "
                    "measurement first'), bad ones are platitudes ('do not be "
                    "careless'). Two to five well-chosen items beat twenty."
                ),
                "items": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                                "why": {"type": "string"},
                                "instead": {"type": "string"},
                            },
                            "required": ["action"],
                        },
                    ]
                },
            },
        },
        "required": ["title", "steps"],
    }

    def __init__(self, task_plan, session_id: str = ""):
        self._plan = task_plan
        self._session_id = session_id

    def execute(self, **kwargs) -> str:
        title = kwargs["title"]
        steps = kwargs["steps"]

        # Normalize: LLM sometimes returns steps as a JSON-encoded string instead of array
        if isinstance(steps, str):
            steps = steps.strip()
            if steps.startswith("["):
                try:
                    steps = json.loads(steps)
                except (json.JSONDecodeError, ValueError):
                    pass
            # If still a string (single step or unparseable), wrap in list
            if isinstance(steps, str):
                steps = [s.strip() for s in steps.split("\n") if s.strip()]

        if not steps or not isinstance(steps, list):
            return "ERROR: At least one step is required."
        
        # Normalize steps: keep dicts as-is, convert strings
        normalized = []
        for s in steps:
            if not s:
                continue
            if isinstance(s, dict):
                # Structured step
                if "title" not in s:
                    return f"ERROR: Step dict must have 'title' field: {s}"
                normalized.append(s)
            else:
                # Plain string step
                normalized.append(str(s))
        
        if not normalized:
            return "ERROR: At least one step is required."

        # Oracle-at-frame: validate the DECLARED shape of a supplied oracle and
        # bind its first exercise to the step the agent designates as load-bearing
        # (F4: a JSON-encoded string is decoded, like `steps` above; F3: the target
        # is explicit, not silently "whatever was listed first").
        oracle = kwargs.get("oracle")
        if isinstance(oracle, str):
            oracle = oracle.strip()
            if oracle.startswith("{"):
                try:
                    oracle = json.loads(oracle)
                except (json.JSONDecodeError, ValueError):
                    pass
        if oracle is not None:
            if not declares_external_premise(oracle):
                return (
                    "ERROR: oracle must DECLARE an external-premise source. Provide "
                    "'premise_source' (one of: "
                    + " | ".join(PREMISE_SOURCES)
                    + "), non-empty 'oracle' and 'first_exercise', and no extra keys. "
                    "A self-authored/unknown premise_source, a missing leg, or an "
                    "annex key is refused. Note: this is a declaration, not proof -- "
                    "name the external artifact so a reader can check it."
                )
            binding = build_oracle_binding_line(oracle["oracle"], oracle["first_exercise"])
            # F3: bind to the step the agent marks load-bearing when it says so,
            # else the first listed step (the convention is stated in the FRAME
            # acceptance text: "list that step first, or pass its index").
            target_idx = 0
            raw_idx = kwargs.get("oracle_step_id")
            if isinstance(raw_idx, int) and 1 <= raw_idx <= len(normalized):
                target_idx = raw_idx - 1
            target = normalized[target_idx]
            if isinstance(target, dict):
                target.setdefault("acceptance", [])
                target["acceptance"] = [binding] + list(target["acceptance"])
            else:
                normalized[target_idx] = {"title": str(target), "acceptance": [binding]}

        # Harness-appended FIXED steps — the plan standard template. The FIRST
        # step frames the plan (divergence check + qualifier/facet extraction +
        # wall-aware first cut + premortem + standard-method), and the LAST two
        # are review (protects THIS deliverable) then RSI (protects the HARNESS,
        # for FUTURE tasks). Their acceptance carries the framing-time and
        # completion-time discipline that previously lived in plan_create/step_done
        # gates. The frame step is PREPENDED so the agent works it first; review
        # and RSI are APPENDED so it works them last.
        normalized.insert(0, {
            "title": _FRAME_STEP_TITLE,
            "acceptance": list(_FRAME_STEP_ACCEPTANCE),
        })
        normalized.append({
            "title": _REVIEW_STEP_TITLE,
            "acceptance": list(_REVIEW_STEP_ACCEPTANCE),
        })
        normalized.append({
            "title": _RSI_STEP_TITLE,
            "acceptance": list(_RSI_STEP_ACCEPTANCE),
        })

        thinking = kwargs.get("thinking", "") or ""
        forbidden = kwargs.get("forbidden")

        try:
            plan = self._plan.create(title, normalized, self._session_id, thinking=thinking,
                                     oracle=oracle, forbidden=forbidden)
            forbidden_report = self._forbidden_report(plan)
            return (
                "Plan created.\n\n"
                "NOTE: this is the FlagScale-Agent Plan STANDARD TEMPLATE. The harness "
                "has prepended one fixed FIRST step and appended two fixed final steps: "
                "Frame (divergence check + qualifier/facet extraction + wall-aware first "
                "cut + premortem + standard-method) → your steps → Review (self-check + "
                "reviewer) → RSI (recursive self-improvement). Work the steps IN ORDER "
                "and provide verification evidence for each acceptance criterion. All "
                "fixed steps are SKIPPABLE with a reasonable reason, but do not skip "
                "them silently.\n\n"
                + forbidden_report
                + self._time_budget_report(normalized)
                + f"{self._plan.summary()}"
            )
        except Exception as e:
            return f"ERROR: {e}"

    def _forbidden_report(self, plan: dict) -> str:
        """Confirm back to the agent the prohibitions it just committed to.

        Echoing them at creation is the point: a list the agent wrote and never
        saw again is a list it will not act on. The plan context re-renders them
        every turn; this one-time echo closes the loop at the moment of writing,
        and an EMPTY list is reported as a gap to fill rather than passing
        silently -- the failure mode this mechanism targets (polishing already
        existing work while the clock runs out) is exactly what an agent that
        wrote no prohibitions leaves itself open to.
        """
        items = (plan or {}).get("forbidden") or []
        if not items:
            return (
                "⚠ 不做清单 (forbidden) 为空：你只写了要做什么，没有写不该做什么。"
                "时间紧张时这份计划没有任何防线。用 "
                "plan_update(action='add_forbidden') 补 2-5 条具体的禁做动作"
                "（每条：动作 + 为什么 + 改做什么）。\n\n"
            )
        lines = [f"不做清单 (forbidden) — 已写下 {len(items)} 条，每轮都会回显在计划顶部："]
        for i, e in enumerate(items, 1):
            seg = f"  {i}. ✗ {e.get('action', '')}"
            if e.get("instead"):
                seg += f" → 改为：{e['instead']}"
            lines.append(seg)
        return "\n".join(lines) + "\n\n"

    def _time_budget_report(self, steps: list) -> str:
        """Frame-time accounting for the plan the agent just committed to.

        The plan is only worth the time it can actually fit, so this reports the
        arithmetic AT CREATION — the one moment when a plan that cannot finish is
        still cheap to replace. It deliberately does NOT reject the plan: a wall
        the agent beats by re-scoping is the agent's call, but it must SEE the
        overrun now rather than discover it at the deadline. With no injected
        budget there is nothing to compare against, so it stays silent rather than
        inventing a deadline.
        """
        budget = self._plan.budget_sec() if hasattr(self._plan, "budget_sec") else None
        if not budget:
            return ""
        estimates = [
            s.get("est_sec") for s in steps
            if isinstance(s, dict) and s.get("est_sec")
        ]
        # Harness-appended fixed steps carry no estimate; they are small but not
        # free, so they are reported as an unnamed slack allowance instead of
        # being silently counted as zero.
        from flagscale_agent.react.plan import (
            DEFAULT_RESERVE_FRACTION, _fmt_sec, is_fixed_step,
        )
        missing = sum(
            1 for s in steps
            if isinstance(s, dict) and not s.get("est_sec")
            and not is_fixed_step(s)
        )
        reserve = round(budget * DEFAULT_RESERVE_FRACTION, 1)
        lines = ["── TIME BUDGET (frame-time accounting) ──"]
        lines.append(
            f"wall {_fmt_sec(budget)} | reserve for wrap-up/verification "
            f"{_fmt_sec(reserve)} | spendable on steps {_fmt_sec(budget - reserve)}"
        )
        if not estimates:
            lines.append(
                "⚠ NONE of your steps carries an est_sec. You have committed to a "
                "sequence of actions without saying what any of them costs against "
                "the wall — that is not a plan, it is a wish list. Give every step "
                "an est_sec (plan_update can amend them) BEFORE you start working: "
                "you cannot cut scope you never priced."
            )
            return "\n".join(lines) + "\n\n"
        total = round(sum(estimates), 1)
        # Same trap as the ledger: summing estimates end to end is only the
        # answer for a pure chain. Two independent steps can run at the same
        # time, so the wall is bounded by the LONGEST chain. Measuring both
        # stops this message from prescribing a scope cut for a plan that only
        # needs a fan-out.
        from flagscale_agent.react.plan import (
            _critical_path_secs, _id_list, _with_default_chain,
        )
        # Compute the critical path over the SAME dependency shape `create()`
        # will persist. `create()` fills in the default serial chain for every
        # step that does not declare `depends_on`, so measuring the raw dicts
        # here would read "no depends_on" as "no dependencies" and report a
        # fan-out saving that the persisted plan does not have — the report and
        # the ledger below would then contradict each other in ONE response.
        cp = _critical_path_secs(
            _with_default_chain([s for s in steps if isinstance(s, dict)])
        )
        path = cp["critical"] if cp["gain"] > 0.5 else total
        lines.append(
            f"your steps estimate to {_fmt_sec(total)} serially "
            f"({len(estimates)} estimated, {missing} unestimated)"
            + (f" + reserve {_fmt_sec(reserve)}" if reserve else "")
        )
        if cp["gain"] > 0.5:
            lines.append(
                f"critical path {_fmt_sec(cp['critical'])} — running "
                f"{_id_list(cp['parallel'])} concurrently saves {_fmt_sec(cp['gain'])}"
            )
        planned = path + reserve
        if planned > budget:
            over = round(planned - budget, 1)
            lines.append(
                f"⚠ THIS PLAN DOES NOT FIT: "
                f"{'critical path' if cp['gain'] > 0.5 else 'estimate'}"
                f"+reserve {_fmt_sec(planned)} exceeds the wall "
                f"{_fmt_sec(budget)} by {_fmt_sec(over)}."
            )
            if cp["gain"] > 0.5:
                lines.append(
                    "DECIDE NOW, not at the deadline. (1) FAN OUT FIRST: the "
                    f"steps {_id_list(cp['parallel'])} do not depend on each "
                    "other — give one branch to a worker (spawn_worker / "
                    "dispatch_many) while you take the longest chain. Fan-out is "
                    "not an optimization here, it is the only way the required "
                    "set fits. (2) Only if the plan STILL does not fit after "
                    "counting the parallel saving, shrink scope: keep the step "
                    f"that bounds the clock (step {cp['constraint']}) at its "
                    "cheapest sufficient version, and drop or defer the single "
                    "most expensive step only if a cheaper path still reaches a "
                    "SCORABLE artifact. Never delete the step that VERIFIES the "
                    "deliverable — an unverified artifact scores zero, so the "
                    "checking step is the last thing to cut, not the first."
                )
            else:
                lines.append(
                    "DECIDE NOW, not at the deadline — you still have the whole "
                    "wall in front of you. The dependency chain is real (no "
                    "independent branches to run concurrently), so parallelism "
                    "cannot rescue it; the fix is scope. Re-scope in this order: "
                    f"(1) the clock is bounded by step {cp['constraint']} — take "
                    "its cheapest sufficient version first; (2) drop or defer the "
                    "single most expensive step only if a cheaper path still "
                    "reaches a SCORABLE artifact; (3) replace an open-ended "
                    "search with a bounded discriminating experiment whose "
                    "outcomes eliminate candidates. Never delete the step that "
                    "VERIFIES the deliverable — an unverified artifact scores "
                    "zero, so the checking step is the last thing to cut. What "
                    "you must NOT do is keep the plan and hope: a plan whose "
                    "estimate already exceeds the wall fails by arithmetic."
                )
        else:
            lines.append(
                f"fits with {_fmt_sec(round(budget - planned, 1))} slack "
                "— re-check this ledger after each step and re-scope the moment "
                "actuals run over their estimates"
            )
        return "\n".join(lines) + "\n\n"
