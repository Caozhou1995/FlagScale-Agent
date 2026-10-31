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
from flagscale_agent.react.instruments import check_calibration
from flagscale_agent.react.instruments import check_instrument_input
from flagscale_agent.react.instruments import check_requirements_read
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
    "OBSERVATION OVER ARGUMENT. Every conclusion points to an output you actually ran and "
    "read (command + result). 'The method should be right' is not evidence. Anything with "
    "no runnable check is stated as NOT VERIFIED.",
    "REQUIREMENTS RECONCILIATION. List each requirement of the task verbatim and give a "
    "command + real output for each, or NOT VERIFIED; confirm the deliverable satisfies "
    "every one at its final path including form constraints (format, naming, version, "
    "timing, counts).",
    "GENERALIZATION. Verify on at least one perturbed or held-out input, not only the "
    "development sample; if there is no generalizable point, say why.",
    "DELIVERY HYGIENE. The delivery path holds only what the task requires (no .bak / "
    "scratch / build byproducts); the real artifact is non-empty and well-formed, not a "
    "placeholder or a truncated file.",
    "UNSETTLED PREMISES. For any load-bearing premise you described as unverifiable, "
    "either give a probe whose design intent was to falsify it, or state the world-state "
    "in which it is false and why that is a safe bet.",
    "PRE-MORTEM. Assume you are wrong: name the single point most likely to fool you and "
    "say how you ruled it out.",
    "REVIEWER, a deliberate trade-off, not a mandate. Know its benefit — a fresh context "
    "surfaces the defect "
    "your own tests cannot construct — and price it against the time budget of a worker's "
    "whole deadline. "
    "Dispatch one when the change can fail SILENTLY (a "
    "parser, permission, config, serialization or cross-call-site boundary, or a "
    "multi-file behaviour change); skip it for a trivial single-file edit and self-verify. "
    "STATE the decision and WHY, tied to that predicate.",
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
    "IMPROVE THE HARNESS, not this deliverable. Its highest-value output is a MECHANISM "
    "change that makes a CLASS of future tasks go better: a proposal(), or a "
    "prompt/guard/tool/knowledge edit. Ask: can this run's gap become a durable change? If "
    "yes, LAND it and name the concrete target (proposal id, file, or the exact edit).",
    "Report the three RSI channels explicitly in the wrap-up, each under its own sub-heading "
    "even when empty (then say why): (a) landed mechanism changes (file:line / proposal id "
    "/ memory key); (b) proposals — newly raised this session AND historical open ones "
    "still awaiting "
    "a decision, by id (proposal(action='list') if needed); never silently drop one; "
    "(c) memory updates landed (key + one line).",
    "Task knowledge is the FALLBACK: only when there is genuinely no mechanism-level gap, "
    "land a memory_write as fuel for future tasks, and say WHY it does not rise to a "
    "harness change. If nothing is reusable at either level, assert that and say why — a "
    "legal close, but it must be asserted.",
    "Point to a concrete artifact, not a vague 'be more careful': a lesson that never "
    "lands breaks the recursion, and an unpersisted improvement is not an improvement.",
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
    "Negative constraints (forbidden list): write 2-5 entries {action, why, instead} "
    "in the SAME plan_create call. The harness re-renders them at the top of your plan "
    "context every turn, so a prohibition decided NOW fires at the decision point — one "
    "you would only think of at the deadline arrives after the mistake. Name ACTIONS "
    "specific to this task ('do not re-run the same command with a bigger timeout — read "
    "the partial output and change the approach'), not virtues ('be careful'). Append "
    "more later with plan_update(action='add_forbidden').",
    ORACLE_ACCEPTANCE_ITEM,
    STATED_CONSTRAINT_ITEM,
    AMBIGUITY_ANCHOR_ITEM,
    "FRAMING. State in one line: (a) the problem CLASS and its standard method — naming "
    "the class hands you the method AND its failure modes at once, and reaching for a "
    "sweep or brute-force enumeration is the tell that the class was never identified; "
    "(b) what observation would prove your answer wrong; (c) which rival explanations you "
    "must exclude; (d) how expensive being wrong is here, which sets verification "
    "strength. Then put 2-3 genuinely DIFFERENT framings of the task on the table — a "
    "better method, a more correct method, or an unconsidered decomposition — and rule on "
    "each (adopt / reject with the reason). You framed this plan, so its premises are "
    "invisible to you; that is why the ruling is written down rather than thought.",
    "DIVERGER (a deliberate trade-off, not a mandate). Know its benefit — it probes the "
    "FRAMING before any artifact exists, the cheapest moment to fix a wrong premise — and "
    "price it against the time budget of a worker's whole deadline. Dispatch spawn_worker with "
    "role=\"diverger\" and a READ-ONLY contract when the framing is genuinely open (a novel "
    "or ambiguous task, several plausible decompositions, an irreversible direction): a "
    "fresh context does not inherit your premises, which is the whole point. The diverger "
    "tag is a LEAF role: no reviewer preset, and it cannot itself spawn. Skip it when "
    "the framing is settled and cheap to state (a small well-specified edit). STATE the "
    "decision and WHY in plan notes. Its findings are CLAIMS — reproduce or refute each. "
    "A reviewer needs a FROZEN artifact and belongs at the Review step, never here.",
    "FIRST CUT AND CONTEXT. (a) Order steps so a minimal but complete, scorable deliverable "
    "lands at the required path BEFORE scaling up, then refine in place and "
    "write-through the same artifact (never rewrite from scratch) — whatever is on "
    "disk when the clock stops is what scores, so keep a validated version at the path at "
    "all times. (b) Validate the standard method on the smallest meaningful sample before "
    "scaling up to the full task; a generic step like 'analyze/implement/test' means the "
    "problem class was never identified. (c) If two candidate paths exist and you "
    "cannot tell which is right, make the first plan a PORTFOLIO: one discriminating "
    "micro-experiment per candidate. (d) PREMORTEM the workspace: if reference or golden "
    "artifacts are present, diff your convention against them before inventing one; if a "
    "test/verify harness or the repo's own suite is present, USE it as the grader, run it "
    "early and after each change — a hand-rolled check that shadows it doubles the cost "
    "and can still miss the contract. Document the deliverable's usage (a README or "
    "docstring) when a consumer will run it without you. (e) Qualifier extraction: name "
    "every qualifier in the "
    "task (a point in "
    "time, a version, a subset, a metric definition, a named tool/model instance), "
    "folding each into "
    "a step or acceptance criterion rather than leaving it as a head-only act; read each "
    "boundary in the direction the task MEANS, not the convenient one, "
    "because a dropped qualifier silently answers a nearby question the task never asked.",
    "Acceptance-facet manifest. BEFORE you build, write analysis/facets.md (state the path "
    "in the plan) with ONE ROW PER VERIFIABLE CLAUSE of the task text — the row count is "
    "fixed by the clause count, not by how many faces you happen to think of, so a short "
    "manifest is a defect, not a judgment call. Segment on the task text's own sentences "
    "and bullets; a numeric bound, a form rule, a named artifact and a named command are "
    "each their own clause, and when in doubt split rather than merge, because a merged "
    "row hides the face it absorbed — never fold several clauses into one row. Each row "
    "carries (a) the clause quoted verbatim, (b) the exact command/input that exercises "
    "that face, (c) the observed result when you ran it. A row with no executed probe is a "
    "sentence, not a check; a row claiming the result you expect is not a check either. "
    "Close-out reviews this manifest row by row, not your memory of it. A face you silence "
    "(a flag that quiets output, a stream you stop printing) is a face the consumer still "
    "reads and you no longer check.",
    "Reading ledger. When the output FORM is genuinely open (two readings that would "
    "produce different bytes), do not pick one and build around it. List the readings "
    "first, each pinned to the task text's own words, plus the artifact each would "
    "produce; then run a dominance test against the consumer's own material — does one "
    "reading make the other impossible, or does its artifact satisfy the other's "
    "acceptance test too? First determine which direction the comparison penalizes: cover "
    "both readings only when extra output is ignored; when extras are penalized, cover "
    "with a switch defaulted to the dominant reading instead of shipping both. When no "
    "reading dominates, cover the top-2 (or top-k) in the same run — an undominated "
    "ambiguity is a coin flip whose wrong side costs a whole run. This is the hedge/bet "
    "rule applied at BUILD time, not a new doctrine.",
    "HARNESS DECLARATION SHAPES (convention / deliverable_reading / assumptions): the "
    "guard message states the exact accepted shape when it fires, so do not pre-engineer "
    "them here. What matters at planning time is only this: a step that rests on a derived "
    "convention must carry a probe whose output could have refuted it; a step that emits a "
    "graded artifact must state its path; a load-bearing premise you could not test is a "
    "labelled gap, not a silent one. A task's EXAMPLE value locks FORM (shape, units, "
    "stride), never a specific value or base.",
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
                    "REQUIRED. The oracle that will decide this task, with a "
                    "premise EXTERNAL to your own implementation. Object with "
                    "keys: `premise_source` (one of 'differential' | 'provided' "
                    "| 'consumer'), `oracle` (what decides it), and "
                    "`first_exercise` (the exact command/check that runs it). "
                    "When supplied, its first exercise is bound into the "
                    "acceptance of the FIRST load-bearing step you list, so the "
                    "acceptance->verification channel forces the oracle's "
                    "evidence to exist early. A self-authored premise_source is "
                    "refused. If the task text names NO deciding tool at all, "
                    "declare that explicitly: premise_source 'consumer', oracle "
                    "text 'none: <why no external decider exists>' -- an "
                    "auditable declaration, not an omission. Omitting this field "
                    "REFUSES plan creation."
                ),
                "properties": {
                    "premise_source": {"type": "string"},
                    "oracle": {"type": "string"},
                    "first_exercise": {"type": "string"},
                    "named_arbiter": {
                        "type": "string",
                        "description": (
                            "The tool/library the TASK TEXT itself names as the "
                            "decider of a disputed quantity — quote it from the "
                            "task text (the sentence saying whose output is the "
                            "ground truth). This is not the hidden checker: it is "
                            "a name the task gives you to read. Write \"none\" only "
                            "after checking that the task text names no deciding "
                            "tool: an OMITTED key reads as 'never looked' and is "
                            "refused at plan close, while \"none\" is an explicit, "
                            "auditable declaration."
                        ),
                    },
                    "deliverable_path": {
                        "type": "string",
                        "description": (
                            "The artifact path you actually deliver. When set, the "
                            "named arbiter must be run against THIS path before the "
                            "plan may close — pointing it at a hand-typed sample or "
                            "a re-derived substring measures the tool's arithmetic, "
                            "not your shipped bytes."
                        ),
                    },
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
            "calibration": {
                "type": "object",
                "description": (
                    "REQUIRED. Instrument 1 of 4: prove your check works BEFORE "
                    "you trust it. Run the instrument you will rely on ONCE on a "
                    "case whose answer you ALREADY know, and record what it said. "
                    "You authored the instrument, so it shares your blind spot — an "
                    "uncalibrated check reporting PASS only tells you it agrees "
                    "with itself. Keys: `instrument` (the exact check/command), "
                    "`known_case` (a case whose answer is already known and "
                    "external to the instrument — a value read from the task's own "
                    "data, a prior observed result, a supplied reference), "
                    "`observed` (what the instrument actually reported on it), "
                    '`matches` = "yes" | "no" | "no_known_case". If no such case '
                    'exists, say "no_known_case" — an honest answer, but it '
                    "obliges you to show a NEGATIVE CONTROL later (the instrument "
                    'must be seen to FIRE on a bad input). If it says "no", your '
                    "instrument is broken and is refused: every PASS it produces "
                    "afterwards is noise."
                ),
                "properties": {
                    "instrument": {"type": "string"},
                    "known_case": {"type": "string"},
                    "observed": {"type": "string"},
                    "matches": {
                        "type": "string",
                        "enum": ["yes", "no", "no_known_case"],
                    },
                },
                "required": ["instrument", "matches"],
            },
            "requirements_read": {
                "type": "array",
                "description": (
                    "REQUIRED when the task text defers its requirements to an "
                    "external standard ('make sure X satisfies <standard>'s "
                    "requirements', 'per the <vendor> protocol', 'as specified by "
                    "<spec>'). Those clauses are NOT in the task text, so a "
                    "checklist built only from the task text looks complete while "
                    "being incomplete. Write down what the standard actually said, "
                    "one entry per clause: {clause, source, covered_by}. "
                    "`covered_by` = how the plan satisfies it, or "
                    "'dropped: <reason>' when you deliberately do not. Dropping is "
                    "allowed; dropping SILENTLY is what this field stops."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "clause": {"type": "string"},
                        "source": {"type": "string"},
                        "covered_by": {"type": "string"},
                    },
                    "required": ["clause"],
                },
            },
        },
        "required": ["title", "steps", "calibration"],
    }

    def __init__(self, task_plan, session_id: str = "", task_text_fn=None):
        self._plan = task_plan
        self._session_id = session_id
        # The ORIGINAL task text, used to decide whether `requirements_read`
        # applies (the task defers to a named external standard). Injected as a
        # callable so the tool reads the live task text rather than a snapshot.
        self._task_text_fn = task_text_fn

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
        # DECLARATION REQUIRED AT CREATION (measured loss, 2026-10-27):
        # the schema exposes `oracle`, the FRAME acceptance demands it, and the
        # model still omitted the kwarg -- the plan was created with oracle: {},
        # so the binding/arbiter machinery never armed and the downstream gates
        # (which fire only on a BOUND oracle) were all inert. A declaration the
        # agent can silently skip is not a declaration: refuse creation until the
        # agent commits one, the same way `steps` is required. This is a SHAPE
        # gate (same honest scope as declares_external_premise) -- it forces the
        # naming of a decider up front; it cannot prove the premise is external.
        if oracle is None or not oracle:
            return (
                "ERROR: plan_create requires an oracle declaration -- the plan's "
                "FRAME acceptance is built around one, and a plan created without "
                "it arms none of the verification machinery. Provide an oracle "
                "dict: 'premise_source' (one of: "
                + " | ".join(PREMISE_SOURCES)
                + "), non-empty 'oracle' (what decides the task's key quantity) "
                "and 'first_exercise' (the exact command that runs it). If the "
                "task text names NO deciding tool at all, say so explicitly in "
                "the 'oracle' text (premise_source 'consumer', oracle text "
                "'none: <why no external decider exists>') -- that is an "
                "auditable declaration, not an omission. This is a declaration, "
                "not proof: name the artifact so a reader can check it."
            )
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
        calibration = kwargs.get("calibration")
        # F: LLM tool arguments can arrive JSON-encoded instead of parsed
        # (same transport drift `steps` and `oracle` already tolerate above).
        # Without this the agent can submit a perfectly standard object and be
        # refused forever with "malformed" while its retries stay identical --
        # observed 6 consecutive refusals with zero new information.
        if isinstance(calibration, str):
            _raw = calibration.strip()
            if _raw.startswith("{"):
                try:
                    calibration = json.loads(_raw)
                except (json.JSONDecodeError, ValueError):
                    pass

        # INSTRUMENT 1 GATE: refuse to create the plan without a calibration
        # declaration. This is the load-bearing half of the mechanism -- the
        # schema only *asks*, this refuses, and only a refusal changes behaviour
        # (measured: the advisory forms of every earlier mechanism were ignored).
        _cal_refusal = check_calibration(calibration)
        if _cal_refusal is not None:
            return f"ERROR: {_cal_refusal[1]}"

        # INSTRUMENT 1b GATE: the calibration binds the INSTRUMENT; this binds
        # the INPUT it is handed. Measured losses in this class were all inputs,
        # not instruments: a correctly calibrated tool fed a span the agent
        # picked (skipping the prefix the consumer's own construction includes)
        # returned a value inside the bound while the consumer's construction of
        # the same quantity crossed it. Both readings were defensible; only one
        # decided the reward. Refusing here -- not at plan close -- because this
        # is the moment the construction is being chosen, and a declaration
        # demanded 20 steps later cannot change that choice.
        _inp_refusal = check_instrument_input(calibration)
        if _inp_refusal is not None:
            return f"ERROR: {_inp_refusal[1]}"

        # NAMED-STANDARD GATE: when the task text defers its requirements to an
        # external standard, the clauses that will decide the reward are NOT in
        # the task text. Measured loss: an agent read a mandatory clause in the
        # standard, judged the checker "less likely to verify it", dropped it --
        # and failed on exactly that clause. No later gate can
        # recover it, because every coverage sweep only reads DECLARED
        # constraints. So the clause list is demanded here, at the moment the
        # plan is frozen; dropping a clause stays legal, but must be written.
        # Accepted inside calibration as well as top-level (instrument_input
        # precedent: the calibration object is the declaration home).
        requirements_read = kwargs.get("requirements_read")
        if requirements_read is None and isinstance(calibration, dict):
            requirements_read = calibration.get("requirements_read")
        if isinstance(requirements_read, str):
            _raw = requirements_read.strip()
            if _raw.startswith("["):
                try:
                    requirements_read = json.loads(_raw)
                except (json.JSONDecodeError, ValueError):
                    pass
        _req_refusal = check_requirements_read(
            self._task_text_fn() if self._task_text_fn else "", requirements_read)
        if _req_refusal is not None:
            return f"ERROR: {_req_refusal[1]}"

        try:
            plan = self._plan.create(title, normalized, self._session_id, thinking=thinking,
                                     oracle=oracle, forbidden=forbidden,
                                     calibration=calibration,
                                     requirements_read=requirements_read)
            forbidden_report = self._forbidden_report(plan)
            oracle_report = self._oracle_report(oracle)
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
                + oracle_report
                + forbidden_report
                + self._time_budget_report(normalized)
                + f"{self._plan.summary()}"
            )
        except Exception as e:
            return f"ERROR: {e}"

    def _oracle_report(self, oracle: dict) -> str:
        """State the oracle declaration back to the agent at creation time.

        Part C of the measured fix (2026-10-27): the agent wrote a
        self-congratulatory 'oracle: ...' line in its thinking text while never
        passing the kwarg, and nothing echoed the declaration state -- so the
        gap was invisible until every downstream gate turned out inert. The
        refusal (part A) closes creation-time silence; this echo closes the
        creation-time loop on the success path, the same one-time echo the
        forbidden list gets. Named tools referenced in the declaration are
        quoted verbatim from the AGENT'S OWN dict -- nothing here invents or
        suggests a tool name (no grader leak).
        """
        decider = str(oracle.get("oracle", "")).strip()
        source = str(oracle.get("premise_source", "")).strip()
        exercise = str(oracle.get("first_exercise", "")).strip()
        arbiter = oracle.get("named_arbiter")
        arbiter_note = ""
        if isinstance(arbiter, str) and arbiter.strip():
            arbiter_note = f"  named arbiter (the tool the task text names): {arbiter.strip()}\n"
        return (
            f"oracle declaration (you promised this; it MUST really be run before completion):\n"
            f"  premise_source: {source}\n"
            f"  oracle (decider): {decider}\n"
            f"  first_exercise: {exercise}\n"
            f"{arbiter_note}"
            "  WARNING - key discipline: an external decider the task text names must be run"
            " against THE DELIVERABLE ITSELF, not against a sample you constructed. When your"
            " reading and its reading disagree, ITS reading wins; do not explain the gap away"
            " as 'my script has a bug' and switch back to your own quantity.\n\n"
        )

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
        from flagscale_agent.react.plan import _MAX_FORBIDDEN
        items = (plan or {}).get("forbidden") or []
        meta = (plan or {}).get("forbidden_meta") or {}
        dropped = int(meta.get("dropped") or 0)
        requested = int(meta.get("requested") or 0)
        if not items:
            if dropped:
                return (
                    f"WARNING: the forbidden list is EMPTY - you submitted {requested} item(s), "
                    f"but {dropped} were dropped for missing an action string (or the whole "
                    "value was malformed JSON). Send them again as dicts "
                    "{'action': ...}.\n\n"
                )
            return (
                "WARNING: the forbidden list is EMPTY - you only wrote what to DO, not what "
                "NOT to do. Under time pressure this plan has no line of defence. "
                "Use plan_update(action='add_forbidden') to add 2-5 concrete prohibited "
                "actions (each: action + why + what to do instead).\n\n"
            )
        lines = [f"forbidden list - {len(items)} item(s) recorded; re-rendered at the top of the plan every turn:"]
        if dropped:
            cap_note = (
                f"  Note: another {dropped}/{requested} item(s) were dropped (missing action "
                "text or duplicates). Duplicates need no action; items missing an action "
                "should be re-sent as dicts {'action': ...}."
            )
            if requested > _MAX_FORBIDDEN and len(items) == _MAX_FORBIDDEN:
                # A third disclosed state (reviewer N3): valid items dropped by
                # the CAP are not malformed and not duplicates — telling the
                # agent to "fix and resend" them just re-drops them.
                cap_note = (
                    f"  Note: another {dropped}/{requested} item(s) were dropped: the list is "
                    f"at its cap of {_MAX_FORBIDDEN}, so only the highest-priority items were "
                    "kept. Do NOT re-send the dropped ones - they will be dropped again. To "
                    "replace one, re-send the SAME action via "
                    "plan_update(action='add_forbidden') to refresh its why/instead."
                )
            lines.append(cap_note)
        for i, e in enumerate(items, 1):
            seg = f"  {i}. X {e.get('action', '')}"
            if e.get("instead"):
                seg += f" -> instead: {e['instead']}"
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
