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
    "Divergence check — do BOTH halves, like the Review step. (1) YOUR OWN "
    "deeper divergence: put 2-3 genuinely different framings of the task's "
    "ORIGINAL statement on the table — a BETTER method (faster/cheaper/simpler), a "
    "MORE CORRECT method (avoids a risk this plan carries), or an UNCONSIDERED "
    "decomposition/reading. (2) A read-only diverger worker — dispatch it (per the "
    "trade-off in the next item) when the framing warrants a fresh set of eyes. "
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
]


class PlanCreateTool(Tool):
    name = "plan_create"
    description = (
        "Create a task plan with ordered steps for multi-step work. "
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

        try:
            plan = self._plan.create(title, normalized, self._session_id, thinking=thinking)
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
                f"{self._plan.summary()}"
            )
        except Exception as e:
            return f"ERROR: {e}"
