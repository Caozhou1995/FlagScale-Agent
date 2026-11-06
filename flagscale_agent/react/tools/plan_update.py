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

"""Plan update tool — modify task plan steps and status."""

from __future__ import annotations

import re

from flagscale_agent.react.tools.base import Tool

_THINKING_ECHO = (
    "Model recorded — it is now your working theory. What does it PREDICT for "
    "your next move, and what is the cheapest observation that could kill it? "
    "Run that."
)


# Pattern to extract integer from strings like "step_1", "step 2", "Step_3", "#4"
_STEP_ID_RE = re.compile(r'(?:step[_\s]?)?#?(\d+)', re.IGNORECASE)


def _parse_step_id(raw) -> int | None:
    """Parse step_id from various LLM formats: 1, "1", "step_1", "step 2", etc."""
    # A bool IS an int in Python; reject it explicitly so the handler and the
    # guard's `_normalize_step_id` agree. Otherwise `step_id=True` would mark
    # step 1 done while the guard normalizes it to None (acceptance-less), a
    # silent bypass of the Mode-1 acceptance-evidence gate.
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        raw = raw.strip()
        # Try direct integer parse first
        try:
            return int(raw)
        except ValueError:
            pass
        # Try regex extraction
        m = _STEP_ID_RE.search(raw)
        if m:
            return int(m.group(1))
    return None


class PlanUpdateTool(Tool):
    name = "plan_update"
    description = (
        "Update the active task plan: mark steps done/skipped, add new steps, "
        "replan, or complete/abandon the plan. Use to track progress as you work."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["step_done", "step_doing", "step_skip", "add_steps", "update_acceptance", "add_forbidden", "predict", "settle", "complete", "abandon", "deactivate", "reactivate", "set_thinking"],
                "description": "What to do: step_done/step_doing/step_skip (update a step), add_steps (insert new steps), update_acceptance (modify step acceptance criteria), complete/abandon (finish the plan), deactivate (pause current plan), reactivate (resume a paused plan by id), set_thinking (rewrite the plan-level problem model — pass the new model in the 'thinking' field).",
            },
            "step_id": {
                "type": "integer",
                "description": "Step number to update (for step_done/step_doing/step_skip/update_acceptance).",
            },
            "notes": {
                "type": "string",
                "description": "Append a note to this step (scratchpad). Use freely to record: attempts, failures, key decisions, things to remember. Each call appends a new line — previous notes are preserved.",
            },
            "verification": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Verification evidence for step_done: what was checked, test results, output samples.",
            },
            "prediction_statement": {
                "type": "string",
                "description": (
                    "For action='predict'. Instrument 3 of 4 (pre-registration): "
                    "write the outcome you EXPECT before running, so the result "
                    "cannot be re-explained after the fact. A prediction recorded "
                    "in advance can be wrong — which is exactly what makes it "
                    "informative. Settle it afterwards with action='settle' "
                    "(outcome hit/miss/not_run); plan_update(action='complete') "
                    "REFUSES while any prediction is unsettled."
                ),
            },
            "prediction_expected": {
                "type": "string",
                "description": "For action='predict': the concrete expected value ('~3350 passed', 'reward=1').",
            },
            "prediction_kind": {
                "type": "string",
                "description": "For action='predict': optional tag grouping related predictions.",
            },
            "prediction_id": {
                "type": "string",
                "description": "For action='settle': the pred_xxx id returned when the prediction was recorded.",
            },
            "outcome": {
                "type": "string",
                "enum": ["hit", "miss", "not_run"],
                "description": (
                    "For action='settle': hit = the result matched your prediction; "
                    "miss = it did not (state the observed value); not_run = the run "
                    "never happened (cut short, abandoned) — an honest outcome, never "
                    "reuse 'hit' for it."
                ),
            },
            "prediction_observed": {
                "type": "string",
                "description": "For action='settle': what was ACTUALLY observed, so hit/miss is auditable.",
            },
            "acceptance": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New acceptance criteria (for update_acceptance action).",
            },            "new_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New step descriptions (for add_steps).",
            },
            "after_step_id": {
                "type": "integer",
                "description": "Insert new steps after this step (for add_steps). Omit to append at end.",
            },
            "plan_id": {
                "type": "string",
                "description": "Plan ID to reactivate (for reactivate action).",
            },
            "forbidden": {
                "type": "array",
                "description": (
                    "For action='add_forbidden': the NEGATIVE constraints to append "
                    "to your plan — the actions you must NOT take. `steps` says what "
                    "to do; this says what not to do, and it is re-rendered at the "
                    "TOP of your plan context every turn, beside the time ledger. "
                    "Append here the moment you discover a trap DURING the work — a "
                    "path you already burned time on, a temptation the clock will "
                    "make stronger. Each item is a string or "
                    "{action, why, instead}: `action` = the concrete thing you must "
                    "not do; `why` = what happened in THIS task that proved it; "
                    "`instead` = what to do when tempted. Specific and falsifiable "
                    "only ('do not re-run the same build with a bigger timeout — "
                    "read the partial output and change the approach'), never a "
                    "platitude. Duplicates are dropped automatically."
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
            "thinking": {
                "type": "string",
                "description": (
                    "Rewrite the plan-level problem MODEL — your current theory of the "
                    "task: the bottleneck, the load-bearing hypothesis, the evidence, and "
                    "a FALSIFIABLE prediction for the next move ('if I change X, metric "
                    "should reach ~Y'). Overwrites the previous model (it is the CURRENT "
                    "understanding, not a log). Can be passed with action='set_thinking' "
                    "on its own, or alongside any other action (e.g. step_done + an updated "
                    "model). This is the HOME for deep/deliberate reasoning — when you catch "
                    "yourself about to try yet another quick tweak, write the model here "
                    "first: what is actually limiting the result, and what does your next "
                    "move PREDICT? A shallow retry that cannot state a new prediction is the "
                    "signal to stop and rebuild the model, not act again."
                ),
            },
            "_override_reason": {
                "type": "string",
                "description": "Override a guard block with a justification (min 5 chars). Some guard checks (step_done premise re-check, task completion re-check) block until you re-issue the same call with this field explaining why proceeding is justified. The reason is recorded, not content-checked.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, task_plan):
        self._plan = task_plan
        # Step id this call touched, set by the branch that touched it, so the
        # delta receipt can name what changed instead of reprinting the plan.
        self._last_touched = None

    def execute(self, **kwargs) -> str:
        action = kwargs["action"]
        thinking = kwargs.get("thinking")
        try:
            if action == "set_thinking":
                if not thinking or not str(thinking).strip():
                    return "ERROR: thinking text required for set_thinking."
                self._plan.set_thinking(str(thinking))
                # set_thinking exists to put the model IN VIEW — a delta that
                # omits it answers a different question.
                return self._plan.summary() + "\n\n" + _THINKING_ECHO
            if action == "step_done":
                step_id = _parse_step_id(kwargs.get("step_id"))
                if not step_id:
                    return "ERROR: step_id required for step_done (integer or 'step_N' format)."
                verification = kwargs.get("verification", [])
                self._last_touched = step_id
                self._plan.update_step(step_id, "done", kwargs.get("notes", ""), verification=verification)
            elif action == "step_doing":
                step_id = _parse_step_id(kwargs.get("step_id"))
                if not step_id:
                    return "ERROR: step_id required for step_doing (integer or 'step_N' format)."
                self._last_touched = step_id
                self._plan.update_step(step_id, "doing", kwargs.get("notes", ""))
            elif action == "step_skip":
                step_id = _parse_step_id(kwargs.get("step_id"))
                if not step_id:
                    return "ERROR: step_id required for step_skip (integer or 'step_N' format)."
                self._last_touched = step_id
                self._plan.skip_step(step_id, kwargs.get("notes", ""))
            elif action == "add_steps":
                new_steps = kwargs.get("new_steps", [])
                if not new_steps:
                    return "ERROR: new_steps required for add_steps."
                after = _parse_step_id(kwargs.get("after_step_id"))
                self._plan.add_steps(new_steps, after)
                self._last_touched = after
            elif action == "update_acceptance":
                step_id = _parse_step_id(kwargs.get("step_id"))
                if not step_id:
                    return "ERROR: step_id required for update_acceptance."
                acceptance = kwargs.get("acceptance", [])
                if not acceptance:
                    return "ERROR: acceptance required for update_acceptance."
                self._last_touched = step_id
                self._plan.update_acceptance(step_id, acceptance)
                # The caller just WROTE these criteria — echoing the stored list
                # is the confirmation it needs, and it is the delta, not the plan.
                out = [
                    f"Acceptance updated for step {step_id} "
                    f"({len(acceptance)} item(s)):"
                ]
                out.extend(f"  • {a}" for a in acceptance)
                return "\n".join(out)
            elif action == "predict":
                # INSTRUMENT 3 (pre-registration): record the expected outcome
                # BEFORE the run. Rejected without a statement because an empty
                # prediction cannot be wrong, and a prediction that cannot be
                # wrong carries none of its cost.
                statement = (kwargs.get("prediction_statement") or "").strip()
                if not statement:
                    return (
                        "ERROR: prediction_statement required for predict — write "
                        "the number/outcome you EXPECT before you run it."
                    )
                self._plan.predict(
                    statement=statement,
                    expected=(kwargs.get("prediction_expected") or ""),
                    kind=(kwargs.get("prediction_kind") or ""),
                )
                pending = self._plan.unsettled_predictions()
                lines = [
                    f"Pre-registered. {len(pending)} unsettled prediction(s) on record:"
                ]
                for pp in pending:
                    lines.append(f"  - {pp['id']}: {pp['statement']}")
                lines.append(
                    "Go run it now. Settle it afterwards (action='settle', outcome is one of "
                    "hit / miss / not_run) — plan complete refuses while a "
                    "pre-registration is unsettled."
                )
                return "\n".join(lines)
            elif action == "settle":
                pid = kwargs.get("prediction_id")
                if not pid:
                    return "ERROR: prediction_id required for settle."
                outcome = kwargs.get("outcome") or kwargs.get("prediction_outcome")
                observed = kwargs.get("prediction_observed") or ""
                try:
                    self._plan.settle(pid, outcome, observed)
                except ValueError as e:
                    return f"ERROR: {e}"
                left = self._plan.unsettled_predictions()
                return (
                    f"Settled {pid} as {str(outcome).strip().lower()}."
                    + (f" {len(left)} still unsettled." if left else " No unsettled predictions left.")
                )
            elif action == "add_forbidden":
                items = kwargs.get("forbidden", [])
                if not items:
                    return "ERROR: forbidden required for add_forbidden."
                res = self._plan.add_forbidden(items)
                added = res.get("added", 0)
                total = res.get("total", 0)
                # Cap-full and duplicate are DIFFERENT situations the agent
                # must not confuse: at the cap the fix is to accept that the
                # list is full (a 9th prohibition is not more protective, it
                # is noise), while a duplicate means the item is already
                # recorded and needs no action. Telling a full list to "append
                # items that are actually new" sends the agent in circles.
                if not added:
                    if res.get("at_cap"):
                        return (
                            f"Forbidden list is full ({total}/{res.get('max', '?')} items). "
                            "The new item was NOT added. The list's value is that every item "
                            "gets re-read every turn — a 9th item is not more protection, "
                            "it is dilution. If the newly discovered trap is deadlier than an "
                            "existing one, re-send with the SAME action name to refresh that "
                            "item's why/instead instead of appending."
                        )
                    skipped = res.get("skipped_duplicates", 0)
                    return (
                        f"No new prohibitions added ({total} already recorded"
                        + (f"; {skipped} duplicate(s) skipped" if skipped else "")
                        + "). The list on disk is unchanged — append items that "
                        "are actually new."
                    )
                out = [f"Appended {added} prohibition(s); {total} now on record:"]
                for i, e in enumerate(res.get("forbidden") or [], 1):
                    seg = f"  {i}. ✗ {e.get('action', '')}"
                    if e.get("instead"):
                        seg += f" → instead: {e['instead']}"
                    out.append(seg)
                if res.get("at_cap"):
                    out.append(
                        f"NOTE: list now at cap ({total}/{res.get('max', '?')}) — "
                        "further appends will be refused."
                    )
                return "\n".join(out)
            elif action == "complete":
                self._closed_plan = self._plan.complete()
            elif action == "abandon":
                self._plan.abandon(kwargs.get("notes", ""))
                self._closed_plan = {"title": "", "status": "abandoned", "steps": []}
            elif action == "deactivate":
                plan = self._plan.deactivate()
                if not plan:
                    return "No active plan to deactivate."
                return f"Plan '{plan['title']}' paused."
            elif action == "reactivate":
                plan_id = kwargs.get("plan_id")
                if not plan_id:
                    return "ERROR: plan_id required for reactivate."
                plan = self._plan.reactivate(plan_id)
                if not plan:
                    return f"ERROR: Could not reactivate plan '{plan_id}'. Not found or not paused."
                # reactivate exists to show the plan you just resumed.
                return self._plan.summary()
            else:
                return f"ERROR: Unknown action '{action}'."
            # thinking may accompany any action (e.g. step_done + updated model).
            # A recorded model earns its keep only by generating the next
            # falsifiable observation — echo the challenge when one is recorded.
            echo = ""
            if thinking and str(thinking).strip():
                self._plan.set_thinking(str(thinking))
                echo = "\n\n" + _THINKING_ECHO
            return self._delta_receipt(kwargs.get("action", ""), echo)
        except Exception as e:
            return f"ERROR: {e}"

    def _delta_receipt(self, action: str, echo: str = "") -> str:
        """What this call CHANGED, not a reprint of the whole plan.

        Every mutating branch used to end in `return self._plan.summary()`,
        which reprinted every step's title, acceptance, evidence and notes.
        Measured on a 32-step plan that was 254,400 chars per receipt — the
        plan's own state, returned to the caller that just wrote it, at a cost
        paid on every single plan_update. The state is not lost: summary() is
        still what plan_status returns, and it is the deliberate read-back path.
        """
        plan = self._plan.get_active()
        closed = getattr(self, "_closed_plan", None)
        if not plan:
            # complete()/abandon() clear the active pointer, but the caller
            # still needs to know WHICH plan it just closed and how it ended.
            if closed:
                steps = closed.get("steps", [])
                done = sum(1 for s in steps if s.get("status") in ("done", "skipped"))
                return (
                    f"Plan '{closed.get('title', '')}' is now "
                    f"{closed.get('status', '')}. {action} applied.\n"
                    f"Progress: {done}/{len(steps)} — no active plan left.\n"
                    "plan_status() shows nothing active; the closed plan remains "
                    "on disk."
                ) + echo
            return "No active plan." + echo
        steps = plan.get("steps", [])
        done = sum(1 for s in steps if s.get("status") in ("done", "skipped"))
        icons = {"pending": " ", "doing": "→", "done": "✓", "skipped": "-", "blocked": "!"}
        out = [
            f"Plan '{plan.get('title', '')}' [{plan.get('status', '')}] — "
            f"{action} applied."
        ]
        touched = self._last_touched
        if touched:
            s = next((x for x in steps if x.get("id") == touched), None)
            if s is not None:
                out.append(
                    f"  Step {s['id']}. [{icons.get(s.get('status'), '?')}] {s['title']}"
                )
                for v in (s.get("verification") or [])[:3]:
                    out.append(f"    ✓ {v}")
                if len(s.get("verification") or []) > 3:
                    out.append(f"    (+{len(s['verification']) - 3} more evidence)")
        out.append(f"Progress: {done}/{len(steps)}")
        nxt = next((s for s in steps if s.get("status") in ("pending", "doing")), None)
        if nxt is not None:
            out.append(f"Next: Step {nxt['id']}. {nxt['title']}")
        else:
            out.append("No open steps left.")
        out.append(
            "Full plan on demand: plan_status() — one step verbatim: "
            "plan_status(step_id=N)."
        )
        return "\n".join(out) + echo
