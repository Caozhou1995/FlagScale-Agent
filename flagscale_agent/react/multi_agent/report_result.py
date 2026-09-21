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

"""Worker-side self-report.

The worker's ONLY added tool. It writes result.json and moves the task to
REPORTED — a *self-report*, not a verdict. The parent NEVER trusts
result.json for acceptance; the parent instead runs the acceptance checks
itself. This tool exists so the parent's reunite has something
to compare against and so a task cannot silently hang in RUNNING after the
worker exits.

Note on "re-run": the acceptance step is the parent running the acceptance
*predicates* (e.g. `test -f out.md`) against the worker's deliverable. This is
VERIFICATION of the product, not re-execution of the task. The parent never
redoes the task itself — the labor stays with the worker; if the parent had to
redo the work, delegating to a worker would be pointless.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from flagscale_agent.react.paths import get_tasks_dir
from flagscale_agent.react.tools.base import Tool

from .contract import _within
from .ledger import LedgerError, REPORTED, RUNNING, TaskLedger


class ReportResultTool(Tool):
    """Report a finished worker task (self-report; NOT acceptance)."""

    name = "report_result"
    description = (
        "A worker MUST call this tool after finishing the task. It writes "
        "result.json and sets the task state to REPORTED. NOTE: this is only a "
        "self-report; the parent independently runs the acceptance checks and "
        "never trusts the self-report. `summary` is required; `files_written` "
        "lists the absolute paths of the files you actually wrote."
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "Brief description of what was done, where the outputs are, "
                    "and whether the goal was met."
                ),
            },
            "files_written": {
                "type": "array",
                "description": (
                    "Absolute paths of the files you actually wrote (must be "
                    "inside the contract's constraints.writable). If omitted, "
                    "only the contract's output_ptr is checked."
                ),
                "items": {"type": "string"},
            },
        },
        "required": ["summary"],
    }

    def __init__(self, ledger: Optional[TaskLedger] = None,
                 tasks_dir: Optional[str] = None):
        self._ledger = ledger or TaskLedger(tasks_dir or get_tasks_dir())

    def execute(self, summary: str = "", files_written: Optional[List[str]] = None,
                **kwargs) -> str:
        # ── 1. must be running INSIDE a worker (env carries the task id) ─────
        task_id = os.environ.get("FLAGSCALE_TASK_ID")
        if not task_id:
            return (
                "ERROR: report_result can only be called inside a worker process "
                "(FLAGSCALE_TASK_ID not found). This is a parent-side internal "
                "tool; the parent verifies via poll/reunite."
            )
        if not summary or not summary.strip():
            return "ERROR: summary must not be empty."

        # ── 2. files_written ∪ output_ptr ⊆ constraints.writable ───────
        rec = self._ledger.get(task_id)
        if rec is None:
            return f"ERROR: task {task_id} not found in the ledger (contract missing?)."
        c = rec.contract
        writable = (c.constraints or {}).get("writable") or [] if c else []
        checks: List[str] = []
        if c and c.output_ptr:
            checks.append(c.output_ptr)
        for p in (files_written or []):
            checks.append(p)
        for p in checks:
            if not any(_within(p, w) for w in writable):
                return (
                    f"ERROR: file {p!r} is not inside the contract's "
                    f"constraints.writable {writable!r}. Only write to "
                    "the allowed directories."
                )

        # ── 3. write result.json + RUNNING → REPORTED ────────────────────────
        payload: Dict[str, Any] = {
            "summary": summary.strip(),
            "files_written": list(files_written or []),
            "output_ptr": c.output_ptr if c else "",
        }
        try:
            self._ledger.write_result(task_id, payload)
        except LedgerError as e:
            return f"ERROR: failed to write result: {e}"

        rec2 = self._ledger.get(task_id)
        status = rec2.status if rec2 else "?"
        return (
            f"reported task {task_id} (status={status}). "
            "The parent will independently run the acceptance checks; this is "
            "a self-report, not a pass."
        )
