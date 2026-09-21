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

"""Parent-side reunite: acceptance verification + poll tool.

The single most important rule lives here: **the parent NEVER trusts the
worker's self-report**. When a
task reaches REPORTED the parent independently runs every `acceptance` check
itself, inside its OWN process, and only the exit codes of those checks decide
DONE vs REJECTED. `result.json` is surfaced to the model as *reference only*
(via `poll_tasks result`); it is never an input to the verdict.

IMPORTANT — what "run the acceptance checks" means and does NOT mean:
  * The parent runs the acceptance PREDICATES (e.g. `test -f out.md`,
    `wc -l < out.md`). That is VERIFICATION of the worker's deliverable.
  * The parent does NOT redo the task. The labor (writing the review, editing
    the code, producing the artifact) stays with the worker — that is the
    whole point of delegating to another agent. The parent only checks the
    artifact that came back; it never regenerates it.

    REPORTED  --parent runs acceptance predicates-->  DONE      (all exit 0)
    REPORTED  --parent runs acceptance predicates-->  REJECTED  (any nonzero)

A task that is not yet REPORTED is not judged (the parent is told to keep
polling). A task already in a terminal state is returned as-is, so repeated
polls are idempotent and side-effect-free.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flagscale_agent.react.paths import get_tasks_dir
from flagscale_agent.react.tools.base import Tool

from .ledger import (
    DONE,
    FAILED,
    REJECTED,
    REPORTED,
    TERMINAL_STATUSES,
    LedgerError,
    TaskLedger,
)

# Per-check subprocess timeout (120s each).
DEFAULT_CHECK_TIMEOUT_S = 120
# How much of a check's output to keep in the verdict / REJECTED note.
STDOUT_TAIL_CHARS = 2000


@dataclass
class CheckEvidence:
    """One acceptance check's result."""
    check: str
    kind: str
    cwd: str
    exit_code: int
    stdout_tail: str
    passed: bool
    timed_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "kind": self.kind,
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "stdout_tail": self.stdout_tail,
            "passed": self.passed,
            "timed_out": self.timed_out,
        }


@dataclass
class Verdict:
    """The parent's independent judgement of one task."""
    task_id: str
    status: str            # ledger status after judging (or current, if pending)
    passed: bool           # True only when ALL checks passed AND status == DONE
    pending: bool          # True when the task is not yet ready to be judged
    checks: List[CheckEvidence] = field(default_factory=list)
    note: str = ""
    # result.json is carried for REFERENCE only — never as acceptance evidence.
    self_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "passed": self.passed,
            "pending": self.pending,
            "note": self.note,
            "checks": [c.to_dict() for c in self.checks],
            "self_report": self.self_report,
        }


def _tail(text: str, n: int = STDOUT_TAIL_CHARS) -> str:
    text = text or ""
    return text[-n:] if len(text) > n else text


def _run_one_check(item: Dict[str, Any], default_cwd: str,
                   timeout_s: int) -> CheckEvidence:
    """Run one acceptance predicate as a subprocess; NEVER trust any self-report.

    `check_command` runs the command with the shell, exit 0 == pass. Any other
    `kind` is unsupported and counts as a failure (fail-closed).
    """
    kind = str(item.get("kind", "check_command") or "check_command")
    cmd = str(item.get("check", "") or "")
    # The check's cwd defaults to the parent's current cwd. An absolute
    # per-item cwd wins; a relative/empty one falls back to the parent cwd.
    item_cwd = str(item.get("cwd", "") or "")
    cwd = item_cwd if os.path.isabs(item_cwd) else default_cwd

    if kind != "check_command":
        return CheckEvidence(
            check=cmd, kind=kind, cwd=cwd, exit_code=-1,
            stdout_tail=f"unsupported acceptance kind: {kind!r}",
            passed=False,
        )
    if not cmd.strip():
        return CheckEvidence(
            check=cmd, kind=kind, cwd=cwd, exit_code=-1,
            stdout_tail="empty check command", passed=False,
        )

    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        out = _tail((proc.stdout or b"").decode("utf-8", "replace"))
        return CheckEvidence(
            check=cmd, kind=kind, cwd=cwd, exit_code=proc.returncode,
            stdout_tail=out, passed=(proc.returncode == 0),
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout or b""
        if isinstance(out, str):
            out = out.encode("utf-8", "replace")
        return CheckEvidence(
            check=cmd, kind=kind, cwd=cwd, exit_code=-1,
            stdout_tail=_tail(out.decode("utf-8", "replace")),
            passed=False, timed_out=True,
        )
    except Exception as e:  # OSError, etc. — fail closed
        return CheckEvidence(
            check=cmd, kind=kind, cwd=cwd, exit_code=-1,
            stdout_tail=f"check execution error: {e}", passed=False,
        )


def check_result(ledger: TaskLedger, task_id: str,
                 timeout_s: int = DEFAULT_CHECK_TIMEOUT_S,
                 default_cwd: Optional[str] = None) -> Verdict:
    """Independently run a task's acceptance predicates and judge it.

    Semantics (this is the whole verdict contract):

      * Task missing                → Verdict(pending=False, status="?")  — error
      * status in (DONE/REJECTED/FAILED, terminal) → return as-is (idempotent)
      * status == REPORTED          → the parent runs every acceptance predicate
                                      itself; all pass → transition DONE,
                                      else → transition REJECTED (note carries
                                      the failing check + stdout tail).
      * any other non-terminal      → Verdict(pending=True); do NOT judge yet.

    The parent runs the PREDICATES (verification), never the task itself. The
    worker's result.json is attached as `self_report` for reference and is
    NEVER consulted for the verdict.
    """
    rec = ledger.get(task_id)
    if rec is None:
        return Verdict(task_id=task_id, status="?", passed=False,
                       pending=False, note=f"no such task: {task_id}")

    # Already judged / dead: return current state unchanged (poll is idempotent).
    if rec.status in TERMINAL_STATUSES:
        self_report = ledger.read_result(task_id)
        return Verdict(
            task_id=task_id, status=rec.status, passed=(rec.status == DONE),
            pending=False, checks=[], self_report=self_report,
            note=f"already terminal: {rec.status}",
        )

    # Not yet ready: worker still running / not reported. Do not judge.
    if rec.status != REPORTED:
        return Verdict(
            task_id=task_id, status=rec.status, passed=False, pending=True,
            checks=[], self_report=ledger.read_result(task_id),
            note=f"not ready (status={rec.status}); poll again later",
        )

    # ── REPORTED: the parent runs the acceptance predicates itself ───────────
    c = rec.contract
    acceptance = list(c.acceptance or []) if c else []
    cwd = default_cwd or os.getcwd()
    checks: List[CheckEvidence] = [
        _run_one_check(a, cwd, timeout_s) for a in acceptance
    ]
    all_pass = bool(checks) and all(ck.passed for ck in checks)

    if all_pass:
        try:
            ledger.transition(task_id, DONE, note="acceptance verified by parent")
            status = DONE
        except LedgerError:
            status = ledger.get(task_id).status
        note = f"all {len(checks)} acceptance checks passed"
    else:
        failed = [ck for ck in checks if not ck.passed]
        bits = []
        for ck in failed[:3]:
            why = "TIMEOUT" if ck.timed_out else f"exit={ck.exit_code}"
            tail = (ck.stdout_tail or "").strip().replace("\n", " ")
            bits.append(f"[{why}] {ck.check}  ::  {tail[:300]}")
        note = (f"acceptance FAILED ({len(failed)}/{len(checks)}): "
                + " | ".join(bits))
        try:
            ledger.transition(task_id, REJECTED, note=note)
            status = REJECTED
        except LedgerError:
            status = ledger.get(task_id).status

    return Verdict(
        task_id=task_id, status=status, passed=all_pass, pending=False,
        checks=checks, self_report=ledger.read_result(task_id), note=note,
    )


class PollTasksTool(Tool):
    """Parent-side poll/reunite tool.

    actions:
      * list               — show all tasks + current status
      * check <task_id>    — run acceptance predicates, judge, write DONE/REJECTED
      * result <task_id>   — show the worker's result.json (REFERENCE ONLY)
    """

    name = "poll_tasks"
    description = (
        "Parent-side poll / acceptance tool. actions: list (list all tasks) / "
        "check <task_id> (the parent runs the acceptance predicates and decides "
        "DONE or REJECTED, writing it back to the ledger) / "
        "result <task_id> (show the worker's self-reported result.json — "
        "reference only, never the basis for acceptance). "
        "Acceptance is decided solely by the parent running the checks; the "
        "worker's self-report never counts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "list | check | result",
                "enum": ["list", "check", "result"],
            },
            "task_id": {
                "type": "string",
                "description": "Required for check/result: the target task id.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, ledger: Optional[TaskLedger] = None,
                 tasks_dir: Optional[str] = None):
        self._ledger = ledger or TaskLedger(tasks_dir or get_tasks_dir())

    def execute(self, action: str = "", task_id: str = "", **kwargs) -> str:
        action = (action or "").strip().lower()
        if action == "list":
            return self._do_list()
        if action == "check":
            return self._do_check(task_id)
        if action == "result":
            return self._do_result(task_id)
        return f"ERROR: unknown action {action!r} (expected list / check / result)."

    # ── actions ──────────────────────────────────────────────────────────────
    def _do_list(self) -> str:
        root = self._ledger._dir
        if not root or not os.path.isdir(root):
            return "Ledger is empty (no tasks dispatched yet)."
        ids = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )
        if not ids:
            return "Ledger is empty (no tasks dispatched yet)."
        lines = []
        for tid in ids:
            rec = self._ledger.get(tid)
            if rec is None:
                continue
            lines.append(f"- {tid}  status={rec.status}  output_ptr={rec.output_ptr}")
        return "task list:\n" + "\n".join(lines)

    def _do_check(self, task_id: str) -> str:
        if not task_id:
            return "ERROR: check requires a task_id."
        try:
            v = check_result(self._ledger, task_id)
        except Exception as e:
            return f"ERROR: check failed: {e}"
        if v.pending:
            return (f"task {task_id}: not ready (status={v.status}); "
                    "poll again later. No ledger changes made.")
        head = ("PASSED" if v.passed else "FAILED")
        body = [f"task {task_id}: {head} (status={v.status})", f"note: {v.note}"]
        for ck in v.checks:
            mark = "OK" if ck.passed else "FAIL"
            why = "timeout" if ck.timed_out else f"exit={ck.exit_code}"
            body.append(f"  [{mark}] ({why}) {ck.check}")
            if not ck.passed and ck.stdout_tail:
                body.append("      stdout_tail: "
                            + ck.stdout_tail.strip().replace("\n", " ")[:500])
        return "\n".join(body)

    def _do_result(self, task_id: str) -> str:
        if not task_id:
            return "ERROR: result requires a task_id."
        payload = self._ledger.read_result(task_id)
        if payload is None:
            return f"task {task_id}: no result.json (worker did not report)."
        import json as _json
        return ("result.json (worker self-report only, NOT acceptance evidence; "
                "use check for acceptance):\n"
                + _json.dumps(payload, ensure_ascii=False, indent=2))
