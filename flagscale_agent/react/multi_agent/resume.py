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

"""Resume-with-message — the parent<->child dialogue channel.

Resume is not only crash-recovery. When a child's result is REJECTED, or the
child says "I am stuck", the PARENT continues that child WITH A MESSAGE. The
child keeps its EXISTING history/state (conversation, swap_store, plans) and
carries on in-context — it is not restarted fresh.

Authorization (isomorphic to contract.parent + the single-writer session lock):
  * the DIRECT parent is the sole default authorizer;
  * a grandparent (or any ancestor) may resume ONLY by ADOPTING an orphaned
    subtree, and only while the immediate parent is DEAD. Adoption writes an
    explicit audit record; the immutable contract.json is never mutated.

A child NEVER resumes upward.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from flagscale_agent.react.paths import get_tasks_dir
from flagscale_agent.react.tools.base import Tool

from .ledger import (
    ACTIVE_STATUSES,
    FAILED,
    REJECTED,
    RUNNING,
    LedgerError,
    TaskLedger,
)
from .spawn import SpawnWorkerTool, _Watchdog
from .wiring import RESUME_PATH_ENV

# The nesting dir under a parent's session dir holding its children.
SUBAGENTS_DIRNAME = "subagents"

# Statuses a child may be resumed out of (source states).
RESUMABLE_STATUSES = (REJECTED, FAILED)


def child_session_dir(parent_session_dir: str, task_id: str) -> str:
    """The nested session dir of `task_id` under `parent_session_dir`."""
    return os.path.join(parent_session_dir, SUBAGENTS_DIRNAME, task_id)


@dataclass
class ResumeTarget:
    task_id: str
    status: str
    session_dir: str
    parent_task_id: Optional[str]
    parent_trace: list
    max_minutes: float


def resolve_target(ledger: TaskLedger, task_id: str,
                   parent_session_dir: str) -> Optional[ResumeTarget]:
    """Read the (immutable) contract + state to build a resume target.

    Returns None when the task does not exist or carries no contract.
    """
    rec = ledger.get(task_id)
    if rec is None or rec.contract is None:
        return None
    c = rec.contract
    parent = c.parent or {}
    try:
        mm = float((c.constraints or {}).get("max_minutes", 0) or 0)
    except (TypeError, ValueError):
        mm = 0.0
    return ResumeTarget(
        task_id=task_id,
        status=rec.status,
        session_dir=child_session_dir(parent_session_dir, task_id),
        parent_task_id=parent.get("task_id"),
        parent_trace=list(parent.get("parent_trace", [])),
        max_minutes=mm,
    )


def _parent_is_dead(ledger: TaskLedger, parent_task_id: Optional[str],
                    adopter_session_dir: str) -> bool:
    """True when the immediate parent shows no sign of life.

    Two independent signals, both must be clear: the parent task is not in an
    ACTIVE ledger status, and no live process holds the parent's session lock
    (derived under the adopter's own subagents/ tree).
    """
    if not parent_task_id:
        return True
    prec = ledger.get(parent_task_id)
    if prec is not None and prec.status in ACTIVE_STATUSES:
        return False
    from flagscale_agent.react.session import get_session_lock_holder
    pdir = child_session_dir(adopter_session_dir, parent_task_id)
    if get_session_lock_holder(pdir) is not None:
        return False
    return True


def authorize(caller_task_id: Optional[str], tgt: ResumeTarget,
              ledger: TaskLedger, adopter_session_dir: str
              ) -> Tuple[bool, str, str]:
    """Decide whether `caller_task_id` may resume `tgt`.

    Returns (allowed, via, reason). `via` is "parent" or "adoption"; `reason`
    is a human-readable refusal cause when allowed is False.

    The caller's own task id (env FLAGSCALE_TASK_ID) is the sole input — a child
    can never resume upward because it is never an ancestor of its target.
    """
    caller = caller_task_id or None
    direct_parent = tgt.parent_task_id or None
    if caller == direct_parent:
        return True, "parent", ""
    if caller is not None and caller in tgt.parent_trace:
        # A strict ancestor: permitted ONLY by adoption, and ONLY when the
        # immediate parent is dead.
        if not _parent_is_dead(ledger, direct_parent, adopter_session_dir):
            return False, "", ("adoption refused: immediate parent "
                               f"{direct_parent} is still alive")
        return True, "adoption", ""
    return False, "", (
        "caller is not the direct parent or an ancestor of the target"
    )


def rewrite_contract_parent(ledger: TaskLedger, task_id: str,
                            adopter_task_id: str,
                            adopter_trace: list) -> Path:
    """Point a task's contract at its adopter — an EXPLICIT adoption.

    The approved design requires adoption to "explicitly rewrite contract.parent
    to the grandparent" so the derivation tree (which reconstructs lineage from
    contract.parent) stays a connected tree rather than orphaning the subtree.
    This is safe to contract IDENTITY: task_id = sha256(goal|constraints|
    acceptance)[:12] EXCLUDES the parent field, so reparenting cannot change the
    id. Only `parent` and `depth` are updated; every other field is preserved
    byte-for-byte. The adopter's own ancestry (parent_trace) is carried through.
    """
    rec = ledger.get(task_id)
    if rec is None or rec.contract is None:
        raise LedgerError(f"no such task or missing contract: {task_id}")
    c = rec.contract
    new_parent = {
        "task_id": adopter_task_id,
        "depth": len(adopter_trace),
        "parent_trace": list(adopter_trace),
    }
    from .ledger import _contract_to_json
    wire = _contract_to_json(c)
    wire["parent"] = new_parent
    wire["depth"] = len(adopter_trace) + 1
    cpath = ledger.task_dir(task_id) / "contract.json"
    ledger._atomic_write_json(cpath, wire)
    return cpath


def write_adoption_audit(ledger: TaskLedger, task_id: str,
                         adopter_task_id: Optional[str],
                         prior_parent_id: Optional[str]) -> Path:
    """Append an audit sidecar recording an adoption.

    The authoritative lineage lives in contract.json, whose `parent` is
    explicitly rewritten by rewrite_contract_parent() during adoption. This
    sidecar records the *history* of adoption events (who adopted what, from
    which prior parent) so the transition is auditable without overwriting it.
    """
    tdir = ledger.task_dir(task_id)
    tdir.mkdir(parents=True, exist_ok=True)
    audit = tdir / "adoption.json"
    prior = []
    if audit.exists():
        try:
            prior = json.loads(audit.read_text(encoding="utf-8")).get("events", [])
        except Exception:
            prior = []
    prior.append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "adopted_by": adopter_task_id,
        "prior_parent": prior_parent_id,
    })
    payload = {"task_id": task_id, "events": prior}
    tmp = audit.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, audit)
    return audit


class ResumeChildTool(Tool):
    """Parent-side resume: continue a rejected/failed child WITH A MESSAGE.

    Reopens the child task and re-enters its EXISTING nested session dir,
    appending the parent's message as the child's next user turn. Parent-only
    (a worker cannot resume its siblings).
    """

    name = "resume_child"
    description = (
        "Resume a directly-spawned child (or, by ADOPTING an orphaned subtree "
        "whose parent is dead, a descendant) WITH A MESSAGE. The child keeps its "
        "existing history and continues in-context — used when its result was "
        "REJECTED or it reported being stuck. Authorization: the direct parent "
        "may always resume; an ancestor only by adoption while the immediate "
        "parent is dead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The child task id to resume.",
            },
            "message": {
                "type": "string",
                "description": "The parent's message, delivered as the child's "
                               "next user turn.",
            },
            "deadline_minutes": {
                "type": "number",
                "description": "New per-run deadline for the resumed child "
                               "(defaults to the contract's max_minutes).",
            },
        },
        "required": ["task_id", "message"],
    }

    def __init__(self, ledger: Optional[TaskLedger] = None,
                 tasks_dir: Optional[str] = None, session_dir: Optional[str] = None,
                 spawn: Optional[SpawnWorkerTool] = None):
        self._ledger = ledger or TaskLedger(tasks_dir or get_tasks_dir())
        self._session_dir = session_dir
        self._spawn = spawn or SpawnWorkerTool(ledger=self._ledger,
                                               session_dir=session_dir)

    def execute(self, task_id: str = "", message: str = "",
                deadline_minutes: float = 0, **kwargs) -> str:
        if not task_id:
            return "ERROR: resume_child requires a task_id."
        if not (message or "").strip():
            return "ERROR: resume_child requires a non-empty message."
        if not self._session_dir:
            return ("ERROR: resume_child needs the parent session dir (none "
                    "configured).")
        tgt = resolve_target(self._ledger, task_id, self._session_dir)
        if tgt is None:
            return f"ERROR: no such task or missing contract: {task_id}"

        caller = os.environ.get("FLAGSCALE_TASK_ID") or None
        allowed, via, reason = authorize(caller, tgt, self._ledger,
                                         self._session_dir)
        if not allowed:
            return f"ERROR: {reason}"
        if tgt.status not in RESUMABLE_STATUSES:
            return (f"ERROR: task {task_id} is {tgt.status}, not resumable "
                    f"(only REJECTED/FAILED). Refuse to resume a DONE task.")

        rec = self._ledger.get(task_id)
        contract = rec.contract

        # Reopen the child (REJECTED/FAILED -> RUNNING) BEFORE the fork so the
        # marker is durable even if the process dies.
        try:
            self._ledger.transition(task_id, RUNNING,
                                    note=f"resumed via {via}")
        except Exception as e:
            return f"ERROR: failed to reopen task {task_id}: {e}"

        if via == "adoption":
            # The approved design: adoption EXPLICITLY rewrites contract.parent
            # to the adopter so the derivation tree stays connected. Identity is
            # preserved (id excludes parent). Record an audit sidecar too.
            try:
                adopter_trace = (json.loads(
                    os.environ.get("FLAGSCALE_PARENT_TRACE", "[]") or "[]")
                    + [caller] if caller else [])
                rewrite_contract_parent(self._ledger, task_id, caller,
                                        adopter_trace)
            except Exception as e:
                return f"ERROR: failed to rewrite contract.parent on adoption: {e}"
            try:
                write_adoption_audit(self._ledger, task_id, caller,
                                     tgt.parent_task_id)
            except Exception as e:
                return f"ERROR: failed to record adoption audit: {e}"

        # The resume message is per-agent state: write it into the child's
        # nested session dir (created if this is the first touch).
        cdir = Path(tgt.session_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        resume_path = cdir / "resume.prompt"
        resume_path.write_text(message, encoding="utf-8")

        # Build the child env (same nested session root + contract path) and add
        # the resume pointer, then fork a fresh process into the SAME session.
        # argv stays the normal spawn form (OPTIONS before the positional
        # contract path, per typer); the child detects RESUME via the env var,
        # not a new CLI flag — no cli.py change needed.
        env = self._spawn._build_env(contract)
        env[RESUME_PATH_ENV] = str(resume_path)
        dm = float(deadline_minutes or 0) or tgt.max_minutes or 10.0
        contract_path = self._ledger.task_dir(task_id) / "contract.prompt"
        log_path = self._ledger.task_dir(task_id) / "worker.log"
        argv = [self._spawn._agent_bin, "--time-budget-sec", str(int(dm * 60)),
                str(contract_path)]
        try:
            log_fh = open(log_path, "a", encoding="utf-8")
        except Exception as e:
            return f"ERROR: failed to open worker.log: {e}"
        try:
            proc = subprocess.Popen(argv, **self._spawn._popen_kwargs(env, log_fh))
        except Exception as e:
            log_fh.close()
            try:
                self._ledger.transition(task_id, FAILED,
                                        note=f"resume Popen failed: {e}")
            except Exception:
                pass
            return f"ERROR: failed to spawn resumed child: {e}"
        # The child holds its own dup of the log fd; close the parent's copy so
        # the descriptor does not leak in the (long-lived) parent process.
        log_fh.close()

        try:
            self._ledger.transition(task_id, RUNNING, note="resumed",
                                    pid=proc.pid)
        except Exception:
            pass
        # A resumed child needs the SAME parent-side watchdog a freshly spawned
        # one gets, otherwise a hung/deadline-exceeded resumed worker is never
        # killed and its RUNNING task (an ACTIVE status) permanently occupies a
        # global concurrency slot and blocks adoption. Deadline is per-run.
        try:
            deadline_epoch = int(time.time()) + int(dm * 60)
            _Watchdog(self._ledger, task_id, proc.pid, deadline_epoch).start()
        except Exception:
            pass
        return (f"resumed task {task_id} via {via} pid={proc.pid} "
                f"session={tgt.session_dir} deadline={dm:g}min")

