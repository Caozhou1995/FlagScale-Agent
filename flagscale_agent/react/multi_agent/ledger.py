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

"""Task ledger — the per-task directory store + status state machine.

Every task is one directory named by its content-addressed id:

    $FLAGSCALE_HOME/tasks/<task_id>/
      contract.json   # immutable, atomic write before spawn (tmp + os.replace)
      state.json      # {"status", "history", "pid", "output_ptr"}
      result.json     # worker self-report (reference only; NEVER trusted for
                      # acceptance)

(NOTE: when the parent has a session dir, a spawned/resumed worker's runtime
trace lands under the parent session tree — <parent_session>/subagents/<task_id>/
worker.log — NOT here; the ledger dir holds only the global, session-independent
bookkeeping files above. Without a session dir, worker.log falls back to this
task dir.)

All reads/writes go through TaskLedger — never hand-edit the files. state.json
is guarded by an exclusive fcntl.flock (the lock file IS state.json), because
the parent side (watchdog / reunite) and the worker side (report_result) are
DIFFERENT processes and both mutate the same state.

Status machine (illegal transitions raise LedgerError):

    SPAWNING      → RUNNING | FAILED
    RUNNING       → REPORTED | DONE | REJECTED | DEADLINE_MISSED | TERMINATED | FAILED
    REPORTED      → DONE | REJECTED | FAILED | TERMINATED
    TERMINATED    → FAILED
    DEADLINE_MISSED → TERMINATED
    DONE | REJECTED | FAILED → (terminal; retry() mints a NEW id)
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contract import Contract

# ── statuses ────────────────────────────────────────────────────────────────
SPAWNING = "SPAWNING"
RUNNING = "RUNNING"
REPORTED = "REPORTED"
DONE = "DONE"
REJECTED = "REJECTED"
FAILED = "FAILED"
DEADLINE_MISSED = "DEADLINE_MISSED"
TERMINATED = "TERMINATED"

ALL_STATUSES = (
    SPAWNING, RUNNING, REPORTED, DONE, REJECTED, FAILED,
    DEADLINE_MISSED, TERMINATED,
)
TERMINAL_STATUSES = (DONE, REJECTED, FAILED)
ACTIVE_STATUSES = (SPAWNING, RUNNING, REPORTED)

# Legal transition table. A state maps to its allowed next states.
_TRANSITIONS: Dict[str, set] = {
    SPAWNING: {RUNNING, FAILED},
    RUNNING: {REPORTED, DONE, REJECTED, DEADLINE_MISSED, TERMINATED, FAILED},
    REPORTED: {DONE, REJECTED, FAILED, TERMINATED},
    TERMINATED: {FAILED},
    DEADLINE_MISSED: {TERMINATED},
    DONE: set(),
    # Resume-with-message reopens a REJECTED or FAILED child so the parent can
    # hand it feedback and it reports a corrected result. DONE is accepted and
    # is NOT resumable.
    REJECTED: {RUNNING},
    FAILED: {RUNNING},
}


class LedgerError(RuntimeError):
    """Raised on an illegal state transition or ledger misuse."""


class DuplicateTask(LedgerError):
    """Raised when creating a task id that already exists and is still active."""


@dataclass
class TaskRecord:
    """A read view of one task's state.json (+ optional contract)."""
    task_id: str
    status: str
    history: List[Dict[str, Any]]
    pid: Optional[int]
    output_ptr: str
    contract: Optional[Contract] = None


class TaskLedger:
    """Flat-directory-per-task store with a status state machine."""

    def __init__(self, tasks_dir: str):
        self._dir = tasks_dir

    # ── paths ────────────────────────────────────────────────────────────────
    def task_dir(self, task_id: str) -> Path:
        if not task_id or "/" in task_id or ".." in task_id:
            raise LedgerError(f"Invalid task id: {task_id!r}")
        return Path(self._dir) / task_id

    def _state_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "state.json"

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ── atomic file helpers ──────────────────────────────────────────────────
    @staticmethod
    def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
        d = path.parent
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(d), prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── create ───────────────────────────────────────────────────────────────
    def create(self, c: Contract, check_inputs_exist: bool = True) -> Path:
        """Create the task directory + contract.json + state.json.

        Raises DuplicateTask if the id already exists AND is still active
        (SPAWNING/RUNNING/REPORTED) — the content-addressing that makes
        duplicate delivery detectable. A terminal same-id dir is left as-is
        (audit) and NOT overwritten; retry() is the way to re-run.
        """
        c.validate(check_inputs_exist=check_inputs_exist)
        tdir = self.task_dir(c.id)
        state_path = tdir / "state.json"

        if tdir.exists():
            existing = self.get(c.id)
            if existing and existing.status in ACTIVE_STATUSES:
                raise DuplicateTask(
                    f"task {c.id} already active (status={existing.status})"
                )
            # Terminal same-id dir: refuse silent overwrite.
            raise DuplicateTask(
                f"task {c.id} already exists (status={existing.status if existing else '?'}); "
                "use retry() for a new id"
            )

        tdir.mkdir(parents=True, exist_ok=False)
        # contract.json is written once and never mutated — the promise the
        # worker was handed must be byte-stable. Do NOT touch after this.
        self._atomic_write_json(tdir / "contract.json", _contract_to_json(c))
        now = self._now_iso()
        self._atomic_write_json(state_path, {
            "status": SPAWNING,
            "history": [{"status": SPAWNING, "ts": now, "note": "created", "pid": None}],
            "pid": None,
            "output_ptr": c.output_ptr,
        })
        return tdir

    # ── read ─────────────────────────────────────────────────────────────────
    def get(self, task_id: str) -> Optional[TaskRecord]:
        """Read a task record. Returns None if it does not exist.

        Reads under a SHARED flock (LOCK_SH) so a cross-process reader (the
        parent watchdog/reunite) can never observe state.json mid-rewrite —
        transition() holds LOCK_EX and rewrites the file in place, and without
        this a reader would occasionally catch the truncate→write window and
        see an empty/partial file (JSONDecodeError).
        """
        state_path = self._state_path(task_id)
        if not state_path.exists():
            return None
        with open(state_path, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                st = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        contract = None
        cpath = self.task_dir(task_id) / "contract.json"
        if cpath.exists():
            try:
                with open(cpath, "r", encoding="utf-8") as cf:
                    contract = _contract_from_json(json.load(cf))
            except Exception:
                contract = None
        return TaskRecord(
            task_id=task_id,
            status=st.get("status", "?"),
            history=st.get("history", []),
            pid=st.get("pid"),
            output_ptr=st.get("output_ptr", ""),
            contract=contract,
        )

    def active_ids(self) -> List[str]:
        """Task ids currently in SPAWNING/RUNNING/REPORTED."""
        out = []
        root = Path(self._dir)
        if not root.exists():
            return out
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            rec = self.get(child.name)
            if rec and rec.status in ACTIVE_STATUSES:
                out.append(child.name)
        return out

    # ── transition (flock-guarded read-modify-write) ─────────────────────────
    def transition(self, task_id: str, new: str, note: str = "",
                   pid: Optional[int] = None) -> TaskRecord:
        """Move a task to `new`. Raises LedgerError on illegal transition.

        The read-modify-write is serialized with an exclusive flock on
        state.json, so a concurrently-running worker (report_result) and the
        parent watchdog cannot interleave and lose an update.
        """
        if new not in ALL_STATUSES:
            raise LedgerError(f"Unknown status: {new!r}")
        state_path = self._state_path(task_id)
        if not state_path.exists():
            raise LedgerError(f"No such task: {task_id}")

        with open(state_path, "r+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                st = json.load(f)
                cur = st.get("status", "?")
                allowed = _TRANSITIONS.get(cur, set())
                if new not in allowed and new != cur:
                    raise LedgerError(
                        f"illegal transition {cur} → {new} for task {task_id} "
                        f"(allowed: {sorted(allowed)})"
                    )
                if new != cur:
                    st["history"].append({
                        "status": new, "ts": self._now_iso(),
                        "note": note or "", "pid": pid,
                    })
                st["status"] = new
                if pid is not None:
                    st["pid"] = pid
                # rewrite in place under the held lock
                f.seek(0)
                f.truncate()
                json.dump(st, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return self.get(task_id)

    # ── worker-side self report ──────────────────────────────────────────────
    def write_result(self, task_id: str, payload: Dict[str, Any]) -> Path:
        """Write result.json (worker self-report) and move state to REPORTED.

        Called ONLY by the worker-side ReportResultTool. The payload's
        `self_report` is reference material — the parent independently VERIFIES
        the deliverable by running the acceptance predicate itself (it never
        redoes the task).
        """
        tdir = self.task_dir(task_id)
        if not tdir.exists():
            raise LedgerError(f"No such task: {task_id}")
        payload = dict(payload)
        payload.setdefault("ts", self._now_iso())
        self._atomic_write_json(tdir / "result.json", payload)
        # RUNNING → REPORTED (or already REPORTED / DONE: leave as-is)
        rec = self.get(task_id)
        if rec and rec.status == RUNNING:
            self.transition(task_id, REPORTED, note="worker reported result")
        return tdir / "result.json"

    def read_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        p = self.task_dir(task_id) / "result.json"
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── prune ────────────────────────────────────────────────────────────────
    def prune(self, older_than_days: int) -> List[str]:
        """Remove terminal task dirs whose state.json mtime is older than N days.

        Manual cleanup entry (no automatic timer is wired). Refuses to
        touch any task still in an active status.
        """
        removed = []
        root = Path(self._dir)
        if not root.exists():
            return removed
        cutoff = time.time() - older_than_days * 86400
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            rec = self.get(child.name)
            if not rec or rec.status not in TERMINAL_STATUSES:
                continue
            sp = child / "state.json"
            try:
                if sp.stat().st_mtime < cutoff:
                    shutil.rmtree(child)
                    removed.append(child.name)
            except OSError:
                continue
        return removed


# ── contract JSON (contract.json is the frozen promise, not the wire form) ───
def _contract_to_json(c: Contract) -> Dict[str, Any]:
    return {
        "id": c.id,
        "goal": c.goal,
        "constraints": c.constraints,
        "acceptance": c.acceptance,
        "inputs": c.inputs,
        "output_ptr": c.output_ptr,
        "deadline_epoch": c.deadline_epoch,
        "depth": c.depth,
        "parent": c.parent,
    }


def _contract_from_json(d: Dict[str, Any]) -> Contract:
    return Contract(
        id=d["id"],
        goal=d["goal"],
        constraints=d.get("constraints", {}),
        acceptance=d.get("acceptance", []),
        inputs=d.get("inputs", []),
        output_ptr=d["output_ptr"],
        deadline_epoch=d.get("deadline_epoch", 0),
        depth=d.get("depth", 1),
        parent=d.get("parent", {}),
    )


# ── derivation-tree audit ────────────────────────────────────────────────────
def render_forest(ledger: "TaskLedger") -> List[Dict[str, Any]]:
    """Reconstruct the audit tree of every task's derivation, from contracts.

    Walks every task dir under the ledger, reads its immutable contract.json,
    and returns a flat, sorted list of nodes:

        {task_id, goal, depth, status, parent_id, parent_trace}

    `parent_trace` is the full ancestry chain (root-first, EXCLUDING the task
    itself); `parent_id` is the immediate parent (last element, or None for the
    orchestrator's direct children). Because each contract is immutable and
    content-addressed, this is a faithful record of "who spawned whom" — the
    auditable-derivation-tree requirement — and needs no extra bookkeeping at spawn
    time beyond the parent field the spawn already writes.
    """
    nodes: List[Dict[str, Any]] = []
    root = Path(ledger._dir)
    if not root.exists():
        return nodes
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        rec = ledger.get(child.name)
        if rec is None or rec.contract is None:
            continue
        c = rec.contract
        trace = list((c.parent or {}).get("parent_trace", []))
        nodes.append({
            "task_id": c.id,
            "goal": c.goal,
            "depth": c.depth,
            "status": rec.status,
            "parent_id": (c.parent or {}).get("task_id") or (trace[-1] if trace else None),
            "parent_trace": trace,
        })
    return nodes
