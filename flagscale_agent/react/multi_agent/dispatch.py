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

"""Parent-side fan-out dispatcher + BOUNDED reunite.

Adds one mechanism on top of the spawn/reunite primitives: an N-worker dispatcher whose
reunite protocol returns POINTERS, not text. Two exit conditions must BOTH hold
Two properties must BOTH hold:

  (a) parallelizable tasks get a REAL wall-clock speedup, and
  (b) the parent context is NOT blown up.

The mechanism therefore never concatenates worker output. A worker's long report
stays on disk (worker.log / result.json / the artifact at output_ptr); what
crosses back into the parent is a compact PointerRecord — task_id + status +
output_ptr + a length-capped note. That is the "pointers, not a text wall" rule.

Scheduling:
  * At most `degree` workers run concurrently. `degree` is clamped to the
    hard concurrency cap (spawn.MAX_CONCURRENT).
  * REPORTED is still an ACTIVE status, so it occupies a concurrency slot. To free a slot
    the dispatcher must JUDGE the task (check_result → DONE/REJECTED). That is
    exactly the bounded-reunite step: judging both frees the slot and produces
    the pointer record.
  * If a spawn is refused because the ledger is full (backpressure from tasks
    not started by this dispatch), the spec is requeued and the loop waits —
    the dispatcher never overruns the cap.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flagscale_agent.react.paths import get_tasks_dir
from flagscale_agent.react.tools.base import Tool

from .contract import compute_id
from .ledger import TaskLedger
from .reunite import check_result
from .spawn import MAX_CONCURRENT, SUBAGENTS_DIRNAME, SpawnWorkerTool

# How often the dispatch loop re-checks task state (seconds).
POLL_INTERVAL_S = 1.0
# Max chars of a note carried into a pointer record (context stays bounded).
NOTE_CAP = 180


@dataclass
class PointerRecord:
    """A bounded, pointer-style reunite record — never a worker text body.

    `output_ptr` and `log_path` are POINTERS the parent (or a human) can open
    on demand; they are not inlined. `note` is a short, capped summary (a
    verdict reason or a spawn error), never the worker's prose.
    """
    task_id: str
    status: str
    passed: bool
    output_ptr: str = ""
    log_path: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "passed": self.passed,
            "output_ptr": self.output_ptr,
            "log_path": self.log_path,
            "note": self.note,
        }


def _clip(text: str, n: int = NOTE_CAP) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


# Statuses that are neither ACTIVE nor terminal but mean "stop waiting".
_SETTLED_NONTERMINAL = {"DEADLINE_MISSED", "TERMINATED"}
# Default upper bound on a single dispatch_many call (seconds).
DEFAULT_MAX_WAIT_S = 1800


def _parse_task_id(spawn_output: str) -> Optional[str]:
    """Extract the task id from SpawnWorkerTool.execute's success string.

    Format (stable, ours): "spawned task <id> pid=<pid> depth=<d> ...".
    """
    parts = (spawn_output or "").split()
    if len(parts) >= 3 and parts[0] == "spawned" and parts[1] == "task":
        return parts[2]
    return None


def _dispatch_many_blocking(
    specs: List[Dict[str, Any]],
    degree: int = MAX_CONCURRENT,
    spawn: Optional[SpawnWorkerTool] = None,
    ledger: Optional[TaskLedger] = None,
    timeout_s: Optional[int] = None,
    poll_interval: float = POLL_INTERVAL_S,
    max_wait_s: float = DEFAULT_MAX_WAIT_S,
) -> List[PointerRecord]:
    """BLOCKING fan-out core: fill `degree` slots, poll+judge until all settle.

    `degree` workers run concurrently (clamped to the hard cap — this
    dispatcher does not raise it). Each spec is a dict of SpawnWorkerTool.execute kwargs: goal,
    constraints, acceptance, inputs, output_ptr, deadline_minutes.

    Returns one PointerRecord per spec, in input order. The records carry
    POINTERS (output_ptr, log_path) + a capped note — never worker prose, so the
    parent context stays bounded regardless of how much each worker wrote.
    """
    from .reunite import DEFAULT_CHECK_TIMEOUT_S
    tmo = DEFAULT_CHECK_TIMEOUT_S if timeout_s is None else timeout_s

    # Degree is clamped by the hard cap; this dispatcher never raises it.
    try:
        degree = int(degree)
    except (TypeError, ValueError):
        degree = MAX_CONCURRENT
    degree = max(1, min(degree, MAX_CONCURRENT))

    if spawn is None:
        spawn = SpawnWorkerTool(ledger=ledger)
    ledger = ledger or spawn._ledger

    n = len(specs)
    records: List[Optional[PointerRecord]] = [None] * n
    pending: List[int] = list(range(n))
    inflight: Dict[str, int] = {}
    wait_deadline = time.time() + max_wait_s

    while pending or inflight:
        # ── 1. fill free slots ───────────────────────────────────────────────
        while pending and len(inflight) < degree:
            idx = pending[0]
            spec = dict(specs[idx])
            out = spawn.execute(**spec)
            if out.startswith("ERROR"):
                # Backpressure (slots full elsewhere / active dup) → wait, retry.
                if ("concurrency slots full" in out) or ("duplicate task" in out):
                    break
                # Hard contract/Popen failure → record and move on.
                records[idx] = PointerRecord(
                    task_id="", status="SPAWN_FAILED", passed=False,
                    output_ptr=str(spec.get("output_ptr", "")), note=_clip(out),
                )
                pending.pop(0)
                continue
            tid = _parse_task_id(out)
            if tid is None:
                records[idx] = PointerRecord(
                    task_id="", status="SPAWN_FAILED", passed=False,
                    output_ptr=str(spec.get("output_ptr", "")),
                    note=_clip("could not parse task id: " + out),
                )
                pending.pop(0)
                continue
            pending.pop(0)
            inflight[tid] = idx
            rec = ledger.get(tid)
            records[idx] = PointerRecord(
                task_id=tid, status=(rec.status if rec else "SPAWNING"),
                passed=False, output_ptr=(rec.output_ptr if rec else ""),
                log_path=_worker_log_ptr(spawn, ledger, tid),
                note="dispatched",
            )

        # ── 2. poll + judge inflight (bounded reunite; judging frees a slot) ──
        for tid in list(inflight):
            v = check_result(ledger, tid, timeout_s=tmo)
            settled = (not v.pending) or (v.status in _SETTLED_NONTERMINAL) \
                or (ledger.get(tid) is None)
            if not settled:
                continue
            idx = inflight.pop(tid)
            trec = ledger.get(tid)
            prev_ptr = records[idx].output_ptr if records[idx] else ""
            records[idx] = PointerRecord(
                task_id=tid,
                status=(v.status if v.status != "?" else
                        (trec.status if trec else "?")),
                passed=v.passed,
                output_ptr=(trec.output_ptr if trec else prev_ptr),
                log_path=_worker_log_ptr(spawn, ledger, tid),
                note=_clip(v.note),
            )

        if not pending and not inflight:
            break
        if time.time() > wait_deadline:
            # Give up: mark whatever is still unstarted as TIMEOUT (bounded).
            for idx in pending:
                records[idx] = PointerRecord(
                    task_id="", status="TIMEOUT", passed=False,
                    output_ptr=str(specs[idx].get("output_ptr", "")),
                    note=_clip(f"dispatch_many exceeded max_wait_s={max_wait_s:g}"),
                )
            for tid, idx in inflight.items():
                trec = ledger.get(tid)
                records[idx] = PointerRecord(
                    task_id=tid, status=(trec.status if trec else "TIMEOUT"),
                    passed=False,
                    output_ptr=(trec.output_ptr if trec else ""),
                    log_path=_worker_log_ptr(spawn, ledger, tid),
                    note="still running at dispatch deadline",
                )
            break
        time.sleep(poll_interval)

    return [r for r in records]  # type: ignore[return-value]


# ── ASYNC dispatch: background fan-out + poll handle ─────────────────────────
#
# Background rationale: the harness tool loop is SYNCHRONOUS (a tool's execute()
# returns a str and the agent blocks until it does). A blocking dispatch_many
# therefore freezes the parent for the WHOLE fan-out — the parent cannot do
# anything else while workers run. We flip the default to ASYNC: dispatch_many
# returns a handle IMMEDIATELY (control returns to the agent) while a background
# daemon thread runs the exact same fill-slot + poll + judge loop. The thread
# keeps judging REPORTED tasks (which still occupy a D9 slot) so all N tasks
# eventually run and every slot is freed. The parent retrieves the bounded
# PointerRecords with a later dispatch_many(action='poll') call — mirroring
# shell(background=true) + shell_jobs(poll).
#
# Dispatch async is SAFE here because a worker is a separate PROCESS whose
# liveness does not depend on the parent's thread. If the parent process exits
# mid-fan-out the daemon thread dies, but every already-spawned worker keeps
# running and writes its own ledger record + result.json — so a later session's
# dispatch_many(action='list')/poll_tasks can still reconcile them from disk.
_DISPATCH_JOBS: Dict[str, Dict[str, Any]] = {}
_DISPATCH_LOCK = threading.Lock()

# Terminal state of the background thread's own bookkeeping.
_JOB_RUNNING = "running"
_JOB_COMPLETE = "complete"


def _jobs_file(tasks_dir: str) -> Path:
    return Path(tasks_dir) / "dispatch_jobs.json"


def _persist_job(tasks_dir: str, job: Dict[str, Any]) -> None:
    """Merge one job into the on-disk dispatch registry (best-effort, atomic)."""
    try:
        path = _jobs_file(tasks_dir)
        os.makedirs(tasks_dir, exist_ok=True)
        with _DISPATCH_LOCK:
            data: Dict[str, Any] = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8")) or {}
                except Exception:
                    data = {}
            data[job["dispatch_id"]] = job
            # Unique tmp per writer: two processes can target the same
            # tasks_dir/dispatch_jobs.json, and a FIXED tmp name would let one
            # publish the other's half-written file. os.replace stays atomic.
            tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
    except Exception:
        # Persistence is a convenience for cross-session reconcile — a failure
        # here must never kill the fan-out.
        pass


def _load_jobs(tasks_dir: str) -> Dict[str, Any]:
    path = _jobs_file(tasks_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _worker_log_ptr(spawn: Optional[SpawnWorkerTool],
                    ledger: TaskLedger, tid: str) -> str:
    """The log pointer for `tid`, derived the same way the spawn itself did.

    When the parent bound a session dir the log lives in the worker's NESTED
    SESSION dir (<parent_session>/subagents/<tid>/worker.log); without one, the
    ledger task dir (legacy fallback). Keeps the parent-side pointer consistent
    with the file spawn.py actually opens.
    """
    if spawn is not None and hasattr(spawn, "worker_log_path"):
        return str(spawn.worker_log_path(tid))
    return str(ledger.task_dir(tid) / "worker.log")


class _RecordingSpawn:
    """Wraps a spawn so the background job records every task id it starts.

    Purely observational: it forwards execute() verbatim and appends the parsed
    task id to the live job record, so a PARTIAL poll can surface already-settled
    tasks before the whole fan-out finishes.
    """

    def __init__(self, spawn, dispatch_id: str, ledger: TaskLedger):
        self._spawn = spawn
        self._dispatch_id = dispatch_id
        self._ledger = ledger

    def __getattr__(self, name):
        # Pass the wrapped spawn's attributes through (e.g. `_session_dir`,
        # `_ledger`, `_agent_bin`) so helpers that derive per-worker paths from
        # the spawn keep working through the recorder wrapper.
        return getattr(self._spawn, name)

    def execute(self, **kw):
        out = self._spawn.execute(**kw)
        tid = _parse_task_id(out)
        if tid:
            # Mutate AND snapshot the job under the SAME lock, so a concurrent
            # finalize (also under the lock) can never interleave a version that
            # drops this just-appended id from the on-disk registry.
            with _DISPATCH_LOCK:
                job = _DISPATCH_JOBS.get(self._dispatch_id)
                if job is not None:
                    ids = job.setdefault("task_ids", [])
                    if tid not in ids:
                        ids.append(tid)
                    snapshot = dict(job)
                else:
                    snapshot = None
            if snapshot is not None:
                _persist_job(str(self._ledger._dir), snapshot)
        return out


def _run_dispatch_job(dispatch_id: str, specs: List[Dict[str, Any]],
                      degree: int, spawn: SpawnWorkerTool,
                      ledger: TaskLedger) -> None:
    """Background thread body: run the blocking core, then bank the results."""
    recorder = _RecordingSpawn(spawn, dispatch_id, ledger)
    try:
        records = _dispatch_many_blocking(
            specs, degree=degree, spawn=recorder, ledger=ledger,
        )
        rec_dicts = [r.to_dict() for r in records]
        state = _JOB_COMPLETE
        err = ""
    except Exception as e:  # never let the thread die silently
        rec_dicts = []
        state = _JOB_COMPLETE
        err = _clip(f"dispatch job error: {e}")
    with _DISPATCH_LOCK:
        prev = dict(_DISPATCH_JOBS.get(dispatch_id, {}))
    job = {
        "dispatch_id": dispatch_id,
        "state": state,
        "n_specs": len(specs),
        "records": rec_dicts,
        "task_ids": prev.get("task_ids", []),
        "error": err,
        "updated": time.time(),
    }
    with _DISPATCH_LOCK:
        _DISPATCH_JOBS[dispatch_id] = job
    _persist_job(str(ledger._dir), job)


# The one handle returned by an async dispatch — a dict, for stable JSON output.
def _new_handle(dispatch_id: str, n_specs: int, degree: int,
                ledger: TaskLedger, spawn: Optional[SpawnWorkerTool] = None,
                ) -> Dict[str, Any]:
    return {
        "dispatch_id": dispatch_id,
        "state": _JOB_RUNNING,
        "n_specs": n_specs,
        "degree": degree,
        "tasks_dir": str(ledger._dir),
        # The parent's session dir at dispatch time: lets a LATER poll (even a
        # different process) derive each worker's nested-session log pointer
        # without holding the spawn object.
        "session_dir": (getattr(spawn, "_session_dir", None) or ""),
        "records": None,
    }


def _live_records(job: Dict[str, Any],
                  ledger: TaskLedger) -> Tuple[List[PointerRecord], List[str]]:
    """Best-effort pointer records from the CURRENT ledger state of a job.

    Before the background thread finishes we cannot know the final records, but
    every spec that already carries a task id (banked in `_DISPATCH_JOBS[...]`
    augmentation below) or that is discoverable from the ledger lets the parent
    watch progress now. The authoritative, complete records arrive when the
    thread banks them (`state == complete`).
    """
    settled_extra = {"DEADLINE_MISSED", "TERMINATED"}
    recs: List[PointerRecord] = []
    # Cross-session reconcile: the poll may run in a process that never held
    # the spawn. The log pointer is derived from the session_dir the JOB
    # recorded at dispatch time (same rule spawn.worker_log_path applies).
    sdir = job.get("session_dir") or ""
    for tid in job.get("task_ids", []) or []:
        trec = ledger.get(tid)
        if trec is None:
            continue
        status = trec.status
        passed = (status == "DONE")
        pending = status not in ("DONE", "REJECTED", "FAILED",
                                 *settled_extra)
        if pending:
            continue  # not yet settleable — omit from the partial view
        if sdir:
            log_ptr = str(Path(sdir) / SUBAGENTS_DIRNAME / tid / "worker.log")
        else:
            log_ptr = str(ledger.task_dir(tid) / "worker.log")
        recs.append(PointerRecord(
            task_id=tid, status=status, passed=passed,
            output_ptr=trec.output_ptr,
            log_path=log_ptr,
            note=_clip("partial (background still running)"),
        ))
    return recs, [r.task_id for r in recs]


def start_dispatch_many(
    specs: List[Dict[str, Any]],
    degree: int = MAX_CONCURRENT,
    spawn: Optional[SpawnWorkerTool] = None,
    ledger: Optional[TaskLedger] = None,
) -> Dict[str, Any]:
    """Async fan-out: START the background fan-out and return a handle AT ONCE.

    The returned handle carries `dispatch_id`; poll it with
    `poll_dispatch(dispatch_id)` (or `dispatch_many(action='poll')`). Control
    returns to the caller immediately — no blocking on worker completion.
    """
    try:
        degree = int(degree)
    except (TypeError, ValueError):
        degree = MAX_CONCURRENT
    degree = max(1, min(degree, MAX_CONCURRENT))

    if spawn is None:
        spawn = SpawnWorkerTool(ledger=ledger)
    ledger = ledger or spawn._ledger

    dispatch_id = "dsp_" + uuid.uuid4().hex[:12]
    handle = _new_handle(dispatch_id, len(specs), degree, ledger, spawn)
    with _DISPATCH_LOCK:
        _DISPATCH_JOBS[dispatch_id] = dict(handle, task_ids=[])
    _persist_job(str(ledger._dir), dict(handle, task_ids=[]))

    th = threading.Thread(
        target=_run_dispatch_job,
        args=(dispatch_id, [dict(s) for s in specs], degree, spawn, ledger),
        name=f"dispatch-{dispatch_id}", daemon=True,
    )
    th.start()
    return handle


def poll_dispatch(dispatch_id: str, ledger: Optional[TaskLedger] = None,
                  tasks_dir: Optional[str] = None) -> Dict[str, Any]:
    """Poll a background dispatch: returns its state + current pointer records.

    state == 'running'  → records hold whatever has settled so far (partial);
                          poll again later.
    state == 'complete' → records are the FINAL bounded PointerRecords, once.
    """
    ledger = ledger or TaskLedger(tasks_dir or get_tasks_dir())
    with _DISPATCH_LOCK:
        job = _DISPATCH_JOBS.get(dispatch_id)
    if job is None:
        job = _load_jobs(str(ledger._dir)).get(dispatch_id)
    if job is None:
        return {"dispatch_id": dispatch_id, "state": "unknown",
                "error": f"no such dispatch: {dispatch_id}"}

    out = dict(job)
    if job.get("state") == _JOB_COMPLETE:
        out["records"] = job.get("records") or []
    else:
        recs, ids = _live_records(job, ledger)
        out["records"] = [r.to_dict() for r in recs]
        out["settled"] = ids
    return out


def dispatch_many(
    specs: Optional[List[Dict[str, Any]]] = None,
    degree: int = MAX_CONCURRENT,
    spawn: Optional[SpawnWorkerTool] = None,
    ledger: Optional[TaskLedger] = None,
    timeout_s: Optional[int] = None,
    poll_interval: float = POLL_INTERVAL_S,
    max_wait_s: float = DEFAULT_MAX_WAIT_S,
    wait: bool = False,
) -> Any:
    """ASYNC by default: launch N workers in the background, return a handle.

    The background daemon thread runs the SAME bounded fan-out (≤ degree
    concurrent, ≥ N tasks scheduled), judging each REPORTED task so D9 slots
    recycle. Returns the handle immediately so the parent keeps control. Retrieve
    the bounded PointerRecords with a later `poll_dispatch(dispatch_id)`.

    Pass `wait=True` for the LEGACY blocking behaviour (returns the list of
    PointerRecords once all tasks settle) — used by tests and any caller that
    explicitly wants to block.
    """
    specs = specs or []
    if spawn is None:
        spawn = SpawnWorkerTool(ledger=ledger)
    ledger = ledger or spawn._ledger
    if wait:
        return _dispatch_many_blocking(
            specs, degree=degree, spawn=spawn, ledger=ledger,
            timeout_s=timeout_s, poll_interval=poll_interval,
            max_wait_s=max_wait_s,
        )
    return start_dispatch_many(specs, degree=degree, spawn=spawn, ledger=ledger)


def dispatch_many_blocking(
    specs: List[Dict[str, Any]],
    degree: int = MAX_CONCURRENT,
    spawn: Optional[SpawnWorkerTool] = None,
    ledger: Optional[TaskLedger] = None,
    timeout_s: Optional[int] = None,
    poll_interval: float = POLL_INTERVAL_S,
    max_wait_s: float = DEFAULT_MAX_WAIT_S,
) -> List[PointerRecord]:
    """LEGACY synchronous dispatch — blocks until every task settles.

    Kept for callers that explicitly want the old behaviour and for the
    in-repo unit tests. New code should use `dispatch_many` (async) +
    `poll_dispatch`.
    """
    return _dispatch_many_blocking(
        specs, degree=degree, spawn=spawn, ledger=ledger,
        timeout_s=timeout_s, poll_interval=poll_interval, max_wait_s=max_wait_s,
    )


def format_pointers(records: List[PointerRecord]) -> str:
    """One bounded line per record — the parent-facing reunite summary."""
    if not records:
        return "dispatch_many: no specs."
    lines = [f"dispatched {len(records)} task(s):"]
    for r in records:
        mark = "PASS" if r.passed else "FAIL"
        loc = r.output_ptr or "(no output_ptr)"
        lines.append(
            f"- [{mark}] task={r.task_id or '-'} status={r.status} "
            f"ptr={loc}" + (f"  note: {r.note}" if r.note else "")
        )
    return "\n".join(lines)


class DispatchManyTool(Tool):
    """Parent-side fan-out dispatcher. Parent-only.

    Spawns N workers (bounded concurrency) and reunites with POINTER records so
    the parent context is not blown up by worker reports.
    """

    name = "dispatch_many"
    description = (
        "Fan out N independent task specs and reunite with bounded POINTER "
        "records (task_id/status/output_ptr + a short note) — never worker "
        "text. ASYNC by default: it starts the fan-out in the background and "
        "returns a dispatch_id IMMEDIATELY (control returns to you; no blocking "
        "on worker completion). Poll the result later with "
        "action='poll' dispatch_id=<id>; poll again until state='complete'. Runs "
        "up to `degree` workers concurrently (clamped to the hard cap) and judges "
        "each as it reports, so slots recycle. Pass wait=true only if you truly "
        "want to block for the legacy synchronous result. Use for parallelizable "
        "work where each task writes its own output path."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "dispatch (default) = start an async fan-out and return a "
                    "dispatch_id; poll = fetch the bounded pointer records for a "
                    "dispatch_id (call repeatedly until state='complete')."
                ),
                "enum": ["dispatch", "poll"],
            },
            "dispatch_id": {
                "type": "string",
                "description": "Required for action='poll': the id returned by dispatch.",
            },
            "specs": {
                "type": "array",
                "description": (
                    "List of task specs. Each spec: {goal, constraints, "
                    "acceptance, inputs, output_ptr, deadline_minutes} — the "
                    "same fields as spawn_worker."
                ),
                "items": {"type": "object"},
            },
            "degree": {
                "type": "integer",
                "description": (
                    "Max workers to run concurrently (default = hard cap). "
                    "Clamped to the hard cap; it is never raised."
                ),
            },
            "wait": {
                "type": "boolean",
                "description": (
                    "Legacy blocking mode: wait for every task to settle and "
                    "return the pointer records directly. Default false (async)."
                ),
            },
        },
        "required": [],
    }

    def __init__(self, ledger: Optional[TaskLedger] = None,
                 tasks_dir: Optional[str] = None,
                 spawn: Optional[SpawnWorkerTool] = None,
                 session_dir: Optional[str] = None):
        self._ledger = ledger or TaskLedger(tasks_dir or get_tasks_dir())
        self._spawn = spawn
        # Thread the parent's session dir into the fan-out so every worker
        # dispatched here nests under <session_dir>/subagents/<task_id>, exactly
        # as a directly-spawned worker does.
        if self._spawn is None and session_dir:
            self._spawn = SpawnWorkerTool(ledger=self._ledger,
                                          session_dir=session_dir)

    def execute(self, specs: Optional[List[Dict[str, Any]]] = None,
                degree: int = MAX_CONCURRENT, action: str = "dispatch",
                dispatch_id: str = "", wait: bool = False, **kwargs) -> str:
        if os.environ.get("FLAGSCALE_TASK_ID"):
            return ("ERROR: dispatch_many is parent-only (a worker cannot fan "
                    "out).")
        action = (action or "dispatch").strip().lower()

        if action == "poll":
            if not dispatch_id:
                return "ERROR: action='poll' requires a dispatch_id."
            info = poll_dispatch(dispatch_id, ledger=self._ledger)
            return _format_poll(info)

        if action != "dispatch":
            return f"ERROR: unknown action {action!r} (expected dispatch / poll)."

        specs = specs or []
        if not isinstance(specs, list) or not specs:
            return "ERROR: dispatch_many requires a non-empty specs list."

        if wait:
            records = _dispatch_many_blocking(
                specs, degree=degree, spawn=self._spawn, ledger=self._ledger,
            )
            return format_pointers(records)

        handle = start_dispatch_many(
            specs, degree=degree, spawn=self._spawn, ledger=self._ledger,
        )
        return (
            f"dispatched {handle['n_specs']} task(s) in the background "
            f"(degree={handle['degree']}).\n"
            f"dispatch_id: {handle['dispatch_id']}\n"
            "Control returned to you now — workers run in the background and "
            "each reported task is judged as it settles.\n"
            f"Poll with: dispatch_many(action='poll', "
            f"dispatch_id='{handle['dispatch_id']}') until state='complete'."
        )


def _format_poll(info: Dict[str, Any]) -> str:
    """Render a poll_dispatch payload as bounded, pointer-shaped text."""
    state = info.get("state")
    did = info.get("dispatch_id", "")
    if state == "unknown":
        return f"ERROR: {info.get('error', 'no such dispatch')} ({did})"
    recs = info.get("records") or []
    if state != "complete":
        lines = [f"dispatch {did}: state=running "
                 f"(settled {len(recs)}/{info.get('n_specs', '?')} so far)"]
        for d in recs:
            # PROVISIONAL: the partial view derives `passed` from status==DONE
            # only — the authoritative PASS/FAIL runs the acceptance check and
            # arrives with state='complete'. Do not bank a partial PASS.
            lines.append(f"- [~{(d.get('status') or '?')}] "
                         f"task={d.get('task_id') or '-'} "
                         f"ptr={d.get('output_ptr') or '(none)'}")
        lines.append("these are PROVISIONAL (status only); poll again until "
                     "state='complete' for acceptance-checked PASS/FAIL.")
        return "\n".join(lines)
    # complete: full bounded pointer records. Surface a job-level error FIRST —
    # a crash in the background thread must never masquerade as a clean
    # "dispatched 0 task(s)" success (the silent-success failure class).
    err = (info.get("error") or "").strip()
    if err:
        lines = [f"dispatch {did}: state=complete — JOB ERROR: {err}",
                 f"- dispatched {len(recs)} task(s) before the failure:"]
    else:
        lines = [f"dispatch {did}: state=complete — "
                 f"dispatched {len(recs)} task(s):"]
    for d in recs:
        mark = "PASS" if d.get("passed") else "FAIL"
        loc = d.get("output_ptr") or "(no output_ptr)"
        note = d.get("note") or ""
        lines.append(f"- [{mark}] task={d.get('task_id') or '-'} "
                     f"status={d.get('status')} ptr={loc}"
                     + (f"  note: {note}" if note else ""))
    return "\n".join(lines)
