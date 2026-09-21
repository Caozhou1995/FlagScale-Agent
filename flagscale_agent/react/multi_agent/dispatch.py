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

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flagscale_agent.react.paths import get_tasks_dir
from flagscale_agent.react.tools.base import Tool

from .contract import compute_id
from .ledger import TaskLedger
from .reunite import check_result
from .spawn import MAX_CONCURRENT, SpawnWorkerTool

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


def dispatch_many(
    specs: List[Dict[str, Any]],
    degree: int = MAX_CONCURRENT,
    spawn: Optional[SpawnWorkerTool] = None,
    ledger: Optional[TaskLedger] = None,
    timeout_s: Optional[int] = None,
    poll_interval: float = POLL_INTERVAL_S,
    max_wait_s: float = DEFAULT_MAX_WAIT_S,
) -> List[PointerRecord]:
    """Fan out N self-contained specs and reunite with POINTER records.

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
                log_path=str(ledger.task_dir(tid) / "worker.log"),
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
                log_path=str(ledger.task_dir(tid) / "worker.log"),
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
                    log_path=str(ledger.task_dir(tid) / "worker.log"),
                    note="still running at dispatch deadline",
                )
            break
        time.sleep(poll_interval)

    return [r for r in records]  # type: ignore[return-value]


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
        "text. Runs up to `degree` workers concurrently (clamped to the hard cap) "
        "and judges each as it reports, so slots recycle. Use for parallelizable "
        "work where each task writes its own output path."
    )
    parameters = {
        "type": "object",
        "properties": {
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
        },
        "required": ["specs"],
    }

    def __init__(self, ledger: Optional[TaskLedger] = None,
                 tasks_dir: Optional[str] = None,
                 spawn: Optional[SpawnWorkerTool] = None):
        self._ledger = ledger or TaskLedger(tasks_dir or get_tasks_dir())
        self._spawn = spawn

    def execute(self, specs: Optional[List[Dict[str, Any]]] = None,
                degree: int = MAX_CONCURRENT, **kwargs) -> str:
        if os.environ.get("FLAGSCALE_TASK_ID"):
            return ("ERROR: dispatch_many is parent-only (a worker cannot fan "
                    "out).")
        specs = specs or []
        if not isinstance(specs, list) or not specs:
            return "ERROR: dispatch_many requires a non-empty specs list."
        records = dispatch_many(
            specs, degree=degree, spawn=self._spawn, ledger=self._ledger,
        )
        return format_pointers(records)
