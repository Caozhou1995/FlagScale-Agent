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

"""Parent-side spawn: fork a worker subprocess + watchdog.

This is the only place that creates a worker process. Two hard rules govern it:

  * A worker may spawn another worker only while UNDER the depth cap. The cap
    is a table-driven constant (env FLAGSCALE_MAX_DEPTH, default
    DEFAULT_MAX_DEPTH) that the agent has no tool to raise. Every spawn
    computes child_depth = own_depth + 1 and refuses EXPLICITLY (with the depth
    and the parent trace in the message) once own_depth >= the cap. The
    parent's env never carries FLAGSCALE_TASK_ID for itself; every spawned
    child DOES carry it, which is how a process knows it is a worker.
  * The child must NEVER inherit the agent's tty. `stdin=DEVNULL` +
    `start_new_session=True` keeps the worker's fd0 off the REPL's tty (the
    exact class of bug fixed by commits 5c15d3b / 8620cb0). start_new_session
    additionally makes the child a session/process-group leader, so a single
    os.killpg() reaps the whole worker tree.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flagscale_agent.react.paths import get_tasks_dir
from flagscale_agent.react.tools.base import Tool

from .contract import Contract, ContractError
from .ledger import (
    ACTIVE_STATUSES,
    DEADLINE_MISSED,
    FAILED,
    RUNNING,
    DuplicateTask,
    LedgerError,
    TaskLedger,
)

# Concurrency cap (constant for now).
MAX_CONCURRENT = 2
# Depth cap default. The EFFECTIVE cap is read per spawn from the env key
# FLAGSCALE_MAX_DEPTH (default DEFAULT_MAX_DEPTH) — a table-driven constant the
# agent has no tool to raise. It is clamped into [MIN_MAX_DEPTH, HARD_MAX_DEPTH]
# so a corrupt env value cannot disable the invariant.
DEFAULT_MAX_DEPTH = 2
MIN_MAX_DEPTH = 1
HARD_MAX_DEPTH = 8
# Watchdog poll cadence (seconds).
WATCH_INTERVAL = 5.0


def _effective_max_depth() -> int:
    """The effective depth cap: env FLAGSCALE_MAX_DEPTH, clamped to safe range."""
    try:
        v = int(os.environ.get("FLAGSCALE_MAX_DEPTH", "") or DEFAULT_MAX_DEPTH)
    except ValueError:
        v = DEFAULT_MAX_DEPTH
    return max(MIN_MAX_DEPTH, min(v, HARD_MAX_DEPTH))


def _render_contract(c: Contract) -> str:
    """Render a contract into a self-sufficient worker prompt.

    Contains goal + constraints + acceptance + inputs + output_ptr + deadline +
    task_id — and NOTHING of the parent session's history (design §2.2).
    """
    cons = c.constraints or {}
    lines: List[str] = []
    lines.append(f"# Task contract task_id={c.id}")
    lines.append("")
    lines.append("## Goal")
    lines.append(c.goal)
    lines.append("")
    lines.append("## Constraints")
    writable = cons.get("writable") or []
    forbidden = cons.get("forbidden") or []
    lines.append(f"- writable (writable dirs): {', '.join(map(str, writable)) or '(none)'}")
    lines.append(f"- forbidden: {', '.join(map(str, forbidden)) or '(none)'}")
    if cons.get("max_minutes") is not None:
        lines.append(f"- max_minutes (time budget): {cons['max_minutes']}")
    lines.append("")
    lines.append("## Acceptance (the parent runs each check independently; "
                 "the self-report does not count)")
    for i, a in enumerate(c.acceptance, 1):
        kind = a.get("kind", "check_command")
        cwd = a.get("cwd", "")
        extra = f" cwd={cwd}" if cwd else ""
        lines.append(f"{i}. [{kind}{extra}] {a.get('check', '')}")
    lines.append("")
    lines.append("## Inputs")
    if c.inputs:
        for inp in c.inputs:
            lines.append(f"- {inp.get('kind', 'value')}: {inp.get('value', '')}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Output output_ptr (you must write to this path)")
    lines.append(c.output_ptr)
    lines.append("")
    dl = datetime.fromtimestamp(c.deadline_epoch, tz=timezone.utc).isoformat()
    lines.append(f"## deadline (UTC): {dl}  (epoch={c.deadline_epoch})")
    lines.append("")
    lines.append(
        "When done you MUST call the report_result tool to report a summary. "
        "You MAY call spawn_worker to delegate part of the work, but ONLY while "
        "under the infrastructure depth cap; a spawn beyond the cap is refused "
        "with an explicit error. You cannot raise the cap."
    )
    return "\n".join(lines)


class _Watchdog(threading.Thread):
    """Parent-side, daemon liveness watcher for one spawned worker (design §2.3).

    A purely observational thread: it never touches the REPL, never blocks
    process exit (daemon=True), and only writes to the ledger + kills the
    worker's process group. That separation is the whole point — the lesson
    from prompt_watchdog is that an observer must not be able to wedge the
    parent.
    """

    def __init__(self, ledger: TaskLedger, task_id: str, pid: int,
                 deadline_epoch: int, interval: float = WATCH_INTERVAL):
        super().__init__(daemon=True, name=f"watchdog-{task_id}")
        self._ledger = ledger
        self._task_id = task_id
        self._pid = pid
        self._deadline_epoch = deadline_epoch
        self._interval = interval

    def _alive(self) -> bool:
        try:
            # start_new_session=True makes the child a process-group leader, so
            # its pgid == pid; signal 0 probes without delivering.
            os.killpg(self._pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but not ours — still alive
        except OSError:
            return False

    def _killpg(self):
        try:
            pgid = os.getpgid(self._pid)
        except OSError:
            pgid = self._pid
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass

    def run(self):
        while True:
            time.sleep(self._interval)
            rec = self._ledger.get(self._task_id)
            if rec is None:
                return
            # Terminal / no-longer-active (REPORTED is still active) → done.
            if rec.status not in ACTIVE_STATUSES:
                return
            alive = self._alive()

            if time.time() >= self._deadline_epoch:
                if rec.status == RUNNING:
                    if alive:
                        self._killpg()
                    try:
                        self._ledger.transition(
                            self._task_id, DEADLINE_MISSED,
                            note="deadline exceeded; worker killed (INV5)",
                        )
                    except LedgerError:
                        pass
                elif rec.status == REPORTED and alive:
                    # Already delivered; just reap a hung process. Its state
                    # (REPORTED) is terminal for the watchdog — the parent
                    # reunite decides DONE/REJECTED.
                    self._killpg()
                return

            if not alive:
                # Process disappeared while state is still RUNNING: it crashed
                # or exited without calling report_result.
                if rec.status == RUNNING:
                    try:
                        self._ledger.transition(
                            self._task_id, FAILED,
                            note="worker exited without report",
                        )
                    except LedgerError:
                        pass
                return


class SpawnWorkerTool(Tool):
    """Spawn one worker subprocess against a freshly-minted contract."""

    name = "spawn_worker"
    description = (
        "Spawn a worker subprocess to execute a self-contained task contract; "
        "returns a task_id. The contract must contain goal/constraints/"
        "acceptance/output_ptr/deadline_minutes. The worker runs as an "
        "independent process and does not inherit the parent REPL's tty; a "
        "parent-side watchdog reaps it on timeout. After the worker reports, "
        "the parent runs the acceptance checks independently."
    )
    parameters = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "Task goal, one sentence (<=200 chars).",
            },
            "constraints": {
                "type": "object",
                "description": (
                    "Constraints. Conventional keys: writable (list of absolute "
                    "dirs; output_ptr must be inside one of them), forbidden "
                    "(list[str]), max_minutes (number)."
                ),
            },
            "acceptance": {
                "type": "array",
                "description": (
                    "List of acceptance items; the parent runs each one "
                    "independently. Each item: "
                    "{kind:'check_command', check:'<shell command>', "
                    "cwd:'<optional>'}. Exit code 0 means pass."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "check": {"type": "string"},
                        "cwd": {"type": "string"},
                    },
                    "required": ["check"],
                },
            },
            "inputs": {
                "type": "array",
                "description": "Input list. Each item: {kind:'path'|'value', value:'...'}",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "value": {"type": "string"},
                    },
                },
            },
            "output_ptr": {
                "type": "string",
                "description": (
                    "Absolute path the artifact must be written to (must be "
                    "inside constraints.writable)."
                ),
            },
            "deadline_minutes": {
                "type": "number",
                "description": (
                    "Task deadline in minutes; on expiry the watchdog kills the "
                    "worker's process group."
                ),
            },
        },
        "required": ["goal", "constraints", "acceptance", "output_ptr", "deadline_minutes"],
    }

    def __init__(self, agent_bin: Optional[str] = None,
                 ledger: Optional[TaskLedger] = None,
                 tasks_dir: Optional[str] = None):
        self._agent_bin = (agent_bin or os.environ.get("FLAGSCALE_AGENT_BIN")
                           or "flagscale-agent")
        self._ledger = ledger or TaskLedger(tasks_dir or get_tasks_dir())

    # ── helpers exposed for testability ──────────────────────────────────────
    def _build_env(self, c: Contract) -> Dict[str, str]:
        env = dict(os.environ)
        env["FLAGSCALE_TASK_ID"] = c.id
        env["FLAGSCALE_TASK_DEPTH"] = str(c.depth)
        env["FLAGSCALE_PARENT_TRACE"] = json.dumps(
            (c.parent or {}).get("parent_trace", []))
        env["FLAGSCALE_CONTRACT_PATH"] = str(self._ledger.task_dir(c.id) / "contract.prompt")
        env["FLAGSCALE_OUTPUT_DIR"] = str(Path(c.output_ptr).parent)
        # The worker resolves its ledger from FLAGSCALE_TASKS_DIR (paths.get_tasks_dir).
        # Propagate the parent's ACTUAL dir so both processes share one ledger even
        # when the parent used a custom/non-default tasks_dir (otherwise report_result
        # in the child cannot find its own task).
        env["FLAGSCALE_TASKS_DIR"] = str(self._ledger._dir)
        return env

    def _popen_kwargs(self, env: Dict[str, str], log_fh) -> Dict[str, Any]:
        # ── THE load-bearing constraint (see module docstring) ───────────────
        # stdin=DEVNULL  → the child's fd0 never touches the REPL tty, so a
        #                  stray stdin read cannot steal keystrokes (5c15d3b).
        # start_new_session=True → child becomes its own session + process
        #                  group leader, so os.killpg() reaps its whole tree
        #                  and it never shares the parent's controlling tty.
        return {
            "stdin": subprocess.DEVNULL,
            "stdout": log_fh,
            "stderr": subprocess.STDOUT,
            "env": env,
            "start_new_session": True,
            "cwd": os.getcwd(),
        }

    # ── main entry ───────────────────────────────────────────────────────────
    def execute(self, goal: str = "", constraints: Optional[Dict[str, Any]] = None,
                acceptance: Optional[List[Dict[str, Any]]] = None,
                inputs: Optional[List[Dict[str, Any]]] = None,
                output_ptr: str = "", deadline_minutes: float = 0,
                task_plan: Any = None, _env: Optional[Dict[str, str]] = None,
                **kwargs) -> str:
        # ── 1. role probe (INV1) ─────────────────────────────────────────────
        # A worker's env carries FLAGSCALE_TASK_ID; the parent's does not. This
        # is a PROBE, not a refusal: being a worker no longer forbids spawning —
        # the depth cap below does. We read the id to build the parent chain and
        # the inherited ancestry so the derivation tree stays auditable.
        own_task_id = os.environ.get("FLAGSCALE_TASK_ID") or None
        try:
            own_trace = json.loads(os.environ.get("FLAGSCALE_PARENT_TRACE", "[]") or "[]")
            if not isinstance(own_trace, list):
                own_trace = []
        except (ValueError, TypeError):
            own_trace = []

        # ── 2. depth check (D10) ─────────────────────────────────────────────
        # The orchestrator is depth 0; the child it spawns is depth 1. A worker
        # at depth d may spawn a child at depth d+1 only while d < max_depth.
        max_depth = _effective_max_depth()
        try:
            cur_depth = int(os.environ.get("FLAGSCALE_TASK_DEPTH", "0") or "0")
        except ValueError:
            cur_depth = 0
        if cur_depth >= max_depth:
            return (
                f"ERROR: depth limit exceeded (D10). current depth={cur_depth}, "
                f"MAX_DEPTH={max_depth}; a process at depth {cur_depth} cannot "
                "spawn (the cap is infrastructure-enforced and cannot be raised "
                f"from here). parent_trace={own_trace}."
            )
        child_depth = cur_depth + 1
        # The child's ancestry = this process's ancestry + this process's own id
        # (present only when this process is itself a worker). The immediate
        # parent is the last element; the whole chain makes the tree auditable.
        child_trace = own_trace + ([own_task_id] if own_task_id else [])

        # ── 3. concurrency check (D9) ────────────────────────────────────────
        try:
            active = self._ledger.active_ids()
        except Exception as e:  # ledger unreadable
            return f"ERROR: failed to read the ledger: {e}"
        if len(active) >= MAX_CONCURRENT:
            return (
                f"ERROR: concurrency slots full (D9). active={len(active)} >= "
                f"MAX_CONCURRENT={MAX_CONCURRENT}; current tasks: "
                f"{', '.join(active)}. Reclaim finished tasks with poll_tasks first."
            )

        # ── 4. build + validate the contract (INV2: content-addressed id) ────
        cons = dict(constraints or {})
        try:
            dm = float(deadline_minutes)
        except (TypeError, ValueError):
            return f"ERROR: invalid deadline_minutes: {deadline_minutes!r}"
        if dm <= 0:
            return "ERROR: deadline_minutes must be a positive number of minutes."
        cons.setdefault("max_minutes", dm)
        deadline_epoch = int(time.time()) + int(dm * 60)

        try:
            c = Contract.build(
                goal=goal,
                constraints=cons,
                acceptance=list(acceptance or []),
                output_ptr=output_ptr,
                inputs=list(inputs or []),
                deadline_epoch=deadline_epoch,
                depth=child_depth,
                parent={
                    "task_id": own_task_id,  # None for the orchestrator
                    "depth": cur_depth,
                    "parent_trace": child_trace,
                },
            )
            c.validate(check_inputs_exist=True)
        except ContractError as e:
            return f"ERROR: contract validation failed: {e}"

        # ── 5. create the ledger entry (INV2 dedup) ──────────────────────────
        try:
            tdir = self._ledger.create(c, check_inputs_exist=True)
        except DuplicateTask as e:
            return (
                f"ERROR: duplicate task (an active instance with the same "
                f"goal/constraints/acceptance already exists): {e}. To re-run, "
                "change the contract (content-addressing mints a new id)."
            )
        except ContractError as e:
            return f"ERROR: contract validation failed: {e}"

        # ── 6. R1: long contract goes to a FILE; argv carries only the path ──
        contract_path = Path(tdir) / "contract.prompt"
        try:
            contract_path.write_text(_render_contract(c), encoding="utf-8")
        except Exception as e:
            self._safe_fail(c.id, f"failed to write contract file: {e}")
            return f"ERROR: failed to write contract file: {e}"

        # ── 7. spawn (tty-safe) ──────────────────────────────────────────────
        env = self._build_env(c)
        if _env:
            env.update(_env)
        env["FLAGSCALE_CONTRACT_PATH"] = str(contract_path)
        log_path = Path(tdir) / "worker.log"
        try:
            log_fh = open(log_path, "w", encoding="utf-8")
        except Exception as e:
            self._safe_fail(c.id, f"failed to open worker.log: {e}")
            return f"ERROR: failed to open worker.log: {e}"

        # NOTE: typer requires OPTIONS before the positional `query` arg —
        # `flagscale-agent <path> --time-budget-sec N` fails with
        # "No such command '--time-budget-sec'". The flag MUST come first.
        argv = [self._agent_bin, "--time-budget-sec", str(int(dm * 60)),
                str(contract_path)]
        try:
            proc = subprocess.Popen(argv, **self._popen_kwargs(env, log_fh))
        except Exception as e:
            log_fh.close()
            self._safe_fail(c.id, f"Popen failed: {e}")
            return f"ERROR: failed to spawn subprocess: {e}"

        # ── 8. SPAWNING → RUNNING, record pid ────────────────────────────────
        try:
            self._ledger.transition(c.id, RUNNING, note="spawned", pid=proc.pid)
        except LedgerError as e:
            # Contract raced to terminal between create and here — rare; kill.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            log_fh.close()
            return f"ERROR: state transition failed: {e}"

        # ── 9. start the parent-side watchdog (daemon; never blocks exit) ────
        wd = _Watchdog(self._ledger, c.id, proc.pid, c.deadline_epoch)
        wd.start()

        return (
            f"spawned task {c.id} pid={proc.pid} depth={c.depth} "
            f"deadline={dm:g}min log={log_path}"
        )

    # ── failure helper ───────────────────────────────────────────────────────
    def _safe_fail(self, task_id: str, note: str):
        """Best-effort move a half-created task to FAILED (SPAWNING→FAILED)."""
        try:
            self._ledger.transition(task_id, FAILED, note=note)
        except Exception:
            pass
