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

"""Parent-side spawn: fork a worker subprocess + watchdog (M2, design §2.2/§2.3).

This is the only place that creates a worker process. Two hard rules govern it:

  * INV1 — a worker must NEVER spawn another worker. Enforced two ways: the
    parent's env never carries FLAGSCALE_TASK_ID for itself (so the check can
    tell parent from worker), and every spawned child DOES carry it, so any
    spawn_worker call inside a worker is refused.
  * The child must NEVER inherit the agent's tty. `stdin=DEVNULL` +
    `start_new_session=True` keeps the worker's fd0 off the REPL's tty (the
    exact class of bug fixed by commits 5c15d3b / 8620cb0). start_new_session
    additionally makes the child a session/process-group leader, so a single
    os.killpg() reaps the whole worker tree.
"""

from __future__ import annotations

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

# D9 — concurrency cap: constant to start, config-driven at M6.
MAX_CONCURRENT = 2
# D10 — depth cap: M2 forbids workers from spawning, so depth is always 1.
MAX_DEPTH = 1
# Watchdog poll cadence (seconds).
WATCH_INTERVAL = 5.0


def _render_contract(c: Contract) -> str:
    """Render a contract into a self-sufficient worker prompt.

    Contains goal + constraints + acceptance + inputs + output_ptr + deadline +
    task_id — and NOTHING of the parent session's history (design §2.2).
    """
    cons = c.constraints or {}
    lines: List[str] = []
    lines.append(f"# 任务契约 task_id={c.id}")
    lines.append("")
    lines.append("## 目标 goal")
    lines.append(c.goal)
    lines.append("")
    lines.append("## 约束 constraints")
    writable = cons.get("writable") or []
    forbidden = cons.get("forbidden") or []
    lines.append(f"- writable(可写目录): {', '.join(map(str, writable)) or '(none)'}")
    lines.append(f"- forbidden(禁止): {', '.join(map(str, forbidden)) or '(none)'}")
    if cons.get("max_minutes") is not None:
        lines.append(f"- max_minutes(时限): {cons['max_minutes']}")
    lines.append("")
    lines.append("## 验收标准 acceptance (父端会逐条重跑，自报不算数)")
    for i, a in enumerate(c.acceptance, 1):
        kind = a.get("kind", "check_command")
        cwd = a.get("cwd", "")
        extra = f" cwd={cwd}" if cwd else ""
        lines.append(f"{i}. [{kind}{extra}] {a.get('check', '')}")
    lines.append("")
    lines.append("## 输入 inputs")
    if c.inputs:
        for inp in c.inputs:
            lines.append(f"- {inp.get('kind', 'value')}: {inp.get('value', '')}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## 产物 output_ptr (必须写到这个路径)")
    lines.append(c.output_ptr)
    lines.append("")
    dl = datetime.fromtimestamp(c.deadline_epoch, tz=timezone.utc).isoformat()
    lines.append(f"## deadline (UTC): {dl}  (epoch={c.deadline_epoch})")
    lines.append("")
    lines.append(
        "完成后【必须】调用 report_result 工具上报 summary；"
        "严禁调用 spawn_worker（worker 不能再派 worker）。"
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
    """Spawn one worker subprocess against a freshly-minted contract (M2)."""

    name = "spawn_worker"
    description = (
        "派生一个 worker 子进程执行一个自足的任务契约，返回 task_id。"
        "契约必须包含 goal/constraints/acceptance/output_ptr/deadline_minutes。"
        "worker 独立进程运行、不继承父 REPL 的 tty；父端 watchdog 负责超时回收。"
        "worker 完成后由父端重跑 acceptance 验收。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": f"任务目标，一句话（<=200 字符）",
            },
            "constraints": {
                "type": "object",
                "description": (
                    "约束。约定键: writable(list[绝对目录]，output_ptr 必须落在其中之一), "
                    "forbidden(list[str]), max_minutes(number)"
                ),
            },
            "acceptance": {
                "type": "array",
                "description": (
                    "验收条目列表，父端会逐条重跑。每项: "
                    "{kind:'check_command', check:'<shell 命令>', cwd:'<可选>'}。"
                    "check 以退出码 0 视为通过。"
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
                "description": "输入列表。每项: {kind:'path'|'value', value:'...'}",
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
                "description": "产物必须写到的绝对路径（必须在 constraints.writable 内）。",
            },
            "deadline_minutes": {
                "type": "number",
                "description": "任务时限（分钟），到点 watchdog 会 killpg 回收。",
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
        # ── 1. role check (INV1) ─────────────────────────────────────────────
        # A worker's env carries FLAGSCALE_TASK_ID; a worker must never spawn.
        if os.environ.get("FLAGSCALE_TASK_ID"):
            return (
                "ERROR: spawn_worker 在 worker 内被禁止 (INV1)。"
                f"当前进程已是 worker (task_id={os.environ['FLAGSCALE_TASK_ID']})，"
                "worker 不能再派 worker。请自行完成任务并调用 report_result。"
            )

        # ── 2. depth check (D10) ─────────────────────────────────────────────
        try:
            want_depth = int(os.environ.get("FLAGSCALE_TASK_DEPTH", "1") or "1") + 1
        except ValueError:
            want_depth = 2
        # In M2 the parent is depth 0; the child it spawns is depth 1. If the
        # caller env already advertises a depth >= MAX_DEPTH we refuse.
        try:
            cur_depth = int(os.environ.get("FLAGSCALE_TASK_DEPTH", "0") or "0")
        except ValueError:
            cur_depth = 0
        if cur_depth >= MAX_DEPTH:
            return (
                f"ERROR: depth 超限 (D10)。当前 depth={cur_depth}, MAX_DEPTH={MAX_DEPTH}; "
                "本里程碑禁止 worker 再派 worker。"
            )
        child_depth = cur_depth + 1

        # ── 3. concurrency check (D9) ────────────────────────────────────────
        try:
            active = self._ledger.active_ids()
        except Exception as e:  # ledger unreadable
            return f"ERROR: 读取账本失败: {e}"
        if len(active) >= MAX_CONCURRENT:
            return (
                f"ERROR: 并发槽位已满 (D9)。active={len(active)} >= MAX_CONCURRENT="
                f"{MAX_CONCURRENT}; 现有任务: {', '.join(active)}。"
                "请先用 poll_tasks 回收已完成的任务。"
            )

        # ── 4. build + validate the contract (INV2: content-addressed id) ────
        cons = dict(constraints or {})
        try:
            dm = float(deadline_minutes)
        except (TypeError, ValueError):
            return f"ERROR: deadline_minutes 非法: {deadline_minutes!r}"
        if dm <= 0:
            return "ERROR: deadline_minutes 必须为正数（分钟）。"
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
                parent={},
            )
            c.validate(check_inputs_exist=True)
        except ContractError as e:
            return f"ERROR: 契约校验失败: {e}"

        # ── 5. create the ledger entry (INV2 dedup) ──────────────────────────
        try:
            tdir = self._ledger.create(c, check_inputs_exist=True)
        except DuplicateTask as e:
            return (
                f"ERROR: 重复任务（相同 goal/constraints/acceptance 已有活跃实例）: {e}。"
                "如需重跑请修改契约（内容寻址会生成新 id）。"
            )
        except ContractError as e:
            return f"ERROR: 契约校验失败: {e}"

        # ── 6. R1: long contract goes to a FILE; argv carries only the path ──
        contract_path = Path(tdir) / "contract.prompt"
        try:
            contract_path.write_text(_render_contract(c), encoding="utf-8")
        except Exception as e:
            self._safe_fail(c.id, f"写契约文件失败: {e}")
            return f"ERROR: 写契约文件失败: {e}"

        # ── 7. spawn (tty-safe) ──────────────────────────────────────────────
        env = self._build_env(c)
        if _env:
            env.update(_env)
        env["FLAGSCALE_CONTRACT_PATH"] = str(contract_path)
        log_path = Path(tdir) / "worker.log"
        try:
            log_fh = open(log_path, "w", encoding="utf-8")
        except Exception as e:
            self._safe_fail(c.id, f"打开 worker.log 失败: {e}")
            return f"ERROR: 打开 worker.log 失败: {e}"

        # NOTE: typer requires OPTIONS before the positional `query` arg —
        # `flagscale-agent <path> --time-budget-sec N` fails with
        # "No such command '--time-budget-sec'". Flag MUST come first.
        argv = [self._agent_bin, "--time-budget-sec", str(int(dm * 60)),
                str(contract_path)]
        try:
            proc = subprocess.Popen(argv, **self._popen_kwargs(env, log_fh))
        except Exception as e:
            log_fh.close()
            self._safe_fail(c.id, f"Popen 失败: {e}")
            return f"ERROR: 派生子进程失败: {e}"

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
            return f"ERROR: 状态迁移失败: {e}"

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
