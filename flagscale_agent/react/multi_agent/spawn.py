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
import re
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flagscale_agent.react.paths import get_tasks_dir
from flagscale_agent.react.tools.base import Tool

from .contract import (
    REVIEWER_MIN_MINUTES,
    ROLE_RE_REVIEWER,
    Contract,
    ContractError,
    is_reviewer_role,
    resolve_role,
    role_spec,
)
from .ledger import (
    ACTIVE_STATUSES,
    DEADLINE_MISSED,
    FAILED,
    REPORTED,
    RUNNING,
    DuplicateTask,
    LedgerError,
    TaskLedger,
)

# Concurrency cap (constant for now).
MAX_CONCURRENT = 10

# The nesting dir under a parent's session dir holding its children's nested
# session dirs (and, since this commit, their worker.log traces).
SUBAGENTS_DIRNAME = "subagents"
# Depth cap default. The EFFECTIVE cap is read per spawn from the env key
# FLAGSCALE_MAX_DEPTH (default DEFAULT_MAX_DEPTH) — a table-driven constant the
# agent has no tool to raise. It is clamped into [MIN_MAX_DEPTH, HARD_MAX_DEPTH]
# so a corrupt env value cannot disable the invariant.
DEFAULT_MAX_DEPTH = 2
MIN_MAX_DEPTH = 1
HARD_MAX_DEPTH = 8
# Watchdog poll cadence (seconds).
WATCH_INTERVAL = 5.0

# ── RoleSpec: reviewer auto-injection + preset ───────────────────────────────
# "the goal is a review" — word-bounded so 'preview' never matches. This
# predicate only TRIGGERS injection when the caller supplied NO explicit role
# tag; suppression is keyed on the explicit tag (presence of `reviewer` or
# `role` in constraints), never on re-detecting prose.
_REVIEW_GOAL_RE = re.compile(r"\b(re-?review|review|audit)\b", re.IGNORECASE)


def _is_review_goal(goal: str) -> bool:
    """True when the goal reads like a review task (word-bounded)."""
    return bool(_REVIEW_GOAL_RE.search(goal or ""))


def maybe_auto_inject_reviewer(goal: str, constraints: Dict[str, Any]) -> None:
    """[b56f8276] Auto-inject `constraints.reviewer = True` for a review goal.

    Injected IN PLACE, and only when the caller left no explicit role tag:
    presence of either `reviewer` or `role` (whatever its value, including
    False) suppresses injection — the explicit tag is the only suppression
    signal.
    """
    if not isinstance(constraints, dict):
        return
    if "reviewer" in constraints or "role" in constraints:
        return
    if _is_review_goal(goal):
        constraints["reviewer"] = True


def _has_rev_marker(constraints: Dict[str, Any],
                    inputs: Optional[List[Dict[str, Any]]]) -> bool:
    """A revision marker: constraints['rev'] non-empty, or a value input
    shaped `rev=<non-empty>`."""
    if str(constraints.get("rev", "") or "").strip():
        return True
    for inp in inputs or []:
        if not isinstance(inp, dict):
            continue
        if inp.get("kind") == "value":
            val = str(inp.get("value", "") or "")
            if re.match(r"^\s*rev\s*=\s*\S+", val, re.IGNORECASE):
                return True
    return False


def apply_reviewer_preset(goal: str, constraints: Dict[str, Any],
                          acceptance: List[Dict[str, Any]], output_ptr: str,
                          deadline_minutes: float,
                          inputs: Optional[List[Dict[str, Any]]] = None,
                          ) -> tuple:
    """[81c81515 + 91aab676] Apply the reviewer preset to a spawn request.

    For a reviewer-class role:
      * raise ContractError unless `inputs` carry the deliverable abs path
        (a kind=path input) AND a revision marker (constraints['rev'] or a
        `rev=...` value input);
      * raise the deadline floor to REVIEWER_MIN_MINUTES (10);
      * inject acceptance checks (the acceptance-suggestion half of 91aab676):
        the report is non-empty, carries a per-finding table or an explicit
        'no findings' line, and has no dangling '(IN PROGRESS' header.
    Mutates `constraints` (max_minutes bump) and returns
    (acceptance, deadline_minutes). Non-reviewer roles pass through untouched.
    """
    spec = role_spec(constraints)
    if not spec.is_review:
        return list(acceptance or []), deadline_minutes
    acc = list(acceptance or [])
    ins = list(inputs or [])

    # inputs must carry the deliverable abs path + rev
    path_inputs = [i for i in ins
                   if isinstance(i, dict) and i.get("kind") == "path"
                   and isinstance(i.get("value"), str)
                   and i["value"].startswith("/")]
    if not path_inputs:
        raise ContractError(
            "reviewer preset: inputs must carry the deliverable's absolute "
            "path — add {\"kind\": \"path\", \"value\": \"<abs path>\"}."
        )
    if not _has_rev_marker(constraints, ins):
        raise ContractError(
            "reviewer preset: inputs must carry the revision under review — "
            "add {\"kind\": \"value\", \"value\": \"rev=<git sha|mtime>\"} or "
            "constraints.rev."
        )

    # deadline floor
    if deadline_minutes < spec.min_minutes:
        deadline_minutes = spec.min_minutes
    try:
        prev = float(constraints.get("max_minutes"))
    except (TypeError, ValueError):
        prev = None
    if prev is None or prev < spec.min_minutes:
        constraints["max_minutes"] = spec.min_minutes

    # acceptance suggestions (91aab676): report-completeness as checks the
    # parent actually runs. The completeness item calls the SAME helper the
    # write-time gate uses (check_report_completeness) so the two can never
    # drift apart; flagscale_agent is importable from any cwd in the parent
    # environment (verified), so the python -c form is safe.
    existing = {(a or {}).get("check") for a in acc if isinstance(a, dict)}
    injected = [
        {"kind": "check_command", "check": f"test -s {output_ptr}"},
        {"kind": "check_command",
         "check": f"python -c \"import sys; from "
                  f"flagscale_agent.react.multi_agent.contract import "
                  f"check_report_completeness; p = "
                  f"check_report_completeness({output_ptr!r}); "
                  f"sys.exit(0 if p is None else 1)\""},
    ]
    for item in injected:
        if item["check"] not in existing:
            acc.append(item)
    return acc, deadline_minutes


def _path_fingerprint(path: str) -> Optional[str]:
    """md5+mtime of a FILE path input, or None (dirs/unreadable are skipped).

    Why: a reviewer is handed a diff/text snapshot of a LIVE worktree. If the
    worktree is edited again after dispatch, the reviewer cites a hunk the
    running system never had — a false finding. Stamping the exact bytes'
    fingerprint into the contract lets the reviewer (and the parent) detect
    drift by re-hashing the same path before trusting a citation.
    """
    try:
        p = Path(path)
        if not p.is_file():
            return None
        import hashlib
        h = hashlib.md5()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        st = p.stat()
        return f"md5={h.hexdigest()} mtime={int(st.st_mtime)} size={st.st_size}"
    except OSError:
        return None


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
    # Reviewer detection: EXPLICIT role tags, via resolve_role() — the same
    # resolver the write-time gate and the preset use, so a role-tag-only
    # contract ({"role": "re_reviewer"} with no boolean flag) gets the same
    # discipline lines. A bare `cons.get("reviewer")` here would silently
    # skip role-only re-reviewers even though report_result gates them.
    if is_reviewer_role(cons):
        lines.append("")
        lines.append("## Reviewer discipline")
        lines.append(
            "Your report must be COMPLETE when you call report_result: a "
            "non-empty report containing >=1 finding (one markdown table row "
            "per finding, e.g. `| F1 | bug | fix |`) or an explicit "
            "'no findings' line, and NO dangling '(IN PROGRESS' header. An "
            "unfinished report is REJECTED at report_result — finalize it "
            "before reporting."
        )
        if resolve_role(cons) == ROLE_RE_REVIEWER:
            lines.append(
                "RE-REVIEW ROLE: verify ONLY that each previously reported "
                "finding is fixed in the diff — do not re-review the whole "
                "artifact. State a per-finding verdict (fixed / not fixed / "
                "partially fixed) in the table; the deadline and input preset "
                "are the same as a full review."
            )
        lines.append(
            "Your file:line citations are LEADS, not evidence: the parent "
            "re-verifies each one before acting. Cite exactly what you saw "
            "(quote <=1 line + number), never from memory; if you could not "
            "open the cited location, say so."
        )
        lines.append(
            "If the artifact may change during your review, note the revision "
            "you reviewed (git rev / mtime / line count) in your report; "
            "findings against a stale revision must say so."
        )
        # Diff-of-diffs: the contract stamps a fingerprint for any file path
        # input so the reviewer can detect a worktree that was edited AFTER
        # dispatch (the source of false findings against a stale snapshot).
        fp_rows = []
        for inp in c.inputs or []:
            if isinstance(inp, dict) and inp.get("kind") == "path":
                fp = _path_fingerprint(str(inp.get("value", "")))
                if fp:
                    fp_rows.append((str(inp.get("value")), fp))
        if fp_rows:
            lines.append(
                "REVISION PINNING: the lines below are the exact bytes of the "
                "path(s) you were handed, stamped at dispatch. If an artifact "
                "may change mid-review, RE-HASH the same path before trusting "
                "it — `md5sum <path>` must equal the stamp. On a mismatch, your "
                "citations address a stale revision the running system never "
                "had: re-read the file and re-run the diff before citing, and "
                "mark any finding you cannot re-confirm as 'possibly stale'."
            )
            for pth, fp in fp_rows:
                lines.append(f"  - {pth} :: {fp}")
        lines.append(
            "For a follow-up review: your session env exposes "
            "FLAGSCALE_SESSION_ROOT/ID of the parent — recall_search the "
            "parent conversation log for prior findings and state whether "
            "each prior finding was fixed, plus your own new findings."
        )
    lines.append("")
    lines.append(
        "## Citation requirement"
    )
    lines.append(
        "Every file:line reference in your report MUST be accompanied by the "
        "verbatim source text of that single line (quote it, <=1 line). The "
        "parent re-checks against the quoted source text with a grep, not "
        "against your line number — a stale or off-by-one line number that "
        "contradicts the quoted text is treated as unverified. If a claim "
        "rests on a symbol that does not exist, say so explicitly."
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
                    "(list[str]), max_minutes (number), reviewer (bool; set "
                    "true for a read-only review task so its contract carries "
                    "the reviewer-discipline lines)."
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
                 tasks_dir: Optional[str] = None,
                 session_dir: Optional[str] = None):
        self._agent_bin = (agent_bin or os.environ.get("FLAGSCALE_AGENT_BIN")
                           or "flagscale-agent")
        self._ledger = ledger or TaskLedger(tasks_dir or get_tasks_dir())
        # The caller's OWN session dir. When set, each spawned worker nests its
        # session under <session_dir>/subagents/<task_id> (see _build_env); when
        # None the child falls back to the global default sessions root.
        self._session_dir = session_dir

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
        # Nested session home: the worker's own session dir becomes a CHILD of
        # the parent's — <parent_session_dir>/subagents/<task_id>. The child
        # agent reads FLAGSCALE_SESSION_ROOT/FLAGSCALE_SESSION_ID when its config
        # has no explicit session_dir. memory/proposals are NOT touched (they key
        # on FLAGSCALE_HOME): only the session dir nests.
        if self._session_dir:
            env["FLAGSCALE_SESSION_ROOT"] = str(
                Path(self._session_dir) / SUBAGENTS_DIRNAME)
            env["FLAGSCALE_SESSION_ID"] = c.id
        return env

    def worker_log_path(self, task_id: str) -> Path:
        """The worker's stdout/stderr log file for `task_id`.

        The log is a PER-AGENT trace (the worker's own ReAct console), so it
        lives in the worker's NESTED SESSION dir —
        <parent_session>/subagents/<task_id>/worker.log — next to its
        conversation/plans, not in the global tasks/ ledger. The ledger dir
        keeps only the cross-task audit records (contract/state/result).
        Fallback: without a known parent session dir (direct tool use in
        tests / legacy callers), the ledger task dir is used.
        """
        if self._session_dir:
            return (Path(self._session_dir) / SUBAGENTS_DIRNAME / task_id
                    / "worker.log")
        return Path(self._ledger.task_dir(task_id)) / "worker.log"

    def recorded_worker_log_path(self, task_id: str) -> Path:
        """The log path FROZEN at spawn time (task-scoped), else recomputed.

        A resume/adoption may run under a different session dir than the
        original spawn; appending must continue the ORIGINAL trace, so prefer
        the path recorded on the task and only fall back to recomputation for
        tasks spawned before this field existed.
        """
        rec = Path(self._ledger.task_dir(task_id)) / "worker_log_path"
        try:
            if rec.exists():
                p = rec.read_text(encoding="utf-8").strip()
                if p:
                    return Path(p)
        except Exception:
            pass
        return self.worker_log_path(task_id)

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

        # ── 4b. RoleSpec: auto-inject + reviewer preset (before freezing) ────
        # [b56f8276] auto-inject the explicit reviewer tag for a review goal
        # (an explicit `reviewer`/`role` tag suppresses injection).
        maybe_auto_inject_reviewer(goal, cons)
        # [81c81515 + 91aab676] apply the preset: deliverable+rev inputs,
        # deadline floor, acceptance-completeness injection.
        try:
            acceptance_list, dm = apply_reviewer_preset(
                goal, cons, list(acceptance or []), output_ptr, dm,
                inputs=list(inputs or []))
        except ContractError as e:
            return f"ERROR: contract validation failed: {e}"
        deadline_epoch = int(time.time()) + int(dm * 60)

        try:
            c = Contract.build(
                goal=goal,
                constraints=cons,
                acceptance=list(acceptance_list or []),
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
        log_path = self.worker_log_path(c.id)
        # Freeze the log location on the TASK, not on the caller's session: a
        # later resume/adoption may run under a DIFFERENT session dir, and it
        # must append to THIS file so the audit trail is not split. Best-effort.
        try:
            (Path(tdir) / "worker_log_path").write_text(
                str(log_path), encoding="utf-8")
        except Exception:
            pass
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "w", encoding="utf-8")
        except Exception as e:
            self._safe_fail(c.id, f"failed to open worker.log: {e}")
            return f"ERROR: failed to open worker.log: {e}"

        # Two-line header: a worker.log read in isolation (tail, artifacts
        # export, forensic sweep) must self-identify its task. Written and
        # flushed BEFORE Popen — the child inherits the shared
        # open-file-description offset, so its output continues AFTER the
        # header rather than overwriting it.
        log_fh.write(f"# task_id: {c.id}\n# contract: {contract_path}\n")
        log_fh.flush()

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
        # The child holds its own dup of the log fd; close the parent's copy so
        # the descriptor does not leak in the (long-lived) parent process.
        log_fh.close()

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
