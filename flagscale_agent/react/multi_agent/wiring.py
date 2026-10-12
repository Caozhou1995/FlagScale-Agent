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

"""Worker-side wiring helpers.

Keeps the changes to agent.py minimal and testable: three small functions that
the agent calls, plus the fixed role prefix. Nothing here spawns; nothing
here touches the REPL.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from flagscale_agent.react.paths import get_tasks_dir

from .ledger import FAILED, RUNNING, TaskLedger
from .contract import is_leaf_role_name

# Fixed role prefix for an ORDINARY worker (may delegate under the depth cap).
WORKER_ROLE_PREFIX = (
    "[worker role] You are a worker dispatched by a parent Agent. The contract "
    "has been injected. When done you MUST call report_result to report a "
    "summary. You MAY call spawn_worker to delegate part of the work, but only "
    "while under the infrastructure depth cap; a spawn beyond the cap is "
    "refused with an explicit error and you cannot raise the cap.\n"
)

# Fixed role prefix for a LEAF worker (reviewer/diverger): no delegation.
WORKER_LEAF_ROLE_PREFIX = (
    "[worker role] You are a LEAF worker (reviewer/diverger) dispatched by a "
    "parent Agent. The contract has been injected. When done you MUST call "
    "report_result to report a summary. You may NOT spawn any worker "
    "(spawn_worker / dispatch_many / resume_child are refused for your role); "
    "do the work yourself and report.\n"
)

# Env keys injected by SpawnWorkerTool.
TASK_ID_ENV = "FLAGSCALE_TASK_ID"
CONTRACT_PATH_ENV = "FLAGSCALE_CONTRACT_PATH"
OUTPUT_DIR_ENV = "FLAGSCALE_OUTPUT_DIR"
# The worker's OWN role tag (empty for an ordinary worker). Stamped by
# SpawnWorkerTool._build_env; read here to pick the leaf-aware role prefix and
# by SpawnWorkerTool.execute to refuse a spawn from a leaf worker.
TASK_ROLE_ENV = "FLAGSCALE_TASK_ROLE"
# Env key injected by ResumeChildTool: a file holding the parent's resume message.
# When present, the worker RE-ENTERS its existing session and treats this text as
# the next user turn instead of the original contract.
RESUME_PATH_ENV = "FLAGSCALE_RESUME_PATH"


def is_worker() -> bool:
    """True when this process is a spawned worker (env carries a task id)."""
    return bool(os.environ.get(TASK_ID_ENV))


def resolve_resume_query() -> Optional[str]:
    """Return the resume message when this worker is being resumed, else None.

    A resumed worker re-enters its EXISTING session dir (same id, inherited via
    env) and must continue in-context with the parent's new message — not start
    over from the original contract. The parent wrote that message to the path
    in FLAGSCALE_RESUME_PATH. Returns the text (stripped) or None when this is
    an ordinary first run.
    """
    rpath = os.environ.get(RESUME_PATH_ENV)
    if not rpath:
        return None
    try:
        text = Path(rpath).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _role_prefix_for_env() -> str:
    """Pick the role prefix from the stamped role env (leaf vs ordinary).

    `TASK_ROLE_ENV` is written by SpawnWorkerTool._build_env with the child's
    OWN role. A leaf role (reviewer/diverger) gets the no-delegation prefix;
    anything else (ordinary worker, unset) gets the delegating prefix.
    """
    env_role = os.environ.get(TASK_ROLE_ENV) or None
    if is_leaf_role_name(env_role):
        return WORKER_LEAF_ROLE_PREFIX
    return WORKER_ROLE_PREFIX


def resolve_worker_query(query: Optional[str]) -> Optional[str]:
    """Return the query the single-shot loop should run.

    The CLI's positional arg is only a PATH (long contracts go to a file,
    argv carries the path). When we're a worker and a contract file is present,
    read it and prepend the fixed role prefix. Otherwise pass the caller's
    query through untouched (normal interactive / single-shot behaviour).
    """
    if not is_worker():
        return query
    prefix = _role_prefix_for_env()
    cpath = os.environ.get(CONTRACT_PATH_ENV)
    if cpath and Path(cpath).exists():
        try:
            contract_text = Path(cpath).read_text(encoding="utf-8")
        except OSError:
            return (prefix + (query or "")).strip()
        return prefix + contract_text
    # Worker but no contract file: still stamp the role so the model knows the
    # rules, even if the body is whatever the caller passed.
    return (prefix + (query or "")).strip()


def finalize_worker_if_no_report(ledger: Optional[TaskLedger] = None) -> Optional[str]:
    """After a worker's react loop ends, close a dangling RUNNING task.

    A worker that exits without calling report_result leaves the ledger stuck
    in RUNNING, which the parent watchdog would later notice — but the worker
    can close it itself, faster and with a better note. Returns the task id if
    a transition happened, else None.
    """
    task_id = os.environ.get(TASK_ID_ENV)
    if not task_id:
        return None
    lg = ledger or TaskLedger(get_tasks_dir())
    rec = lg.get(task_id)
    if rec is None or rec.status != RUNNING:
        return None
    try:
        lg.transition(task_id, FAILED, note="worker exited without report")
    except Exception:
        return None
    return task_id
