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
the agent calls, plus the fixed role prefix (D7). Nothing here spawns; nothing
here touches the REPL.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from flagscale_agent.react.paths import get_tasks_dir

from .ledger import FAILED, RUNNING, TaskLedger

# D7 — fixed role prefix. The contract body is already the query; terse.
WORKER_ROLE_PREFIX = (
    "[worker role] You are a worker dispatched by a parent Agent. The contract "
    "has been injected. When done you MUST call report_result to report a "
    "summary. You MAY call spawn_worker to delegate part of the work, but only "
    "while under the infrastructure depth cap; a spawn beyond the cap is "
    "refused with an explicit error and you cannot raise the cap.\n"
)

# Env keys injected by SpawnWorkerTool.
TASK_ID_ENV = "FLAGSCALE_TASK_ID"
CONTRACT_PATH_ENV = "FLAGSCALE_CONTRACT_PATH"
OUTPUT_DIR_ENV = "FLAGSCALE_OUTPUT_DIR"


def is_worker() -> bool:
    """True when this process is a spawned worker (env carries a task id)."""
    return bool(os.environ.get(TASK_ID_ENV))


def resolve_worker_query(query: Optional[str]) -> Optional[str]:
    """Return the query the single-shot loop should run.

    The CLI's positional arg is only a PATH (R1: long contracts go to a file,
    argv carries the path). When we're a worker and a contract file is present,
    read it and prepend the fixed role prefix. Otherwise pass the caller's
    query through untouched (normal interactive / single-shot behaviour).
    """
    if not is_worker():
        return query
    cpath = os.environ.get(CONTRACT_PATH_ENV)
    if cpath and Path(cpath).exists():
        try:
            contract_text = Path(cpath).read_text(encoding="utf-8")
        except OSError:
            return (WORKER_ROLE_PREFIX + (query or "")).strip()
        return WORKER_ROLE_PREFIX + contract_text
    # Worker but no contract file: still stamp the role so the model knows the
    # rules, even if the body is whatever the caller passed.
    return (WORKER_ROLE_PREFIX + (query or "")).strip()


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
