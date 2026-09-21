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

"""Worker-side self-report (M2, design §2.5).

The worker's ONLY added tool. It writes result.json and moves the task to
REPORTED — a *self-report*, not a verdict. Per INV3 the parent NEVER trusts
result.json for acceptance; it re-runs the acceptance checks itself (§1.4
rule 5). This tool exists so the parent's reunite has something to compare
against and so a task cannot silently hang in RUNNING after the worker exits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from flagscale_agent.react.paths import get_tasks_dir
from flagscale_agent.react.tools.base import Tool

from .contract import _within
from .ledger import LedgerError, REPORTED, RUNNING, TaskLedger


class ReportResultTool(Tool):
    """Report a finished worker task (self-report; NOT acceptance)."""

    name = "report_result"
    description = (
        "worker 完成任务后【必须】调用本工具上报。写入 result.json 并把任务状态"
        "置为 REPORTED。注意：这只是自报，父端会重跑 acceptance 独立验收（自报不算数）。"
        "summary 必填；files_written 列出你实际写出的文件绝对路径。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "对完成情况的简述（做了什么、产物在哪、是否达成目标）。",
            },
            "files_written": {
                "type": "array",
                "description": (
                    "实际写出的文件绝对路径列表（必须落在契约 constraints.writable 内）。"
                    "不传则仅按契约 output_ptr 校验。"
                ),
                "items": {"type": "string"},
            },
        },
        "required": ["summary"],
    }

    def __init__(self, ledger: Optional[TaskLedger] = None,
                 tasks_dir: Optional[str] = None):
        self._ledger = ledger or TaskLedger(tasks_dir or get_tasks_dir())

    def execute(self, summary: str = "", files_written: Optional[List[str]] = None,
                **kwargs) -> str:
        # ── 1. must be running INSIDE a worker (env carries the task id) ─────
        task_id = os.environ.get("FLAGSCALE_TASK_ID")
        if not task_id:
            return (
                "ERROR: report_result 只能在 worker 进程内调用（未检测到 "
                "FLAGSCALE_TASK_ID）。这是父端/父Agent 的内部工具，父端验收走 poll/reunite。"
            )
        if not summary or not summary.strip():
            return "ERROR: summary 不能为空。"

        # ── 2. INV4: files_written ∪ output_ptr ⊆ constraints.writable ───────
        rec = self._ledger.get(task_id)
        if rec is None:
            return f"ERROR: 账本中找不到任务 {task_id}（契约丢失？）。"
        c = rec.contract
        writable = (c.constraints or {}).get("writable") or [] if c else []
        checks: List[str] = []
        if c and c.output_ptr:
            checks.append(c.output_ptr)
        for p in (files_written or []):
            checks.append(p)
        for p in checks:
            if not any(_within(p, w) for w in writable):
                return (
                    f"ERROR: 文件 {p!r} 不在契约 constraints.writable 内 "
                    f"{writable!r} (INV4)。请只写到允许目录。"
                )

        # ── 3. write result.json + RUNNING → REPORTED ────────────────────────
        payload: Dict[str, Any] = {
            "summary": summary.strip(),
            "files_written": list(files_written or []),
            "output_ptr": c.output_ptr if c else "",
        }
        try:
            self._ledger.write_result(task_id, payload)
        except LedgerError as e:
            return f"ERROR: 写入 result 失败: {e}"

        rec2 = self._ledger.get(task_id)
        status = rec2.status if rec2 else "?"
        return (
            f"reported task {task_id} (status={status})。"
            "父端将重跑 acceptance 验收；这是自报，不代表已通过 (INV3)。"
        )
