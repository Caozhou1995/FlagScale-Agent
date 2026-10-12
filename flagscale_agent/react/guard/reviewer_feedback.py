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

"""ReviewerFeedbackGuard — a reviewer must be ADJUDICATED before completion.

Failure this closes (observed end-to-end): the parent spawned a reviewer to
independently check its own work, the reviewer reported back, and the parent
then emitted [TASK_COMPLETE] WITHOUT ever reading the reviewer's findings — the
review cost a worker's wall-clock and bought nothing. The reviewer was a
decoration, not a check.

Root cause: nothing ties an un-adjudicated reviewer child to the moment the
parent declares the task done. The global task ledger cannot supply the scope —
it is shared across sessions (ledger.py:25), so "the reviewers THIS session
spawned and has not finalized" is not expressible against it. The spawn site
therefore records each child in a PER-SESSION registry
(<session_dir>/subagents/spawned.jsonl, written by spawn.record_spawn); this
guard reads THAT at the text-only completion path.

Design (single block, overridable):
- check_pre fires ONLY at completion consultation (tool_name == "" and
  llm_responded — the fresh text-only [TASK_COMPLETE] path, kernel.py) so it can
  never interfere with ordinary tool execution.
- It blocks when any reviewer-class child (role reviewer/re_reviewer) recorded
  in THIS session's registry is still in an ACTIVE status
  (SPAWNING/RUNNING/REPORTED). The message names the tasks and tells the parent
  to poll_tasks(action='check') (REPORTED -> DONE/REJECTED) and fold the
  findings into the wrap-up, or to override deliberately if the reviewer truly
  cannot land.
- accept_override demands the reason cite at least one of those reviewer
  task_ids: a bare re-emit of [TASK_COMPLETE] never releases it, and a generic
  "done" cannot stand in for having actually handled the named reviewer.

Fail-silent rules (a guard must never break the run):
- No session_dir / registry unreadable -> silent.
- Ledger unreadable -> silent.
- Worker subprocess -> silent (a worker has no parent completion to gate).
"""

from __future__ import annotations

import os
import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

_BLOCK = (
    "[ReviewerFeedback] Task completion is gated: {n} reviewer-class worker(s) "
    "spawned by THIS session are still un-finalized — {tasks}. A reviewer that "
    "reports and is never read is not a check, just cost. Before completing, "
    "poll_tasks(action='check', task_id=<id>) for each (REPORTED -> DONE/"
    "REJECTED finalizes it and runs the acceptance predicates), READ its "
    "findings, and fold them into the wrap-up (fix what is real, state what you "
    "refute and why). If a reviewer genuinely cannot land, override with "
    "_override_reason naming the reviewer task_id(s) you are abandoning and why "
    "the evidence you already hold is sufficient."
)


def _is_worker() -> bool:
    return os.environ.get("FLAGSCALE_TASK_ID", "") != ""


class ReviewerFeedbackGuard(Guard):
    """Completion-time gate: reviewer-class children must be adjudicated."""

    name = "reviewer_feedback"
    # Just below reviewer_deadline (94): the deadline/poll discipline governs the
    # WAIT; this gate governs the CLOSE. Both are reviewer-scoped.
    priority = 95

    def __init__(self, session_dir_fn=None, ledger=None):
        # session_dir_fn is a zero-arg callable returning the LIVE session dir
        # (it can be reassigned on session restore) — never a frozen string.
        self._session_dir_fn = session_dir_fn
        self._ledger = ledger

    # ── plumbing ─────────────────────────────────────────────────────────────
    def _session_dir(self):
        try:
            return self._session_dir_fn() if self._session_dir_fn else None
        except Exception:
            return None

    def _ledger_obj(self):
        if self._ledger is not None:
            return self._ledger
        try:
            from flagscale_agent.react.multi_agent.ledger import TaskLedger
            from flagscale_agent.react.multi_agent.wiring import get_tasks_dir

            return TaskLedger(get_tasks_dir())
        except Exception:
            return None

    def _live_reviewers(self):
        """Task_ids of THIS session's reviewer-class children not yet terminal.

        Returns [(task_id, status), ...]; [] when nothing is pending or any
        lookup fails (fail-silent).
        """
        try:
            from flagscale_agent.react.multi_agent.spawn import read_session_spawns
            from flagscale_agent.react.multi_agent.contract import (
                ROLE_REVIEWER, ROLE_RE_REVIEWER)
            from flagscale_agent.react.multi_agent.ledger import (
                ACTIVE_STATUSES, DEADLINE_MISSED)
        except Exception:
            return []
        sd = self._session_dir()
        if not sd:
            return []
        reviewer_roles = (ROLE_REVIEWER, ROLE_RE_REVIEWER)
        ledger = self._ledger_obj()
        if ledger is None:
            return []
        out = []
        for rec in read_session_spawns(sd):
            role = rec.get("role") or ""
            if role not in reviewer_roles:
                continue
            tid = rec.get("task_id")
            if not tid:
                continue
            try:
                trec = ledger.get(tid)
            except Exception:
                continue
            if trec is None:
                continue
            if trec.status in ACTIVE_STATUSES or trec.status == DEADLINE_MISSED:
                # DEADLINE_MISSED is in NEITHER ACTIVE_STATUSES nor
                # TERMINAL_STATUSES (ledger.py:75-76): a reviewer that died of
                # its wall-clock budget still holds an un-adjudicated report —
                # often only in its worker.log — and poll_tasks(check) can
                # still finalize it (check requires merely non-terminal,
                # ledger.py:327). Silently ignoring it would repeat the exact
                # failure this guard closes.
                out.append((tid, trec.status))
        return out

    # ── guard protocol ───────────────────────────────────────────────────────
    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if _is_worker():
            return None
        # Only the text-only completion path (kernel.py). A tool call is never
        # this guard's business, so ordinary execution is untouched.
        if ctx.tool_name:
            return None
        if not ctx.llm_responded:
            return None
        live = self._live_reviewers()
        if not live:
            return None
        tasks = ", ".join(f"{tid}({st})" for tid, st in live)
        return GuardVerdict.block(
            message=_BLOCK.format(n=len(live), tasks=tasks),
            reason="reviewer_feedback_pending",
            category="reviewer_feedback",
            overridable=True,
        )

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """A release must NAME at least one of the pending reviewer task_ids.

        This blocks the two cheap escapes: a bare re-emit of [TASK_COMPLETE]
        (no reason) and a generic "done" that never engages the reviewer. It
        does NOT require the reviewer to have landed — abandoning a stuck
        reviewer is a legitimate choice, but it must be an explicit one.
        """
        if not reason or len(reason.strip()) <= 5:
            return False
        live = self._live_reviewers()
        if not live:
            # Nothing pending anymore (e.g. the parent polled it to terminal
            # between the block and the override turn): nothing to gate.
            return True
        for tid, _ in live:
            if re.search(r"\b" + re.escape(tid) + r"\b", reason):
                return True
        return False
