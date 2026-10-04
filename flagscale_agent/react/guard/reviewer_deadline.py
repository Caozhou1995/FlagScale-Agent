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

"""ReviewerDeadlineGuard — hard-stop review waiting that is eating the budget.

Failure observed end-to-end (distributed-training trial, 1800s enforced
wall-clock): the agent received a reviewer demand, spawned a reviewer, and then
polled it "not ready" again and again until the harness killed the whole task.
Each poll round is a full LLM iteration; the waiting itself consumed the tail
of the budget that was supposed to be spent adjudicating and reporting.
Root cause: NOTHING on the agent side bounds reviewer waiting — the poll tool's
static hint and the spacing advisory both keep the loop alive, and the deadline
the reviewer eventually hits exists only in the subprocess layer.

Design (two layers, both scoped to REVIEWER targets):
- L1 soft inject (check_pre on poll_tasks): when the enforced wall-clock budget
  is nearly exhausted (remaining below max(300s, 15% of budget)), a poll of a
  reviewer-class task gets an advisory demanding adjudication from the evidence
  already collected.
- L2 hard block (check_pre on poll_tasks): after 2 not-ready polls of the SAME
  reviewer-class task — counted only while the budget is actually thin (the
  streak resets whenever the budget is far from the wall) — the next poll of
  that reviewer is refused outright. The block message states the contract the
  override must satisfy, and accept_override ENFORCES it: the cited reviewer
  report file must EXIST on disk and the reason must quote it (a >=5-word
  verbatim span of the reason appears in the file). A text-only reason can
  never release the block; the run-out-of-time alternative (accept the partial
  evidence, finish the deliverable) requires NO further poll at all.

Fail-silent rules (a guard must never break tool execution or fire in
environments that cannot know the deadline):
- stats_fn missing / raising / returning None (no external wall) -> guard silent.
- Ledger unreadable / task unknown -> not a reviewer target, guard silent for
  that poll (PollSpacingGuard still covers generic poll discipline).

Scoped to reviewer-class targets only: TaskRecord.contract.constraints
["reviewer"] is the explicit flag the spawn-side contract rendering uses
(spawn.py renders the "## Reviewer discipline" section from exactly this flag).
Non-reviewer polls keep their normal behavior.
"""

from __future__ import annotations

import os
import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.poll_spacing import _poll_round_not_ready

# L2: not-ready polls of the same reviewer task allowed before the next one is
# refused. 2 means the agent has already spent >=3 full iterations waiting on
# the same reviewer with zero findings to show.
BLOCK_THRESHOLD = 2

# L1: fire the deadline advisory when remaining wall-clock falls below this
# fraction of the budget (or below this absolute floor, whichever is larger).
REMAINING_FRACTION = 0.15
REMAINING_FLOOR_SEC = 300.0

_ADJUDICATE = (
    "[ReviewerDeadline] ~{remain_fmt} of the enforced wall-clock budget is left "
    "and task {tid} (a reviewer) is still not ready. Do NOT wait for it: "
    "adjudicate from the evidence you already have — you ran the tests and read "
    "the results yourself; a reviewer is an independent second opinion, not the "
    "acceptance authority. If the reviewer lands later, fold its findings into "
    "the wrap-up; if it never lands, say so explicitly instead of stalling."
)

_BLOCK = (
    "[ReviewerDeadline] Hard stop: reviewer task {tid} has returned not-ready "
    "for {n} consecutive polls while the wall-clock budget runs down "
    "(~{remain_fmt} left). Polling it again is refused. Exit the wait loop one "
    "of two ways: (a) PREFER the no-poll path — adjudicate from the evidence "
    "you already hold and finish the deliverable/wrap-up; or (b) if the "
    "reviewer's report is genuinely required, override with _override_reason "
    "pointing at the reviewer's report file path (it must EXIST on disk and you "
    "must quote what it says). Waiting passively is not an override."
)


def _is_reviewer_target(task_id: str) -> bool:
    """True when the ledger says this task was spawned as a reviewer.

    Read straight from the on-disk contract (TaskRecord.contract carries
    constraints, which preserve the spawn-side reviewer flag).
    """
    if not task_id:
        return False
    try:
        from flagscale_agent.react.multi_agent.ledger import TaskLedger
        from flagscale_agent.react.multi_agent.wiring import get_tasks_dir

        ledger = TaskLedger(get_tasks_dir())
        rec = ledger.get(task_id)
    except Exception:
        return False
    if rec is None or rec.contract is None:
        return False
    try:
        cons = rec.contract.constraints or {}
    except Exception:
        return False
    return bool(cons.get("reviewer"))


def _is_reviewer_worker() -> bool:
    """Worker subprocesses have no parent budget to protect — stay silent.

    Broadly keyed on FLAGSCALE_TASK_ID (any worker), not reviewers only: a
    worker cannot poll the parent's ledger anyway (poll_tasks is registered
    parent-side only), so the distinction is moot today — the docstring must
    not claim narrower than the predicate.
    """
    return os.environ.get("FLAGSCALE_TASK_ID", "") != ""


class ReviewerDeadlineGuard(Guard):
    """Pre-tool guard: bound reviewer waiting by the enforced time budget."""

    name = "reviewer_deadline"
    priority = 94  # Just above first_e2e_run (93): deadline beats pacing.

    def __init__(self, stats_fn=None):
        self._stats_fn = stats_fn
        # task_id -> consecutive not-ready poll count (persisted across turns;
        # reset only when the poll settles or the reviewer reports).
        self._not_ready: dict[str, int] = {}

    def reset_turn(self):
        # A fresh user turn may legitimately start a new review wait; keep the
        # per-task counters (a restart of the SAME review is the same wait),
        # but nothing else is carried.
        return

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Enforce the contract the L2 block message states.

        The reason must point at the reviewer's report file: a path that
        EXISTS on disk, plus a quote of what the report says (a >=5-word
        verbatim span of the reason must appear in the file). A text-only
        reason can never satisfy this — passively claiming the report exists
        without reading it is exactly the poll-chaining this guard exists to
        stop. Trailing punctuation after the cited path (`,` `.` or backticks)
        is tolerated.
        """
        if not reason:
            return False
        # A reason may cite the path in ordinary prose: pull every path-shaped
        # token, trim trailing punctuation/backticks/quotes, and accept if ANY
        # of them names a real report file that the reason then quotes.
        candidates = re.findall(r"/(?:[^\s'\"`])+", reason)
        for raw in candidates:
            path = raw.rstrip(".,;:!?`'\")]}")
            try:
                with open(path, "r", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if not text.strip():
                continue
            # The reason must quote the report: some >=5-word verbatim span of
            # the reason must appear in the file (whitespace-normalized on both
            # sides; the path token itself is the pointer, not the evidence).
            norm_text = " ".join(text.split())
            tokens = re.sub(r"/(?:[^\s'\"`])+", " ", reason).split()
            for i in range(max(0, len(tokens) - 4)):
                window = " ".join(tokens[i : i + 5])
                if window in norm_text:
                    return True
        return False

    # ── plumbing ─────────────────────────────────────────────────────────────
    def _stats(self) -> dict | None:
        try:
            return self._stats_fn() if self._stats_fn else None
        except Exception:
            # A stats_fn hiccup must never break tool execution — fail silent.
            return None

    @staticmethod
    def _deadline_armed(stats: dict) -> bool:
        """L1/L2 active only when remaining budget is actually thin.

        far-from-deadline -> (False, None): the ordinary wait is legitimate and
        generic poll discipline stays PollSpacingGuard's job.
        """
        remaining = stats.get("remaining")
        if remaining is None:
            return False
        budget = stats.get("budget") or 0.0
        floor = max(REMAINING_FLOOR_SEC, REMAINING_FRACTION * budget)
        if remaining > floor:
            return False
        return True

    def _bump(self, task_id: str, tool_result: str | None) -> int:
        """Advance/reset the per-task not-ready streak for THIS poll round."""
        if tool_result and tool_result.strip().startswith(
            ("[BLOCKED BY GUARD]", "ERROR:")):
            return self._not_ready.get(task_id, 0)
        if _poll_round_not_ready("poll_tasks", tool_result):
            self._not_ready[task_id] = self._not_ready.get(task_id, 0) + 1
        else:
            # Settled (PASSED/FAILED) or a not-a-poll-round result — wait over.
            self._not_ready.pop(task_id, None)
        return self._not_ready.get(task_id, 0)

    # ── guard protocol ───────────────────────────────────────────────────────
    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # In a worker subprocess (a reviewer polling ITS children) there is no
        # parent budget to protect — stay out of the way.
        if _is_reviewer_worker():
            return None
        stats = self._stats()
        if not stats or not self._deadline_armed(stats):
            # Budget far from the wall: never fire. check_post also resets the
            # streak in this state — only thin-budget polls count toward L2.
            return None
        name = ctx.tool_name or ""
        if name != "poll_tasks":
            return None
        task_id = str(ctx.tool_args.get("task_id", "") or "")
        if not task_id or not _is_reviewer_target(task_id):
            return None  # non-reviewer polls are not this guard's business
        from flagscale_agent.react.guard.time_budget import _fmt

        remain_fmt = _fmt(stats.get("remaining", 0.0))
        n = self._not_ready.get(task_id, 0)
        if n >= BLOCK_THRESHOLD:
            return GuardVerdict.block(
                message=_BLOCK.format(tid=task_id, n=n, remain_fmt=remain_fmt),
                reason=f"reviewer_deadline_{task_id}",
                category="reviewer_deadline",
                overridable=True,
            )
        # Below the block threshold but near the deadline: soft-demand
        # adjudication (L1). This rides the allowed poll's result.
        return GuardVerdict.inject(
            _ADJUDICATE.format(tid=task_id, remain_fmt=remain_fmt),
            reason=f"reviewer_deadline_soft_{task_id}",
            category="reviewer_deadline",
        )

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Track the not-ready streak AFTER each executed poll round so the
        # NEXT check_pre sees it. Blocked calls carry a [BLOCKED BY GUARD]
        # tool_result and advance nothing (mirrors PollSpacingGuard).
        if _is_reviewer_worker():
            return None
        name = ctx.tool_name or ""
        if name != "poll_tasks":
            # Any intervening non-poll tool does not end a REVIEW wait (the
            # reviewer runs for minutes; the parent should work meanwhile) —
            # but a settled poll does.
            return None
        task_id = str(ctx.tool_args.get("task_id", "") or "")
        if not task_id:
            return None
        stats = self._stats()
        if not stats or not self._deadline_armed(stats):
            # Far from the wall: a not-ready round is NOT evidence of deadline
            # pressure. Reset the streak so "consecutive" means consecutive
            # polls taken while the budget is actually thin.
            self._not_ready.pop(task_id, None)
            return None
        self._bump(task_id, ctx.tool_result)
        return None
