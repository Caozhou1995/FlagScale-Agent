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
  never interfere with ordinary tool execution. It further requires the text to
  actually CARRY the completion sentinel: the kernel consults completion guards
  for BOTH [TASK_COMPLETE] and [NEED_USER_INPUT] (kernel.py:381), and llm_responded
  is True in both cases — so without a sentinel check this gate wrongly blocked a
  legitimate mid-task QUESTION while a reviewer was still running. A question is
  not a completion; release it.
- It blocks when any reviewer-class child (role reviewer/re_reviewer) recorded
  in THIS session's registry is not yet in a TERMINAL status. Two classes:
  * ACTIVE (SPAWNING/RUNNING/REPORTED) — poll_tasks(action='check') finalizes
    a REPORTED reviewer (REPORTED -> DONE/REJECTED);
  * SETTLED-WITHOUT-REPORT (DEADLINE_MISSED/TERMINATED) — the worker is gone
    and can never reach REPORTED, so poll_tasks returns a settled verdict with
    NO ledger transition (reunite.py:259). Polling does NOT clear these; only an
    explicit override naming the task_id does. The message says so.
  Both sets are in neither ACTIVE_STATUSES nor TERMINAL_STATUSES.
- accept_override demands the reason cite at least one of those reviewer
  task_ids: a bare re-emit of [TASK_COMPLETE] never releases it, and a generic
  "done" cannot stand in for having actually handled the named reviewer.
- An accepted override is DURABLE: the named task_ids are persisted to
  <session>/subagents/adjudicated.jsonl (spawn.record_adjudication) and
  _live_reviewers filters them out on every later turn. Without this a
  SETTLED-WITHOUT-REPORT reviewer — which has no terminal ledger status to
  reach — would be re-listed on EVERY completion turn, forcing the agent to
  re-pass an override forever (an infinite leash). The file, not an extra field
  on the append-only spawned.jsonl, so recording an adjudication can never
  rewrite the spawn registry.

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
    "spawned by THIS session are still un-adjudicated — {tasks}. A reviewer that "
    "reports and is never read is not a check, just cost. For a REPORTED "
    "reviewer, poll_tasks(action='check', task_id=<id>) finalizes it (REPORTED "
    "-> DONE/REJECTED) and runs the acceptance predicates; READ its findings and "
    "fold them into the wrap-up (fix what is real, state what you refute and "
    "why). A reviewer in DEADLINE_MISSED/TERMINATED is settled-without-report "
    "(it can never reach REPORTED, so polling will NOT clear this) — its "
    "analysis often lives only in its worker.log; sweep that log, then override "
    "with _override_reason naming the reviewer task_id(s) you are abandoning and "
    "why the evidence you already hold (including anything recovered from the "
    "log) is sufficient. An accepted override SETTLES the named reviewer "
    "permanently — you will not be re-asked."
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
            from flagscale_agent.react.multi_agent.spawn import (
                read_session_spawns, read_adjudications)
            from flagscale_agent.react.multi_agent.contract import (
                ROLE_REVIEWER, ROLE_RE_REVIEWER)
            from flagscale_agent.react.multi_agent.ledger import (
                ACTIVE_STATUSES, DEADLINE_MISSED, TERMINATED)
        except Exception:
            return []
        sd = self._session_dir()
        if not sd:
            return []
        reviewer_roles = (ROLE_REVIEWER, ROLE_RE_REVIEWER)
        ledger = self._ledger_obj()
        if ledger is None:
            return []
        # Task_ids this session has ALREADY adjudicated (released via a prior
        # override). Persisted in <session>/subagents/adjudicated.jsonl so the
        # release survives across turns AND across a process restart — without
        # it a SETTLED-WITHOUT-REPORT reviewer is re-listed every completion
        # turn forever (the ledger has no terminal status to move to).
        adjudicated = read_adjudications(sd)
        out = []
        for rec in read_session_spawns(sd):
            role = rec.get("role") or ""
            if role not in reviewer_roles:
                continue
            tid = rec.get("task_id")
            if not tid:
                continue
            if tid in adjudicated:
                continue
            try:
                trec = ledger.get(tid)
            except Exception:
                continue
            if trec is None:
                continue
            if trec.status in ACTIVE_STATUSES:
                # REPORTED is pollable: poll_tasks(action='check') finalizes it.
                out.append((tid, trec.status))
            elif trec.status in (DEADLINE_MISSED, TERMINATED):
                # SETTLED-WITHOUT-REPORT (neither ACTIVE nor TERMINAL —
                # ledger.py:75-76). The worker is gone and can never reach
                # REPORTED, so poll_tasks returns a settled verdict with NO
                # ledger transition (reunite.py:259, _SETTLED_WITHOUT_REPORT);
                # polling does NOT clear this — only an explicit override
                # naming the task_id does. Both statuses are the same class
                # (reunite.py:67, dispatch.py:98): including only DEADLINE_MISSED
                # would leave the TERMINATED twin silently ungated.
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
        # The kernel consults completion guards for BOTH [TASK_COMPLETE] and
        # [NEED_USER_INPUT] (kernel.py:381): llm_responded is True in both. This
        # gate is about COMPLETION only, so require the completion sentinel in the
        # text — mirroring the kernel's own _is_completion (trailing sentinel OR
        # the override form). Without this, a legitimate mid-task QUESTION ("which
        # config?") was wrongly blocked while a reviewer was still running.
        from flagscale_agent.react.display import (
            trailing_sentinel, has_override_completion)
        _text = (ctx.assistant_text or "").rstrip()
        if not (trailing_sentinel(_text) == "[TASK_COMPLETE]"
                or has_override_completion(_text)):
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
        """Release on ANY substantive reply (user directive 2026-10-28).

        The old contract demanded the reason cite a reviewer task_id AND the
        adjudication to persist successfully before release — measured on the
        dna-assembly line this cost the parent three blocked completion turns
        even though its reason was substantively correct each time. The gate's
        purpose (the parent must ENGAGE the reviewer at least once) is met by
        any non-trivial reply: the reply itself is the engagement, and the
        block message already told the parent what a real adjudication looks
        like. So: any reply longer than a bare "done" releases. Persistence is
        still attempted (best-effort) so a SETTLED-WITHOUT-REPORT reviewer is
        not re-listed next turn — but a persistence failure no longer rejects
        the release; the reply happened, which is what the user contract is.
        """
        if not reason or len(reason.strip()) <= 5:
            return False
        live = self._live_reviewers()
        if not live:
            # Nothing pending anymore (e.g. the parent polled it to terminal
            # between the block and the override turn): nothing to gate.
            return True
        matched = []
        for tid, _ in live:
            if re.search(r"\b" + re.escape(tid) + r"\b", reason):
                matched.append(tid)
        # The reply is the engagement, so settle EVERYTHING currently live:
        # naming a tid is the precise form, but an unnamed substantive reply
        # still releases — and it must release PERMANENTLY, otherwise a
        # SETTLED-WITHOUT-REPORT reviewer (no terminal ledger status to reach)
        # is re-listed on every completion turn: an infinite leash, the exact
        # defect the adjudication registry exists to prevent.
        settle = matched if matched else [tid for tid, _ in live]
        try:
            from flagscale_agent.react.multi_agent.spawn import (
                record_adjudication)
            sd = self._session_dir()
            for tid in settle:
                record_adjudication(sd, tid)
        except Exception:
            pass
        return True
