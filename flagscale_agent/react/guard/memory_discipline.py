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

"""MemoryDisciplineGuard — reminds the agent to use memory proactively.

Logic (call-count driven, ADVISORY ONLY — this guard never blocks):
- Track tool calls since last memory read/write.
- Every 10 calls → inject the standard reminder.
- On the call that crosses 30 calls → inject an ESCALATED reminder once
  (stronger wording, same cadence afterwards: the 10-call rhythm resumes).
- If LLM reads/writes memory, reset counter and escalation state.

The 30-call threshold used to return GuardVerdict.block(). It is an inject
now: the gate still fires on the same call count, but being late on a memory
read is not an irreversibly-wrong action, so interrupting execution bought
nothing the reminder text did not — while the block cost a full round trip on
every triggering turn. Escalation is a ONE-SHOT event (a flag), not a
`calls_since >= 30` inject: a bare threshold compare would re-fire on every
subsequent call and nag harder than the block ever did.

Note: The [TASK_COMPLETE] completion-time memory review was moved to
VerificationGuard's _TEXT_COMPLETE_HYGIENE gate (block, not inject) so it
actually stops the agent before completion.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class MemoryDisciplineGuard(Guard):
    """Remind agent to read/write memory if it hasn't done so recently."""

    name = "memory_discipline"
    priority = 90  # Low priority — advisory only

    INJECT_THRESHOLD = 10
    BLOCK_THRESHOLD = 30

    def __init__(self):
        self._calls_since_memory = 0
        # One-shot escalation latch for the 30-call mark (see module docstring):
        # it fires the moment the count crosses BLOCK_THRESHOLD, then stays set
        # until a memory op resets the counter, so the escalated text lands once
        # instead of on every call past the threshold.
        self._escalated = False
        # True while an escalation latch was consumed building a verdict that a
        # concurrent block may suppress; block_not_surfaced() hands it back.
        self._escalation_unreturned = False

    _MEMORY_TOOLS = frozenset((
        "memory_write", "memory_read", "memory_list",
        "plan_status", "plan_create", "plan_update",
    ))

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            # The [TASK_COMPLETE] completion-time memory review was moved to
            # VerificationGuard's _TEXT_COMPLETE_HYGIENE gate (block, not inject)
            # so it actually stops the agent before completion. An inject here
            # was useless — the agent had already emitted [TASK_COMPLETE] and
            # would not act on advisory text.
            return None

        if ctx.tool_name in self._MEMORY_TOOLS:
            self._calls_since_memory = 0
            self._escalated = False
            return None

        self._calls_since_memory += 1

        if self._calls_since_memory >= self.BLOCK_THRESHOLD and not self._escalated:
            self._escalated = True
            # The escalation latch above is a ONE-SHOT side effect consumed while
            # building a verdict that a concurrent block may then suppress. Hand
            # it back via block_not_surfaced() so the heavy mark is not lost for
            # the rest of the run.
            self._escalation_unreturned = True
            return GuardVerdict.inject(
                f"[MemoryDiscipline] {self._calls_since_memory} tool calls without any memory "
                "operation — this is the heavy mark. Stop and run ONE concrete recall "
                "action before your next irreversible step: "
                "memory_read(key='pitfall/<domain>/') for the domain you are working in "
                "(whole-domain pitfall read), or memory_list(keyword='...') if you only have "
                "a symptom keyword. Primary sources beat memory where they disagree, so a "
                "recalled entry is a claim to re-verify, not an answer. "
                "And if you have findings worth saving, memory_write() them — a finding "
                "not written is a finding lost at the next eviction.",
                reason=f"no_memory_ops_{self.BLOCK_THRESHOLD}_calls",
                category="memory_discipline",
            )

        if self._calls_since_memory % self.INJECT_THRESHOLD == 0:
            return GuardVerdict.inject(
                f"[MemoryDiscipline] {self._calls_since_memory} tool calls without "
                "reading or writing memory. Pick ONE: "
                "(1) RECALL — memory_read(key='pitfall/<current-domain>/') before your next "
                "risky action (launch/build/deploy/new host); "
                "(2) WRITE — memory_write() a finding from this stretch (path/config/error fix); "
                "(3) DIGEST — recurring pitfall → insight, evidenced insight → skill/knowledge.",
                reason="no_memory_ops_recently",
                category="memory_discipline",
            )

        return None

    def block_not_surfaced(self):
        """Hand back an escalation latch consumed by a verdict that never surfaced.

        A generic periodic inject is suppressed when a higher-ranked block wins
        the same resolve. This guard is one-shot at BLOCK_THRESHOLD, so the
        suppressed escalation is otherwise dead for the rest of the run.
        """
        if self._escalation_unreturned:
            self._escalated = False
            self._escalation_unreturned = False

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Clear the counter on a substantive reason (>5 chars).

        This guard no longer blocks, so there is no block to override — the hook
        is kept because the guard registry calls it uniformly and the reset
        semantics are still meaningful (a stated reason means the agent has
        consciously decided memory is not needed right now).
        """
        if reason and len(reason.strip()) > 5:
            self._calls_since_memory = 0
            self._escalated = False
            return True
        return False


