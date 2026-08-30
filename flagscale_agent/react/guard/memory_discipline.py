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

Logic:
- Track tool calls since last memory read/write
- Every 10 calls without memory operation → inject a reminder
- Every 30 calls without memory operation → block (overridable)
- Before TASK_COMPLETE without memory review → inject evolution reminder
- If LLM reads/writes memory, reset counter
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
        self._evolution_reminded = False
        self._has_memory_review = False

    _MEMORY_TOOLS = frozenset((
        "memory_write", "memory_read", "memory_list",
        "plan_status", "plan_create", "plan_update",
    ))

    _MEMORY_READ_TOOLS = frozenset(("memory_read", "memory_list"))

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            # Check if assistant is about to emit TASK_COMPLETE without memory review
            if (ctx.assistant_text
                    and "[TASK_COMPLETE]" in ctx.assistant_text
                    and not self._evolution_reminded
                    and not self._has_memory_review):
                self._evolution_reminded = True
                return GuardVerdict.inject(
                    "[MemoryDiscipline] About to TASK_COMPLETE but no memory review this session. "
                    "Before completing, run memory_list() and check:\n"
                    "(1) Any new fact/pitfall/insight to save?\n"
                    "(2) Can any existing pitfall be elevated to an insight (recurring pattern)?\n"
                    "(3) Can any existing insight be digested into a concrete artifact — "
                    "create/improve a skill, knowledge doc, or agent code?\n"
                    "(4) Any existing fact invalidated by this session's work?\n\n"
                    "Report [Memory suggestions] to user with proposed actions; "
                    "do NOT self-execute digest/delete without confirmation.",
                    reason="evolution_check_before_complete",
                    category="memory_evolution_reminder",
                )
            return None

        if ctx.tool_name in self._MEMORY_TOOLS:
            self._calls_since_memory = 0
            if ctx.tool_name in self._MEMORY_READ_TOOLS:
                self._has_memory_review = True
            return None

        self._calls_since_memory += 1

        if self._calls_since_memory >= self.BLOCK_THRESHOLD:
            # Do NOT reset counter here — only reset in accept_override if override succeeds
            return GuardVerdict.block(
                f"[MemoryDiscipline] {self.BLOCK_THRESHOLD} tool calls without any memory operation. "
                "You likely have findings worth saving (facts, pitfalls, insights) or existing "
                "memories that could help. Run memory_list() or memory_write() before continuing.",
                reason=f"no_memory_ops_{self.BLOCK_THRESHOLD}_calls",
                category="memory_discipline",
            )

        if self._calls_since_memory % self.INJECT_THRESHOLD == 0:
            return GuardVerdict.inject(
                f"[MemoryDiscipline] {self._calls_since_memory} tool calls without "
                "reading or writing memory. Consider: saving key findings as fact/pitfall/insight, "
                "or checking existing memories to avoid repeating past work. "
                "If a pitfall recurs, elevate to insight; "
                "if an insight has enough evidence, digest into skill/knowledge/agent code.",
                reason="no_memory_ops_recently",
                category="memory_discipline",
            )

        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Allow override of block if LLM provides a reason."""
        if reason and len(reason.strip()) > 5:
            self._calls_since_memory = 0
            return True
        return False


