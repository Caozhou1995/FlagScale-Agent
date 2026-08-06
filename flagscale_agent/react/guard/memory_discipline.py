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
- Track tool calls since last memory read/write (or last reminder)
- Every 10 calls without memory operation → inject a reminder
- After memory_read/memory_list returns content → inject staleness check reminder
- If LLM reads/writes memory, reset counter
- If LLM overrides, reset counter
- No cap — keeps reminding every 10 calls as long as memory isn't used
"""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


_STALENESS_MSG = (
    "[MemoryDiscipline] You just read memories. Verify each entry against "
    "current code/state. If any memory is outdated (environment changed, "
    "bug fixed, path moved), supersede or delete it NOW via "
    "memory_write(supersedes=['old/key/here']). "
    "Do NOT leave stale memories uncorrected."
)


class MemoryDisciplineGuard(Guard):
    """Remind agent to read/write memory if it hasn't done so recently.

    Also enforces self-evolution: before TASK_COMPLETE, remind agent to
    review memory for new findings, digestible insights, or stale facts.
    """

    name = "memory_discipline"
    priority = 90  # Low priority — advisory only
    overridable = True

    # How many tool calls without memory ops before reminding
    reminder_threshold = 10

    def __init__(self):
        super().__init__()
        self._calls_since_memory = 0
        self._staleness_reminded = False
        self._evolution_reminded = False
        self._has_memory_review = False  # whether agent did memory_list this session

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
                    "Evolution example:\n"
                    "  pitfall/nccl/nic_exclude_syntax (hit 2+ times, same root cause)\n"
                    "  → elevate to insight/nccl/whitelist_over_exclude\n"
                    "  → digest: add step to skill 'train-run': always use NCCL_IB_HCA=<list> whitelist, "
                    "排除式(^dev) in NCCL 2.28+ has known bugs\n\n"
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
            if ctx.tool_name == "memory_write":
                self._staleness_reminded = False
            return None

        self._calls_since_memory += 1

        if self._calls_since_memory >= self.reminder_threshold:
            self._calls_since_memory = 0
            return GuardVerdict.inject(
                f"[MemoryDiscipline] {self.reminder_threshold} tool calls without "
                "reading or writing memory. Consider: saving key findings as fact/pitfall/insight, "
                "or checking existing memories to avoid repeating past work. "
                "If a pitfall recurs, elevate to insight; "
                "if an insight has enough evidence, digest into skill/knowledge/agent code.",
                reason="no_memory_ops_recently",
                category="memory_idle_reminder",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """After memory_read/memory_list returns content, remind to verify staleness."""
        if ctx.tool_name not in self._MEMORY_READ_TOOLS:
            return None

        if self._staleness_reminded:
            return None

        result = ctx.tool_result or ""
        if not result or len(result) < 20:
            return None
        if "no entries" in result.lower() or "not found" in result.lower():
            return None

        self._staleness_reminded = True
        return GuardVerdict.inject(
            _STALENESS_MSG,
            reason="memory_staleness_check",
            category="memory_staleness_reminder",
        )

    def was_inject_effective(self, ctx: GuardContext) -> bool | None:
        if ctx.tool_name in self._MEMORY_TOOLS:
            return True
        return False

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        if reason and len(reason.strip()) > 5:
            self._calls_since_memory = 0
            return True
        return False

    def reset_state(self):
        super().reset_state()
        self._calls_since_memory = 0
        self._staleness_reminded = False
        self._evolution_reminded = False
        self._has_memory_review = False

    def reset_turn(self):
        pass

    def reset_new_turn(self):
        """New user message — reset staleness flag so next read batch gets reminder."""
        self._staleness_reminded = False
