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
    "current code/state. If any memory is outdated (bug fixed, file deleted, "
    "architecture changed), supersede or delete it NOW via memory_write(supersedes=[old_key]). "
    "Do NOT leave stale memories uncorrected."
)


class MemoryDisciplineGuard(Guard):
    """Remind agent to read/write memory if it hasn't done so recently."""

    name = "memory_discipline"
    priority = 90  # Low priority — advisory only
    overridable = True

    # How many tool calls without memory ops before reminding
    reminder_threshold = 10

    def __init__(self):
        super().__init__()
        self._calls_since_memory = 0
        self._staleness_reminded = False  # Only remind once per read batch

    _MEMORY_TOOLS = frozenset((
        "memory_write", "memory_read", "memory_list",
        "plan_status", "plan_create", "plan_update",
        "workspace_experiment",
    ))

    _MEMORY_READ_TOOLS = frozenset(("memory_read", "memory_list"))

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Skip pre-iteration check (no specific tool being attempted)
        if not ctx.tool_name:
            return None

        # Memory tool call — reset counter
        if ctx.tool_name in self._MEMORY_TOOLS:
            self._calls_since_memory = 0
            # memory_write after a read resets staleness reminder
            if ctx.tool_name == "memory_write":
                self._staleness_reminded = False
            return None

        self._calls_since_memory += 1

        # Remind every N calls — then reset counter so next reminder is N calls later
        if self._calls_since_memory >= self.reminder_threshold:
            self._calls_since_memory = 0  # Reset — next reminder in another N calls
            return GuardVerdict.inject(
                f"[MemoryDiscipline] {self.reminder_threshold} tool calls without "
                "reading or writing memory. Consider saving key findings or "
                "checking existing memories to avoid repeating past work.",
                reason="no_memory_ops_recently",
                category="memory_idle_reminder",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """After memory_read/memory_list returns content, remind to verify staleness."""
        if ctx.tool_name not in self._MEMORY_READ_TOOLS:
            return None

        # Don't remind if already reminded this batch (multiple reads in one turn)
        if self._staleness_reminded:
            return None

        # Only trigger if there's actual content returned (not empty/error)
        result = ctx.tool_result or ""
        if not result or len(result) < 20:
            return None
        # Skip if result is just "No entries found" or similar
        if "no entries" in result.lower() or "not found" in result.lower():
            return None

        self._staleness_reminded = True
        return GuardVerdict.inject(
            _STALENESS_MSG,
            reason="memory_staleness_check",
            category="memory_staleness_reminder",
        )

    def was_inject_effective(self, ctx: GuardContext) -> bool | None:
        """If agent used a memory tool after our inject, it was effective.
        Any non-memory tool means the agent ignored the reminder."""
        if ctx.tool_name in self._MEMORY_TOOLS:
            return True
        return False

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Accept any non-trivial reason — reset counter."""
        if reason and len(reason.strip()) > 5:
            self._calls_since_memory = 0
            return True
        return False

    def reset_state(self):
        """Full reset (decay/override)."""
        super().reset_state()
        self._calls_since_memory = 0
        self._staleness_reminded = False

    def reset_turn(self):
        """Per-iteration — nothing to do."""
        pass

    def reset_new_turn(self):
        """New user message — reset staleness flag so next read batch gets reminder."""
        self._staleness_reminded = False
