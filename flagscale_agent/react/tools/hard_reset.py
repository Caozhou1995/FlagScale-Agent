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

"""HardReset tool — LLM-initiated full context reset.

When context pressure is high and eviction alone won't help (few evictable
messages remaining), the LLM should save progress to memory/plan and then
call this tool to reset the conversation context.

The tool:
1. Generates a summary of the current conversation state
2. Clears all messages except system prompt + last 4
3. Injects a continuation message so work can resume seamlessly

This is the LLM's escape hatch when ContextPressureGuard signals that
eviction is exhausted and a fresh context is needed.
"""

from flagscale_agent.react.tools.base import Tool


class HardResetTool(Tool):
    name = "hard_reset"
    description = (
        "Reset the conversation context when context pressure is critical "
        "and eviction alone cannot free enough space. "
        "Before calling: save progress to memory_write() and plan_update(notes=...). "
        "After reset: the conversation restarts with a summary of prior work, "
        "preserving the last 4 messages for continuity."
    )
    parameters = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why the reset is needed (e.g. 'context pressure 92%, only 5 evictable messages').",
            },
        },
        "required": ["reason"],
    }

    def __init__(self, agent):
        """Takes a reference to the agent to call _hard_reset_context()."""
        self._agent = agent

    def execute(self, **kwargs) -> str:
        reason = kwargs.get("reason", "LLM-initiated hard reset")

        # Check if reset is actually needed (guard against unnecessary calls)
        hm = self._agent.history
        pressure = hm.get_context_pressure()
        evictable = hm.get_evictable_indexes()

        if pressure < 0.75 and len(evictable) > 50:
            return (
                f"Hard reset not needed: pressure={int(pressure*100)}%, "
                f"evictable={len(evictable)}. Use evict() instead."
            )

        # Execute hard reset
        self._agent._hard_reset_context()

        return (
            f"Hard reset complete. Reason: {reason}. "
            f"Context cleared and rebuilt with continuation summary. "
            f"Use plan_status() and memory_list() to re-orient."
        )
