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

"""KnowledgeFirstGuard — reminds agent to load domain knowledge proactively.

The agent has access to load_knowledge() and load_skill() for domain expertise,
but often forgets to use them before diving into implementation. This guard
tracks tool calls and injects a reminder every N calls if no knowledge/skill
has been loaded recently.

Design: inject-only, never blocks. High frequency reminders to build the habit.
"""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# How many tool calls without knowledge loading before reminding
REMINDER_INTERVAL = 8

# Tools that satisfy the "loaded knowledge" requirement
_KNOWLEDGE_TOOLS = frozenset((
    "load_knowledge", "load_skill",
))

# Tools that don't count toward the interval (meta-operations)
_META_TOOLS = frozenset((
    "evict", "evict_list", "recall",
    "plan_status", "plan_create", "plan_update",
    "memory_read", "memory_list", "memory_write",
    "workspace_experiment",
))


class KnowledgeFirstGuard(Guard):
    """Remind agent to load knowledge/skills if it hasn't done so recently.

    Injects a gentle reminder every REMINDER_INTERVAL tool calls when no
    load_knowledge or load_skill has been called. Never blocks — always
    inject-only. Resets counter when knowledge is loaded.
    """

    name = "knowledge_first"
    priority = 85  # Low priority — advisory
    overridable = True

    def __init__(self):
        super().__init__()
        self._calls_since_knowledge = 0
        self._total_reminders = 0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Before tool execution, check if knowledge reminder is needed."""
        if not ctx.tool_name:
            return None

        # Knowledge/skill loaded — reset counter
        if ctx.tool_name in _KNOWLEDGE_TOOLS:
            self._calls_since_knowledge = 0
            return None

        # Meta tools don't count
        if ctx.tool_name in _META_TOOLS:
            return None

        self._calls_since_knowledge += 1

        if self._calls_since_knowledge >= REMINDER_INTERVAL:
            self._calls_since_knowledge = 0
            self._total_reminders += 1
            return GuardVerdict.inject(
                "[KnowledgeFirst] You haven't loaded domain knowledge or skills recently. "
                "Consider whether the current task involves a specialized domain "
                "(parallelism, training config, NCCL, data pipeline, model porting, etc.) "
                "that would benefit from load_knowledge() or load_skill(). "
                "Loading knowledge BEFORE acting prevents avoidable mistakes. "
                "If the current task is straightforward and doesn't need domain expertise, "
                "proceed as normal.",
                reason="no_knowledge_loaded_recently",
                category="knowledge_first_reminder",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def reset_state(self):
        super().reset_state()
        self._calls_since_knowledge = 0
        self._total_reminders = 0

    def reset_turn(self):
        """Don't reset per-turn — the reminder interval spans turns."""
        pass

    def reset_new_turn(self):
        """New user message — don't reset, knowledge need persists."""
        pass
