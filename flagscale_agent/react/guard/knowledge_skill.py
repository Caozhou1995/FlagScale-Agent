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

"""KnowledgeSkillGuard — reminds agent to load domain knowledge/skills proactively.

Logic:
- Track tool calls since last knowledge/skill load
- Every 15 calls without knowledge/skill load → inject a reminder
- Every 40 calls without knowledge/skill load → block (overridable)
- If LLM loads knowledge/skill, reset counter
- Meta tools (evict, plan, memory) don't count toward threshold

Design parallel to MemoryDisciplineGuard but with looser thresholds,
because not every task requires domain knowledge.
"""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class KnowledgeSkillGuard(Guard):
    """Remind agent to load knowledge/skills if it hasn't done so recently."""

    name = "knowledge_skill"
    priority = 85  # Low priority — advisory

    INJECT_THRESHOLD = 15
    BLOCK_THRESHOLD = 40

    _KNOWLEDGE_TOOLS = frozenset((
        "load_knowledge", "load_skill",
    ))

    # Tools that don't count toward threshold (meta-operations)
    _META_TOOLS = frozenset((
        "evict", "recall",
        "plan_status", "plan_create", "plan_update",
        "memory_read", "memory_list", "memory_write",
    ))

    def __init__(self):
        self._calls_since_knowledge = 0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        # Knowledge/skill loaded — reset counter
        if ctx.tool_name in self._KNOWLEDGE_TOOLS:
            self._calls_since_knowledge = 0
            return None

        # Meta tools don't count
        if ctx.tool_name in self._META_TOOLS:
            return None

        self._calls_since_knowledge += 1

        if self._calls_since_knowledge >= self.BLOCK_THRESHOLD:
            # Do NOT reset counter here — only reset in accept_override if override succeeds
            return GuardVerdict.block(
                f"[KnowledgeSkill] {self.BLOCK_THRESHOLD} tool calls without loading "
                "domain knowledge or skills. Consider whether load_knowledge() or "
                "load_skill() would help the current task. "
                "If the current task genuinely does not need domain knowledge, "
                "override with a reason explaining why.",
                reason=f"no_knowledge_load_{self.BLOCK_THRESHOLD}_calls",
                category="knowledge_skill",
            )

        if self._calls_since_knowledge % self.INJECT_THRESHOLD == 0:
            return GuardVerdict.inject(
                f"[KnowledgeSkill] {self._calls_since_knowledge} tool calls without "
                "loading domain knowledge or skills. Consider whether the current task "
                "involves a specialized domain (parallelism, training config, NCCL, data "
                "pipeline, model porting, etc.) that would benefit from load_knowledge() or "
                "load_skill(). Loading knowledge BEFORE acting prevents avoidable mistakes. "
                "If the current task is straightforward and doesn't need domain expertise, "
                "proceed as normal.",
                reason="no_knowledge_load_recently",
                category="knowledge_skill",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Allow override of block if LLM explains why knowledge isn't needed."""
        if reason and len(reason.strip()) > 5:
            self._calls_since_knowledge = 0
            return True
        return False

    def reset_turn(self):
        """Don't reset per-turn — knowledge need persists across turns."""
        pass
