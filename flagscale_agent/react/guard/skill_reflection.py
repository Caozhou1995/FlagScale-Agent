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

"""SkillReflectionGuard — periodic self-review mechanism for loaded skills.

Design Philosophy:
    Not "don't trust LLM", but "long context makes instruction-following hard".
    
    Solution: Periodic reminder (not real-time judge) for LLM to self-reflect:
    - Every N tool calls (default 8), block and prompt LLM to review
    - LLM reads its own conversation history
    - LLM decides: "Did I follow the skill? Should I adjust?"
    - LLM can choose: "Skill doesn't fit, I deviated intentionally" (OK)
                   or "I drifted unintentionally, need to correct" (fix)

Advantages over Constraint system:
    ✅ Self-awareness: LLM reviews its own behavior, not external judge
    ✅ Full context: LLM has complete history, knows why it did things
    ✅ Flexibility: Can choose to not follow skill if skill is wrong
    ✅ Low overhead: Periodic trigger (every 8 calls), not per-call judge
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class SkillReflectionGuard(Guard):
    """Periodically prompt LLM to review its adherence to loaded skills."""

    name = "skill_reflection"
    priority = 5  # Lower than safety guards, higher than advisory

    def __init__(self, review_interval: int = 8):
        """Initialize reflection guard.
        
        Args:
            review_interval: Trigger review every N tool calls. Default 8.
                            Set to 0 or negative to disable.
        """
        self._review_interval = review_interval
        self._tool_call_count = 0
        self._loaded_skills: list[str] = []  # Track active skills
        self._last_review_count = 0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Check if review interval reached, trigger self-reflection."""
        # Only trigger if skills are loaded and interval is positive
        if not self._loaded_skills or self._review_interval <= 0:
            return None

        # Only count actual tool calls (not empty tool_name)
        if not ctx.tool_name:
            return None

        self._tool_call_count += 1

        # Check if review interval reached
        calls_since_review = self._tool_call_count - self._last_review_count
        if calls_since_review < self._review_interval:
            return None

        # Trigger review
        self._last_review_count = self._tool_call_count

        return GuardVerdict.block(
            self._make_review_prompt(),
            reason="Periodic skill adherence review",
            category="skill_reflection",
        )

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """No post-execution checks needed."""
        return None

    def _make_review_prompt(self) -> str:
        """Generate the self-reflection prompt."""
        skills_list = ", ".join(f"/{s}" for s in self._loaded_skills)

        return f"""[SKILL ADHERENCE REVIEW]

You have loaded these skills: {skills_list}

**Before proceeding**, review your recent actions:

1. Check your conversation history (recent turns in context, or read conversation_full.json for full history)
2. Review if your approach aligns with the loaded skills' guidance
3. Answer these questions:
   - Am I following the workflow/principles defined in the skills?
   - If not, is it because:
     a) The skill guidance doesn't fit this specific situation? (OK to deviate)
     b) I forgot or overlooked the skill instructions? (Need to correct)

**If you deviated intentionally** (case a): Briefly explain why the skill guidance didn't apply, then continue.

**If you drifted unintentionally** (case b): Acknowledge what you overlooked and adjust your next steps accordingly.

After your reflection (2-3 sentences), continue with your next tool call.

This is NOT a judgment or violation — it's a periodic reminder to stay aligned with loaded guidance.
"""

    # ── Public API ──

    def on_skill_loaded(self, skill_name: str):
        """Called when a skill is loaded. Adds skill to tracking list.
        
        Args:
            skill_name: Name of the skill that was loaded (without / prefix)
        """
        if skill_name not in self._loaded_skills:
            self._loaded_skills.append(skill_name)

    def on_skill_unloaded(self, skill_name: str):
        """Called when a skill is unloaded. Removes skill from tracking list.
        
        Args:
            skill_name: Name of the skill that was unloaded
        """
        if skill_name in self._loaded_skills:
            self._loaded_skills.remove(skill_name)

    def reset_turn(self):
        """Reset on new user message.
        
        Note: We keep skill list and counter across turns — reflection
        persists throughout the task, not just within one turn.
        """
        # Keep everything across turns — reflection is session-level
        pass

    @property
    def loaded_skills(self) -> list[str]:
        """Currently loaded skills being tracked."""
        return list(self._loaded_skills)

    @property
    def tool_call_count(self) -> int:
        """Total tool calls since last reset."""
        return self._tool_call_count

    @property
    def calls_until_next_review(self) -> int:
        """Number of tool calls until next review triggers."""
        if self._review_interval <= 0 or not self._loaded_skills:
            return -1  # Disabled
        calls_since = self._tool_call_count - self._last_review_count
        return max(0, self._review_interval - calls_since)
