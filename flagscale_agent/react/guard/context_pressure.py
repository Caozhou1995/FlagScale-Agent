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

"""Context pressure guard — advisory-only, LLM decides what to evict.

Monitors context token usage and injects reminders for the LLM to call evict().
Never auto-evicts. Never suggests reducing work quality.
"""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Thresholds
SOFT_LIMIT_RATIO = 0.75
HARD_LIMIT_RATIO = 0.90


class ContextPressureGuard(Guard):
    """Guard that monitors context pressure and reminds LLM to evict.

    Design principles:
    - NEVER auto-evict — LLM decides what to evict
    - NEVER suggest reducing quality, skipping steps, or being concise
    - Persistently remind until LLM takes action
    - Provide actionable guidance: use evict_list to browse, then evict
    """

    name = "context_pressure"
    priority = 10  # High priority
    overridable = False
    escalate_after = 5  # After 5 blocks, escalate

    # How many inject reminders before switching to block
    INJECT_LIMIT = 5

    def __init__(self, working_window_tokens: int = 0):
        super().__init__()
        self._soft_warned = False
        self._hard_remind_count = 0
        self._working_window_tokens = working_window_tokens

    @property
    def working_window_tokens(self) -> int:
        """Return the working window size for display. Fallback to 120K if not set."""
        return self._working_window_tokens or 120_000

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Check context pressure after tool execution."""
        pressure = ctx.context_pressure
        if pressure < SOFT_LIMIT_RATIO * 0.9:
            # Below hysteresis threshold — fully reset
            self._soft_warned = False
            self._hard_remind_count = 0
            return None

        ww = self.working_window_tokens
        estimated_tokens = int(pressure * ww)

        # Hard limit — check if eviction is possible
        if pressure >= HARD_LIMIT_RATIO:
            evictable = ctx.evictable_indexes
            
            # If no evictable indexes, all old messages are already evicted
            # Only the last 4 (protected tail) remain
            if not evictable:
                # Warn once with actionable guidance, then stop nagging
                if self._hard_remind_count == 0:
                    self._hard_remind_count = 999  # Prevent infinite reminders
                    # Estimate tokens in protected tail (last 4 messages)
                    tail_tokens = max(0, int((pressure - 0.6) * ww))
                    return GuardVerdict.inject(
                        f"[Context pressure CRITICAL: {int(pressure * 100)}% "
                        f"({estimated_tokens}/{ww} tokens)] "
                        f"All old messages already evicted. Only the last 4 messages remain (protected).\n\n"
                        f"Why: The last 4 messages are always protected to preserve your recent working context. "
                        f"They currently contain ~{tail_tokens} tokens of tool results and responses.\n\n"
                        f"Next steps:\n"
                        f"1. Use evict_list() to browse what's already evicted\n"
                        f"2. If you need old content: recall(index=N) to retrieve, use it, then re-evict\n"
                        f"3. If you don't need old content: summarize current progress to memory_write(), "
                        f"wrap up this step, and let the next turn start with clean context\n"
                        f"4. Continue work if critical task is incomplete — the last 4 messages are sufficient "
                        f"for focused execution. Don't abandon work due to context pressure.",
                        category="context_pressure_fully_evicted",
                    )
                # Don't block if nothing can be evicted — that creates a deadlock
                return None
            
            # Normal case: evictable content exists
            self._hard_remind_count += 1
            idx_hint = f"Evictable indexes: {evictable} ({len(evictable)} total)."

            msg = (
                f"[Context pressure CRITICAL: {int(pressure * 100)}% "
                f"({estimated_tokens}/{ww} tokens)] "
                f"Call evict(indexes=[...]) to free at least 30% of context. "
                f"You can evict ANY message (user, assistant, tool_result) except index 0 and the last 4. "
                f"Evict aggressively — use wide ranges from the list below. "
                f"Do NOT reduce work quality — recall(index=N) can retrieve evicted content if needed later. "
                f"{idx_hint}"
            )

            if self._hard_remind_count >= self.INJECT_LIMIT:
                return GuardVerdict.block(msg, category="context_pressure")
            else:
                return GuardVerdict.inject(msg, category="context_pressure")

        # Soft limit — first advisory
        if pressure >= SOFT_LIMIT_RATIO and not self._soft_warned:
            self._soft_warned = True
            evictable = ctx.evictable_indexes
            if evictable:
                idx_hint = f" Evictable indexes: {evictable} ({len(evictable)} total)."
            else:
                idx_hint = ""
            return GuardVerdict.inject(
                f"[Context pressure: {int(pressure * 100)}% "
                f"({estimated_tokens}/{ww} tokens)] "
                f"Consider calling evict(indexes=[...]) to free space. "
                f"You can evict ANY message except index 0 and the last 4."
                f"{idx_hint}"
                f" Reminder: if you need previously-seen content, use recall(index=N) instead of re-reading files.",
                category="context_pressure",
            )

        return None

    def reset(self):
        """Full reset of guard state."""
        self._soft_warned = False
        self._hard_remind_count = 0

    def reset_turn(self):
        """Per-turn reset."""
        pass
