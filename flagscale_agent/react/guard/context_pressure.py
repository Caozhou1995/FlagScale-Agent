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
HARD_RESET_RATIO = 0.85  # pressure > 85% AND evictable < 50 → block for hard_reset


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

    # How many inject reminders before switching to block
    INJECT_LIMIT = 5

    # Tools allowed through the hard_reset block (save progress + reset)
    _HARD_RESET_ALLOWED_TOOLS = frozenset({
        "memory_write", "memory_read", "memory_list",
        "plan_update", "plan_status",
        "hard_reset",
        "evict", "evict_list", "recall",
    })

    def __init__(self, working_window_tokens: int = 0):
        self._soft_warned = False
        self._hard_remind_count = 0
        self._working_window_tokens = working_window_tokens
        self._hard_reset_needed = False

    @property
    def working_window_tokens(self) -> int:
        """Return the working window size for display. Fallback to 120K if not set."""
        return self._working_window_tokens or 120_000

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Block non-save tools when hard_reset is needed."""
        if not self._hard_reset_needed:
            return None

        # Re-check: maybe pressure dropped (e.g. after evict calls)
        pressure = ctx.context_pressure
        evictable = ctx.evictable_indexes
        if pressure < HARD_RESET_RATIO or len(evictable) >= 50:
            self._hard_reset_needed = False
            return None

        # Allow save-progress and reset tools through
        if ctx.tool_name in self._HARD_RESET_ALLOWED_TOOLS:
            return None

        return GuardVerdict.block(
            f"[Context pressure {int(pressure * 100)}% with only {len(evictable)} "
            f"evictable messages] 需要 hard reset。\n"
            f"允许的操作：memory_write, plan_update, hard_reset, evict\n"
            f"请先保存进度，然后调用 hard_reset(reason='...')",
            reason="hard_reset_required",
            category="context_pressure_hard_reset",
        )

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
        evictable = ctx.evictable_indexes

        # Hard reset condition: pressure > 85% AND few evictable messages left
        # Set flag so check_pre blocks non-save tools on next call
        if pressure >= HARD_RESET_RATIO and len(evictable) < 50:
            self._hard_reset_needed = True
            return GuardVerdict.inject(
                f"[Context pressure {int(pressure * 100)}% with only {len(evictable)} "
                f"evictable messages] Eviction 无法释放足够空间。\n"
                f"请执行：\n"
                f"1. memory_write() 保存关键发现\n"
                f"2. plan_update(notes='...') 记录当前状态\n"
                f"3. hard_reset(reason='...') 重置上下文\n"
                f"下一次非保存类工具调用将被 block。",
                reason="hard_reset_required",
                category="context_pressure_hard_reset",
            )

        # Hard limit — check if eviction is possible
        if pressure >= HARD_LIMIT_RATIO:
            
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
                        reason="context_pressure",
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
                return GuardVerdict.block(msg, reason="context_pressure", category="context_pressure")
            else:
                return GuardVerdict.inject(msg, reason="context_pressure", category="context_pressure")

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
                reason="context_pressure",
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
