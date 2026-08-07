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

"""Context pressure guard — monitors token usage and reminds LLM to evict or hard_reset.

Design:
- All detection in check_pre (before tool execution, not after)
- Never auto-evicts — LLM decides what to evict
- Never suggests reducing work quality
- Escalation: soft inject → hard inject → block → escalate (force hard_reset)
"""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Thresholds
SOFT_LIMIT_RATIO = 0.75
HARD_LIMIT_RATIO = 0.90
HARD_RESET_RATIO = 0.85  # pressure >= 85% AND evictable < 50 → need hard_reset
HYSTERESIS_RATIO = 0.675  # SOFT * 0.9 — reset all flags when below this


class ContextPressureGuard(Guard):
    """Guard that monitors context pressure and reminds LLM to evict.

    Escalation path:
    - >= 75%: inject soft reminder (once)
    - >= 90%: inject hard reminder (every call), block after 5 reminders
    - >= 85% + evictable < 50: escalate (force hard_reset, not overridable)
    """

    name = "context_pressure"
    priority = 10  # High priority — context exhaustion is critical

    # How many hard inject reminders before blocking
    INJECT_LIMIT = 5

    # Max consecutive blocks before escalating to force hard_reset
    MAX_BLOCKS_BEFORE_ESCALATE = 3

    # Tools allowed through when hard_reset is needed
    _ALLOWED_TOOLS = frozenset({
        "memory_write", "memory_read", "memory_list",
        "plan_update", "plan_status", "plan_create",
        "hard_reset",
        "evict", "evict_list", "recall",
    })

    def __init__(self, working_window_tokens: int = 0):
        self._soft_warned = False
        self._hard_remind_count = 0
        self._consecutive_blocks = 0
        self._fully_evicted_warned = False
        self._working_window_tokens = working_window_tokens

    @property
    def working_window_tokens(self) -> int:
        return self._working_window_tokens or 120_000

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """All pressure detection happens here — before tool execution."""
        pressure = ctx.context_pressure
        if pressure <= 0:
            return None

        # Hysteresis: fully reset when pressure drops well below soft limit
        if pressure < HYSTERESIS_RATIO:
            self._soft_warned = False
            self._hard_remind_count = 0
            self._consecutive_blocks = 0
            self._fully_evicted_warned = False
            return None

        ww = self.working_window_tokens
        estimated_tokens = int(pressure * ww)
        evictable = ctx.evictable_indexes

        # ─── Hard reset path: pressure >= 85% AND few evictable messages ───
        # This means eviction can't free enough space — need hard_reset.
        if pressure >= HARD_RESET_RATIO and len(evictable) < 50:
            # Allow save-progress and reset tools through
            if ctx.tool_name in self._ALLOWED_TOOLS:
                self._consecutive_blocks = 0
                return None

            self._consecutive_blocks += 1

            # After MAX_BLOCKS_BEFORE_ESCALATE consecutive blocks, escalate
            # (force hard_reset — LLM clearly isn't responding to blocks)
            if self._consecutive_blocks >= self.MAX_BLOCKS_BEFORE_ESCALATE:
                return GuardVerdict.escalate(
                    f"[Context pressure {int(pressure * 100)}% with only "
                    f"{len(evictable)} evictable messages] "
                    f"Blocked {self._consecutive_blocks} times but hard_reset not executed. "
                    f"You MUST call hard_reset(reason='...') now. "
                    f"Save critical state first: memory_write() and plan_update(notes='...').",
                    reason="hard_reset_forced",
                    category="context_pressure_hard_reset",
                )

            # Block non-allowed tools with clear guidance
            return GuardVerdict.block(
                f"[Context pressure {int(pressure * 100)}% with only "
                f"{len(evictable)} evictable messages] "
                f"Eviction cannot free enough space. Execute in order:\n"
                f"1. memory_write() — save key findings\n"
                f"2. plan_update(notes='...') — record current state\n"
                f"3. hard_reset(reason='...') — reset context\n"
                f"Allowed tools: memory_write, plan_update, hard_reset, evict, evict_list, recall",
                reason="hard_reset_required",
                category="context_pressure_hard_reset",
            )

        # ─── Hard path: pressure >= 90% but evictable >= 50 ───
        # Eviction can still help — remind aggressively, then block.
        if pressure >= HARD_LIMIT_RATIO:
            if not evictable:
                # Edge case: 90%+ but nothing to evict (all protected)
                if not self._fully_evicted_warned:
                    self._fully_evicted_warned = True
                    return GuardVerdict.inject(
                        f"[Context pressure CRITICAL: {int(pressure * 100)}% "
                        f"({estimated_tokens}/{ww} tokens)] "
                        f"All old messages already evicted. Only protected messages remain.\n"
                        f"Options:\n"
                        f"1. Wrap up current step and let next turn start fresh\n"
                        f"2. Use recall(index=N) if you need old content\n"
                        f"3. Continue if task is nearly complete",
                        reason="context_fully_evicted",
                        category="context_pressure",
                    )
                return None

            self._hard_remind_count += 1
            idx_hint = f"Evictable indexes: {evictable} ({len(evictable)} total)."

            msg = (
                f"[Context pressure CRITICAL: {int(pressure * 100)}% "
                f"({estimated_tokens}/{ww} tokens)] "
                f"Call evict(indexes=[...]) to free at least 30% of context. "
                f"You can evict ANY message (user, assistant, tool_result) "
                f"except index 0 and the last 4. "
                f"Evict aggressively — use wide ranges. "
                f"Do NOT reduce work quality — recall(index=N) retrieves evicted content. "
                f"{idx_hint}"
            )

            if self._hard_remind_count >= self.INJECT_LIMIT:
                return GuardVerdict.block(msg, reason="context_pressure", category="context_pressure")
            return GuardVerdict.inject(msg, reason="context_pressure", category="context_pressure")

        # ─── Soft path: pressure >= 75% ───
        if pressure >= SOFT_LIMIT_RATIO and not self._soft_warned:
            self._soft_warned = True
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
                f" Reminder: if you need previously-seen content, "
                f"use recall(index=N) instead of re-reading files.",
                reason="context_pressure",
                category="context_pressure",
            )

        return None

    def reset_turn(self):
        pass
