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

"""ShellSafetyGuard — shell command safety (dual-level) + error escalation.

Two-level shell safety via LLM judge (no regex):
  - is_fatal: irreversible catastrophic commands → escalate (cannot override)
  - is_dangerous: risky but potentially valid commands → block (can override)

When Judge is unavailable, blocks all shell commands conservatively.
Also tracks consecutive tool errors for escalation across all tools.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.utils import (
    get_judge_result as _get_judge_result,
    SOURCE_LLM as _SOURCE_LLM,
    SOURCE_CACHE as _SOURCE_CACHE,
    SOURCE_DEFAULT as _SOURCE_DEFAULT,
    SOURCE_UNAVAILABLE as _SOURCE_UNAVAILABLE,
)


class ShellSafetyGuard(Guard):
    """Shell command safety (dual-level LLM judge) + error escalation.

    Checked first (priority=10). Uses LLM judge for all safety decisions.
    Two-level shell safety:
      - is_fatal → escalate (cannot override, irreversible catastrophe)
      - is_dangerous → block (can override with reason)
    """

    name = "safety"
    priority = 10


    # Escalation thresholds
    _ERROR_ESCALATE_WARN = 3
    _ERROR_ESCALATE_HARD = 5

    def __init__(self):
        self._consecutive_errors: int = 0
        self._root_cause_recorded_since_error: bool = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Only check shell commands for danger
        if ctx.tool_name != "shell":
            return None
        cmd = ctx.tool_args.get("command", "")
        if not cmd:
            return None

        classify = ctx.classify_fn
        if not classify:
            return GuardVerdict.block(
                "[Safety] Safety classifier unavailable — blocking shell command. "
                "Re-run with a working LLM provider, or use /mode confirm to manually approve.",
                reason="classify_fn not available for safety pre-check",
            )

        # Level 1: is_fatal — irreversible catastrophic commands (escalate, cannot override)
        is_fatal, fatal_source = _get_judge_result(
            classify, "is_fatal", {"command": cmd}, default=False,
        )
        if fatal_source in (_SOURCE_DEFAULT, _SOURCE_UNAVAILABLE):
            return GuardVerdict.block(
                "[Safety] Safety judge unavailable — blocking shell command. "
                f"Judge returned default value (source={fatal_source}). "
                "Re-run with a working LLM provider.",
                reason=f"safety classifier unavailable (source={fatal_source})",
            )
        if is_fatal:
            return GuardVerdict.escalate(
                "[Safety] FATAL: This command would cause irreversible catastrophic damage "
                "(e.g. destroy filesystems, wipe databases, brick systems). "
                "This cannot be overridden. Use a safer, more targeted approach.",
                reason="fatal command blocked by LLM judge — irreversible damage",
            )

        # Level 2: is_dangerous — risky but potentially valid (block, can override)
        is_dangerous, danger_source = _get_judge_result(
            classify, "is_dangerous", {"command": cmd}, default=False,
        )
        if is_dangerous:
            return GuardVerdict.block(
                "[Safety] Dangerous command detected and blocked. "
                "If this is intentional, explain why and use a "
                "more targeted approach.",
                reason="dangerous command blocked by LLM judge",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        result = ctx.tool_result or ""
        classify = ctx.classify_fn

        # Use LLM to determine if this is a real error
        is_error = False
        error_source = ""
        if classify and ctx.tool_name in ("shell", "write_file", "edit_file"):
            is_error, error_source = _get_judge_result(
                classify, "is_error", {
                    "tool_name": ctx.tool_name,
                    "command": ctx.tool_args.get("command", ""),
                    "result": result,
                }, default=False,
            )

        error_trustworthy = error_source in (_SOURCE_LLM, _SOURCE_CACHE)

        # Track memory_write as root-cause documentation (regardless of error status)
        if ctx.tool_name == "memory_write" and self._consecutive_errors > 0:
            self._root_cause_recorded_since_error = True

        if is_error:
            self._consecutive_errors += 1

            if self._consecutive_errors >= self._ERROR_ESCALATE_HARD:
                return GuardVerdict.escalate(
                    f"[Safety] {self._consecutive_errors} consecutive tool errors. "
                    "The current approach is not working. Stop, diagnose the root "
                    "cause, and reformulate your strategy before continuing.",
                    reason=f"hard escalation: {self._consecutive_errors} errors",
                )

            if self._consecutive_errors >= self._ERROR_ESCALATE_WARN:
                if not self._root_cause_recorded_since_error:
                    return GuardVerdict.inject(
                        f"[Safety] {self._consecutive_errors} consecutive tool errors "
                        "without recording root cause. Use memory_write to document "
                        "what's failing and why before retrying.",
                        reason="error escalation warn: no root cause recorded",
                    )
        else:
            if error_trustworthy:
                if self._consecutive_errors > 0:
                    self._consecutive_errors = 0
                self._root_cause_recorded_since_error = False

        # Track recovery via LLM success check
        if ctx.tool_name == "shell" and classify:
            is_success, success_source = _get_judge_result(
                classify, "is_success", {
                    "command": ctx.tool_args.get("command", ""),
                    "result": result,
                }, default=False,
            )
            if is_success and success_source in (_SOURCE_LLM, _SOURCE_CACHE):
                self._consecutive_errors = 0

        return None

    def reset_turn(self):
        pass  # Error state accumulates within a turn

    def reset_new_turn(self):
        """Decay consecutive errors on new user message.
        
        Halve the counter so cross-turn patterns still accumulate
        but a fresh topic gets some breathing room.
        """
        self._consecutive_errors = self._consecutive_errors // 2
        self._root_cause_recorded_since_error = False
