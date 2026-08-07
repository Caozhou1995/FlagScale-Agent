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

"""ShellSafetyGuard — shell command safety via dual-level LLM judge.

Two-level shell safety (no regex):
  - is_fatal: irreversible catastrophic commands → escalate (cannot override)
  - is_dangerous: risky but potentially valid commands → block (can override)

When Judge is unavailable, blocks all shell commands conservatively.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.utils import (
    get_judge_result as _get_judge_result,
    SOURCE_DEFAULT as _SOURCE_DEFAULT,
    SOURCE_UNAVAILABLE as _SOURCE_UNAVAILABLE,
)


class ShellSafetyGuard(Guard):
    """Shell command safety via dual-level LLM judge.

    Checked first (priority=10). Uses LLM judge for all safety decisions.
    Two-level shell safety:
      - is_fatal → escalate (cannot override, irreversible catastrophe)
      - is_dangerous → block (can override with reason)
    """

    name = "safety"
    priority = 10

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
                category="safety",
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
                category="safety",
            )
        if is_fatal:
            return GuardVerdict.escalate(
                "[Safety] FATAL: This command would cause irreversible catastrophic damage "
                "(e.g. destroy filesystems, wipe databases, brick systems). "
                "This cannot be overridden. Use a safer, more targeted approach.",
                reason="fatal command blocked by LLM judge — irreversible damage",
                category="safety",
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
                category="safety",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def reset_turn(self):
        pass
