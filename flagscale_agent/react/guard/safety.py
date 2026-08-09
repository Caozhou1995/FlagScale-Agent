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

"""ShellSafetyGuard — shell command safety via LLM judge.

Two-level shell safety:
  - is_fatal: irreversible catastrophic commands → escalate (cannot override)
  - is_dangerous: risky but potentially valid commands → block (can override)
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class ShellSafetyGuard(Guard):
    """Shell command safety via LLM judge.

    Checked first (priority=10).
    Two-level shell safety:
      - is_fatal → escalate (cannot override, irreversible catastrophe)
      - is_dangerous → block (can override with reason)
    """

    name = "safety"
    priority = 10

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None
        cmd = ctx.tool_args.get("command", "")
        if not cmd:
            return None

        classify = ctx.classify_fn
        if not classify:
            return None  # No judge = agent not functional, skip

        # Level 1: is_fatal — irreversible catastrophic commands
        if classify("is_fatal", {"command": cmd}, default=False):
            return GuardVerdict.escalate(
                "[Safety] FATAL: This command would cause irreversible catastrophic damage "
                "(e.g. destroy filesystems, wipe databases, brick systems). "
                "This cannot be overridden. Use a safer, more targeted approach.",
                reason="fatal command blocked by LLM judge — irreversible damage",
                category="safety",
            )

        # Level 2: is_dangerous — risky but potentially valid
        if classify("is_dangerous", {"command": cmd}, default=False):
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
