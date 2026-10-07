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

"""DeliverableSurvivalGuard — post-destructive-command survival recheck.

Fires after a destructive command SUCCEEDED (tool_result shows the command
ran) and demands the agent re-verify that its deliverables still exist at
their delivery paths. Evidence class (2026-10-07): an agent killed its own
shell with a self-matching `pkill -f` pattern mid-run — every earlier verified
state (solution file on disk) was destroyed with the process, and the agent's
belief that "the deliverable exists" was never re-checked before it claimed
done. Injection-only: it never blocks; it re-arms the survival question the
moment anything destructive lands.
"""

from __future__ import annotations

import re

from . import Guard, GuardContext, GuardVerdict

# Command shapes that can destroy prior work. Matched against the command
# string only — scope judgments belong to the agent, the guard only re-arms
# the question. `kill`-family patterns cover the self-kill case (the shell
# itself dying destroys unflushed state and orphans the run); rm/mv/truncate/
# redirection cover direct artifact destruction; git clean/checkout/reset
# --hard cover tree destruction.
_DESTRUCTIVE_RE = re.compile(
    r"(^|[;&|]\s*)(sudo\s+)?rm\s"
    r"|(^|[;&|]\s*)(sudo\s+)?mv\s+\S+\s+/"
    r"|(^|[;&|]\s*)truncate\s"
    r"|(^|[;&|]\s*)(pkill|killall|kill)\b"
    r"|(^|[;&|]\s*)git\s+(clean|checkout\s+--|reset\s+--hard|restore\s)"
    r"|(^|[;&|]\s*)mkfs\."
    r"|(^|[;&|]\s*)dd\s+.*of=/dev/"
    r"|>\s*/\S"  # redirection clobbering an absolute path
    r"|\bshred\b",
)

_SURVIVAL_MESSAGE = """
[DeliverableSurvival] A destructive command just ran. Do NOT treat your prior
belief that the deliverable exists as still valid — re-verify survival NOW:
ls -la the exact delivery path(s) named by the task (and any symlink targets)
and read enough bytes to confirm it is the real artifact, not a truncated or
placeholder file. If anything is missing or corrupted, rebuild it from the
last working state BEFORE continuing — and if the command was a self-kill
(pkill/killall matching this shell's own cmdline), treat everything since the
last verified checkpoint as untrusted and re-run the survival check first.
"""


class DeliverableSurvivalGuard(Guard):
    """Post-shell guard: demand a deliverable survival recheck after destruction."""

    name = "deliverable_survival"
    # Advisory tier, same family as post_edit_far_end (70) — inject-only,
    # never blocks; runs late so it observes the FINAL command text.
    priority = 72

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None  # Inject-only

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None
        command = str(ctx.tool_args.get("command") or "")
        if not command or not _DESTRUCTIVE_RE.search(command):
            return None
        result = (ctx.tool_result or "").lstrip()
        # Fire only when the command actually EXECUTED. A blocked/erroring
        # command changed nothing, so there is nothing to re-verify.
        if not result or result.startswith("ERROR") or result.startswith(
            "[BLOCKED BY GUARD]"
        ):
            return None
        return GuardVerdict.inject(
            _SURVIVAL_MESSAGE,
            reason="deliverable_survival_recheck",
            # Independent category — registry deduplicates injects by
            # category; a shared one would be swallowed by another guard's
            # inject in the same pass.
            category="deliverable_survival",
        )
