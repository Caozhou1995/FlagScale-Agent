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

"""FindGuard — block `find` invocations in favor of internal retrieval.

`find` over a large directory tree (especially / or a shared/NFS mount) is slow
and hammers the filesystem. Most path/file lookups are already answered by the
internal information-retrieval order (conversation logs → memory → scoped grep →
ask the user). This guard blocks any shell command that invokes `find` and
points the agent at those cheaper channels first.

The block is overridable: a genuinely necessary, tightly-scoped find can proceed
with an _override_reason.
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Match `find` as a command word: at start of the command, or after a shell
# separator/pipe (so it catches `find ...`, `cd x && find ...`, `... | find`),
# but NOT substrings like `findutils`, `myfind`, or `--find`.
_FIND_RE = re.compile(r"(?:^|[|;&]|&&|\|\||\bthen\b|\bdo\b)\s*find\s+", re.IGNORECASE)


_FIND_MESSAGE = (
    "[FindGuard] `find` on a large directory tree is slow and often the wrong "
    "tool. Before running find, follow the internal information-retrieval order:\n"
    "\n"
    "  1. conversation_full.json / conversation.json (session dir) — grep for the "
    "path/file you're chasing. Near-zero cost, it may already be recorded.\n"
    "  2. memory — memory_list(keyword=...) or memory_read(key='fact/<domain>/'). "
    "A path you discovered before is likely already saved.\n"
    "  3. scoped search tools — locate files by name with a fast indexed search; "
    "search contents with `grep -rn <pattern> <specific_dir>`. Both beat a bare "
    "`find /` walk.\n"
    "  4. ask the user for the path if it's a package/source location.\n"
    "\n"
    "A broad `find /` or a find over a big shared/NFS tree can take minutes and "
    "hammer the filesystem — that is what this guard blocks.\n"
    "\n"
    "If your find is genuinely necessary AND tightly scoped (bounded root, "
    "-maxdepth / -name filters keeping the walk cheap), override with "
    "_override_reason explaining the root is bounded and the cheaper channels "
    "above don't apply."
)


class FindGuard(Guard):
    """Block shell commands that invoke `find`; overridable when scoped."""

    name = "find_guard"
    priority = 25

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None

        command = ctx.tool_args.get("command", "")
        if not command:
            return None

        # Block on EVERY find invocation (not once-per-turn): each new find that
        # lacks an override should be stopped. The registry's override mechanism
        # releases a single call when _override_reason is supplied for it.
        if _FIND_RE.search(command):
            return GuardVerdict.block(
                _FIND_MESSAGE,
                reason="find_invocation",
                category="find_guard",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None
