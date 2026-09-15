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

Detection is shell-aware so it does not over- or under-block:

  * A `find` command word is recognized at the start of the command OR after a
    shell statement separator: `|`, `;`, `&`, `&&`, `||`, a NEWLINE (a command on
    its own line is a real invocation), or the keywords `then`/`do`. Substrings
    like `findutils`, `myfind`, or `--find` are NOT matched.
  * Quoted regions ('...' / "...") and heredoc bodies (<<EOF ... EOF) are
    stripped BEFORE matching, because a `find` token appearing there is data
    (an echo string, a python docstring piped via a heredoc), not a command
    invocation. Without this, `echo 'a | find b'` would be a false positive.
  * Newlines are statement boundaries (re.MULTILINE), so a bare
    `\\nfind ...` on its own line is blocked — previously such a find escaped
    the guard entirely.
  * A RECURSIVE `grep` whose walk starts at a broad root (`/`, `/mnt`,
    `/public-nvme`, ...) is blocked the same way — a whole-tree recursive grep
    is as slow and filesystem-heavy as a bare find. Scoped greps
    (`grep -rn PATTERN ./src`), non-recursive greps, and greps into a specific
    subdirectory pass freely; they are what the guard recommends instead.
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Match `find` as a command word: at start of the command/line (re.MULTILINE so
# `^` also matches after a newline), or after a shell separator/pipe/keyword
# (so it catches `find ...`, `cd x && find ...`, `... | find`, `x\nfind ...`),
# but NOT substrings like `findutils`, `myfind`, or `--find`.
_FIND_RE = re.compile(
    r"(?:^|[|;&]|&&|\|\||\n|\bthen\b|\bdo\b)\s*find(?:\s|$)",
    re.IGNORECASE | re.MULTILINE,
)

# Heredoc introducer: `<<WORD`, `<<-WORD`, `<<'WORD'`, `<<"WORD"`. The word must
# start with a letter/underscore so a bitshift like `2 << 3` is not mistaken for
# a heredoc.
_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z_]\w*)['\"]?")

# Statement separators inside a shell command (newline included, since a command
# on its own line is a separate statement).
_STMT_SPLIT_RE = re.compile(r"\|\||&&|[;|&\n]")

# Recursive-grep flags. `-r`/`-R` and the long forms, plus combined short flags
# (e.g. `-rn`, `-Rl`) whose letters include r/R.
_GREP_RECURSIVE_LONG = {"--recursive", "--dereference-recursive"}

# A grep is "broad" when its recursive walk starts at one of these roots — a
# whole system tree or the top of a shared/NFS mount. Scanning one of these can
# take minutes and hammer the filesystem. A specific SUBdirectory (e.g.
# /public-nvme/proj/src) is scoped and allowed.
_GREP_BROAD_ROOTS = {
    "/", "/bin", "/boot", "/data", "/etc", "/home", "/lib", "/mnt", "/opt",
    "/proc", "/root", "/run", "/sbin", "/srv", "/sys", "/usr", "/var",
    "/clistorage", "/public-nvme", "/public-mixed",
}


# Prefix words that may precede the real command word (`sudo grep ...`).
_CMD_PREFIXES = {"sudo", "env", "command", "nohup", "time", "nice", "stdbuf"}


def _grep_is_broad(sanitized: str) -> bool:
    """True if `sanitized` invokes a RECURSIVE grep over a broad root.

    Only recursive greps targeting a whole system tree / shared-mount root are
    flagged. Scoped recursive greps (`grep -rn PATTERN ./src`), non-recursive
    greps (`grep PATTERN /etc/hosts`), and subdirectory targets all pass — the
    guard message itself recommends scoped `grep -rn <pattern> <dir>`.

    `grep` must be the COMMAND word of its statement (optionally after a prefix
    like `sudo`), so `echo grep -rn foo /` is not mistaken for a real grep.
    """
    for stmt in _STMT_SPLIT_RE.split(sanitized):
        toks = stmt.split()
        if not toks:
            continue
        # Locate the command word: skip leading VAR=val assignments and prefixes.
        idx = 0
        while idx < len(toks) and (
            "=" in toks[idx] or toks[idx] in _CMD_PREFIXES
        ):
            idx += 1
        if idx >= len(toks) or toks[idx] != "grep":
            continue

        recursive = False
        broad = False
        for t in toks[idx + 1:]:
            if t in _GREP_RECURSIVE_LONG:
                recursive = True
            elif t.startswith("-") and not t.startswith("--") and re.search(r"[rR]", t[1:]):
                recursive = True
            elif (t.rstrip("/") or "/") in _GREP_BROAD_ROOTS:
                broad = True
        if recursive and broad:
            return True
    return False


def _strip_heredocs(cmd: str) -> str:
    """Blank out heredoc bodies so `find` inside them is not seen as a command.

    A heredoc body is data fed to a program's stdin, not a shell command, so any
    `find` token there is not an invocation. Only blank lines up to a matching
    terminator; if no terminator is found, leave the text untouched (avoids
    mangling a line that merely contains `<<`).
    """
    lines = cmd.split("\n")
    n = len(lines)
    i = 0
    while i < n:
        delims = _HEREDOC_RE.findall(lines[i])
        if not delims:
            i += 1
            continue
        pending = list(delims)
        end = None
        j = i + 1
        while j < n and pending:
            tok = lines[j].strip()
            if tok in pending:
                pending.remove(tok)
                if not pending:
                    end = j
            j += 1
        if end is not None:
            for k in range(i + 1, end + 1):
                lines[k] = ""
            i = end + 1
        else:
            i += 1
    return "\n".join(lines)


def _strip_quoted(cmd: str) -> str:
    """Blank out single/double-quoted regions (respecting backslash escapes).

    A `find` token inside a quoted string is string data, not a command word.
    Newlines inside quotes are blanked too, so an unterminated/spanning quote
    cannot accidentally manufacture a statement boundary.
    """
    out: list[str] = []
    i = 0
    n = len(cmd)
    quote: str | None = None
    while i < n:
        c = cmd[i]
        if quote is None:
            if c == "\\" and i + 1 < n:
                out.append(c)
                out.append(cmd[i + 1])
                i += 2
                continue
            if c in ("'", '"'):
                quote = c
                out.append(" ")
                i += 1
                continue
            out.append(c)
            i += 1
        else:
            if c == "\\" and quote == '"' and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == quote:
                quote = None
                out.append(" ")
                i += 1
                continue
            out.append(" ")
            i += 1
    return "".join(out)


def _sanitize(cmd: str) -> str:
    """Remove heredoc bodies and quoted regions, leaving only executable text."""
    return _strip_quoted(_strip_heredocs(cmd))


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


_GREP_MESSAGE = (
    "[FindGuard] a RECURSIVE `grep` over a broad root (`/`, `/mnt`, "
    "`/public-nvme`, ...) is slow and hammers the filesystem — same cost class "
    "as a bare `find`. Prefer the internal information-retrieval order:\n"
    "\n"
    "  1. conversation_full.json / conversation.json (session dir) — the earlier "
    "hit may already be recorded. Near-zero cost.\n"
    "  2. memory — memory_list(keyword=...) or memory_read(key='fact/<domain>/').\n"
    "  3. scope the search — `grep -rn <pattern> <specific_subdir>` (NOT a whole "
    "tree); locate files by name with a fast indexed search.\n"
    "  4. ask the user for the path if it's a package/source location.\n"
    "\n"
    "If the broad recursive grep is genuinely necessary, override with "
    "_override_reason explaining why no scoped root applies."
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

        sanitized = _sanitize(command)

        # Block on EVERY find invocation (not once-per-turn): each new find that
        # lacks an override should be stopped. The registry's override mechanism
        # releases a single call when _override_reason is supplied for it.
        if _FIND_RE.search(sanitized):
            return GuardVerdict.block(
                _FIND_MESSAGE,
                reason="find_invocation",
                category="find_guard",
            )

        # Broad recursive grep: same cost class as a bare find.
        if _grep_is_broad(sanitized):
            return GuardVerdict.block(
                _GREP_MESSAGE,
                reason="broad_recursive_grep",
                category="find_guard",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None
