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


# ── Hidden-find detection: payloads handed to remote/shell EXECUTORS ──
#
# The sanitize pass strips quoted regions to avoid false-positives on string
# data (`echo 'a | find b'`). But a quoted region passed to an EXECUTOR is not
# data — it is a payload the executor will run (possibly on a remote host or
# inside a container, i.e. on NFS trees even slower than local ones):
#
#   ssh host "docker exec c bash -lc 'find / -name x'"
#
# After stripping quotes, the guard sees only `ssh` and `head`. This block
# closes that gap: when a statement's command word is an executor, its quoted
# arguments and heredoc body are re-scanned recursively for the same
# violations (find / broad recursive grep), bounded depth.

_EXECUTORS = {
    "ssh", "scp", "sftp", "mosh", "kubectl", "docker", "podman", "nerdctl",
    "ctr", "crictl", "nsenter", "su", "sudo", "doas", "setsid", "stdbuf",
    "nohup", "xargs", "parallel", "env",
}
_SHELL_WRAPPERS = {"bash", "sh", "zsh", "fish", "csh", "tcsh", "ksh", "dash"}

_MAX_PAYLOAD_DEPTH = 6


def _extract_payloads(sanitized: str) -> list[str]:
    """Collect quoted regions from `sanitized` (heredocs already blanked).

    Returns each quoted region's raw contents, so inner layers of nested
    quoting remain visible to the re-scanner (ssh "docker ... 'find ...'").
    """
    payloads: list[str] = []
    i, n = 0, len(sanitized)
    quote = None
    buf: list[str] = []
    while i < n:
        c = sanitized[i]
        if quote is None:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c in ("'", '"'):
                quote = c
                i += 1
                continue
            i += 1
        elif c == "\\" and quote == '"' and i + 1 < n:
            buf.append(c)
            buf.append(sanitized[i + 1])
            i += 2
        elif c == quote:
            payloads.append("".join(buf))
            buf = []
            quote = None
            i += 1
        else:
            buf.append(c)
            i += 1
    if quote is not None and buf:
        payloads.append("".join(buf))
    return payloads


def _statement_command_word(toks: list[str]) -> str | None:
    """First real command word of a token list, skipping VAR=val and prefixes."""
    idx = 0
    while idx < len(toks) and ("=" in toks[idx] or toks[idx] in _CMD_PREFIXES):
        idx += 1
    return toks[idx] if idx < len(toks) else None


def _cmd_base(word: str | None) -> str:
    """Basename of a command word, quote-stripped.

    `/usr/bin/find` -> `find`; `find"` (closing quote glued to the token
    after quote-region splitting) -> `find`.
    """
    return word.strip("'\"").rsplit("/", 1)[-1] if word else ""


def _statement_find_violation(text: str) -> GuardVerdict | None:
    """find as a statement's command word.

    Covers forms the anchored _FIND_RE misses: absolute paths
    (`/usr/bin/find ...`) and prefix forms (`sudo find ...`, `nohup find
    ...`, `VAR=val find ...`). Used at both top level and inside payloads.
    """
    for stmt in _STMT_SPLIT_RE.split(text):
        if _cmd_base(_statement_command_word(stmt.split())) == "find":
            return GuardVerdict.block(
                _FIND_MESSAGE, reason="find_invocation", category="find_guard",
            )
    return None


def _violation_in_shell_text(text: str) -> GuardVerdict | None:
    """Scan an already-sanitized shell fragment for find / broad grep."""
    v = _statement_find_violation(text)
    if v is not None:
        return v
    if _FIND_RE.search(text):
        return GuardVerdict.block(
            _FIND_MESSAGE, reason="find_invocation", category="find_guard",
        )
    if _grep_is_broad(text):
        return GuardVerdict.block(
            _GREP_MESSAGE, reason="broad_recursive_grep", category="find_guard",
        )
    return None


def _cmd_subst_violation(sanitized: str) -> GuardVerdict | None:
    """find inside `$( ... )` or backticks (command substitution executes)."""
    for m in re.finditer(r"\$\(([^()]*)\)|`([^`]*)`", sanitized):
        inner = m.group(1) or m.group(2) or ""
        # Prefix with a dummy separator so a leading find is a command word.
        v = _violation_in_shell_text("dummy_sep; " + inner)
        if v is not None:
            return v
    return None


def _xargs_find_violation(toks: list[str]) -> GuardVerdict | None:
    """find as a bare token after an executor: `xargs find` (stdin-driven
    walk) and `kubectl exec pod -- find ...` (no quoting layer to carry the
    payload, the find token rides bare in the statement)."""
    head = _cmd_base(_statement_command_word(toks))
    if head not in _EXECUTORS and head != "xargs":
        return None
    for tok in toks[1:]:
        if _cmd_base(tok) == "find":
            return GuardVerdict.block(
                _FIND_MESSAGE, reason="find_invocation", category="find_guard",
            )
    return None


def _hidden_violation(cmd: str, depth: int = 0) -> GuardVerdict | None:
    """Detect find / broad grep hidden inside executor payloads.

    `cmd` is the RAW command; this function blanks heredoc bodies itself (a
    heredoc is stdin data, never executed) but KEEPS quoted regions — they
    are payloads an executor (ssh, docker, kubectl, bash -lc, ...) will run,
    often against remote hosts or NFS trees where a stray recursive find is
    the slowest of all.

    Recursion: a payload's own statement may itself be an executor (ssh ->
    docker exec -> bash -lc), so payloads are re-scanned with the same rule,
    bounded by _MAX_PAYLOAD_DEPTH. Pure data payloads (echo / python -c) are
    never re-scanned — their command word is not an executor.
    """
    if depth >= _MAX_PAYLOAD_DEPTH:
        return None
    blanked = _strip_heredocs(cmd)
    for stmt in _STMT_SPLIT_RE.split(blanked):
        toks = stmt.split()
        v = _xargs_find_violation(toks)
        if v is not None:
            return v
        word = _cmd_base(_statement_command_word(toks))
        if word in _EXECUTORS or word in _SHELL_WRAPPERS:
            for payload in _extract_payloads(stmt):
                v = _violation_in_shell_text(payload)
                if v is None:
                    v = _hidden_violation(payload, depth + 1)
                if v is not None:
                    return v
    return _cmd_subst_violation(blanked)


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
        # _statement_find_violation adds token-anchored coverage for absolute
        # paths (/usr/bin/find) and prefix forms (sudo/nohup/VAR=val find).
        if _FIND_RE.search(sanitized) or _statement_find_violation(sanitized):
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

        # Hidden invocations: find / broad grep inside payloads handed to
        # remote/shell executors (ssh, docker, kubectl, bash -lc, ...) or
        # command substitution. Quote-stripping above makes these invisible;
        # they still execute (often on remote NFS trees, where a stray
        # recursive find is the slowest of all). Pass the RAW command —
        # _hidden_violation re-derives its own quote-preserving view.
        hidden = _hidden_violation(command)
        if hidden is not None:
            return hidden

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None
