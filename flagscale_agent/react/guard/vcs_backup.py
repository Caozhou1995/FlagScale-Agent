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

"""VcsBackupGuard — targeted interception of destructive git operations.

Scope is deliberately narrow: only git commands that can permanently destroy
work — uncommitted changes, recoverable VCS state, or history / the object
store. The upfront backup reminder lives in StartupGuard's BackupPhase (first
tool call) and is NOT touched by this guard.

Matching is SEGMENT-level, not whole-string substring: the command is lexed
into statement segments (shlex with punctuation chars — an ``&&``/``;``
inside quotes does NOT split), each segment is stripped of env assignments
and wrapper binaries (``sudo``/``env``/``nohup``/...), and only segments that
actually INVOKE git are classified. Consequences:
  - a git command merely MENTIONED in ``grep``/``echo``/``cat`` never fires
    (that segment does not run git);
  - the read-only ``git filter-repo --analyze`` exemption is decided per
    segment at token level, so a chained ``;``/``&&`` statement can never
    borrow another statement's exemption, and a file argument that merely
    contains the text (``/tmp/--analyze.txt``) never grants it;
  - command substitutions (``$(...)`` / backticks run at runtime even inside
    quotes) and ``bash -c`` bodies are recursed into.
Known fail-open limitations (documented): a heredoc body is not re-parsed as
shell; indirection through an unexpanded variable (``"$CMD"``) or a wrapper
outside the wrapper list (``xargs``) is not seen.

Patterns are deterministic (style law 2: no LLM) and organized by DAMAGE
CLASS, not by command name — new commands must be classified first, then
covered by their class. Safe variants are excluded by construction (see
tests).
"""

from __future__ import annotations

import re
import shlex

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

# ----------------------------------------------------------------- lexing --

_PUNCT = ";&|"
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_WRAPPER_CMDS = {"env", "sudo", "nohup", "timeout", "nice"}
# Wrapper flags that consume the NEXT token as their value.
_WRAPPER_VALUE_FLAGS = {"-u", "-g", "--user", "--group", "--unset", "-n", "-k"}
_WRAPPER_DURATION_RE = re.compile(r"^\d+(?:\.\d+)?[smhdw]?$")  # timeout 30 / 30s
_SHELL_BINARIES = {"bash", "sh", "zsh", "ksh", "dash"}
_SUBST_RE = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")
_MAX_DEPTH = 2  # bash -c / $(...) recursion budget
def _lex_segments(command: str) -> list[str]:
    """Split a command into statement segments at ; & | (and newlines).

    Quoted operators do not split (shlex honors quotes), so
    ``echo 'run git gc --prune=now'`` stays one segment headed by ``echo``.
    Unbalanced quoting fails closed: raw split on the operators.
    """
    text = command.replace("\n", "; ")  # a newline is a statement boundary
    lex = shlex.shlex(text, posix=True, punctuation_chars=_PUNCT)
    lex.whitespace_split = True
    lex.commenters = ""
    try:
        tokens = list(lex)
    except ValueError:
        return [s.strip() for s in re.split(r"[;&|]", text) if s.strip()]
    segments: list[str] = []
    current: list[str] = []
    for tok in tokens:
        if tok and all(ch in _PUNCT for ch in tok):
            if current:
                segments.append(" ".join(current))
                current = []
        else:
            current.append(tok)
    if current:
        segments.append(" ".join(current))
    return segments


def _strip_wrappers(tokens: list[str]) -> list[str]:
    """Drop leading env assignments and wrapper binaries (``sudo -u w git ...``)."""
    for _ in range(len(tokens) + 1):  # every pass consumes at least one token
        if not tokens:
            return tokens
        head = tokens[0]
        if _ENV_ASSIGN_RE.match(head):
            tokens = tokens[1:]
            continue
        if head not in _WRAPPER_CMDS:
            return tokens
        tokens = tokens[1:]
        while tokens and tokens[0] in _WRAPPER_VALUE_FLAGS:
            tokens = tokens[2:]  # flag plus its value
        if tokens and tokens[0] == "--":
            tokens = tokens[1:]
        if head == "timeout" and tokens and _WRAPPER_DURATION_RE.match(tokens[0]):
            tokens = tokens[1:]
    return tokens


# git global options that consume the next token (``git -C <path> log ...``)
_GIT_GLOBAL_VALUE_OPTS = {
    "-C", "-c", "--git-dir", "--work-tree", "--namespace",
    "--super-prefix", "--shallow-file",
}


def _git_subcommand(tokens: list[str]) -> tuple:
    """(subcommand, args) of a ``git ...`` token list; (None, []) if absent."""
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in _GIT_GLOBAL_VALUE_OPTS:
            i += 2
        elif tok.startswith("-"):
            i += 1  # unknown flag: never assume it swallows a value
        else:
            return tok, tokens[i + 1:]
    return None, []
def _units(command: str, depth: int = 0):
    """Yield each token list that would RUN: statement segments, plus
    ``bash -c`` bodies and command substitutions (they execute at runtime
    even inside quotes)."""
    for seg in _lex_segments(command):
        # Command substitutions run at runtime even inside quotes — extract
        # them from the RAW segment: an env-assign head like ``x=$(git ...)``
        # would otherwise be dropped as an assignment, losing the body.
        if depth < _MAX_DEPTH:
            for match in _SUBST_RE.finditer(seg):
                body = (match.group(1) or match.group(2) or "").strip()
                if body:
                    yield from _units(body, depth + 1)
        toks = _strip_wrappers(seg.split())
        if not toks:
            continue
        head = toks[0]
        if depth < _MAX_DEPTH and head in _SHELL_BINARIES and "-c" in toks[1:]:
            script = " ".join(toks[toks.index("-c", 1) + 1:])
            if script.strip():
                yield from _units(script, depth + 1)
                continue
        yield toks


def _git_units(command: str):
    """(subcommand, args) for every git invocation the command would run."""
    for toks in _units(command):
        head = toks[0]
        if head.endswith("/git"):
            head = "git"
        elif head.endswith("/git-filter-repo"):
            head = "git-filter-repo"
        if head == "git":
            sub, args = _git_subcommand(toks)
        elif head == "git-filter-repo":
            sub, args = "filter-repo", toks[1:]
        else:
            continue  # a grep/echo/cat segment mentions git; it does not RUN git
        yield sub, args


# --------------------------------------------------------------- matchers --
# One matcher per damage form, decided from (subcommand, args) of ONE git
# invocation. Token-exact matching is what excludes safe variants (mentions,
# dry-runs, report-only modes) by construction.

def _m_checkout_discard(sub, args):
    """``git checkout`` that discards working-tree changes."""
    if sub != "checkout":
        return False
    if "--" in args:
        return True  # explicit pathspec form overwrites those paths
    positional = [t for t in args if not t.startswith("-")]
    if not positional:
        return True  # bare checkout (legacy tripwire kept)
    return all(t in (".", "..") for t in positional)  # discard-all dot forms


def _m_reset_hard(sub, args):
    return sub == "reset" and "--hard" in args


def _m_clean_force(sub, args):
    """``git clean`` without a dry-run flag (an ``n`` short-flag or --dry-run)."""
    if sub != "clean":
        return False
    pre = args[: args.index("--")] if "--" in args else args
    if "--dry-run" in pre:
        return False
    return not any(
        t.startswith("-") and not t.startswith("--") and "n" in t[1:] for t in pre
    )


def _m_restore_discard(sub, args):
    """``git restore`` touching the working tree (anything not staged-only)."""
    if sub != "restore":
        return False
    if "--staged" in args and not ({"--worktree", "-W"} & set(args)):
        return False  # unstage only
    return True
def _m_stash_destroy(sub, args):
    return sub == "stash" and bool(args) and args[0] in ("drop", "clear")


def _m_branch_delete(sub, args):
    return sub == "branch" and bool({"-d", "-D", "--delete"} & set(args))


def _m_push_force(sub, args):
    if sub != "push":
        return False
    return any(
        t in ("--force", "-f", "-F") or t.startswith("--force-with-lease")
        for t in args
    )


# Class 3 — REWRITE or irreversibly DESTROY history / the object store.
# Commit hashes change or objects vanish; a stash does not protect these, so
# the backup ritual does not discharge them (see check_pre).

def _m_rebase(sub, args):
    return sub == "rebase"


def _m_filter_branch(sub, args):
    return sub == "filter-branch"


def _m_filter_repo(sub, args):
    """git-filter-repo rewrites objects unless in report-only --analyze mode.

    The exemption is a per-segment TOKEN check: an ``--analyze*`` option token
    in THIS invocation. A path argument that merely contains the text
    (``/tmp/--analyze.txt``) does not start with ``--analyze`` and never
    exempts; a later ``git filter-repo --analyze`` segment cannot whitewash an
    earlier rewriting segment (F1). A longer ``--analyze*`` word is an
    unrecognized option — git aborts before doing anything, so it is harmless
    (upstream sanity-check refuses --analyze combined with rewriting options).
    """
    if sub != "filter-repo":
        return False
    return not any(t.startswith("--analyze") for t in args)


def _m_reflog_destroy(sub, args):
    return sub == "reflog" and bool(args) and args[0] in ("expire", "delete")


def _m_gc_prune_now(sub, args):
    """``git gc --prune`` keeps the default 2-week grace — same damage class
    as bare ``git gc``; only the immediate ``--prune=now`` destroys now (F2)."""
    return sub == "gc" and "--prune=now" in args
_CLASS1_MATCHERS = (
    _m_checkout_discard, _m_reset_hard, _m_clean_force, _m_restore_discard,
)
_CLASS2_MATCHERS = (_m_stash_destroy, _m_branch_delete, _m_push_force)
_CLASS3_MATCHERS = (
    _m_rebase, _m_filter_branch, _m_filter_repo, _m_reflog_destroy,
    _m_gc_prune_now,
)
_DESTRUCTIVE_MATCHERS = _CLASS1_MATCHERS + _CLASS2_MATCHERS + _CLASS3_MATCHERS

_DESTRUCTIVE_MESSAGE = """[VcsBackupGuard] This git command can destroy work IRREVERSIBLY — it may wipe uncommitted changes, a recoverable state you may need (stash/branch/reflog), or rewrite history / delete objects so commit hashes and old content are gone for good (lesson: `git checkout` erased an uncommitted checkpoint patch — cost days of experiments to recover).

Before running it, snapshot the dirty tree — it costs one command and keeps the working tree unchanged:
  git stash push -u -m "backup-before-destructive"   # includes untracked (-u)
  # ... run your destructive command ...
  git stash apply                                     # restore; drop ONLY after verified recovery

If a stash/backup already exists, or you keep patches another way (bundle, patch file, remote branch), proceed:
  _override_reason: "backup exists: <stash id / patch path / branch name>"

If the tree is already clean or the deleted data is regenerable, override with that explanation."""
class VcsBackupGuard(Guard):
    """Targeted guard for destructive git operations."""

    name = "vcs_backup"
    priority = 15  # after StartupGuard(5)/Backup reminder, before shell safety(20+)

    def __init__(self):
        # Matcher ids already acknowledged via a valid override this turn.
        self._acked: set[int] = set()

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None
        command = str(ctx.tool_args.get("command", ""))
        if not command:
            return None
        units = list(_git_units(command))
        if not units:
            return None
        # The backup ritual (a real stash-push STATEMENT, not a mention)
        # discharges only the damage classes a stash actually protects
        # (Classes 1-2: uncommitted work, recoverable VCS state) — so the
        # recipe itself never blocks. Class 3 rewrites history / deletes
        # objects a stash cannot restore: it still blocks and needs an
        # explicit override reason. The drop/clear carve-out stays (never
        # whitelist destruction of pre-existing backups).
        ritual = any(s == "stash" and a and a[0] == "push" for s, a in units)
        stomps = any(
            s == "stash" and a and a[0] in ("drop", "clear") for s, a in units
        )
        if ritual and not stomps and not any(
            m(s, a) for m in _CLASS3_MATCHERS for s, a in units
        ):
            return None
        for idx, matcher in enumerate(_DESTRUCTIVE_MATCHERS):
            if idx in self._acked:
                continue
            if any(matcher(s, a) for s, a in units):
                return GuardVerdict.block(
                    message=_DESTRUCTIVE_MESSAGE,
                    reason=f"destructive_git_pattern_{idx}",
                    category="vcs_backup",
                    overridable=True,
                )
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Release only when the reason cites a concrete backup channel
        (deterministic keyword check, style law 4). One matcher per override:
        a new destructive form requires its own acknowledgment."""
        if not reason or len(reason.strip()) <= 5:
            return False
        lowered = reason.lower()
        if not any(
            kw in lowered
            for kw in ("stash", "backup", "bak", "patch", "bundle", "branch",
                       "clean", "regenerat")
        ):
            return False
        command = str(ctx.tool_args.get("command", "")) if ctx else ""
        for idx, matcher in enumerate(_DESTRUCTIVE_MATCHERS):
            if any(matcher(s, a) for s, a in _git_units(command)):
                self._acked.add(idx)
        return True

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def reset_turn(self):
        self._acked.clear()
