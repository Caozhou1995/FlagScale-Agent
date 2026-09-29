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
_WRAPPER_CMDS = {"env", "sudo", "nohup", "timeout", "nice", "xargs"}
# Per-wrapper flags that consume the NEXT token as their value. Any OTHER
# ``-...`` token under a wrapper is treated as valueless (dropped): an
# unknown valueless flag must not bail the scan (``sudo -H git ...``), and
# glued values (``-n5``, ``-I{}``) are single tokens. A value flag missing
# from this table makes the FOLLOWING token the head — a documented
# fail-open (keep the table current).
_WRAPPER_VALUE_FLAGS = {
    "env": {"-u", "--unset", "-S", "--split-string", "--file"},
    "sudo": {"-u", "-g", "-p", "-r", "-t", "-C", "-D",
             "--user", "--group", "--prompt", "--role", "--type",
             "--close-from", "--chdir"},
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "xargs": {"-I", "-E", "-n", "-P", "-L", "-s", "-a", "-e", "-d"},
}
_WRAPPER_DURATION_RE = re.compile(r"^\d+(?:\.\d+)?[smhdw]?$")  # timeout 30 / 30s
_SHELL_BINARIES = {"bash", "sh", "zsh", "ksh", "dash"}
_SUBST_RE = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")
_MAX_DEPTH = 2  # bash -c / $(...) recursion budget


def _quote_mask(text: str) -> list[bool]:
    """Per-character quote state: True = literal (single-quoted, or
    double-quoted with \\/$`` unescaped). Backslash escapes everywhere."""
    mask = [False] * len(text)
    i, n = 0, len(text)
    state = ""  # '', "'"
    while i < n:
        ch = text[i]
        if state == "'":
            if ch == "'":
                state = ""
            else:
                mask[i] = True
            i += 1
        else:
            if ch == "'":
                state = "'"
                mask[i] = True
                i += 1
            elif ch == "\\" and i + 1 < n:
                mask[i] = mask[i + 1] = True  # escaped char is literal
                i += 2
            else:
                i += 1
    return mask


def _substitution_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, body) of RUNTIME command substitutions — unquoted
    ``$(...)`` and backticks only. Single-quoted (F2/F3) never substitutes;
    inside double quotes they DO."""
    spans: list[tuple[int, int, str]] = []
    mask = _quote_mask(text)
    i, n = 0, len(text)
    while i < n:
        if mask[i]:
            i += 1
            continue
        if text.startswith("$(", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            spans.append((i, j, text[i + 2:j - 1] if depth == 0 else text[i + 2:]))
            i = j
        elif text[i] == "`":
            j = text.find("`", i + 1)
            if j == -1:
                break
            spans.append((i, j + 1, text[i + 1:j]))
            i = j + 1
        else:
            i += 1
    return spans


_HEREDOC_RE = re.compile(r"<<(-?)(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
_SHELL_FEED_RE = re.compile(r"\|\s*(?:sudo\s+|env\s+\S+\s+|nice\s+\S+\s+)?(?:ba|z|da|k)?sh\b")


def _inquote_mask(text: str) -> list[bool]:
    """Per-character 'inside quotes' state (single or double, backslash
    escapes honored) — used to ignore heredoc operators that appear inside
    quoted text (``echo \"use <<EOF syntax\"`` is not a heredoc)."""
    inside = [False] * len(text)
    i, n = 0, len(text)
    state = ""
    while i < n:
        ch = text[i]
        if state == "'":
            inside[i] = True
            if ch == "'":
                state = ""
        elif state == '"':
            inside[i] = True
            if ch == '"' and (i == 0 or text[i - 1] != "\\"):
                state = ""
        elif ch == "'" or ch == '"':
            inside[i] = True
            state = ch
        elif ch == "\\" and i + 1 < n:
            inside[i] = inside[i + 1] = True
            i += 2
            continue
        i += 1
    return inside


def _split_heredocs(text: str):
    """Split into (kind, text) pieces. Semantics of a heredoc BODY:
    - quoted delimiter (<<'EOF')  -> literal data, nothing runs ('data');
    - unquoted delimiter          -> data, BUT unquoted ``$(...)``/backtick
      substitutions still run ('subst-body': scan those only);
    - unquoted body piped into a shell ('| sh') -> the whole body executes
      ('body-shell': full re-lex, fail-closed).
    Unterminated bodies are dropped fail-open. Operators inside quoted
    text are not operators."""
    pieces: list[tuple[str, str]] = []
    pos = 0
    inquote = _inquote_mask(text)
    for match in _HEREDOC_RE.finditer(text):
        if inquote[match.start()]:
            continue
        start = match.start()
        if start >= pos:
            pieces.append(("code", text[pos:start]))
        delim = match.group(3)
        quoted = bool(match.group(2))
        body_re = re.compile(r"(?:\n|^)" + re.escape(delim) + r"(?=\n|$)")
        body_match = body_re.search(text, match.end())
        if not body_match:
            pieces.append(("data", text[match.end():]))  # unterminated
            return pieces, True
        body = text[match.end():body_match.start()]
        if quoted:
            pieces.append(("data", body))
        elif _SHELL_FEED_RE.search(text[body_match.end():]):
            pieces.append(("body-shell", body))
        else:
            pieces.append(("subst-body", body))
        pos = body_match.end()
    pieces.append(("code", text[pos:]))
    return pieces, False


def _split_segments_quoted(text: str) -> list[str]:
    """Split into statement segments at UNQUOTED ``; & |`` and newlines,
    PRESERVING the original characters (quotes stay in the segment) —
    what ``_substitution_spans`` needs to see single-quoted text as
    literal (shlex-based ``_lex_segments`` strips quotes)."""
    segments: list[str] = []
    current: list[str] = []
    i, n = 0, len(text)
    state = ""  # '', "'", '"'
    while i < n:
        ch = text[i]
        if state == "'":
            if ch == "'":
                state = ""
        elif state == '"':
            if ch == '"' and (i == 0 or text[i - 1] != "\\"):
                state = ""
        elif ch == "'" or ch == '"':
            state = ch
        elif ch in ";&|" or ch == "\n":
            segments.append("".join(current))
            current = []
            i += 1
            continue
        elif ch == "\\" and i + 1 < n:
            current.append(text[i:i + 2])
            i += 2
            continue
        current.append(ch)
        i += 1
    if current:
        segments.append("".join(current))
    return [s.strip() for s in segments if s.strip()]


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


def _strip_wrapper(tokens: list[str], wrapper: str) -> list[str]:
    """Strip ONE wrapper invocation head; caller re-reduces the result."""
    value_flags = _WRAPPER_VALUE_FLAGS.get(wrapper, set())
    tokens = tokens[1:]
    while tokens:
        tok = tokens[0]
        if tok == "--":
            return tokens[1:]
        if tok in value_flags:
            tokens = tokens[2:]  # flag plus its value
        elif wrapper == "timeout" and _WRAPPER_DURATION_RE.match(tok):
            tokens = tokens[1:]  # positional duration before the command
        elif tok.startswith("-"):
            tokens = tokens[1:]  # valueless flag (sudo -H, env -i, nice -5)
        else:
            break
    return tokens


# `eval` re-parses its body — recurse; transparent pass-throughs; and the
# keyword/group heads that may precede a git statement in a compound command.
_EVAL_PASSTHROUGH = {"command", "exec"}
_KEYWORD_HEADS = {"then", "do", "!", "{", "time", "coproc"}
_GROUP_HEADS = {"(", "(("}


def _reduce_head(tokens: list[str], depth: int, out: list):
    """Reduce a statement's head to the effective program: peel keywords,
    groups, env assignments and wrappers, recursing into `eval` bodies.
    What survives is appended to `out` as token lists."""
    for _ in range(len(tokens) + 1):
        if not tokens:
            return
        # group chars glued to tokens: ``(git ... --hard)`` — one shlex token
        while tokens and tokens[0][:1] in "({":
            tokens[0] = tokens[0][1:]
            if not tokens[0]:
                tokens = tokens[1:]
        if tokens and tokens[-1][-1:] in ")}":
            tokens[-1] = tokens[-1][:-1]
            if not tokens[-1]:
                tokens = tokens[:-1]
        if not tokens:
            return
        # backslash escape on the program head: ``\git ...`` runs git
        if len(tokens[0]) > 1 and tokens[0][0] == "\\":
            tokens[0] = tokens[0][1:]
        head = tokens[0]
        if _ENV_ASSIGN_RE.match(head):
            tokens = tokens[1:]
            continue
        if head in _KEYWORD_HEADS:
            tokens = tokens[1:]
            continue
        if head in _GROUP_HEADS:
            tokens = tokens[1:]
            continue
        if head in _EVAL_PASSTHROUGH:
            tokens = tokens[1:]
            continue
        if head == "eval":
            if depth >= _MAX_DEPTH:  # recursion budget exhausted
                out.append(tokens)
                return
            # `eval` runs TWO parses: (1) its line is word-split into argv
            # (one layer of quotes stripped), then (2) the joined argv is
            # re-parsed as a FULL command line — statements, substitutions,
            # nested evals all come alive. Re-run the whole pipeline on it.
            # (Unknown-variable bodies like eval "$X" are invisible to any
            # static lexer — out of scope, same as v1.)
            try:
                argv = shlex.split(" ".join(tokens[1:]))
            except ValueError:
                argv = tokens[1:]  # fail-closed: keep raw fragments
            if argv:
                out.extend(_units(" ".join(argv), depth + 1))
            return
        if head in _WRAPPER_CMDS:
            tokens = _strip_wrapper(tokens, head)
            continue
        out.append(tokens)
        return
    out.append(tokens)  # pathological peel loop


def _reduce_to_program(text_tokens: list[str], depth: int = 0) -> list:
    """Reduce a statement's raw tokens to the effective program token lists
    (see _reduce_head). An eval body may contain several statements."""
    out: list = []
    _reduce_head(list(text_tokens), depth, out)
    return out


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
    """Yield each token list that would RUN. Statement segments (after
    keyword/group/wrapper head reduction), ``bash -c`` bodies, and command
    substitutions (they execute at runtime even inside double quotes).
    Heredoc body semantics (F1): quoted delimiter = literal data; unquoted
    = data, but unquoted ``$(...)`` still runs; a body piped into a shell
    is fully re-lexed (fail-closed)."""
    if depth > _MAX_DEPTH:
        return
    pieces, _unterminated = _split_heredocs(command)
    for kind, text in pieces:
        if kind == "data":
            continue
        if kind == "subst-body":
            for _s, _e, body in _substitution_spans(text):
                if body.strip():
                    yield from _units(body, depth + 1)
            continue
        if kind == "body-shell":
            yield from _units(text, depth + 1)
            continue
        for seg in _split_segments_quoted(text):
            # Command substitutions run at runtime — extract UNQUOTED ones
            # from the raw segment (an env-assign head like ``x=$(git ...)``
            # would otherwise be dropped as an assignment, losing the body).
            for _s, _e, body in _substitution_spans(seg):
                if body.strip():
                    yield from _units(body, depth + 1)
            try:
                raw_toks = shlex.split(seg)
            except ValueError:
                raw_toks = seg.split()  # unbalanced quotes: fail closed
            for toks in _reduce_to_program(raw_toks, depth):
                if not toks:
                    continue
                head = toks[0]
                if head in _SHELL_BINARIES:
                    # unwrap bash -c: EVERY trailing token after the last
                    # -c/--command flag is the script (one or many args)
                    for _i in range(len(toks) - 1, 0, -1):
                        if toks[_i] in ("-c", "--command"):
                            script = " ".join(toks[_i + 1:])
                            if script.strip():
                                yield from _units(script, depth + 1)
                            break
                    else:
                        yield toks
                    continue
                yield toks


def _git_units(command: str):
    """Token lists of every git invocation the command would run. The head
    is already reduced to ``git`` / ``git-filter-repo`` — a grep/echo/cat
    segment merely mentions git; it does not RUN git."""
    for toks in _units(command):
        head = toks[0]
        if head.endswith("/git"):
            head = "git"
        elif head.endswith("/git-filter-repo"):
            head = "git-filter-repo"
        if head in ("git", "git-filter-repo"):
            yield toks


# --------------------------------------------------------------- matchers --
# One matcher per damage form, decided from the FULL token list of ONE git
# invocation (head token = "git"). Token-exact matching is what excludes
# safe variants (mentions, dry-runs, report-only modes) by construction.
# New commands are classified first, then covered by their class.


def _m_checkout_discard(toks):
    """``git checkout`` that discards working-tree changes."""
    if toks[1:2] != ["checkout"]:
        return False
    args = toks[2:]
    if "-f" in args or "--force" in args or "--" in args:
        return True
    positional = [t for t in args if not t.startswith("-")]
    if not positional:
        return True  # bare checkout (legacy tripwire kept)
    return all(t in (".", "..") for t in positional)  # discard-all dot forms


def _m_reset_hard(toks):
    return toks[1:2] == ["reset"] and "--hard" in toks[2:]


def _m_clean_force(toks):
    """``git clean`` without a dry-run flag (an ``n`` short-flag or --dry-run)."""
    if toks[1:2] != ["clean"]:
        return False
    args = toks[2:]
    pre = args[: args.index("--")] if "--" in args else args
    if "--dry-run" in pre:
        return False
    return not any(
        t.startswith("-") and not t.startswith("--") and "n" in t[1:] for t in pre
    )


def _m_restore_discard(toks):
    """``git restore`` touching the working tree (anything not staged-only)."""
    if toks[1:2] != ["restore"]:
        return False
    args = toks[2:]
    if "--staged" in args and not ({"--worktree", "-W"} & set(args)):
        return False  # unstage only
    return True


def _m_switch_discard(toks):
    """``git switch`` discarding working-tree changes (same class as
    checkout -- / restore; -f and --discard-changes overwrite paths)."""
    if toks[1:2] != ["switch"]:
        return False
    args = toks[2:]
    return "-f" in args or "--force" in args or "--discard-changes" in args


def _m_stash_destroy(toks):
    return toks[1:2] == ["stash"] and toks[2:3] and toks[2] in ("drop", "clear")


def _m_branch_delete(toks):
    if toks[1:2] != ["branch"]:
        return False
    args = toks[2:]
    flags = [t for t in args if t.startswith("-") and not t.startswith("--")]
    return any(
        "d" in t[1:] or "D" in t[1:] for t in flags
    ) or "--delete" in args


def _m_push_force(toks):
    if toks[1:2] != ["push"]:
        return False
    args = toks[2:]
    if any(
        t in ("--force", "-f", "-F") or t.startswith("--force-with-lease")
        for t in args
    ):
        return True
    # refspec damage: :branch deletes the remote ref; +refspec force-updates
    return any(
        t == "--" or t.startswith(":") or t.startswith("+")
        for t in args if not t.startswith("-")
    ) or bool({"--delete", "-d", "--all", "--tags", "--mirror"} & set(args))


def _m_update_ref_delete(toks):
    """``git update-ref -d`` deletes a ref (same damage class as branch -D)."""
    if toks[1:2] != ["update-ref"]:
        return False
    args = toks[2:]
    return "-d" in args or "--delete" in args


# Class 3 — REWRITE or irreversibly DESTROY history / the object store.
# Commit hashes change or objects vanish; a stash does not protect these, so
# the backup ritual does not discharge them (see check_pre).

def _m_rebase(toks):
    return toks[1:2] == ["rebase"]


def _m_filter_branch(toks):
    return toks[1:2] == ["filter-branch"]


def _m_filter_repo(toks):
    """git-filter-repo rewrites objects unless in report-only --analyze mode.

    The exemption is a TOKEN check: an ``--analyze*`` option token in THIS
    invocation. A path argument that merely contains the text
    (``/tmp/--analyze.txt``) does not start with ``--analyze`` and never
    exempts; a later ``git filter-repo --analyze`` segment cannot whitewash an
    earlier rewriting segment (F1). A longer ``--analyze*`` word is an
    unrecognized option — git aborts before doing anything, so it is harmless
    (upstream sanity-check refuses --analyze combined with rewriting options).
    """
    if toks[0] == "git-filter-repo":
        args = toks[1:]
    elif toks[0] == "git":
        # locate the subcommand: global options (some consume a value) and
        # unknown dash tokens precede it (``git -C <path> filter-repo ...``)
        i = 1
        while i < len(toks):
            if toks[i] in _GIT_GLOBAL_VALUE_OPTS:
                i += 2
            elif toks[i].startswith("-"):
                i += 1
            else:
                break
        if toks[i:i + 1] != ["filter-repo"]:
            return False
        args = toks[i + 1:]
    else:
        return False
    return not any(t.startswith("--analyze") for t in args)


def _m_reflog_destroy(toks):
    return toks[1:2] == ["reflog"] and toks[2:3] and toks[2] in ("expire", "delete")


def _m_gc_prune_now(toks):
    """``git gc --prune`` keeps the default 2-week grace — same damage class
    as bare ``git gc``; only the immediate ``--prune=now`` destroys now (F2)."""
    return toks[1:2] == ["gc"] and "--prune=now" in toks[2:]


def _m_prune_now(toks):
    """``git prune --expire=now`` destroys unreachable objects immediately;
    bare ``git prune`` keeps the default 2-week grace (git-prune(1): the
    ``--expire`` default grace) and is NOT in this class (same ruling as
    ``git gc --prune``)."""
    if toks[1:2] != ["prune"]:
        return False
    args = toks[2:]
    return any(t in ("--expire=now", "--expire=all") for t in args)


_CLASS1_MATCHERS = (
    _m_checkout_discard, _m_reset_hard, _m_clean_force, _m_restore_discard,
    _m_switch_discard,
)
_CLASS2_MATCHERS = (
    _m_stash_destroy, _m_branch_delete, _m_push_force, _m_update_ref_delete,
)
_CLASS3_MATCHERS = (
    _m_rebase, _m_filter_branch, _m_filter_repo, _m_reflog_destroy,
    _m_gc_prune_now, _m_prune_now,
)
_DESTRUCTIVE_MATCHERS = _CLASS1_MATCHERS + _CLASS2_MATCHERS + _CLASS3_MATCHERS
_CLASS12_MATCHERS = _CLASS1_MATCHERS + _CLASS2_MATCHERS
# A stash-push ritual must never shield stash drop/clear: the "backup" being
# destroyed may be the very one the ritual just created.
_STOMP_IDXS = frozenset(
    i for i, m in enumerate(_DESTRUCTIVE_MATCHERS) if m is _m_stash_destroy
)
_CLASS3_IDXS = frozenset(
    range(len(_CLASS12_MATCHERS), len(_DESTRUCTIVE_MATCHERS))
)

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
        # discharges only the damage classes a stash actually protects, and
        # only the units that run AFTER it (E1/E2: destruction running
        # FIRST is not backed up by a later stash-push). Classes 1-2 cover
        # uncommitted work / recoverable VCS state; Class 3 rewrites
        # history / deletes objects a stash cannot restore: it always
        # blocks and needs an explicit override reason. The drop/clear
        # carve-out stays (never whitelist destruction of pre-existing
        # backups).
        ritual_pos: list[int] = []
        stomps = False
        hits: list[int] = []
        for pos, toks in enumerate(units):
            if toks[1:2] == ["stash"] and toks[2:3] and toks[2] == "push":
                ritual_pos.append(pos)
            if toks[1:2] == ["stash"] and toks[2:3] and toks[2] in ("drop", "clear"):
                stomps = True
            for idx, matcher in enumerate(_DESTRUCTIVE_MATCHERS):
                if idx not in hits and matcher(toks):
                    hits.append(idx)

        def _discharged(pos: int) -> bool:
            """A class 1/2 unit is discharged iff a stash-push STATEMENT
            (never a mention) precedes it in run order."""
            return any(p < pos for p in ritual_pos)

        blocking = []
        for pos, toks in enumerate(units):
            for idx, matcher in enumerate(_DESTRUCTIVE_MATCHERS):
                if matcher(toks) and (
                    idx in _CLASS3_IDXS or idx in _STOMP_IDXS
                    or not _discharged(pos)
                ):
                    blocking.append(idx)
        if ritual_pos and not stomps and not blocking:
            return None
        for idx in blocking:
            if idx in self._acked:
                continue
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
            if any(matcher(toks) for toks in _git_units(command)):
                self._acked.add(idx)
        return True

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def reset_turn(self):
        self._acked.clear()
