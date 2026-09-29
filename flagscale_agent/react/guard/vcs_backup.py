# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on the "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VcsBackupGuard — targeted interception of destructive git operations.

Scope is deliberately narrow: only git commands that
can permanently destroy UNCOMMITTED work. The upfront backup reminder lives in
StartupGuard's BackupPhase (first tool call) and is NOT touched by this guard.

Fires on shell commands whose destructive git form matches. Does NOT fire on:
  - `git checkout <branch>` (safe; only `checkout -- <path>` discards changes)
  - `git stash push` (this IS the backup ritual — required to release the block)
  - `git clean -n` (dry-run), `git restore --staged` (unstage only)
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

# Deterministic destructive-git patterns (style law 2: regex only, no LLM).
# Organized by DAMAGE CLASS, not by command name — new commands must be
# classified first, then covered by their class. Safe variants are excluded
# by construction (see tests).
_CLASS1_PATTERNS = (
    # Class 1 — destroy UNCOMMITTED work in the working tree / index.
    re.compile(r"\bgit\s+checkout\b[^&;|]*?--\s"),
    re.compile(r"\bgit\s+checkout\s+(?:-\S+\s+)*\.?\s*$"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+clean\b(?![^&;|]*\s-[a-zA-Z]*n)"),
    re.compile(r"\bgit\s+restore\b(?!\s+--staged\b)"),
)
_CLASS2_PATTERNS = (
    # Class 2 — destroy RECOVERABLE VCS state (stashes, local/remote refs).
    re.compile(r"\bgit\s+stash\s+(?:drop|clear)\b"),
    re.compile(r"\bgit\s+branch\s+-[dD]\b"),
    re.compile(r"\bgit\s+push\b[^&;|]*?(?:--force\b|--force-with-lease=|-[fF]\b)"),
)
_CLASS3_PATTERNS = (
    # Class 3 — REWRITE or irreversibly DESTROY history / the object store.
    # Commit hashes change or objects vanish; a stash does not protect these,
    # so the backup ritual does not discharge them (see check_pre).
    re.compile(r"\bgit\s+rebase\b"),
    re.compile(r"\bgit\s+filter-branch\b"),
    re.compile(r"\bgit(?:-|\s+)filter-repo\b(?!.*--analyze\b)"),  # subcommand + standalone script; --analyze is report-only
    re.compile(r"\bgit\s+reflog\s+(?:expire|delete)\b"),          # destroys the recovery net itself
    re.compile(r"\bgit\s+gc\b[^&;|]*?--prune\b"),                 # immediate prune (default keeps a 2-week grace); --prune=now matches
)
_DESTRUCTIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    _CLASS1_PATTERNS + _CLASS2_PATTERNS + _CLASS3_PATTERNS
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
        # Pattern ids already acknowledged via a valid override this turn.
        self._acked: set[int] = set()

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None
        command = str(ctx.tool_args.get("command", ""))
        if not command:
            return None
        # The backup ritual (pure stash push) discharges only the damage
        # classes a stash actually protects (Classes 1-2: uncommitted work,
        # recoverable VCS state) — so the recipe itself never blocks. Class 3
        # rewrites history / deletes objects a stash cannot restore: it still
        # blocks and needs an explicit override reason. The drop/clear
        # carve-out stays (never whitelist destruction of pre-existing
        # backups).
        if re.search(r"\bgit\s+stash\s+push\b", command) and not re.search(
            r"\bgit\s+stash\s+(?:drop|clear)\b", command
        ):
            if not any(pattern.search(command) for pattern in _CLASS3_PATTERNS):
                return None
        for idx, pattern in enumerate(_DESTRUCTIVE_PATTERNS):
            if pattern.search(command) and idx not in self._acked:
                return GuardVerdict.block(
                    message=_DESTRUCTIVE_MESSAGE,
                    reason=f"destructive_git_pattern_{idx}",
                    category="vcs_backup",
                    overridable=True,
                )
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Release only when the reason cites a concrete backup channel
        (deterministic keyword check, style law 4). One pattern per override:
        a new destructive form requires its own acknowledgment."""
        if not reason or len(reason.strip()) <= 5:
            return False
        lowered = reason.lower()
        if not any(
            kw in lowered
            for kw in ("stash", "backup", "bak", "patch", "bundle", "branch", "clean", "regenerat")
        ):
            return False
        command = str(ctx.tool_args.get("command", "")) if ctx else ""
        for idx, pattern in enumerate(_DESTRUCTIVE_PATTERNS):
            if pattern.search(command):
                self._acked.add(idx)
        return True

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def reset_turn(self):
        self._acked.clear()
