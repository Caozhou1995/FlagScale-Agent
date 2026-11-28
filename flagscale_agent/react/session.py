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

"""Session persistence — save/load conversation history.

Layout:
  <project_root>/.flagscale/sessions/{session_id}/conversation.json
"""

import fcntl
import json
import os
import shutil
import sys
import tempfile
import time

from pathlib import Path
from typing import Any, Dict, List, Optional

from flagscale_agent.react.paths import get_sessions_root


def _sessions_root() -> str:
    return get_sessions_root()


def get_session_dir(session_id: str) -> str:
    return os.path.join(_sessions_root(), session_id)


# ── Session directory lock (concurrency guard) ──────────────────────────────
# A session directory is shared mutable state: conversation.json and
# conversation_full.json are fully rewritten by every save (last-writer-wins),
# swap_store/{index}.json is keyed by external message index (two processes
# evicting the same index silently overwrite each other's stored content),
# and plans/active.yaml is a single pointer. Two live agent processes bound
# to the same directory therefore corrupt each other's history and recall
# garbage across timelines. The lock makes that state single-owner.
#
# Mechanism: flock(LOCK_EX|LOCK_NB) on <session_dir>/.session.lock.
# Verified on the session NFS store: cross-process acquisition is denied
# (BlockingIOError), and the kernel releases the lock automatically when the
# holder process dies (SIGKILL included) — no stale-lock cleanup needed.

SESSION_LOCK_FILE = ".session.lock"


class SessionLockedError(Exception):
    """Another live agent process already holds the session directory."""

    def __init__(self, session_dir: str, holder: Optional[dict] = None):
        self.session_dir = session_dir
        self.holder = holder or {}
        desc = ""
        if self.holder:
            desc = (f" (holder PID {self.holder.get('pid')}, "
                    f"started {self.holder.get('start', '?')}: "
                    f"{self.holder.get('cmd', '?')})")
        super().__init__(f"session directory is locked by another agent process{desc}: {session_dir}")


def _read_lock_owner(session_dir: str) -> Optional[dict]:
    """Best-effort read of holder diagnostics written by the lock owner."""
    try:
        with open(os.path.join(session_dir, SESSION_LOCK_FILE), "r",
                  encoding="utf-8") as f:
            info = json.load(f)
        return info if isinstance(info, dict) else None
    except Exception:
        return None


def acquire_session_lock(session_dir: str) -> int:
    """Take an exclusive advisory lock on session_dir.

    Returns the lock fd — the caller MUST keep it open for the lifetime of
    the binding and pass it to release_session_lock(). flock is released
    automatically by the kernel if the process dies.

    Raises SessionLockedError if another live process holds the lock.
    """
    os.makedirs(session_dir, exist_ok=True)
    lock_path = os.path.join(session_dir, SESSION_LOCK_FILE)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        os.close(fd)
        raise SessionLockedError(session_dir, _read_lock_owner(session_dir)) from e
    try:
        # Diagnostics only — the flock itself is the source of truth.
        os.truncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps({
            "pid": os.getpid(),
            "start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cmd": " ".join(os.sys.argv[1:])[:200],
        }).encode())
    except Exception:
        pass
    return fd


def release_session_lock(fd: int) -> None:
    """Release a lock acquired by acquire_session_lock (idempotent-ish)."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def conversation_has_real_content(conv_path: str) -> bool:
    """True if conversation.json records at least one non-system message.

    A session dir whose conversation.json holds ONLY the system prompt carries
    nothing a user could resume: no user turn, no assistant turn, turn_count 0.
    Such a file is a byproduct of process startup (the system prompt is
    persisted by the first save / atexit flush before any user input), not a
    conversation. Treating it as content is exactly what pinned 278 empty
    shells on disk (measured 2026-11-01).

    Unreadable/absent file → False (no recoverable content).
    """
    try:
        with open(conv_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    for m in data.get("messages") or []:
        if isinstance(m, dict) and m.get("role") != "system":
            return True
    return False


def dir_empty_except_lock(session_dir: str) -> bool:
    """True if session_dir holds nothing recoverable: only the lock file,
    dotfiles, EMPTY subdirectories, and/or a system-prompt-only
    conversation.json.

    Empty subdirs (a freshly-created swap_store/ or plans/ that never
    received data) carry no information and must not pin an abandoned
    session dir on disk forever — that was the accumulation bug: __init__
    always materializes swap_store/, so a reload's old dir could never
    satisfy the old dotfile-only predicate. Any real content at any depth
    (a conversation.json with a real turn, a swap shard, a plan yaml) still
    counts as non-empty and protects the directory.

    A conversation.json holding only the system prompt is admitted as empty
    for the same reason: it is a startup artifact, not a conversation.
    """
    try:
        entries = os.listdir(session_dir)
    except OSError:
        return False
    for e in entries:
        if e.startswith("."):
            continue
        p = os.path.join(session_dir, e)
        if os.path.isdir(p) and not os.listdir(p):
            continue  # empty subdir — ignorable
        if e == "conversation.json" and not conversation_has_real_content(p):
            continue  # system-prompt-only session — startup artifact
        return False
    return True


# A candidate must be at least this old before clean_stale_empty_sessions
# will remove it: guards against racing a concurrently booting process in
# the window between its makedirs(session_dir) and acquire_session_lock.
CLEAN_MIN_AGE_SEC = 600


def clean_stale_empty_sessions(sessions_root: str, min_age_sec: int = CLEAN_MIN_AGE_SEC) -> list:
    """Remove session dirs that are empty-except-lock, unlocked, and stale.

    Reload/resume used to leave behind empty shell dirs (dotfiles + an empty
    swap_store/). This is the one-off sweeper for the accumulated backlog and
    the standing hook that keeps the root clean. A directory is removed only
    when ALL hold:
      - dir_empty_except_lock() is True (no recoverable content),
      - no live process holds its lock (get_session_lock_holder is None),
      - its mtime is at least min_age_sec old (not a boot-in-progress).
    Returns the list of removed dir paths. Best-effort: failures are skipped.
    """
    removed: list = []
    try:
        entries = os.listdir(sessions_root)
    except OSError:
        return removed
    now = time.time()
    for e in entries:
        d = os.path.join(sessions_root, e)
        if not os.path.isdir(d):
            continue
        if not dir_empty_except_lock(d):
            continue
        if get_session_lock_holder(d) is not None:
            continue
        try:
            if now - os.stat(d).st_mtime < min_age_sec:
                continue
        except OSError:
            continue
        try:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d)
        except Exception:
            continue
    return removed


def get_session_lock_holder(session_dir: str) -> Optional[dict]:
    """Return live holder info if session_dir is locked by another process, else None.

    Cheap gate on file presence, then a non-blocking probe. A leftover lock
    file with no live holder returns None (lock is auto-released on death).
    """
    lock_path = os.path.join(session_dir, SESSION_LOCK_FILE)
    if not os.path.isfile(lock_path):
        return None
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return _read_lock_owner(session_dir)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def save_conversation(session_dir: str, session_id: str, messages: List[Dict[str, Any]],
                      loaded_skills: List[str] = None, metadata: Dict = None,
                      completed: bool = False, session_summary: str = None,
                      session_input_tokens: int = 0, session_output_tokens: int = 0,
                      turn_count: int = 0, session_input_history: List[str] = None) -> str:
    """Save conversation to session_dir/conversation.json. Overwrites on each call.

    Uses atomic write (tmp file + rename) to prevent corruption on crash.
    """
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, "conversation.json")
    data = {
        "session_id": session_id,
        "timestamp": time.time(),
        "completed": completed,
        "messages": messages,
        "loaded_skills": loaded_skills or [],
        "metadata": metadata or {},
        "session_input_tokens": session_input_tokens,
        "session_output_tokens": session_output_tokens,
        "turn_count": turn_count,
        "session_input_history": session_input_history or [],
    }
    if session_summary:
        data["session_summary"] = session_summary
    # Preserve existing summary if not overwritten
    elif os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("session_summary"):
                data["session_summary"] = existing["session_summary"]
        except Exception:
            pass
    # Atomic write: write to tmp then rename
    fd, tmp = tempfile.mkstemp(dir=session_dir, prefix=".tmp_conversation_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_conversation(session_dir: str) -> Optional[Dict[str, Any]]:
    """Load conversation.json from a session directory. Returns None if not found."""
    path = os.path.join(session_dir, "conversation.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mark_completed(session_dir: str):
    """Mark a session as completed (normal exit). Uses atomic write."""
    path = os.path.join(session_dir, "conversation.json")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["completed"] = True
    data["timestamp"] = time.time()
    fd, tmp = tempfile.mkstemp(dir=session_dir, prefix=".tmp_conversation_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Real user input vs harness-authored user-role messages ──────────────────
#
# Not every `role == "user"` message is something the human typed. The kernel and
# the guards append user-role messages constantly (completion nudges, guard
# blocks, NEED_USER_INPUT auto-appends, context-compaction summaries, worker
# contracts, eviction placeholders). A session LIST that shows "the last user
# message" therefore shows harness plumbing, and — worse — when that last message
# is a `tool_result` block it has no text at all, so the preview renders EMPTY and
# a busy session looks like an empty one. MEASURED on the live store
# (110 sessions): 84 previews were blank for this reason while ZERO sessions
# were genuinely empty. Treating "blank preview" as "empty session" would have
# deleted every real conversation.

# Fixed shapes the harness itself authors into `role: user`. Matched as PREFIXES
# at the start of the message text. Deliberately narrow: a shape we fail to list
# degrades to "shown as the preview", which is the harmless direction; the
# harmful direction is hiding a human's own words.
_HARNESS_USER_PREFIXES = (
    "[system:",            # auto_appended NEED_USER_INPUT / continuation notices
    "[worker role]",       # spawned-worker contract (see multi_agent/wiring.py)
    "[Context Hard Reset",
    "[evicted |",          # eviction placeholder written back into history
    "[UltraThink",
    "[blind-spot reader]",  # background blind-spot adoption notice (agent.py:2896)
)

# Tool RESULTS that the harness appends into `role: user` as a plain string
# rather than a tool_result block. Measured case 087f42df / 924e95f3: a 3-message
# session whose real prompt sits at index 1 and whose newest user message is this
# tool output, so a backward scan shows tool output where the human's own words
# belong. The header string is the render contract of
# tools/read_task_statement.py:_render — keep the two in sync.
_TOOL_RESULT_PREFIXES = (
    "independent readings of the task statement",  # matched after a leading count
)


def _is_tool_result_text(text: str) -> bool:
    """True when a user-role string is a tool's rendered output, not a human turn."""
    t = " ".join(text.lstrip().split()[:8])
    return any(p in t for p in _TOOL_RESULT_PREFIXES)

# Harness-authored messages that carry a bracketed guard/plan tag, e.g.
# "[StartupGuard/Backup] ...", "[Plan] You are about to write ...",
# "[BackupGuard] ...", "[CompletionGate] ...". A leading "[Tag]" is the house
# style for injected reminders; a human turn starting with "[" is possible but
# rare, and showing it is the failure we accept.
_BRACKETED_TAG_RE = None  # compiled lazily to keep module import cheap


def _msg_text(msg: Dict[str, Any]) -> str:
    """Concatenate the TEXT blocks of one message. '' when it has none.

    A `tool_result`-only user message legitimately has no text — that is the bug
    that made previews blank, not an error to paper over.
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _is_harness_authored(text: str) -> bool:
    """True when this user-role text was written by the harness, not the human."""
    global _BRACKETED_TAG_RE
    t = text.lstrip()
    if not t:
        return True
    for prefix in _HARNESS_USER_PREFIXES:
        if t.startswith(prefix):
            return True
    # "Your previous response contained text but no tool calls..." and
    # "You have responded N+ times with text only..." are the completion-gate
    # reminders: prose, no bracket tag, so they need their own predicate.
    if t.startswith("Your previous response contained text"):
        return True
    if t.startswith("You have responded ") and "no tool calls and no stop signal" in t:
        return True
    if _BRACKETED_TAG_RE is None:
        import re
        _BRACKETED_TAG_RE = re.compile(r"^\[[A-Za-z][\w ./&-]{0,60}\]")
    return bool(_BRACKETED_TAG_RE.match(t))


def real_user_input(data: Dict[str, Any], limit: int = 120) -> str:
    """The human's own most recent input for this session ('' when unknown).

    Two sources, in order of authority:
      1. `session_input_history` — literally what was typed at the prompt. It is
         the strongest signal but only exists for interactive sessions
         (measured: 23/110), because batch runs never go through the prompt loop.
      2. otherwise scan `messages` BACKWARDS for the newest user message that has
         text and is NOT harness-authored. Skipping backwards is what fixes the
         blank previews: the newest user message is usually a `tool_result` block.
    """
    history = data.get("session_input_history") or []
    for item in reversed(history):
        if isinstance(item, str) and item.strip():
            return item.strip()[:limit]

    messages = data.get("messages") or []
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = _msg_text(msg).strip()
        if not text or _is_harness_authored(text) or _is_tool_result_text(text):
            continue
        return text[:limit]
    return ""


def is_worker_session(data: Dict[str, Any], session_dir: str = "") -> bool:
    """True when this session belongs to a spawned worker (a sub-agent).

    Worker sessions are real conversations, but they are the PARENT's machinery —
    listing them next to human conversations floods the resume list with entries
    the user never started. `metadata` cannot decide this (measured: it is `{}`
    for every session), so the only durable signal is the fixed contract prefix
    the harness prepends to the worker's first user message. Imported from
    wiring.py rather than re-typed, so a change there cannot silently desync.

    Eviction complicates it: once history is compacted, the original first user
    message is replaced by a `[evicted | index=N | role=user | <first 100 chars>
    | N tokens]` placeholder. The contract prefix survives inside that
    placeholder (truncated), so the placeholder's own head is inspected too —
    otherwise a worker session that was compacted becomes indistinguishable from
    a human one and reappears in the list.
    """
    from flagscale_agent.react.multi_agent.wiring import (
        WORKER_ROLE_PREFIX, WORKER_LEAF_ROLE_PREFIX)

    def _looks_like_contract(text: str) -> bool:
        t = text.lstrip()
        if t.startswith(WORKER_ROLE_PREFIX) or t.startswith(WORKER_LEAF_ROLE_PREFIX):
            return True
        # An eviction placeholder embeds the original message's head right after
        # `role=user | `. Match the contract prefix anywhere in the placeholder.
        if t.startswith("[evicted |"):
            return (WORKER_ROLE_PREFIX[:40] in t) or (WORKER_LEAF_ROLE_PREFIX[:40] in t)
        return False

    messages = data.get("messages") or []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = _msg_text(msg)
        if not text.strip():
            continue
        # Only the FIRST text-bearing user message can be the contract; later
        # ones are the parent's resume messages.
        if _looks_like_contract(text):
            return True
        # A hard reset REPLACES the original first turn with a compaction
        # summary, so the live history no longer carries the contract — but the
        # untouched transcript is still on disk next to it under
        # `conversation_full.json`. Check that before concluding "human session":
        # measured case 74791ab1 has an all-summary live history and a full
        # transcript whose first message is the LEAF contract.
        return _worker_by_full_transcript(session_dir)
    return False


def _worker_by_full_transcript(session_dir: str) -> bool:
    """Last-resort worker check against conversation_full.json, else False."""
    if not session_dir:
        return False
    path = os.path.join(session_dir, "conversation_full.json")
    if not os.path.isfile(path):
        return False
    try:
        from flagscale_agent.react.multi_agent.wiring import (
            WORKER_ROLE_PREFIX, WORKER_LEAF_ROLE_PREFIX)
        with open(path, "r", encoding="utf-8") as f:
            full = json.load(f)
        msgs = full.get("messages") if isinstance(full, dict) else full
        for msg in msgs or []:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            text = _msg_text(msg).lstrip()
            if not text:
                continue
            return (text.startswith(WORKER_ROLE_PREFIX)
                    or text.startswith(WORKER_LEAF_ROLE_PREFIX))
    except Exception:
        return False
    return False


def find_resumable_sessions(sessions_root: str = None) -> List[Dict[str, Any]]:
    """Sessions a human can meaningfully resume, newest first.

    Three filters beyond `completed == false`, each measured against the live
    store (110 sessions) where the naive version returned 110 rows of which 84
    had a BLANK preview and 75 were sub-agent machinery:

      · worker sessions are excluded — they are the parent's sub-agents, not
        conversations the user started (see `is_worker_session`);
      · a session with no recoverable human input is excluded — that is what
        "empty" actually means here (nothing to show, nothing to resume to);
        note the count of human messages is NOT the test, because every session
        in the store had some;
      · the preview is the human's own last input (`real_user_input`), not "the
        last message with role=user" — the latter is usually a tool_result block
        with no text at all, which is what rendered 84 previews blank.
    """
    root = sessions_root or _sessions_root()
    if not os.path.isdir(root):
        return []
    results = []
    for entry in os.listdir(root):
        entry_path = os.path.join(root, entry)
        if not os.path.isdir(entry_path):
            continue
        conv_path = os.path.join(entry_path, "conversation.json")
        if not os.path.isfile(conv_path):
            continue
        try:
            with open(conv_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("completed", True):
                continue
            if is_worker_session(data, entry_path):
                continue
            last_user = real_user_input(data, limit=100)
            if not last_user:
                # No human input recoverable (history was fully compacted away
                # and no input history was recorded): nothing to resume.
                continue
            messages = data.get("messages", [])
            user_msgs = [m for m in messages
                         if isinstance(m, dict) and m.get("role") == "user"]
            results.append({
                "session_id": data.get("session_id", entry),
                "session_dir": entry_path,
                "timestamp": data.get("timestamp", 0),
                "loaded_skills": data.get("loaded_skills", []),
                "user_turns": data.get("turn_count") or len(user_msgs),
                "last_user_msg": last_user,
                "session_summary": data.get("session_summary", ""),
            })
        except Exception:
            continue
    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return results


def list_sessions(sessions_root: str = None) -> List[Dict[str, Any]]:
    """List all sessions by scanning sessions/*/conversation.json."""
    root = sessions_root or _sessions_root()
    if not os.path.isdir(root):
        return []
    sessions = []
    for entry in sorted(os.listdir(root), reverse=True):
        entry_path = os.path.join(root, entry)
        if not os.path.isdir(entry_path):
            continue
        conv_path = os.path.join(entry_path, "conversation.json")
        if not os.path.isfile(conv_path):
            continue
        try:
            with open(conv_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            sessions.append({
                "session_id": data.get("session_id", entry),
                "session_dir": entry_path,
                "timestamp": data.get("timestamp", 0),
                "completed": data.get("completed", True),
                "turns": len([m for m in messages
                              if isinstance(m, dict) and m.get("role") == "user"]),
            })
        except Exception:
            continue
    return sessions
