"""The reasoning log persists WHENEVER the conversation does.

User directive (2026-11-28):
    "thinking log落盘除了在post-edit inject之前外，我还想让其和对话文件落盘
     时机一致，也就是对话文件落盘时它也跟着落盘。"

WHY this is a real gap and not a nicety: the post-edit dumps fire on a WRITE.
A long stretch of decisive reasoning often ends in a read, a test run, or a
plain turn boundary — no edit, so no dump. Those are exactly the stretches a
later "point at the turn where you decided X" question needs, and they were the
ones with no file on disk. Tying the dump to the conversation save closes it:
any moment the transcript is durable, the reasoning behind it is too.

Three triggers now exist and ALL go through ONE writer (write_thinking_dump), so
thinking_search parses a single file shape:
    1. before a post-edit far-end inject        (pre-existing)
    2. on _save_conversation (turn boundary / auto-save / SIGTERM handler)
    3. on _maybe_periodic_save's intra-turn checkpoint
The tests below pin (2) and (3) plus the shared-writer invariant.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flagscale_agent.react.agent import WorkerAgent  # noqa: E402
from flagscale_agent.react.guard import GuardContext  # noqa: E402
from flagscale_agent.react.guard.post_edit_far_end import (  # noqa: E402
    THINKING_DUMP_FILENAME,
    _write_thinking_dump,
    write_thinking_dump,
)
from flagscale_agent.react.history import HistoryManager  # noqa: E402

REASON = "DECISION: chose approach Z because the B arm returned 0"


def _agent(session_dir: str) -> WorkerAgent:
    """A real WorkerAgent with a real HistoryManager (no mocks of the seams)."""
    a = WorkerAgent.__new__(WorkerAgent)
    a._session_dir = session_dir
    a._session_id = "t"
    a.turn_count = 1
    a._session_input_tokens = 0
    a._session_output_tokens = 0
    a._session_input_history = ["go"]
    a._loaded_skills = set()
    a._last_periodic_save_at = 0.0
    a._periodic_save_thread = None
    a._PERIODIC_SAVE_INTERVAL_SEC = 300.0
    h = HistoryManager()
    h.append({"role": "system", "content": "sys"})
    h.append({"role": "user", "content": "go"})
    h.append({"role": "assistant", "content": [
        {"type": "thinking", "thinking": REASON}]})
    a.history = h
    return a


# ── trigger 2: conversation save ────────────────────────────────────────────

def test_conversation_save_also_writes_the_reasoning_log(tmp_path) -> None:
    d = str(tmp_path / "s")
    WorkerAgent._save_conversation(_agent(d))
    assert os.path.isfile(os.path.join(d, "conversation.json"))
    log = os.path.join(d, THINKING_DUMP_FILENAME)
    assert os.path.isfile(log), "reasoning log did not follow the conversation save"
    assert REASON in open(log, encoding="utf-8").read()


def test_same_instant_not_just_eventually(tmp_path) -> None:
    """The directive says SAME time, so a 1-hour-later dump does not satisfy it."""
    d = str(tmp_path / "s")
    WorkerAgent._save_conversation(_agent(d))
    lag = abs(os.path.getmtime(os.path.join(d, "conversation.json"))
              - os.path.getmtime(os.path.join(d, THINKING_DUMP_FILENAME)))
    assert lag < 2.0, f"reasoning log written {lag}s away from the conversation"


def test_without_the_hook_no_log_is_written(tmp_path, monkeypatch) -> None:
    """DIFFERENTIAL: disable the new call and the log must NOT appear.

    Without this a green could mean 'the file exists for an unrelated reason'.
    """
    d = str(tmp_path / "s")
    a = _agent(d)
    monkeypatch.setattr(WorkerAgent, "_save_thinking_log", lambda self: None)
    WorkerAgent._save_conversation(a)
    assert os.path.isfile(os.path.join(d, "conversation.json"))
    assert not os.path.isfile(os.path.join(d, THINKING_DUMP_FILENAME))


# ── trigger 3: periodic intra-turn checkpoint ───────────────────────────────

def test_periodic_checkpoint_also_writes_the_reasoning_log(tmp_path) -> None:
    """On a timeout kill the periodic save is the ONLY save that ran — a
    transcript without its reasoning there is the worst case."""
    d = str(tmp_path / "s")
    os.makedirs(d, exist_ok=True)  # a live session dir always exists
    a = _agent(d)
    WorkerAgent._maybe_periodic_save(a)
    t = getattr(a, "_periodic_save_thread", None)
    if t is not None:
        t.join(timeout=15)
    assert os.path.isfile(os.path.join(d, "conversation.json"))
    log = os.path.join(d, THINKING_DUMP_FILENAME)
    assert os.path.isfile(log), "periodic checkpoint did not carry the reasoning"
    assert REASON in open(log, encoding="utf-8").read()


# ── one writer, one file shape ──────────────────────────────────────────────

def test_all_triggers_share_one_file_name_and_format(tmp_path) -> None:
    d = str(tmp_path / "s")
    os.makedirs(d, exist_ok=True)
    ctx = GuardContext(tool_name="write_file", tool_args={"path": "a.py"},
                       messages=[{"role": "assistant",
                                  "content": [{"type": "thinking", "thinking": REASON}]}])
    post_edit = _write_thinking_dump(d, ctx, "a.py")
    assert post_edit and os.path.basename(post_edit) == THINKING_DUMP_FILENAME
    conv = write_thinking_dump(d, [REASON], "conversation save")
    assert conv and os.path.basename(conv) == THINKING_DUMP_FILENAME
    blob = open(conv, encoding="utf-8").read()
    # One parser shape: the header's bullet list + "## reasoning N" markers.
    assert "## reasoning 1" in blob
    assert "- entries: 1" in blob
    assert "thinking_search(query=" in blob


def test_trigger_line_differs_but_format_does_not(tmp_path) -> None:
    """The header must say WHY it was written — otherwise a reader cannot tell
    a post-edit dump from a conversation-save dump."""
    d1, d2 = str(tmp_path / "a"), str(tmp_path / "b")
    ctx = GuardContext(tool_name="write_file", tool_args={"path": "x.py"},
                       messages=[{"role": "assistant", "content": [
                           {"type": "thinking", "thinking": "R"}]}])
    p1 = _write_thinking_dump(d1, ctx, "x.py")
    p2 = write_thinking_dump(d2, ["R"], "Written whenever the conversation is saved")
    s1, s2 = open(p1, encoding="utf-8").read(), open(p2, encoding="utf-8").read()
    assert "post-edit far-end reminder" in s1
    assert "whenever the conversation is saved" in s2
    # The structural skeleton is identical in both.
    for s in (s1, s2):
        assert s.startswith("# Session reasoning log")
        assert "## reasoning 1" in s


# ── best-effort contract: never disturbs a save ─────────────────────────────

def test_failure_never_breaks_the_conversation_save(tmp_path, monkeypatch) -> None:
    d = str(tmp_path / "s")
    a = _agent(d)

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        "flagscale_agent.react.guard.post_edit_far_end.write_thinking_dump", boom)
    WorkerAgent._save_conversation(a)  # must not raise
    assert os.path.isfile(os.path.join(d, "conversation.json"))


def test_no_reasoning_means_no_file(tmp_path) -> None:
    d = str(tmp_path / "s")
    os.makedirs(d, exist_ok=True)
    assert write_thinking_dump(d, [], "t") is None
    assert not os.path.isfile(os.path.join(d, THINKING_DUMP_FILENAME))


def test_eviction_rewritten_messages_do_not_lose_old_reasoning(tmp_path) -> None:
    """Reads _full_log, not the live messages: eviction rewrites the latter,
    and the reasoning is precisely what compaction drops."""
    d = str(tmp_path / "s")
    a = _agent(d)
    early = "EARLY-DECISION: the format must be raw float32"
    a.history.append({"role": "assistant", "content": [
        {"type": "thinking", "thinking": early}]})
    # Simulate eviction: live list loses it, the full log keeps it.
    a.history.messages.clear()
    a.history.messages.append({"role": "user", "content": "later"})
    WorkerAgent._save_conversation(a)
    blob = open(os.path.join(d, THINKING_DUMP_FILENAME), encoding="utf-8").read()
    assert early in blob
