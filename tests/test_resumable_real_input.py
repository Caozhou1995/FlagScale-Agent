"""Resumable-list contract: only real human sessions, previewed with the human's own words.

Three properties, each measured on the live store before being encoded here:
  1. a worker sub-session must not appear in the resumable list;
  2. the preview must be a HUMAN turn, not a tool's rendered output or a harness
     notice (live store: 84/110 previews were blank, and the remaining leaks were
     tool output such as "Ten independent readings of the task statement");
  3. a session with real human input must still be listed and previewed with it —
     the positive bound, so a filter that hides everything cannot pass.
"""

import json

import pytest

from flagscale_agent.react.multi_agent.wiring import (
    WORKER_LEAF_ROLE_PREFIX, WORKER_ROLE_PREFIX)
from flagscale_agent.react.session import (
    find_resumable_sessions, real_user_input, save_conversation)


def _save(root, sid, messages, **kw):
    save_conversation(str(root / sid), sid, messages, completed=False, **kw)


def test_worker_session_is_not_listed(tmp_path):
    _save(tmp_path, "worker1", [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": WORKER_ROLE_PREFIX + " do the thing"},
    ])
    assert find_resumable_sessions(sessions_root=str(tmp_path)) == []


def test_leaf_worker_session_is_not_listed(tmp_path):
    _save(tmp_path, "leaf1", [
        {"role": "user", "content": WORKER_LEAF_ROLE_PREFIX + " leaf contract"},
    ])
    assert find_resumable_sessions(sessions_root=str(tmp_path)) == []


def test_worker_contract_survives_eviction_placeholder(tmp_path):
    """A compacted worker session keeps only a placeholder — still filtered."""
    head = WORKER_LEAF_ROLE_PREFIX[:100]
    _save(tmp_path, "evicted_worker", [
        {"role": "user",
         "content": f"[evicted | index=1 | role=user | {head} | 900 tokens]"},
        {"role": "user", "content": "parent resume message"},
    ])
    assert find_resumable_sessions(sessions_root=str(tmp_path)) == []


def test_human_session_survives_and_previews_human_turn(tmp_path):
    """Positive bound: a real conversation is listed AND keeps its own words."""
    _save(tmp_path, "human1", [
        {"role": "user", "content": "为什么最后没有发结束 marker？"},
        {"role": "assistant", "content": "because ..."},
    ])
    res = find_resumable_sessions(sessions_root=str(tmp_path))
    assert len(res) == 1
    assert res[0]["session_id"] == "human1"
    assert res[0]["last_user_msg"] == "为什么最后没有发结束 marker？"


def test_preview_skips_tool_output_written_as_user_string(tmp_path):
    """Measured live shape: real prompt at index 1, tool output appended after it."""
    _save(tmp_path, "toolout", [
        {"role": "user", "content": "Write me data.comp"},
        {"role": "user",
         "content": "Ten independent readings of the task statement (10 ran). "
                    "These are INPUTS, not conclusions: ..."},
    ])
    res = find_resumable_sessions(sessions_root=str(tmp_path))
    assert len(res) == 1
    assert res[0]["last_user_msg"] == "Write me data.comp"


def test_preview_skips_eviction_placeholder(tmp_path):
    _save(tmp_path, "ph", [
        {"role": "user", "content": "真正的提问"},
        {"role": "user",
         "content": "[evicted | index=5 | role=user | old text | 100 tokens]"},
    ])
    res = find_resumable_sessions(sessions_root=str(tmp_path))
    assert res[0]["last_user_msg"] == "真正的提问"


def test_preview_skips_blindspot_notice(tmp_path):
    _save(tmp_path, "bs", [
        {"role": "user", "content": "真实问题"},
        {"role": "user",
         "content": "[blind-spot reader] A background read of the task statement "
                    "has finished."},
    ])
    res = find_resumable_sessions(sessions_root=str(tmp_path))
    assert res[0]["last_user_msg"] == "真实问题"


def test_real_user_input_prefers_typed_history(tmp_path):
    """session_input_history is the prompt loop — it outranks the message scan."""
    data = {
        "session_input_history": ["first typed", "newest typed"],
        "messages": [{"role": "user", "content": "from messages"}],
    }
    assert real_user_input(data) == "newest typed"


def test_real_user_input_empty_when_nothing_human(tmp_path):
    data = {"messages": [{"role": "user", "content": "[system: auto notice]"}]}
    assert real_user_input(data) == ""
