"""Forward-orphan repair: a tool_use with no adjacent tool_result must get one.

MEASURED FAILURE (2026-11-27, official api.deepseek.com/anthropic): a user
interjection while a tool was still running abandons the turn before the result
is appended, leaving a bare ``tool_use`` in history. From then on EVERY request
fails with

    messages.N: `tool_use` ids were found without `tool_result` blocks
    immediately after: call_XX. Each `tool_use` block must have a corresponding
    `tool_result` block in the next message.

and it never self-heals: the orphan is replayed with the history.
``_validate_tool_pairs`` used to repair only the REVERSE orphan (a tool_result
with no tool_use). These tests pin the forward direction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flagscale_agent.react.history import (  # noqa: E402
    _ORPHAN_TOOL_USE_RESULT,
    _validate_tool_pairs,
)


def _violations(messages):
    """Abstraction of the endpoint's rule: every tool_use answered next message."""
    bad = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        uses = [b for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use"]
        if not uses:
            continue
        have = set()
        if i + 1 < len(messages):
            nxt = messages[i + 1]
            if nxt.get("role") == "user" and isinstance(nxt.get("content"), list):
                have = {b.get("tool_use_id") for b in nxt["content"]
                        if isinstance(b, dict) and b.get("type") == "tool_result"}
        bad.extend((i, b.get("id")) for b in uses if b.get("id") not in have)
    return bad


def _tooluse(tid):
    return {"role": "assistant",
            "content": [{"type": "tool_use", "id": tid, "name": "shell", "input": {}}]}


def test_bare_tool_use_gets_synthesized_result():
    """The exact failing shape: assistant tool_use followed by user interjection."""
    msgs = [
        {"role": "user", "content": "go"},
        _tooluse("call_A"),
        {"role": "user", "content": "stop, do something else"},  # interjection
    ]
    assert len(_violations(msgs)) == 1, "precondition: the input IS broken"

    out = _validate_tool_pairs(msgs)
    assert _violations(out) == [], "repaired history must satisfy the endpoint rule"


def test_synthesized_result_names_the_interruption():
    """The filler must say WHY it is empty, so a later turn is not misled."""
    msgs = [
        {"role": "user", "content": "go"},
        _tooluse("call_A"),
        {"role": "user", "content": "interjection"},
    ]
    out = _validate_tool_pairs(msgs)
    results = [b for m in out if m.get("role") == "user"
               and isinstance(m.get("content"), list)
               for b in m["content"]
               if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "call_A"
    assert results[0]["content"] == _ORPHAN_TOOL_USE_RESULT
    assert "interrupted" in results[0]["content"].lower()


def test_assistant_text_and_thinking_are_preserved():
    """Repair inserts a result; it must NOT delete the turn's real content."""
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "deliberating"},
            {"type": "text", "text": "the answer is 42"},
            {"type": "tool_use", "id": "call_A", "name": "shell", "input": {}},
        ]},
        {"role": "user", "content": "interjection"},
    ]
    out = _validate_tool_pairs(msgs)
    types = [b.get("type") for m in out if m.get("role") == "assistant"
             for b in (m.get("content") or []) if isinstance(b, dict)]
    assert "thinking" in types, "thinking block was dropped"
    assert "text" in types, "assistant text was dropped"
    assert "tool_use" in types, "tool_use block was dropped"


def test_partially_answered_multi_tool_use_is_completed():
    """Only the unanswered ids get fillers; answered ones are not duplicated."""
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_A", "name": "shell", "input": {}},
            {"type": "tool_use", "id": "call_B", "name": "shell", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_A", "content": "real output"},
        ]},
    ]
    out = _validate_tool_pairs(msgs)
    assert _violations(out) == []
    ids = [b.get("tool_use_id") for m in out if m.get("role") == "user"
           and isinstance(m.get("content"), list)
           for b in m["content"]
           if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert ids.count("call_A") == 1, "answered id must not be duplicated"
    assert ids.count("call_B") == 1, "unanswered id must be filled exactly once"


def test_reverse_orphan_still_pruned():
    """The PRE-EXISTING behaviour must survive: a result with no use is dropped."""
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_GHOST", "content": "orphan"},
        ]},
        {"role": "user", "content": "next"},
    ]
    out = _validate_tool_pairs(msgs)
    ghost = [b for m in out if m.get("role") == "user"
             and isinstance(m.get("content"), list)
             for b in m["content"]
             if isinstance(b, dict) and b.get("type") == "tool_result"
             and b.get("tool_use_id") == "call_GHOST"]
    assert ghost == [], "orphaned tool_result must still be removed"


def test_well_formed_history_is_untouched():
    """No orphan -> byte-identical pass-through (no spurious filler appended)."""
    msgs = [
        {"role": "user", "content": "go"},
        _tooluse("call_A"),
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_A", "content": "ok"},
        ]},
    ]
    out = _validate_tool_pairs([dict(m) for m in msgs])
    assert _violations(out) == []
    fillers = [b for m in out if m.get("role") == "user"
               and isinstance(m.get("content"), list)
               for b in m["content"]
               if isinstance(b, dict) and b.get("content") == _ORPHAN_TOOL_USE_RESULT]
    assert fillers == [], "a valid pair must not receive a synthesized result"
