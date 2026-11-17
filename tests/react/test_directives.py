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

"""Tests for the user-directive store, tool, and dashboard rendering.

Three mechanisms carry this channel, and each gets an assertion that can turn
red on its own:

1. SESSION SCOPE — a directive recorded in session A must never appear in
   session B. The dashboard is a per-session artifact; a rule leaking across
   sessions is an order the user never gave for that task.
2. PROVENANCE — an entry without the user's verbatim words is rejected. If a
   blank quote is accepted, the channel silently becomes a second memory store:
   the agent's own judgement presented as a user order.
3. RESIDENCY — the block reaches the SYSTEM PROMPT via the dashboard, not only
   the tool reply. A block that renders into the tool result is invisible at
   decision time, which is the exact failure the channel exists to fix.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture()
def store():
    from flagscale_agent.react import directives

    return directives


@pytest.fixture()
def sess(tmp_path):
    """One session dir — the same one that holds conversation.json."""
    d = tmp_path / "session_a"
    d.mkdir()
    return str(d)


@pytest.fixture()
def sess_b(tmp_path):
    d = tmp_path / "session_b"
    d.mkdir()
    return str(d)


# ── session scope ───────────────────────────────────────────────────────


def test_store_lives_inside_the_session_dir(store, sess):
    store.upsert_directive(sess, "do", "rule x", "quote x")
    assert Path(store.directives_path(sess)) == Path(sess) / "directives.json"
    assert (Path(sess) / "directives.json").is_file()


def test_directive_does_not_leak_into_another_session(store, sess, sess_b):
    """SESSION SCOPE — the property that forced this design."""
    store.upsert_directive(sess, "dont", "不要在部署树改", "你犯了好多次")
    assert "不要在部署树改" in store.render_directives_block(sess)
    assert store.render_directives_block(sess_b) == ""
    assert store.list_directives(sess_b) == []


def test_session_dir_is_required(store):
    """No silent default: a default would reinstate a deployment-wide store."""
    with pytest.raises(ValueError):
        store.directives_path("")


def test_render_without_session_renders_nothing(store):
    assert store.render_directives_block(None) == ""


# ── store behaviour ─────────────────────────────────────────────────────


def test_upsert_and_render_do_and_dont(store, sess):
    store.upsert_directive(sess, "do", "跑相邻回归即可", "追求效率", "全量太慢")
    store.upsert_directive(sess, "dont", "不要在部署树改", "这你犯了好多次", "LIVE 读副本")
    block = store.render_directives_block(sess)
    assert "DO:" in block
    assert "DO NOT:" in block
    assert "跑相邻回归即可" in block
    assert "不要在部署树改" in block
    assert "全量太慢" in block  # the consequence clause survives


def test_empty_store_renders_empty_string(store, sess):
    """An empty heading every turn would be noise that teaches skipping."""
    assert store.render_directives_block(sess) == ""


def test_dedupe_ignores_punctuation_and_whitespace(store, sess):
    """The user restates the same order with a different comma."""
    store.upsert_directive(sess, "dont", "在源码树改，不要在部署树改")
    store.upsert_directive(sess, "dont", "在源码树改, 不要在部署树改")
    assert len(store.list_directives(sess)) == 1


def test_cap_drops_oldest_not_newest(store, sess):
    """A block rendered every turn must be bounded; the newest order wins."""
    for i in range(store.MAX_DIRECTIVES + 5):
        store.upsert_directive(sess, "do", f"rule-{i}", f"quote-{i}")
    rules = [d["rule"] for d in store.list_directives(sess)]
    assert len(rules) == store.MAX_DIRECTIVES
    assert rules[0] == f"rule-{store.MAX_DIRECTIVES + 4}"
    assert "rule-0" not in rules


def test_corrupt_file_degrades_to_empty(store, sess):
    """A rendering hiccup must never break a run."""
    (Path(sess) / "directives.json").write_text("{not json", encoding="utf-8")
    assert store.list_directives(sess) == []
    assert store.render_directives_block(sess) == ""


def test_remove_returns_false_when_absent(store, sess):
    assert store.remove_directive(sess, "never recorded") is False


def test_atomic_write_leaves_no_partial_file(store, sess):
    store.upsert_directive(sess, "do", "x rule", "x quote")
    json.loads((Path(sess) / "directives.json").read_text(encoding="utf-8"))
    assert not [p for p in os.listdir(sess) if p.endswith(".tmp")]


# ── tool ────────────────────────────────────────────────────────────────


def test_tool_rejects_entry_without_user_quote(store, sess):
    """PROVENANCE guard: no user words -> no directive."""
    from flagscale_agent.react.tools.directive import DirectiveTool

    out = DirectiveTool(sess).execute(action="add", kind="dont", rule="do not X")
    assert "ERROR" in out
    assert store.render_directives_block(sess) == ""


def test_tool_add_then_list_then_remove(store, sess):
    from flagscale_agent.react.tools.directive import DirectiveTool

    tool = DirectiveTool(sess)
    assert "(no directives recorded yet)" in tool.execute(action="list")
    out = tool.execute(action="add", kind="dont", rule="跑全量前先问", quote="去跑全量")
    assert "Recorded (DO NOT)" in out
    assert "跑全量前先问" in tool.execute(action="list")
    assert "Removed" in tool.execute(action="remove", rule="跑全量前先问")
    assert "(no directives recorded yet)" in tool.execute(action="list")


def test_tool_remove_without_rule_is_an_error(sess):
    from flagscale_agent.react.tools.directive import DirectiveTool

    assert "ERROR" in DirectiveTool(sess).execute(action="remove")


def test_tool_writes_only_into_its_own_session(store, sess, sess_b):
    """The agent rebinds its session dir on resume; a callable keeps it honest."""
    from flagscale_agent.react.tools.directive import DirectiveTool

    tool = DirectiveTool(lambda: sess)
    tool.execute(action="add", rule="A 会话的规则", quote="用户原话")
    assert store.list_directives(sess_b) == []
    assert len(store.list_directives(sess)) == 1


def test_tool_without_session_reports_error(store):
    from flagscale_agent.react.tools.directive import DirectiveTool

    assert "ERROR" in DirectiveTool("").execute(action="list")


# ── residency: the block must reach the SYSTEM PROMPT ───────────────────


def _dashboard(session_dir, plan_context=""):
    from flagscale_agent.react.prompt_builder import PromptBuilder

    pb = PromptBuilder(skill_manager=None)
    pb._turn_count = 1
    return pb._build_dashboard(plan_context, session_dir)


def test_dashboard_carries_the_directive_block(store, sess):
    """RESIDENCY: rendered by the prompt builder, not only by the tool."""
    store.upsert_directive(sess, "dont", "不要在部署树改", "你犯了好多次", "LIVE 读副本")
    out = _dashboard(sess)
    assert "Directives from the USER" in out
    assert "不要在部署树改" in out


def test_dashboard_of_another_session_has_no_directive_block(store, sess, sess_b):
    """The dashboard is per-session — this is what the user's correction fixed."""
    store.upsert_directive(sess, "dont", "不要在部署树改", "你犯了好多次")
    assert "Directives from the USER" not in _dashboard(sess_b)


def test_directives_render_above_self_authored_forbidden(store, sess):
    """Authority order, asserted on position in the bytes rather than by eye."""
    store.upsert_directive(sess, "dont", "不要在部署树改", "你犯了好多次")
    out = _dashboard(
        sess,
        '<active-plan title="T"><current-forbidden>mine</current-forbidden></active-plan>',
    )
    assert out.index("Directives from the USER") < out.index("Forbidden list")
    assert out.index("不要在部署树改") < out.index("mine")


def test_dashboard_without_directives_is_unchanged(store, sess):
    """No directives -> no empty section (and no stray blank lines)."""
    out = _dashboard(
        sess,
        '<active-plan title="T"><current-forbidden>mine</current-forbidden></active-plan>',
    )
    assert "Directives from the USER" not in out
    assert "Forbidden list" in out
    assert "\n\n\n" not in out


def test_directive_failure_does_not_break_the_dashboard(store, sess, monkeypatch):
    """Best-effort contract: a rendering failure degrades, never raises."""
    def boom(*a, **kw):
        raise RuntimeError("store exploded")

    monkeypatch.setattr(store, "render_directives_block", boom)
    out = _dashboard(sess)
    assert "Turn: 1R" in out  # dashboard still built
