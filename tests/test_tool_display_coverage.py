"""Terminal tool-line readability: every registered tool must render a
human-readable arg summary (not just 'icon + name') and a dedicated icon.

Motivation: `tool_display_summary` fell through to "" for a whole class of
tools, so a tool call printed only `{icon} {name}` and the reader could not
tell WHAT the agent did. This suite locks the coverage in.
"""

import pytest

from flagscale_agent.react.display import _tool_icon
from flagscale_agent.react.tool_executor import tool_display_summary

FALLBACK = _tool_icon("some_unknown_tool_xyz")


# ── Arg summary: previously-empty tools must now be non-empty ───────────

@pytest.mark.parametrize("name,args", [
    ("load_knowledge", {"doc": "cluster_management/01.md", "name": "know-x"}),
    ("load_knowledge", {"name": "know-nccl-core"}),
    ("load_knowledge", {"name": "list"}),
    ("memory_list", {"keyword": "nccl", "domain_filter": "cluster"}),
    ("memory_list", {}),
    ("proposal", {"action": "update", "proposal_id": "prop_1", "status": "done"}),
    ("proposal", {"action": "add", "topic": "topic-x"}),
    ("proposal", {"action": "list"}),
    ("recall_search", {"query": "flock session"}),
    ("inspect_checkpoint", {"path": "/a/b/ckpt.pt"}),
    ("hard_reset", {"reason": "ctx 92%"}),
    ("spawn_worker", {"goal": "review the diff"}),
    ("dispatch_many", {"specs": [{}, {}], "degree": 2}),
    ("poll_tasks", {"action": "check", "task_id": "t1"}),
    ("poll_tasks", {"action": "list"}),
    ("resume_child", {"task_id": "t9"}),
    ("report_result", {"summary": "all 3 e2e passed"}),
])
def test_summary_nonempty(name, args):
    s = tool_display_summary(name, args)
    assert s, f"{name} produced an empty summary -> terminal line unreadable"


def test_summary_carries_the_discriminating_field():
    """The summary must surface the one field a reader needs to tell two
    calls of the same tool apart."""
    assert tool_display_summary("load_knowledge", {"doc": "a/b.md"}) == "a/b.md"
    assert tool_display_summary("recall_search", {"query": "flock session"}) == "flock session"
    assert tool_display_summary("spawn_worker", {"goal": "review the diff"}) == "review the diff"
    assert tool_display_summary("resume_child", {"task_id": "t9"}) == "t9"
    assert tool_display_summary("report_result", {"summary": "ok"}) == "ok"
    assert tool_display_summary("poll_tasks", {"action": "check", "task_id": "t1"}) == "check t1"
    assert tool_display_summary("hard_reset", {"reason": "ctx 92%"}) == "ctx 92%"


def test_summary_proposal_update_shows_target_and_status():
    s = tool_display_summary("proposal", {"action": "update", "proposal_id": "prop_1", "status": "done"})
    assert "prop_1" in s and "done" in s


def test_summary_dispatch_many_counts_workers():
    s = tool_display_summary("dispatch_many", {"specs": [{}, {}, {}], "degree": 2})
    assert "3" in s
    assert "2" in s  # degree surfaced


def test_summary_memory_list_filters():
    s = tool_display_summary("memory_list", {"keyword": "nccl", "domain_filter": "cluster"})
    assert "nccl" in s and "cluster" in s
    # bare memory_list still readable
    assert tool_display_summary("memory_list", {}) == "all"


def test_summary_proposal_batch_update():
    """Batch form (updates list, no profile_id) must stay readable — not
    'update  ->' with dangling arrows (regression: flip-check found it)."""
    s = tool_display_summary("proposal", {"action": "update",
                                          "updates": [{"proposal_id": "p1", "status": "done"},
                                                      {"proposal_id": "p2", "status": "rejected"}]})
    assert s and "->" not in s
    assert "2" in s  # count surfaced


def test_summary_proposal_update_without_status():
    s = tool_display_summary("proposal", {"action": "update", "proposal_id": "p1"})
    assert s == "update p1"


def test_summary_missing_args_does_not_crash():
    for name in ("load_knowledge", "memory_list", "proposal", "recall_search",
                 "inspect_checkpoint", "hard_reset", "spawn_worker",
                 "dispatch_many", "poll_tasks", "resume_child", "report_result"):
        # Must not raise on empty args.
        tool_display_summary(name, {})


# ── Icons: dedicated, non-fallback ──────────────────────────────────────

@pytest.mark.parametrize("name", [
    "load_knowledge", "memory_list", "proposal", "recall_search",
    "inspect_checkpoint", "hard_reset", "resume_child",
])
def test_dedicated_icon_not_fallback(name):
    icon = _tool_icon(name)
    assert icon, f"{name} has no icon"
    assert icon != FALLBACK, f"{name} fell back to the generic gear icon"


def test_new_icons_distinct_from_each_other():
    names = ["load_knowledge", "memory_list", "proposal", "recall_search",
             "inspect_checkpoint", "hard_reset", "resume_child"]
    icons = [_tool_icon(n) for n in names]
    assert len(set(icons)) == len(icons), "two new tools share one icon"
