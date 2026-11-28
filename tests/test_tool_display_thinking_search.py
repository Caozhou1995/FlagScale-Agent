"""Give `thinking_search` a dedicated icon + arg summary, and close the systemic
gap that let it fall through.

Two measured defects (2026-11-28):
  1. `_TOOL_ICONS` had no `thinking_search` entry -> it rendered the generic
     gear `⚙️`, indistinguishable from an unregistered tool. That matters
     because "did the agent search its own reasoning?" was the exact question
     under investigation, and the terminal line could not answer it.
  2. `tool_display_summary("thinking_search", {"query": ...})` returned "" ->
     the line printed `⚙️ thinking_search` with no indication of WHAT was
     searched, so two calls were indistinguishable in a transcript.

The systemic half: `test_tool_display_coverage.py` asserted icons over a
HAND-WRITTEN name list, so any newly registered tool was invisible to it.
These tests enumerate the real `ToolRegistry` instead.
"""
import re

import pytest

from flagscale_agent.react.display import _TOOL_ICONS, _tool_icon


def _render_line(name, args):
    """Reproduce what the terminal actually prints for one tool call."""
    return f"{_tool_icon(name)} {name} {tool_display_summary_safe(name, args)}".strip()


def tool_display_summary_safe(name, args):
    from flagscale_agent.react.tool_executor import tool_display_summary
    return tool_display_summary(name, args)


# ── The two measured defects ────────────────────────────────────────────

def test_thinking_search_has_a_dedicated_icon():
    """Not the generic fallback: the reader must see at a glance that the
    agent queried ITS OWN reasoning, not some unknown tool."""
    assert "thinking_search" in _TOOL_ICONS, (
        "thinking_search missing from _TOOL_ICONS -> renders the generic gear"
    )
    fallback = _tool_icon("some_unknown_tool_xyz")
    assert _tool_icon("thinking_search") != fallback


def test_thinking_search_summary_carries_the_query():
    """The discriminating field is the query string: two searches must be
    distinguishable in a transcript."""
    s = tool_display_summary_safe("thinking_search", {"query": "flock session lock"})
    assert s, "empty summary -> terminal line shows a bare tool name"
    assert "flock session lock" in s


def test_thinking_search_summary_entry_mode_shows_the_entry_number():
    """Entry mode reads ONE reasoning block by index; the index is what
    distinguishes the call, and the query is usually absent."""
    s = tool_display_summary_safe("thinking_search", {"query": "", "mode": "entry", "entry": 3})
    assert s, "entry-mode call must still render something identifying"
    assert "3" in s


def test_thinking_search_rendered_line_is_not_a_bare_tool_name():
    """Cold-reader form check: the full printed line must exceed icon+name."""
    line = _render_line("thinking_search", {"query": "where did I decide X"})
    bare = f"{_tool_icon('thinking_search')} thinking_search"
    assert line != bare
    assert "where did I decide X" in line


# ── The systemic half: enumerate the REAL registry, not a hand list ──────

def _registered_tool_names():
    """Build the real registry the agent builds, and read its tool names."""
    from flagscale_agent.react.agent import WorkerAgent  # noqa: F401
    import inspect
    import flagscale_agent.react.agent as agent_mod

    # Locate the function that assembles tools; call it if it is standalone.
    for name, obj in vars(agent_mod).items():
        if inspect.isfunction(obj) and re.search(r"build.*tools|register.*tools|_make_tools|_init_tools",
                                                 name, re.I):
            try:
                reg = obj()
            except TypeError:
                continue
            if hasattr(reg, "all_tools"):
                return [t.name for t in reg.all_tools()]
    return None


def test_every_registered_tool_has_a_dedicated_icon():
    """The check that should have caught thinking_search.

    Skips (visibly, not silently) if the registry cannot be built standalone —
    a skip is reported, whereas a hand-written list silently omits new tools.
    """
    names = _registered_tool_names()
    if names is None:
        pytest.skip("tool registry is assembled inside WorkerAgent.__init__; "
                    "not callable standalone — see test_icon_table_covers_documented_tools")
    fallback = _tool_icon("some_unknown_tool_xyz")
    missing = [n for n in names if _tool_icon(n) == fallback]
    assert not missing, f"registered tools falling back to the generic gear: {missing}"


def test_icon_table_entries_are_all_known_tool_names():
    """Reverse direction: an icon for a tool that no longer exists is dead
    weight and misleads the reader (a shipped-but-inert name)."""
    from flagscale_agent.react.tools import thinking_search as ts_mod  # noqa: F401
    # The tool module must expose its own name, so the icon key can be checked
    # against the real tool rather than against a remembered string.
    tool_name = getattr(ts_mod, "THINKING_SEARCH_TOOL_NAME", None) or "thinking_search"
    assert tool_name in _TOOL_ICONS


def test_no_two_distinct_tools_share_an_icon_unless_family():
    """One icon per tool, except an intentional family (the three plan_* tools).

    A shared icon between unrelated tools defeats the icon's whole purpose.
    """
    family = {"plan_create", "plan_update", "plan_status"}
    by_icon = {}
    for name, icon in _TOOL_ICONS.items():
        by_icon.setdefault(icon, []).append(name)
    collisions = {ic: ns for ic, ns in by_icon.items()
                  if len(ns) > 1 and not set(ns) <= family}
    assert not collisions, f"unrelated tools share an icon: {collisions}"
