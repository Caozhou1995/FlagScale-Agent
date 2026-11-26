"""Regression guard for the two §8 memory clauses added 2026-11-23 (user request).

WHY THIS FILE EXISTS. `test_prompt_memory_doctrine_antidecay.py` pins the
deletions that a compression pass had eaten. This file pins the two clauses that
the same session's evidence said were MISSING from the doctrine altogether:

  (1) THE RETRIEVAL ORDER — the five-step checklist existed once, as
      "## Information Retrieval — Before You Search", added in be6a880 and
      deleted wholesale in 2185737 ("Rewrite system prompt to principles").
      Its only surviving copy was inside the find_guard *block message*, i.e.
      it was shown only AFTER a broad grep had already been blocked. The read
      half of the doctrine therefore had no order at all.

  (2) A MEMORY ENTRY CARRIES ITS OWN FALSIFIER — measured in this session: two
      entries written earlier had to be superseded because they recorded a
      CONCLUSION ("temperature causes the batch noise", "the ladder is
      ineffective") without the condition it was measured under, so the next
      session inherited them as established fact and could only undo them by
      re-running the whole measurement.

Either clause disappearing silently is the failure mode: §8 is the section no
benchmark literal pins, so it is the first place a "compress to pay for a
clause" round reaches for. Hence exact substrings, not a semantic check.
"""
from __future__ import annotations

import pytest

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

_START = "MEMORY pays in BOTH directions."
# The §8 memory material runs from the opener all the way to the §9 header: the
# retrieval-order and refuted-list clauses sit AFTER the "Context is finite"
# sign-off line, which is a closing sentence, not the end of the section. Using
# that sentence as the boundary is what made the first version of this file
# unable to see the clauses it exists to protect (measured: 4 failed).
_END = "════ 9. THE LOOP ON A PLAN"


def _memory_paragraph() -> str:
    src = SYSTEM_PROMPT_STATIC
    i = src.find(_START)
    assert i != -1, "§8 lost its MEMORY opener entirely"
    j = src.find(_END, i)
    assert j != -1, "§8 lost the §9 header boundary — section unreadable"
    return src[i:j]


# ── (1) the retrieval order ─────────────────────────────────────────────────

# Each step is pinned by the channel it names, in the order it must be tried.
# Order matters: a checklist that keeps all five channels but reorders them
# (shell before memory) is exactly the failure the original section prevented.
# Each step is pinned by the CHANNEL it names and by a truthful marker of that
# channel — never by a harness filename. Measured 2026-11-23: an earlier version
# of this file pinned "conversation_full.json" / "conversation.json", which made
# the guard itself the thing that coupled the prompt to the current harness's
# on-disk layout (rename the file, and the prompt must be edited to stay green).
# A channel is durable; a filename is not.
RETRIEVAL_ORDER = [
    ("THIS SESSION'S OWN LOG", "log-search tool"),
    ("(2) MEMORY", "READ FIRST routine"),
    ("A SCOPED SEARCH", "grep -rn"),
    ("ASK THE USER", "package or source location"),
]


def test_retrieval_clause_names_no_harness_file_literal():
    """The clause must describe channels, not this harness's file names.

    A prompt sentence that names conversation_full.json / conversation.json rots
    the moment the harness renames them, and the rot is silent: the clause keeps
    asserting a file that no longer exists. `grep -rn` is pinned instead because
    it is the tool's own invocation, not a data layout.
    """
    s = _memory_paragraph()
    for lit in ("conversation_full.json", "conversation.json"):
        assert lit not in s, (
            f"§8 retrieval clause names the harness file {lit!r}. Describe the "
            "channel by what it is FOR; the tool description already carries "
            "the filename, so this copy is pure coupling."
        )


def test_retrieval_checklist_is_present_and_ordered():
    s = _memory_paragraph()
    assert "THE RETRIEVAL ORDER, CHEAPEST FIRST" in s, (
        "§8 lost the retrieval-order clause again; it was deleted wholesale once "
        "(2185737) and only survived inside find_guard's block message"
    )
    positions = []
    for channel, marker in RETRIEVAL_ORDER:
        p = s.find(channel)
        assert p != -1, f"retrieval step {channel!r} is missing from §8"
        assert marker in s, f"retrieval step {channel!r} no longer names {marker!r}"
        positions.append(p)
    assert positions == sorted(positions), (
        "retrieval channels are out of order — cheapest-first is the whole point "
        f"(found positions {positions})"
    )


def test_retrieval_clause_keeps_the_broad_search_rule():
    """The rule that actually costs a run: memory BEFORE walking a tree."""
    s = _memory_paragraph()
    assert "Read that file directly instead of scanning a tree" in s
    assert "never the first move" in s


def test_retrieval_clause_keeps_the_log_search_layer():
    """The log-search layer was what the earlier §8 rewrite dropped entirely.

    Pinned by the mechanism (keywords ANDed, anchors fed to recall), not by the
    tool name: if the tool is renamed the DERIVED block still carries the name,
    and this clause still describes the same channel.
    """
    s = _memory_paragraph()
    assert "anchors you can feed to recall(index=N)" in s
    assert "ANDs them" in s


# ── (2) the self-falsifying entry ───────────────────────────────────────────

def test_entry_must_carry_its_falsifier():
    s = _memory_paragraph()
    assert "A MEMORY ENTRY CARRIES ITS OWN FALSIFIER" in s
    # The three faces that make it usable, not just assertive.
    assert "measured under" in s
    assert "what would make it false" in s
    # And why a bare conclusion is worse than nothing.
    assert "guess wearing a fact's clothes" in s


def test_refuted_list_reads_before_the_hypothesis():
    """The companion rule: read the domain's refutations before proposing."""
    s = _memory_paragraph()
    assert "READ THE REFUTED LIST BEFORE YOU PROPOSE" in s
    assert "before the hypothesis, not after the measurement" in s


def test_new_clauses_stay_inside_the_budget():
    from flagscale_agent.react import prompt as prompt_mod
    assert prompt_mod.prompt_len() <= prompt_mod.PROMPT_BUDGET
