"""Falsifiable tests for the SEARCH-LOCATED OFFSET clause in the static prompt.

The clause closes the failure where the grader derives a boundary by SEARCHING
for your fragment inside a concatenation it builds, so the offset it finds is the
FIRST match and can differ from the offset you intended whenever your fragment
shares an edge character with its surroundings. Every derived quantity (a length,
a paired tolerance) then shifts by that difference.

Each assertion below is written so that removing the clause turns it red.
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _flat(text: str) -> str:
    return " ".join(text.split())


FLAT = _flat(SYSTEM_PROMPT_STATIC)


def test_clause_states_grader_searches_for_the_fragment():
    assert "located where you INTENDED it" in FLAT
    assert "finds a boundary by SEARCHING for your fragment" in FLAT


def test_clause_names_first_match_as_the_offset_actually_found():
    assert "the offset it finds is the first match" in FLAT


def test_clause_names_the_shared_edge_cause():
    for token in ("repeated substring", "a common prefix or suffix", "degenerate repeat"):
        assert token in FLAT, token


def test_clause_states_the_shift_consequence():
    assert "Every downstream quantity then shifts" in FLAT
    assert "a paired tolerance you computed at your intended offset can be wrong" in FLAT


def test_clause_requires_reproducing_the_search_and_printing_the_offset():
    assert "Reproduce the SEARCH itself" in FLAT
    assert "run the lookup, and print the offset it returns next to the one you intended" in FLAT


def test_clause_requires_worst_case_over_candidate_offsets():
    assert "if your fragment can match at more than one offset" in FLAT
    assert "evaluate the derived quantities at EVERY candidate offset" in FLAT
    assert "judge against the worst one" in FLAT


def test_clause_forbids_trusting_the_intended_position():
    assert "judge against the worst one" in FLAT


def test_clause_does_not_leak_instance_specific_values():
    """The clause must stay generic: no task-specific numeric or sequence literals."""
    import re as _re

    window = FLAT[FLAT.index("located where you INTENDED it"):]
    window = window[: window.index("Which artifacts does the grader OPEN")]
    leaks = _re.findall(r"\b\d+(?:\.\d+)?\s*(?:nt|bp|degrees)\b", window)
    assert not leaks, leaks
    for literal in ("primers_concat", "insert_start", "primers.fasta", "oligotm"):
        assert literal not in window, literal
