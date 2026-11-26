"""Pin §3(b)'s "remove the span ambiguity" half.

The diagnosis half of this clause ("the span the consumer measures is the
MATCHED region") is pinned in test_prompt_measured_span_clause.py. It named
the mechanism but left the agent with nothing to DO about it: the remedy it
offered was "compute on the consumer's span", which requires knowing the
consumer's matching rule — unavailable in the very cases that matter.

This half makes it executable WITHOUT that knowledge, by attacking the
ambiguity at design time: choose the free parameter so the two readings
collapse, or measure both and require every reading to clear the bound.

Pinned verbatim: it is content-bearing prose, and a topical "contains the
word span" assertion would pass on any rewording that lost the action.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as PROMPT

FLAT = re.sub(r"\s+", " ", PROMPT)


def test_states_the_design_time_preference():
    # The ruling: removing the ambiguity beats satisfying it.
    assert "cheaper to remove than to satisfy" in FLAT


def test_names_the_parameter_and_the_collapse():
    assert "When one free parameter decides whether the consumer's span extends" in FLAT
    assert "the two readings collapse to a single value" in FLAT


def test_names_the_unsatisfiable_case_fallback():
    # When the parameter is NOT ours: cover both readings, not just the
    # intended one.
    assert "Where that parameter is not yours to choose" in FLAT
    assert "compute the quantity under BOTH spans" in FLAT
    assert "not only the one you meant" in FLAT


def test_requires_a_test_of_the_collapse():
    # A claim of collapse that was never exercised is the failure mode.
    assert "test the collapse on one real instance" in FLAT
    assert "a different parameter is in control" in FLAT


def test_extends_the_measured_span_clause_it_depends_on():
    """It must follow the diagnosis it operationalises, in the same paragraph."""
    i = FLAT.find("Compute the measurement on the consumer's matched span")
    j = FLAT.find("Span ambiguity is cheaper to remove than to satisfy")
    assert i != -1 and j != -1
    assert 0 < j - i < 400, "must attach to the span clause, not float elsewhere"


def test_clause_stays_task_agnostic():
    """No benchmark vocabulary: the clause is a general rule, not this task's."""
    banned = [
        "dna", "bsai", "ggtctc", "primer", "overhang", "anneal",
        "oligotm", "primer3", "plasmid", "egfp", "clamp", "fasta",
        "terminal-bench", "dna-assembly",
    ]
    low = FLAT.lower()
    for word in banned:
        assert not re.search(r"\b" + re.escape(word) + r"\b", low), (
            f"clause leaked task vocabulary: {word}"
        )
