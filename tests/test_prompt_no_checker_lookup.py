"""The delivered system prompt must not point the agent at the hidden checker.

A leak of this kind does not crash and does not show up functionally: the model
keeps working, it just scores by transcription instead of capability. These tests
therefore assert on the PROMPT BYTES the agent actually receives, and each has a
negative control so the suite cannot degrade into a no-op.
"""

import re

import pytest

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as SYSTEM_PROMPT


# Phrases that, in the delivered prompt, instruct the reader to go obtain the
# hidden checker / official answer rather than build one from the specification.
LEAK_DIRECTIVES = [
    r"obtain the (?:real )??grader instead of",
    r"check whether the real checker is REACHABLE",
    r"task's upstream repository",
    r"benchmark's published task directory",
    r"official solution script",
    r"a grader you can READ or RUN",
]

# Task-specific vocabulary that must never appear: naming one task's objects,
# libraries, metrics or thresholds tells the model which task it is looking at.
TASK_SPECIFIC = [
    r"primers",
    r"\bmteb\b",
    r"oligotm",
    r"stolen_A1",
    r"forward\.py",
    r"HumanEval",
    r"nucleotide",
    r"1\.8e-2",
    r"0\.999997",
]


def _hits(patterns, text):
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


def test_prompt_does_not_direct_agent_to_the_hidden_checker():
    hits = _hits(LEAK_DIRECTIVES, SYSTEM_PROMPT)
    assert hits == [], f"prompt still points at the hidden checker: {hits}"


def test_prompt_has_no_task_specific_vocabulary():
    hits = _hits(TASK_SPECIFIC, SYSTEM_PROMPT)
    assert hits == [], f"prompt leaks task-specific vocabulary: {hits}"


def test_prompt_states_the_boundary_explicitly():
    """A prohibition must be present, not merely absent leak wording."""
    lowered = SYSTEM_PROMPT.lower()
    assert "transcription test" in lowered
    assert "do not fetch" in lowered


@pytest.mark.parametrize("dirty", [
    "check whether the real checker is REACHABLE and use it.",
    "read the benchmark's published task directory tests/ directory",
    "oligotm computes Tm for the primers",
    "a per-element ratio spread of 1.8e-2 against a 1e-4 tolerance",
])
def test_detectors_are_not_vacuous(dirty):
    """Negative control: each leak family, re-injected, must be caught."""
    pats = LEAK_DIRECTIVES + TASK_SPECIFIC
    assert _hits(pats, dirty) != [], f"detector missed injected leak: {dirty!r}"
