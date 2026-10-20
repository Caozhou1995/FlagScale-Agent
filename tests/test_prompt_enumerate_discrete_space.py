"""Falsifiable tests for the 'enumerate a small discrete design space' clause.

Deleting that clause from the static prompt must turn these RED.
"""
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

CLAUSE_HEAD = "Is my design space SMALL and DISCRETE enough to enumerate instead of search?"
TRAP1 = "STOPPING AT THE FIRST STRUCTURALLY-VALID POINT"
TRAP2 = "ASSUMING MONOTONICITY"


def test_clause_present():
    assert CLAUSE_HEAD in SYSTEM_PROMPT_STATIC


def test_both_traps_named():
    assert TRAP1 in SYSTEM_PROMPT_STATIC
    assert TRAP2 in SYSTEM_PROMPT_STATIC


def test_feasible_region_is_not_the_answer():
    """The clause must distinguish 'feasible region' from 'the verdict'."""
    t = SYSTEM_PROMPT_STATIC
    assert "describe a FEASIBLE REGION, not the answer" in t


def test_requires_exhaustive_scoring_with_grader_evaluator():
    t = SYSTEM_PROMPT_STATIC
    assert "score EVERY point against the FULL constraint set" in t
    assert "score all of it and report how many points pass" in t


def test_requires_reporting_margin_not_only_pass_fail():
    assert "Measure margin, not just pass/fail" in SYSTEM_PROMPT_STATIC


def test_non_monotone_warning_present():
    """The specific wrong intuition (longer => different Tm monotonically) must be stated."""
    t = SYSTEM_PROMPT_STATIC
    assert "need NOT be monotone in your design parameter" in t
    assert "walks in the wrong direction" in t


def test_clause_carries_no_worked_instance():
    """Convention change: this clause now states the method with NO worked instance.

    Earlier the prompt embedded a concrete failing case to make the lesson vivid.
    That case was drawn from one graded task, so it doubled as a hint about that
    task's own arithmetic — a leak. The clause must therefore teach the rule
    abstractly: no sample numbers, and no vocabulary that identifies a graded task.
    """
    t = SYSTEM_PROMPT_STATIC
    i = t.index(CLAUSE_HEAD)
    j = t.index("\n\n", i)
    clause = t[i:j]

    # no worked-instance framing at all
    assert "Real failure:" not in clause
    assert "Real instance:" not in clause
    # and nothing that identifies a graded task or hands over its arithmetic
    for leaked in ("837", "Lr", "Lf", "oligotm", "primer", "melting temperature",
                   "Tm", "degrees", "anneal", "1e-4", "1.8e-2", "cosine"):
        assert leaked not in clause, f"instance-specific token in clause: {leaked}"
