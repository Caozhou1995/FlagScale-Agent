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
    assert "score EVERY point with the grader's own evaluator" in t
    assert "score all of it with the grader's EXACT evaluator" in t


def test_requires_reporting_margin_not_only_pass_fail():
    assert "Measure margin, not just pass/fail" in SYSTEM_PROMPT_STATIC


def test_non_monotone_warning_present():
    """The specific wrong intuition (longer => different Tm monotonically) must be stated."""
    t = SYSTEM_PROMPT_STATIC
    assert "need NOT be monotone in your design parameter" in t
    assert "walks in the wrong direction" in t


def test_instance_example_is_framed_as_illustration():
    """This prompt file's convention: concrete numbers appear ONLY inside a
    'Real instance:' rationale (cf. the cosine/ratio clause), never as a
    directive the agent must reproduce. Assert the framing, not the absence
    of digits — the digits are what make the lesson concrete."""
    t = SYSTEM_PROMPT_STATIC
    i = t.index(CLAUSE_HEAD)
    j = t.index("\n\n", i)
    clause = t[i:j]
    assert "Real failure:" in clause
    # The instance must be explicitly labelled as an example, not an instruction.
    # It must ALSO not name the graded task: a worked instance that identifies the
    # task hands the agent the answer's shape (that is leakage, not teaching).
    rule_body = clause.split("Real failure:")[0]
    for leaked in ("837", "Lr", "Lf", "oligotm", "primer", "melting temperature",
                   "Tm", "degrees", "anneal"):
        assert leaked not in clause, f"instance-specific token in clause: {leaked}"
