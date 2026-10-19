"""Falsifier tests for the named-tool REGISTRY discipline in prompt.py.

The discipline: when a task names a library, that library's own metadata registry
defines the task-typed call. 'The task text never mentions the parameter' is NOT
a licence to omit it.
"""
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

PROMPT = SYSTEM_PROMPT_STATIC


def test_registry_clause_present():
    assert "REGISTRY say the run takes a parameter" in PROMPT


def test_registry_clause_demands_a_lookup_and_its_output():
    low = PROMPT.lower()
    assert "query that registry" in low
    assert "get_model_meta" in PROMPT
    assert "say which registry lookup you ran and what it printed" in low


def test_registry_clause_names_the_invalid_inference():
    low = PROMPT.lower()
    # the exact anti-pattern that made mteb fail three times
    assert "never mentions a prompt/instruction" in low
    assert "a missing mention is not a prohibition" in low


def test_registry_clause_asserts_authority_over_own_reading():
    low = PROMPT.lower()
    assert "treat that printed value as the authority" in low


def test_registry_clause_has_no_instance_leaks():
    """The clause must stay generic: no concrete task names or numbers."""
    start = PROMPT.index("REGISTRY say the run takes a parameter")
    clause = PROMPT[start:PROMPT.index("\n", start)]
    for leak in ("mteb", "SciFact", "bge", "MTEB:", "HumanEval", "0.3", "30, 10"):
        assert leak not in clause, f"instance leak {leak!r} in clause"
    # no bare numbers other than the parameter name in backticks
    assert not re.search(r"\b\d{2,}\b", clause), "numeric leak in clause"
