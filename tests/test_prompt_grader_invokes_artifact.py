"""Falsifiable tests for the prompt discipline:
"The grader may RUN my script itself, possibly after rewriting the world it reads"

Motivation (measured, not hypothesised): a task asked for a script that reconstructs a
weight matrix by querying a local module. The grader, before invoking the script,
OVERWRITES that local module with a different seed and a different shape, then re-runs
the agent's script and loads whatever it saved. A script that imported the local module
to get its reference, or that hardcoded the shape it observed while developing, therefore
reports a self-check "all rows recovered" against a DECOY while the graded matrix differs.
"""

import pytest

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_discipline_present():
    """The discipline must exist and be phrased generically."""
    assert "Does the grader RUN my script itself" in SYSTEM_PROMPT_STATIC


def test_names_the_rewrite_hazard():
    """It must name the specific hazard: the grader may replace the inputs."""
    assert "AFTER replacing the very inputs your script consumes" in SYSTEM_PROMPT_STATIC
    assert "DECOY" in SYSTEM_PROMPT_STATIC


def test_forbids_trusting_local_reference():
    """It must forbid importing/trusting a local reference to define the answer."""
    assert "do not import and trust the local reference" in SYSTEM_PROMPT_STATIC
    assert "never hardcode a dimension, a count or a constant" in SYSTEM_PROMPT_STATIC


def test_requires_shape_agnostic_discovery():
    assert "DISCOVER the size and the parameters at run time" in SYSTEM_PROMPT_STATIC


def test_requires_perturbed_self_check():
    """The self-check must reproduce the invoke in a perturbed world."""
    assert "run the script yourself in a CLEAN copy of the environment" in SYSTEM_PROMPT_STATIC
    assert "compare what it emits against the truth OF THAT PERTURBED WORLD" in SYSTEM_PROMPT_STATIC


def test_requires_time_budget_awareness():
    assert "inside the grader's implied time budget" in SYSTEM_PROMPT_STATIC


def test_no_instance_specific_leakage():
    """Guard against overfitting: no concrete task numbers/names may leak into the prompt."""
    for token in ("forward.py", "steal.py", "stolen_A1", "randn(30", "30, 10", "19x10"):
        assert token not in SYSTEM_PROMPT_STATIC, f"leaked instance token: {token}"


def test_discipline_is_placed_with_the_grader_contract_items():
    """It belongs next to the other 'what does the grader actually do' items."""
    i = SYSTEM_PROMPT_STATIC.index("Does the grader RUN my script itself")
    j = SYSTEM_PROMPT_STATIC.index("Have I replayed the derivation the task implies")
    k = SYSTEM_PROMPT_STATIC.index("Which artifacts does the grader OPEN?")
    # sits in the same block as the grader-contract disciplines
    assert min(j, k) < i < max(j, k) + 4000
