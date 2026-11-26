"""Pin §3(b)'s measured-span clause.

The gap this closes was found on a real failed trail (a consumer whose
reference-matching auto-extended the span it measured, so a self-check that
measured "the region I intended to write" stayed green while the consumer
read a different, longer region and failed).

The clause is content-bearing, so it must be pinned VERBATIM, not by topic.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as PROMPT

FLAT = re.sub(r"\s+", " ", PROMPT)


def test_measured_span_clause_present():
    assert "the span it measures is the MATCHED region" in FLAT
    assert "not the region you meant to write" in FLAT


def test_measured_span_clause_states_the_mechanism():
    # The mechanism, not just the label: why the span differs.
    assert "auto-extends over adjacent identical context" in FLAT
    assert "silently grow" in FLAT


def test_measured_span_clause_states_the_failure_mode():
    # The load-bearing sentence: a self-check can stay green while wrong.
    assert "your own check still reads green" in FLAT


def test_measured_span_clause_states_the_repair():
    assert "Compute the measurement on the consumer's matched span" in FLAT
    assert "with the same matching rule" in FLAT


def test_measured_span_clause_sits_in_the_axis_paragraph():
    """It must extend §3(b)'s axis rule, not float elsewhere."""
    i = FLAT.find("name how the consumer fixes its span")
    assert i != -1
    j = FLAT.find("Compute the measurement on the consumer's matched span")
    assert j != -1
    assert 0 < j - i < 600, "clause must be adjacent to the span rule it extends"
