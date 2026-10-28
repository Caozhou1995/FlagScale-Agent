"""Anti-decay pins for the PRINCIPLE 2.1b clause (semantic co-sourcing).

The clause encodes a scientific pattern distilled from real failures where a
self-consistent simulator disagreed with the consumer because the semantics of
a load-bearing quantity were invented, not derived from the named standard.
The pins below lock the MECHANISM, not the wording. The slice is anchored to
the next section header (2.2), same convention as the 2.1a pins.

Scientific mode, not leakage: every pinned phrase is domain-generic. A sweep
asserts no task-specific token appears in the slice.
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

import pytest


def _slice() -> str:
    start = SYSTEM_PROMPT_STATIC.index("2.1b SEMANTIC CO-SOURCING")
    end = SYSTEM_PROMPT_STATIC.index("2.2 VERIFY AT THE ALTITUDE")
    return " ".join(SYSTEM_PROMPT_STATIC[start:end].split())


@pytest.fixture(scope="module")
def clause() -> str:
    return _slice()


def test_clause_states_definition_not_reference(clause):
    # the core inversion: named standard DEFINES the quantity, it is not advisory
    assert "IS THE DEFINITION, NOT A" in clause
    assert "derived from that standard's own stated behavior" in clause
    assert "never" in clause and "invented by your own implementation" in clause


def test_clause_states_simulator_is_defendant_not_judge(clause):
    # why a green self-simulator proves nothing about compatibility
    assert "second defendant" in clause
    assert "never a judge" in clause
    assert "self-consistent" in clause  # the trap: internal coherence is not compatibility
    assert "negative controls" in clause


def test_clause_requires_the_writeback_line(clause):
    # the falsifier: an unwritable line marks an untested premise
    assert "write one line" in clause
    assert "UNTESTED premise" in clause


def test_clause_silence_is_surfaced_not_filled(clause):
    # standard-silent gaps become plan risks, never intuition fills
    assert "silent" in clause
    assert "surface" in clause and "plan" in clause
    assert "intuition" in clause


def test_clause_invisible_judge_posture(clause):
    # anchor to the highest VISIBLE authority; an unseen judge is anticipated, not queried
    assert "not visible to you at solve time" in clause
    assert "highest authority you CAN see" in clause
    assert "only" in clause and "anticipated" in clause


def test_conservation_invariant_pin_in_2_4():
    # the definition-independent tie-breaker lives inside 2.4
    start = SYSTEM_PROMPT_STATIC.index("2.4 MEASURE THE QUANTITY")
    end = SYSTEM_PROMPT_STATIC.index("2.5 THE COMPARISON")
    s = " ".join(SYSTEM_PROMPT_STATIC[start:end].split())
    assert "CONSERVATION INVARIANT" in s
    assert "definition-independent" in s
    assert "both your own" in s  # names the co-source failure both-definitions share


def test_no_task_identifiers_leak():
    # scientific mode: the clause must be class-generic, zero task vocab
    import re

    s = _slice().lower()
    for leak in ("dna", "golden gate", "overhang", "primer", "taat",
                 "bsai", "egfp", "melting", "plasmid"):
        assert not re.search(r"\b" + re.escape(leak), s), f"leaked task token: {leak}"
