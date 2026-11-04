"""Anti-decay pins for the semantic co-sourcing doctrine (was PRINCIPLE 2.1b).

The clause encodes a scientific pattern distilled from real failures where a
self-consistent simulator disagreed with the consumer because the semantics of
a load-bearing quantity were invented, not derived from the named standard.
The numbered 2.1b/2.4/2.5/2.9 sections no longer exist in the current prompt
(2026-11 ruling: adapt the tests to the current bytes, never the production
code). The pins below lock the MECHANISM via exact current phrases; the sweep
runs over the whole SYSTEM_PROMPT_STATIC.

Scientific mode, not leakage: every pinned phrase is domain-generic. A sweep
asserts no task-specific token appears in the slice.
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

import pytest


def _slice() -> str:
    # The numbered "2.1b ... 2.2" anchor text is gone from the current prompt,
    # so the sweep runs over the whole static prompt, whitespace-normalized.
    return " ".join(SYSTEM_PROMPT_STATIC.split())


@pytest.fixture(scope="module")
def clause() -> str:
    return _slice()


def test_clause_states_definition_not_reference(clause):
    # the core inversion: the definition you were given is checked, not assumed
    assert "Verification asks whether the thing matches the definition you were given" in clause
    assert "validation asks whether that definition was the real goal" in clause


def test_clause_states_simulator_is_defendant_not_judge(clause):
    # why a self-consistent checker proves nothing about the real artifact
    assert "Your own constructed checker ranks last as a judge of itself" in clause
    assert "A green whose verdict you authored is not evidence" in clause
    assert "split code-vs-solution verification" in clause


def test_clause_requires_the_writeback_line(clause):
    # the falsifier: a labelled gap marks an untestable premise
    assert "written down as a labelled gap rather than passed over in silence" in clause
    assert "load-bearing premise you could not test is written down as a labelled gap" in clause


def test_clause_silence_is_surfaced_not_filled(clause):
    # silent gaps are stated honestly, never filled with a silent assumption
    assert "that licence covers only a bound YOU chose" in clause
    assert "written down as a labelled gap rather than passed over in silence" in clause


def test_clause_invisible_judge_posture(clause):
    # an unseen gate is anticipated by sweeping its axes, not queried
    assert "an unseen gate's own axes" in clause
    assert "a gate built on one unexamined axis verifies only that axis" in clause


def test_conservation_invariant_pin_in_2_4():
    # definition-independent tie-breaker: the artifact on disk outranks the story
    start = SYSTEM_PROMPT_STATIC.index("The artifact on disk outranks the story in your head")
    end = SYSTEM_PROMPT_STATIC.index("UNDERDETERMINED CHOICES.")
    s = " ".join(SYSTEM_PROMPT_STATIC[start:end].split())
    assert "THE DELIVERABLE IS THE ONLY SOURCE OF TRUTH" in s
    assert "default to the DELIVERABLE being wrong" in s
    assert "the target's own bytes" in s
    assert "never compute it locally" in s


def test_quantity_is_a_family_not_two_readings():
    """Anti-decay: the open-ended enumeration must survive rewording.

    A clause that caps the candidate set structurally loses the deciding
    member. The current wording keeps the FAMILY framing: enumerate every
    defensible reading, never anchor on a counted number, and sweep the axes
    you did not try before trusting agreement.
    """
    start = SYSTEM_PROMPT_STATIC.index("(b) Which independent axes")
    end = SYSTEM_PROMPT_STATIC.index("(c) Could this check have gone red")
    s = " ".join(SYSTEM_PROMPT_STATIC[start:end].split())
    assert "axes multiply" in s
    assert "Never anchor on a counted number of variants" in s
    assert "enumerate the ones you can name" in s


def test_candidates_killed_by_object_fact():
    """Anti-decay: enumeration alone does not decide — object facts eliminate."""
    start = SYSTEM_PROMPT_STATIC.index("THE PROVENANCE RULE.")
    end = SYSTEM_PROMPT_STATIC.index("THE LAST GATE RUNS ON THE DELIVERED BYTES.")
    s = " ".join(SYSTEM_PROMPT_STATIC[start:end].split())
    assert "the named authority's own output" in s
    assert "if it still passes, the green was shared-source, not evidence" in s


def test_check_set_shape_covers_relational_constraints():
    """Anti-decay: the check set must be shaped like the spec's own — single
    axes AND the axes the consumer varies along (format, boundary, unit,
    scale, order, defaults), not one unexamined axis."""
    start = SYSTEM_PROMPT_STATIC.index("One example is not a held-out test set")
    end = SYSTEM_PROMPT_STATIC.index("ACCUMULATION IS A MEMORY CLAIM.")
    s = " ".join(SYSTEM_PROMPT_STATIC[start:end].split())
    assert "Scale is one axis only" in s
    assert "sweep the boundary types" in s
    assert "an unseen gate" not in s  # axis-sweep language lives in 3(b), not here
    assert "duplicate keys, out-of-order, non-ASCII" in s


def test_satisfy_rule_is_not_pair_scoped():
    """Anti-decay: the whole undominated reading set must be covered and,
    where readings conflict, settled at the intersection instead of betting."""
    start = SYSTEM_PROMPT_STATIC.index("UNDERDETERMINED CHOICES.")
    end = SYSTEM_PROMPT_STATIC.index("THE TRAP INVENTORY")
    s = " ".join(SYSTEM_PROMPT_STATIC[start:end].split())
    assert "ADMITS TWO READINGS, SATISFY BOTH" not in s
    assert "cover the top-2 in the same run" in s
    assert "cover with a switch defaulted to the dominant reading" in s
    assert "Record the bet in the delivery notes" in s


def test_no_task_identifiers_leak():
    # scientific mode: the clause must be class-generic, zero task vocab
    import re

    s = _slice().lower()
    for leak in ("dna", "golden gate", "overhang", "primer", "taat",
                 "bsai", "egfp", "melting", "plasmid"):
        assert not re.search(r"\b" + re.escape(leak), s), f"leaked task token: {leak}"
