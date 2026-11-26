"""The SINGLE VALUE ladder must open by ENUMERATING the judged quantities.

Evidence this pins: job 2026-11-25__19-19-01 / dna-assembly__uRoM8or. The prompt
already carried four strictness clauses (never grounds to veto stricter / no
clause cancels another / Strictest = the WORST case / a named standard's clauses
bind separately) and the agent still shipped 8/8 primers with the BsaI site at
index 0 (verifier: `assert 0 >= 1`). Diagnosis: it never enumerated "does the
primer carry bases 5' of the site" as a JUDGED quantity at all, so no ladder ever
ran on that axis. This test keeps the enumeration step in the bytes.
"""
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

FLAT = re.sub(r"\s+", " ", SYSTEM_PROMPT_STATIC).strip()


def test_ladder_opens_with_the_enumeration_step():
    assert "FIRST LIST THE JUDGED QUANTITIES" in FLAT


def test_enumeration_names_the_kinds_of_judged_quantity():
    # the list must span more than "a field": a length, a position, an offset,
    # a count and present-or-absent are all routinely judged on their own.
    i = FLAT.find("FIRST LIST THE JUDGED QUANTITIES")
    assert i > 0
    window = FLAT[i:i + 300]
    for kind in ("length", "position", "offset", "count", "present-or-absent"):
        assert kind in window, kind


def test_enumeration_states_the_omission_consequence():
    i = FLAT.find("FIRST LIST THE JUDGED QUANTITIES")
    window = FLAT[i:i + 420]
    assert "absent from every check behind it" in window


def test_ladder_is_per_quantity_not_once():
    i = FLAT.find("FIRST LIST THE JUDGED QUANTITIES")
    window = FLAT[i:i + 430]
    assert "walk the ladder PER quantity" in window


def test_enumeration_step_precedes_the_tie_rule():
    # ordering matters: enumeration is a step BEFORE resolving anything, so it
    # must sit ahead of the TIE rung in the bytes, not after it.
    i_enum = FLAT.find("FIRST LIST THE JUDGED QUANTITIES")
    i_tie = FLAT.find("that is a TIE", i_enum)
    assert i_enum > 0 and i_tie > i_enum


def test_prior_strictness_clauses_survive():
    # the four clauses that were already present must not be traded away for it
    assert "never grounds to veto stricter" in FLAT
    assert "no clause cancels another" in FLAT
    assert "An element a named standard requires is never optional." in FLAT
    assert "the WORST case the standard must survive" in FLAT
