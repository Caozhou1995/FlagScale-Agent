"""E9 — AN UNREAD STANDARD IS NOT A PROBABILITY.

Provenance: dna-assembly, job 2026-11-26__23-26-52, trial 8ZXZ9gz, reward=0.
The verifier's own bytes (/tests/test_outputs.py:49):

    assert i >= 1, "Primer must have clamp of at least 1 nucleotide before BsaI site."
    E  AssertionError: Primer must have clamp of at least 1 nucleotide before BsaI site.
    E  assert 0 >= 1

Delivered primer: 'ggtctcatgaggatcccgggaattc' — BsaI site at index 0, clamp = 0.

The agent was NOT ignorant of the requirement. Its own trajectory holds the
reasoning that lost the run:

  :1297 "A grader simulating digest will strip everything before and including
         the cleavage. Adding a clamp is standard and harmless. But there's risk
         if the grader checks the primer starts with GGTCTC exactly."
  :1301 "A clamp before GGTCTC means startswith fails. Requirement is only
         'satisfy NEB's requirements'. ... Which risk is higher?"
  :4114 "Safer: start the primer with GGTCTC (no clamp)."

It never read the checker (/tests, test_outputs, pytest: 0 hits in the 712KB
trajectory). It ranked two HYPOTHESISED checkers against each other and shipped
the minimal form. E6 already said an element "no stated rule forbids and one
reading demands is free" — the agent read E6, then treated a GUESS about the
consumer as the cost that made the element not-free. E9 closes that seam: under
an unread standard, the guessed veto is a coin flip, not evidence.
"""

import re

import pytest

from flagscale_agent.react.prompt import PROMPT_BUDGET, SYSTEM_PROMPT_STATIC, prompt_len

FLAT = re.sub(r"\s+", " ", SYSTEM_PROMPT_STATIC)


def test_e9_is_present_and_named():
    assert "E9 AN UNREAD STANDARD IS NOT A PROBABILITY." in FLAT


def test_e9_is_ordered_after_e8_not_merely_present():
    """A rank placed before E8 is a different rank. Assert the ORDER in the bytes."""
    i8 = FLAT.index("E8 A CARRIER'S SILENCE IS NOT A NEGATIVE FINDING.")
    i9 = FLAT.index("E9 AN UNREAD STANDARD IS NOT A PROBABILITY.")
    assert i8 < i9, "E9 must follow E8 in the ranked entry list"


def test_e9_names_the_failure_shape_not_a_virtue():
    """The clause must be executable: it names the coin flip, the licence, and the bet."""
    for face in (
        '"Which risk is higher"',      # the exact move the agent made
        "coin flip wearing evidence",   # what that move IS
        "no quoted rule forbids",       # the licence to ship the element
        "SHIPS",                        # the action
        "costs the run",                # the cost that settles the ranking
        "minimal form is a bet",        # the named default that must not be taken silently
        "ship the superset",            # the resolution
    ):
        assert face in FLAT, f"E9 lost its executable face: {face!r}"


def test_e9_survives_a_stoned_reader():
    """A stranger holding only the prompt must be able to act on it without context."""
    seg = FLAT[FLAT.index("E9 AN UNREAD STANDARD"):]
    seg = seg.split("Each rank exists because ignoring it lost a run.")[0]
    assert len(seg) > 200, "E9 too short to carry its rule"
    assert "checker you never read" in seg, "must scope itself to the unread-standard case"


def test_earlier_ranks_were_not_evicted_to_seat_e9():
    """The budget's usual casualty. E9 must not have displaced a ranked entry."""
    for head in (
        "E1 The task text's own named authority",
        "E2 The task text's own wording",
        "E3 A recorded death",
        "E4 Your own constructed checker",
        "E5 Where an authority names a variant",
        "E6 When a requirement is only REMEMBERED",
        "E7 A named authority's call that errors",
        "E8 A CARRIER'S SILENCE IS NOT A NEGATIVE FINDING.",
    ):
        assert head in FLAT, f"ranked entry lost while seating E9: {head!r}"


def test_budget_holds_after_e9():
    assert prompt_len() <= PROMPT_BUDGET, f"{prompt_len()} > {PROMPT_BUDGET}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
