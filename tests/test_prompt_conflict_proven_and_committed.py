"""Ablation gates for the (a) conflict-must-be-proven / (b) ruling-must-be-a-value clause.

Why these tests exist
---------------------
Measured on the dna-assembly task over 39 trials (4 pass / 35 fail, same task,
same checksum): the pass/fail difference was NOT awareness of the NEB 5' flank
rule. The failing runs mentioned it MORE often (30 mentions) than the passing run
(15.5). The difference was the ANCHOR of the ruling at the commit moment:

  * passing run  -> anchored on the standard's own words; the apparent conflict
                    (flank vs "primer starts with the site") dissolved because the
                    standard's words never made them mutually exclusive.
  * failing run  -> anchored on an IMAGINED grader ("a grader checking starts with
                    GGTCTC would fail if I add extra bases") plus silence read as
                    negation; it manufactured an either/or that did not exist, then
                    wrote the ruling as prose that never reached the deliverable.

So the clause has two independently testable halves, and each gets its own gate:
  (a) does the prompt demand that a requirements conflict be PROVEN from the
      standard's words (or both sides satisfied together)?
  (b) does the prompt demand the ruling be committed as a VALUE THE DELIVERABLE
      CONSUMES rather than prose?

Every test below is written to go RED when its half is softened. Verified by
ablating each half separately; see the ablation notes beside each assertion.
"""

import re

import pytest

from flagscale_agent.react import prompt as P


def _flat() -> str:
    """Render the static prompt with runs of whitespace collapsed.

    The literal lives across source line breaks, so probing the raw source with a
    single-line phrase fails for reasons that have nothing to do with the clause
    (a defect that cost a full round earlier). Collapsing first removes that
    axis instead of working around it.
    """
    return re.sub(r"\s+", " ", P.SYSTEM_PROMPT_STATIC)


def test_clause_is_present():
    """The clause exists in the DELIVERED bytes (not just in this test file)."""
    flat = _flat()
    assert "the collision is itself a claim" in flat, (
        "half (a) missing: nothing makes the collision a claim"
    )
    assert "prove it from the standard's own words, or satisfy both" in flat, (
        "half (a) missing: no proof obligation / no satisfy-both exit"
    )
    assert "The ruling must become a value the deliverable consumes" in flat, (
        "half (b) missing: nothing demands the ruling reach the deliverable"
    )


def test_half_a_rejects_unproven_conflict():
    """Ablation: soften (a) to 'consider the conflict' and this goes red.

    The clause must make the conflict a CLAIM carrying a proof obligation. If it
    only says 'resolve the conflict', the failing behaviour (manufacturing an
    either/or from an imagined grader) satisfies it and the gate cannot fire.
    """
    flat = _flat()
    assert "the collision is itself a claim" in flat
    # the escape hatch must be present in the same breath, otherwise the rule
    # reads as "always declare a conflict", which is the opposite failure mode
    assert "or satisfy both" in flat, (
        "half (a) lost its alternative: without 'or satisfy both', the wording "
        "licenses declaring a conflict instead of dissolving it"
    )


def test_half_a_forbids_grader_anchored_rulings():
    """The ruling may not be anchored on a guessed consumer.

    This is the exact failing sentence from trial iWuSWMr: 'A grader checking
    "starts with GGTCTC" would fail if I add extra bases.' The prompt already has
    a general 'an imagined grader weighs zero' rule elsewhere; this clause has to
    connect it to the conflict decision specifically, or the model satisfies the
    general rule while still reasoning from the imagined grader here.
    """
    flat = _flat()
    assert "from the standard's own words" in flat, (
        "the anchor the clause requires is not named; a ruling anchored on a "
        "remembered convention or a guessed grader would pass this prompt"
    )


def test_half_b_requires_a_consumed_value_not_prose():
    """Ablation: replace 'a value the deliverable consumes' with 'a note' -> red.

    Measured shape of the failure: the ruling was written into the reasoning
    transcript (~66% into the run) and the deliverable was emitted later (~78%)
    from a separate code path that never read the ruling back. Prose cannot
    survive that gap; a consumed value can.
    """
    flat = _flat()
    assert "a value the deliverable consumes" in flat, (
        "half (b) weakened: a prose ruling is what already failed"
    )
    assert "not prose" in flat, (
        "the contrast that makes (b) binding is missing; without it 'commit the "
        "ruling' can be satisfied by writing another sentence"
    )


def test_clause_sits_in_the_undetermined_choices_section():
    """Placement: the clause must live where open choices are ruled on.

    Not cosmetic. The clause is about what to do when two readings of a standard
    compete, which is exactly the subject of UNDERDETERMINED CHOICES. Put it in
    the closing sections and a model deciding mid-run may never re-read it.
    """
    flat = _flat()
    under = flat.find("UNDERDETERMINED CHOICES")
    clause = flat.find("WHEN TWO REQUIREMENTS COLLIDE")
    trap = flat.find("THE TRAP INVENTORY")
    assert under != -1 and clause != -1 and trap != -1
    assert under < clause < trap, (
        f"clause out of place: under={under} clause={clause} trap={trap}"
    )


def test_prompt_still_within_budget():
    """The budget is a hard gate (asserted by the repo's own suite too)."""
    assert P.prompt_len() <= P.PROMPT_BUDGET, (
        f"over budget: {P.prompt_len()} > {P.PROMPT_BUDGET}"
    )
