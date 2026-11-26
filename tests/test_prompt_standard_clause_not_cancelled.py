"""Ablation: the cancel-by-sibling clause must be present and must forbid using
one clause of a named standard to cancel another.

Measured evidence this guards (dna trial QD2fVYC, prompt md5 4c9c364d):
the previous 'never optional' clause removed the word "optional" from the
trajectory (count 0) yet the same downgrade reappeared as "Actually ... no extra
bases needed for BsaI since it cuts outside" -- the agent used a sibling NEB
clause (spacer between site and overhang) to cancel the graded clause (extra
bases 5' of the site). The delivered primers then had 8/8 site_at = 0.
So the clause to guard is the ACTION (cancelling a sibling clause), not a word.
"""
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_cancel_clause_is_present():
    assert "A named standard's clauses each bind separately" in SYSTEM_PROMPT_STATIC


def test_cancel_clause_forbids_the_action_not_just_the_word():
    # the clause must name the ACTION (cancel) rather than only banning a word
    assert "no clause cancels another" in SYSTEM_PROMPT_STATIC


def test_cancel_clause_still_accompanies_never_optional():
    # the earlier clause stays; this one is additive, not a replacement
    assert "An element a named standard requires is never optional." in SYSTEM_PROMPT_STATIC


def test_prompt_stays_within_budget():
    assert len(SYSTEM_PROMPT_STATIC) <= 50000
