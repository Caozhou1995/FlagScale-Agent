"""Pin the three obligations the 2026-11-25 dna-assembly run showed are missing.

Evidence from that trial (agent/flagscale-agent.txt + verifier/test-stdout.txt):
the agent had the right instinct twice and talked itself out of it both times,
and when the named authority's page would not load it stopped after ONE route.

1. AUTHORITY IS AN OPEN OBLIGATION, not a note. It wrote "If unreachable, I'll
   include a 4-nt clamp" and then dropped it, reasoning from remembered NEB
   guidance instead of a fetched rule. The prompt let it label the field
   UNCONSULTED and proceed.
2. A FAILED ROUTE IS NOT A MISSING FACT. It tried neb.com once (failed), never
   tried a mirror/archive/local copy, even though the guard said to.
3. SELF-PERSUASION MUST CITE. Its deciding argument was "the emphasis is on
   'cut-sites'", i.e. a reading of the task's own wording with no standard
   quoted and no derivation given.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as PROMPT

FLAT = re.sub(r"\s+", " ", PROMPT)


def test_unfetched_authority_leaves_the_obligation_open():
    """A standard you could not reach stays owed; memory may not stand in."""
    assert "unfetched, the obligation stays open" in FLAT
    assert "nothing remembered stands in for it" in FLAT


def test_authority_is_discharged_only_by_an_executed_read():
    assert "the clause is satisfied only by an EXECUTED READ of that source" in FLAT


def test_failed_route_is_not_a_missing_fact():
    assert "A failed ROUTE is not a missing FACT" in FLAT


def test_the_route_set_is_named_not_left_to_improvisation():
    """The four route kinds the run never tried."""
    assert "exhaust the routes (mirror, alternate host, archive, local copy)" in FLAT


def test_self_persuasion_must_quote_or_derive():
    assert (
        "Self-persuasion quoting neither the standard's words nor a derivation "
        "is that same bet"
    ) in FLAT


def test_memory_is_still_not_defensible():
    """The pre-existing rule this clause extends must survive."""
    assert "A remembered convention is NOT defensible and never outranks a sourced reading" in FLAT


def test_clause_stays_task_agnostic():
    banned = [
        "dna", "bsai", "ggtctc", "primer", "overhang", "anneal", "oligotm",
        "primer3", "plasmid", "egfp", "clamp", "fasta", "neb", "golden gate",
        "terminal-bench", "dna-assembly",
    ]
    low = FLAT.lower()
    for word in banned:
        assert not re.search(r"\b" + re.escape(word) + r"\b", low), (
            f"clause leaked task vocabulary: {word}"
        )
