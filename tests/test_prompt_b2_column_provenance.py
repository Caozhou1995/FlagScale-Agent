"""Pin the (b2) column-provenance clause.

Origin: dna-assembly__u3sjafW (job 2026-11-26__22-32-17), reward=0.
The agent DID build the two-column table (prompt §3(b2) worked: its log line 2199
says "That's the two-column table approach from Section 3(b2)", line 2201 "the
pair-difference must hold under both readings", line 4332 "Tm 58-72 -> 61.33-69.82
(every value, both readings); pair delta-Tm <=5 -> max 4.12").  It still failed:
the verifier measured fwd/rev Tm from the full annealing tract INCLUDING the
overhang-matched 4 nt, delta = 7.637 > 5.

Both of its columns were anchored on template matches IT picked itself.  The
missing dimension is that the column boundaries must come from a datum the agent
READ (consumer code / stated definition / sample value), not be constructed.

This test pins the clause so a future length-budget compression cannot silently
drop the one dimension whose absence the evidence identified.
"""
from flagscale_agent.react import prompt as prompt_mod


def _static() -> str:
    return prompt_mod.SYSTEM_PROMPT_STATIC


def test_column_boundary_must_be_read_not_chosen():
    s = _static()
    assert "a datum you READ and can cite" in s, (
        "the (b2) column-provenance clause is missing"
    )
    assert "a construction you chose yourself fills no column" in s


def test_clause_sits_inside_b2_not_floating():
    """The clause must be part of the (b2) paragraph, not an orphan elsewhere."""
    s = _static()
    b2 = s.find("A REJECTED READING IS A COLUMN YOU STILL HAVE TO FILL")
    assert b2 != -1
    clause = s.find("a datum you READ and can cite")
    assert clause != -1
    # must appear after (b2) opens and before (c) begins
    c = s.find("(c) Could this check have gone red", b2)
    assert c != -1
    assert b2 < clause < c, (b2, clause, c)


def test_clause_states_the_failure_mode():
    """It must name WHY self-chosen columns are not enough, not just assert a rule."""
    s = _static()
    assert "two self-chosen columns can both clear the bound while the consumer's span does not" in s


def test_prompt_still_within_budget():
    assert _static() and len(_static()) <= prompt_mod.PROMPT_BUDGET
