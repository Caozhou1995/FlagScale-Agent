"""Pin the "a criterion that cannot go RED is not a criterion" clause.

Measured gap this was written from (gcode-to-text, three runs of the SAME task,
correct string fixed at `flag{gc0d3_iz_ch4LLenGiNg}` by the verifier):

  run 14-09-38  -> correct                                  reward 1
  run 16-25-04  -> `gcod3 / l l / Ging` (8 chars wrong)     reward 0
  run 17-03-43  -> `GinG` (2 chars wrong)                   reward 0

The pass was NOT obtained by a better criterion: at higher zoom it read
"Whoa — this shows the last part as 'i' then 'N' (capital N!)", i.e. it
recognised the glyph by RESEMBLANCE and admitted the earlier read came from a
lower zoom where n's arch had swallowed N's diagonal. Nothing predicted a
number; every zoom "looks like" something, so the resemblance criterion could
never have returned the wrong answer — which is why it did return the wrong
answer in two of three runs.

Worse, run 16-25-04 explicitly identified the discriminating method
("compare each glyph's shape against rendered candidates") and rejected it
twice because "PIL absent" — then wrote its own stdlib rasteriser anyway,
i.e. it had the primitives and declined to invert. The clause's second half
("in the verdict, not a note") covers the sibling shape where a criterion is
selected and then not applied at the decision point.

Budget: the user authorized +200 for this clause alone (51_700 -> 51_900,
generous; the clause costs 194).
"""
from flagscale_agent.react import prompt as prompt_mod

CLAUSE = ' A "looks like X" check cannot go RED'
BODY = ('anything resembles something somewhere. Demand a NUMBER predictable '
        'under BOTH readings; equal under both = no discrimination; in the '
        'verdict, not a note.')


def _static():
    return prompt_mod.SYSTEM_PROMPT_STATIC


def test_clause_present_exactly_once():
    text = _static()
    assert text.count(CLAUSE) == 1, (
        f"clause must appear once, found {text.count(CLAUSE)}")


def test_clause_states_the_cannot_go_red_reason():
    """The WHY is load-bearing: without it the rule reads as a style preference."""
    text = _static()
    assert "cannot go RED" in text
    assert "anything resembles something somewhere" in text


def test_clause_demands_a_number_predictable_under_both_readings():
    """The positive obligation: replace resemblance with a quantity whose value
    differs between the rival hypotheses."""
    text = _static()
    assert "NUMBER predictable under BOTH readings" in text
    assert "equal under both = no discrimination" in text


def test_clause_requires_application_at_the_decision_point():
    """The sibling failure: a criterion that was selected but never used in the
    verdict. Both halves must survive -- an assertion on one alone would let the
    other silently decay."""
    text = _static()
    assert "in the verdict, not a note" in text


def test_clause_sits_in_the_measurement_loop_not_the_evidence_ranks():
    """Placement is part of the rule: it belongs beside "could this check have
    gone red?" (§3c), not among the source-ranking rules (§2)."""
    text = _static()
    anchor = "parse it out of the delivered bytes, so a wrong value can turn that check red."
    start = text.find(anchor)
    assert start != -1, "the §3(c) anchor must still exist"
    window = text[start:start + len(anchor) + len(CLAUSE) + 10]
    assert CLAUSE.strip() in window, (
        "clause must follow §3(c)'s 'could this check have gone red' sentence")


def test_clause_does_not_regress_the_adjacent_measurement_rule():
    """Positive bound: the clause was ADDED, so the sentence it qualifies must
    survive verbatim."""
    text = _static()
    assert "An axis your check hard-codes as a constant offset is fixed, not measured" in text


def test_prompt_within_the_reauthorised_budget():
    n, b = prompt_mod.prompt_len(), prompt_mod.PROMPT_BUDGET
    assert n <= b, f"over budget: {n} > {b}"


def test_clause_costs_no_more_than_the_authorised_slack():
    """It shipped at 194 of the 200 granted. A rewrite must not quietly exceed it."""
    assert len(CLAUSE + BODY) <= 200, f"clause is {len(CLAUSE + BODY)} chars (slack 200)"


def test_no_task_vocabulary_leaked_into_the_prompt():
    """prompt.py is scanned wholesale by the leak fixtures, so no task nouns --
    the gcode flag string, the font names, the trial ids -- may enter it."""
    src = open(prompt_mod.__file__, encoding="utf-8").read()
    for noun in ("gc0d3", "ch4LLen", "gcode", "DejaVu", "BsaI"):
        assert noun.lower() not in src.lower(), f"task vocabulary leaked: {noun}"
