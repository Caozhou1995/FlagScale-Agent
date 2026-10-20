"""The comparison-direction clause must state WHY merging loses.

When the criterion is per-item precision and extra items are not penalised,
clustering near-parallel observations into one averaged representative cannot
help: it removes no penalty and lowers every representative's precision. The
clause must carry the arithmetic (a ratio tolerance IS a direction tolerance)
so the rule is derivable rather than a bare prohibition.
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as P

CLAUSE = "merging is therefore pure loss"


def test_clause_present():
    assert CLAUSE in P


def test_ratio_tolerance_is_direction_tolerance():
    assert "really a DIRECTION tolerance" in P
    assert "positive scale factor already zeroes the ratio" in P


def test_states_the_order_of_magnitude_gap():
    i = P.index(CLAUSE)
    seg = P[max(0, i - 900):i + 900]
    assert "0.99" in seg and "1e-1" in seg and "1e-4" in seg


def test_prescribes_unmerged_emission():
    assert "un-merged and un-averaged" in P


def test_no_task_specific_identifiers_in_clause():
    i = P.index(CLAUSE)
    seg = P[max(0, i - 900):i + 900].lower()
    for tok in ("mteb", "stolen_a1", "primers", "cli_tool", "oligotm", "seed5"):
        assert tok not in seg, tok
