"""The comparison-direction clause must state WHY merging loses.

When the criterion is per-item precision and extra items are not penalised,
clustering near-parallel observations into one averaged representative cannot
help: it removes no penalty and lowers every representative's precision. The
clause must carry the arithmetic (a ratio tolerance IS a direction tolerance)
so the rule is derivable rather than a bare prohibition.

Deliberately NO numeric tolerance and no task identifier appears in the clause:
the relation must hold for whatever tolerance the check uses, so encoding one
particular value here would both overfit and pin a checker constant into the
repo.
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as P

CLAUSE = "merging is therefore pure loss"


def test_clause_present():
    assert CLAUSE in P


def test_ratio_tolerance_is_direction_tolerance():
    assert "IS a direction tolerance under another name" in P
    assert "positive scale factor already zeroes every ratio" in P


def test_states_the_generic_threshold_relation():
    i = P.index(CLAUSE)
    seg = P[max(0, i - 1200):i + 400]
    # The relation must be stated symbolically, not with one checker's numbers.
    assert "sqrt(2" in seg
    assert "τ" in seg and "τ²/2" in seg


def test_no_numeric_tolerance_literals_in_clause():
    i = P.index(CLAUSE)
    seg = P[max(0, i - 1200):i + 400]
    for tok in ("1e-4", "1e-1", "1e-9", "4e-5", "0.99"):
        assert tok not in seg, tok


def test_prescribes_verification_against_original_criterion():
    assert "never against the cluster's own cosine score" in P


def test_prescribes_unmerged_emission():
    assert "un-merged and un-averaged" in P


def test_no_task_specific_identifiers_in_clause():
    i = P.index(CLAUSE)
    seg = P[max(0, i - 1200):i + 400].lower()
    for tok in ("mteb", "stolen_a1", "primers", "cli_tool", "oligotm", "seed5"):
        assert tok not in seg, tok
