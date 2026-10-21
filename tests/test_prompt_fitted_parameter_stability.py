"""G5: a fitted parameter reported without a stability check is one point estimate
from one starting guess -- and the grader may admit a tolerance the task never states.

Evidence (raman-fitting trial, reward=0, batch 2026-10-21__08-10-31): the task
instruction states no tolerance. The grader's test_outputs.py applies four of them
(|x0 err|<5%, |gamma err|<1, |amplitude err|<5%, |offset err|<10%). The agent fit
once with curve_fit defaults and delivered offset=990.7 against a true 1239.09 --
a 20% miss on the ONE parameter with the loosest allowance, i.e. a local optimum,
while two of three grader tests passed.

The clause must be domain-neutral: it talks about fitted parameters and starting
guesses, never about any particular spectrum type.
"""

from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
HEAD = "A FITTED PARAMETER SHIPS WITH ITS STABILITY CHECK"


def _text() -> str:
    return PROMPT.read_text()


def test_clause_present():
    """The discriminative phrase must exist verbatim."""
    assert HEAD in _text()


def test_clause_is_where_quantitative_constraints_are_discussed():
    """It belongs next to the constraint-loyalty rule, not bolted onto the end."""
    text = _text()
    i_clause = text.index(HEAD)
    i_loyalty = text.index("Constraint loyalty")
    assert abs(i_clause - i_loyalty) < 4000, (i_clause, i_loyalty)


def test_clause_names_the_multi_start_requirement():
    """The actionable part: refit from several starting guesses / intervals."""
    text = _text()
    i = text.index(HEAD)
    window = text[i : i + 1600]
    assert "starting" in window and ("several" in window or "multiple" in window)
    assert "agree" in window


def test_clause_requires_reporting_the_spread_not_a_bare_value():
    """A value whose neighbours disagree must be reported WITH the spread."""
    text = _text()
    i = text.index(HEAD)
    window = text[i : i + 1600]
    assert "spread" in window or "interval" in window
    assert "tolerance" in window


def test_no_task_identifying_words_added():
    """Anti-leak: the clause must not name the task, its data file or its fields."""
    text = _text()
    for leak in (
        "raman",
        "graphene",
        "curve_fit",
        "least_squares",
        "x0=",
        "gamma=",
        "amplitude=",
        "offset=",
    ):
        assert leak not in text, leak
