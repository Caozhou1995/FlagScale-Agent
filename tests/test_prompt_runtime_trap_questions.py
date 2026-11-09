"""Pin the RUN-TIME trap-discovery questions in the TRAP INVENTORY.

These were lost once to a budget-driven paragraph cut: the maxim
"A trap you never enumerated is a trap you are standing in" survived but the
two concrete self-check questions did not, leaving the agent told WHAT to
believe (traps exist) but not WHAT TO ASK after a green run. A maxim is not a
procedure; this test keeps the procedure.
"""
import importlib

import flagscale_agent.react.prompt as P


def _clause():
    importlib.reload(P)
    return P.SYSTEM_PROMPT_STATIC


def test_runtime_rung_present():
    assert "RUN-TIME:" in _clause(), "RUN-TIME rung of the trap inventory is missing"


def test_runtime_asks_which_result_contradicts_expectation():
    assert "which result contradicts my expectation" in _clause()


def test_runtime_asks_which_number_has_no_independent_source():
    assert "which number has no independent source" in _clause()


def test_runtime_is_anchored_to_the_green_run_not_to_design_time():
    c = _clause()
    i = c.find("RUN-TIME:")
    assert i > 0
    window = c[i:i + 400]
    assert "first green run" in window, (
        "RUN-TIME must say WHEN to ask (after the first green run); otherwise it "
        "collapses back into the design-time ENUMERATE rung"
    )


def test_the_maxim_alone_is_not_sufficient():
    """Both the maxim and the two questions must coexist: the maxim states the
    hazard, the questions are the entry point that makes it actionable."""
    c = _clause()
    assert "A trap you never enumerated is a trap you are standing in." in c
    assert "which number has no independent source" in c


def test_prompt_still_within_budget():
    importlib.reload(P)
    assert P.prompt_len() <= P.PROMPT_BUDGET
