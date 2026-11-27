"""Pin the clause that a SELF-AUTHORED prohibition is subordinate to the authority.

Why this file exists (2026-11-27, dna-assembly 9L8TfcV):
the agent wrote its own plan forbidden entry #1 as

    "Do not add extra bases 5' of the BsaI site ... WITHOUT VERIFYING a tail
     parser strips it — instead put GGTCTC/GAGACC at a fixed known offset"

and then twice cited it (log :1329, :1913) to REFUSE the NEB requirement.
It had converted its own unverified guess ("maybe the parser does not strip")
into a prohibition, and the self-authored list silently outranked the task's
named authority. Prompt E9 then ended after "or ship the superset", saying
nothing about who outranks whom when the two collide.

These tests fail if that sentence is removed or weakened, so the clause
cannot quietly disappear in a later prompt rewrite (which is exactly what
happened to the v21 hard-ceiling clause in a4b3cc6).
"""
import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from flagscale_agent.react.prompt import PROMPT_BUDGET, prompt_len  # noqa: E402

PROMPT_PY = os.path.join(REPO, "flagscale_agent", "react", "prompt.py")


def _flat():
    src = open(PROMPT_PY, encoding="utf-8").read()
    return re.sub(r"\s+", " ", src.replace('"', " ").replace("\\n", " "))


def _e9_body():
    """E9's own text, cut at its terminator so we test THIS clause only."""
    return _flat().split("E9 AN UNREAD STANDARD IS NOT A PROBABILITY.", 1)[1].split("Each rank exists")[0]


def test_clause_is_present_in_e9():
    body = _e9_body()
    assert "A self-authored prohibition is subordinate to the task text and its authorities" in body, (
        "E9 lost the self-authored-prohibition clause; the dna failure shape returns"
    )


def test_clause_states_own_guess_forbids_nothing():
    """The load-bearing half: YOUR uncertainty must not become a prohibition."""
    body = _e9_body()
    assert "your own unverified guess forbids nothing" in body
    assert "it asks for a LABEL" in body, (
        "the remedy (label, do not forbid) is what changes behaviour; without it "
        "the sentence only says 'be subordinate' with no action"
    )


def test_clause_names_the_authority_order_not_a_virtue():
    """A virtue ('respect the task') would never have stopped the agent."""
    body = _e9_body()
    assert "subordinate to the task text and its authorities" in body


def test_clause_sits_inside_e9_after_the_superset_rule():
    """ORDER matters: it must follow 'ship the superset', not precede it.

    Read forwards: the section first says an unread standard's missing element
    SHIPS, then says a self-authored ban cannot veto it. Reversed, the ban
    would be read first and colour the whole paragraph.
    """
    body = _e9_body()
    i_superset = body.find("or ship the superset")
    i_clause = body.find("A self-authored prohibition is subordinate")
    assert i_superset != -1, "E9 lost its superset rule"
    assert i_clause != -1, "clause missing"
    assert i_superset < i_clause, "clause was placed BEFORE the superset rule"


def test_rule_is_task_agnostic():
    """No task-specific literal may leak into the general prompt."""
    body = _e9_body()
    for bad in (
        "BsaI", "GGTCTC", "oligotm", "primer", "clamp", "gpt2", "gcode",
        "dna", "flag{", "codegolf", "relu",
    ):
        assert bad.lower() not in body.lower(), f"task literal leaked into E9: {bad!r}"


def test_budget_holds():
    assert prompt_len() <= PROMPT_BUDGET, f"{prompt_len()} > {PROMPT_BUDGET}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
