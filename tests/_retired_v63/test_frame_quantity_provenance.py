"""The Frame acceptance must carry the measured-quantity PROVENANCE rule.

The static prompt clause (flagscale_agent/react/prompt.py) is advisory text an
agent may skim; the Frame step's acceptance is the ENFORCED surface -- it is
prepended to every plan and gated by the step_done evidence check. A discipline
that only lives in the prompt has no enforcement, so this test pins the same
rule into STATED_CONSTRAINT_ITEM and fails if a future edit drops it.
"""
import re

from flagscale_agent.react.oracle import STATED_CONSTRAINT_ITEM
from flagscale_agent.react.tools.plan_create import _FRAME_STEP_ACCEPTANCE


def _low() -> str:
    return " ".join(STATED_CONSTRAINT_ITEM.split()).lower()


def test_frame_constraint_item_names_quantity_provenance():
    low = _low()
    # the requirement itself
    assert "where the measured quantity's definition comes from" in low
    # the two admissible sources (mechanical construction OR the named tool)
    assert "mechanical order it states" in low and "tool it names" in low
    # the falsifier: both definitions computed and diffed, not argued
    assert "compute both and diff the numbers" in low
    # the tie-break direction: grader-side definition wins
    assert "grader-side definition wins" in low


def test_frame_step_acceptance_includes_the_item():
    joined = " ".join(_FRAME_STEP_ACCEPTANCE)
    assert STATED_CONSTRAINT_ITEM in joined


def test_no_task_specific_leak():
    low = _low()
    # the loss examples are genericised: no trial ids / grader internals
    for leak in ("dna-insert", "mteb-retrieve", "primers_concat", "humaneval"):
        assert leak not in low, f"Frame acceptance leaked task identifier: {leak}"
    assert not re.search(r"0x[0-9a-f]{4,}", low)
