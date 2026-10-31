"""Regression pin for the G3 sub-clause added on 2026-10-29.

Without this pin the clause can be deleted or softened with every test still green,
which is exactly the silent-removal gap already seen twice in this file's history.
"""
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
FLAT = " ".join(PROMPT.read_text(encoding="utf-8").split())


def test_g3_build_toward_the_bound_from_a_legal_version():
    """Accepting an out-of-bound value as a verdict is necessary but not sufficient.

    Measured loss: an agent measured its artifact over a stated ceiling, correctly
    refused to call the value acceptable, enumerated the cheapest reductions — and the
    wall arrived with the over-ceiling file still at the delivery path. Accepting the
    verdict did not help because the fix chain converged toward the bound from above,
    so the clock decided the landing point.
    """
    # the rule, as a greppable anchor (UPPERCASE title, matched verbatim)
    assert "BUILD TOWARD THE BOUND FROM A BOUND-LEGAL VERSION" in FLAT
    # the direction of construction is the whole content: up from a legal minimum
    assert "NOT DOWN FROM AN OVER-BOUND ONE" in FLAT
    # and the failure shape it closes must be stated, or the clause is advice
    assert "converges toward the bound from above is the wrong shape" in FLAT
    # it must be typed as a SUFFICIENCY gap, not a repeat of the adjudication rule
    assert "necessary but not sufficient" in FLAT
