"""Regression tests: the prompt's hedge doctrine must cover ambiguity in the
CONSTRAINT SATISFIED (what the deliverable must conform to), not only ambiguity
in the READING OF THE OUTPUT.

Motivation (measured): dna-assembly__cdmem3p produced a correct primer pair for
one definition of the annealing region, while the grader used another
(flag 66.06/60.61 delta 5.45 > 5, vs the agent's own self-check 60.9/60.6
delta 1.4). Adding a single clamp base costs nothing and would have satisfied
both definitions. Historical win rate for that task: 2/18.

The clause must be domain-neutral: no task names, no library names, no numeric
tolerance copied out of a grader (that would itself be leakage).

NOTE ON THE TESTS: they pin the CONCEPT by its discriminating phrase, not by a
token that already appears elsewhere in the prompt (an earlier draft asserted
bare "CONSTRAINT", which the cgroup paragraph at L271 already contains — a test
that passes before the clause exists proves nothing).
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
MARKER = "THE CONSTRAINT ITSELF MAY HAVE MORE THAN ONE READING"


def _text() -> str:
    return PROMPT.read_text()


def test_marker_present():
    assert MARKER in _text().upper(), (
        "the hedge doctrine must state that the requirement itself (not just "
        "the output) can have more than one defensible reading"
    )


def test_clause_requires_union_when_satisfying_both_is_free():
    text = _text()
    idx = text.upper().find(MARKER)
    assert idx != -1
    window = text[idx : idx + 1800].lower()
    assert "union" in window or "satisfy both" in window or "both readings" in window, (
        "clause must instruct satisfying BOTH readings rather than choosing one"
    )
    assert ("zero cost" in window or "costs nothing" in window or "adds nothing" in window
            or "free" in window), (
        "clause must tie the union to the case where the second reading costs nothing"
    )


def test_clause_makes_cost_the_discriminator_not_confidence():
    text = _text()
    idx = text.upper().find(MARKER)
    window = text[idx : idx + 1800]
    assert re.search(r"cost", window, re.I), (
        "the discriminator must be the COST of the extra satisfaction, not the "
        "agent's confidence in one reading"
    )


def test_clause_is_an_obligation():
    text = _text()
    idx = text.upper().find(MARKER)
    window = text[idx : idx + 1800]
    assert re.search(r"\b(must|never|always|take|prefer)\b", window, re.I), (
        "clause must be phrased as an obligation"
    )


def test_no_task_identifying_words():
    low = _text().lower()
    for leak in ("dna", "primer", "anneal", "oligotm", "clamp", "raman",
                 "terminal-bench", "harbor"):
        assert re.search(rf"\b{re.escape(leak)}\b", low) is None, f"leaks {leak}"
