"""Regression tests: the prompt must require a single best-guess value for a
machine-checked field, instead of letting the agent write its uncertainty into
the field itself.

Motivation (measured, not invented): gcode-to-text__YJgPBjm produced a 10k-line
trajectory with 49 tool calls, correctly identified that the exact text could
not be fully recovered, and then wrote its uncertainty INTO the delivery.
The grader reads a single value from that field, so the delivery scored 0
despite the analysis being right. Historical win rate for that task across all
batches: 0/26.

The clause must be domain-neutral: no task names, no library names.
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


def _prompt_text() -> str:
    return PROMPT.read_text()


def test_clause_present():
    text = _prompt_text()
    assert "BEST GUESS" in text.upper(), (
        "prompt must state that a machine-checked single-valued field gets a "
        "single best-guess value"
    )


def test_clause_separates_value_from_uncertainty():
    text = _prompt_text().lower()
    # The point is not "be confident" but "put the uncertainty OUTSIDE the field".
    assert "outside" in text or "alongside" in text or "beside" in text, (
        "clause must say where the uncertainty goes instead (outside the "
        "checked field)"
    )


def test_clause_requires_parseability():
    text = _prompt_text().lower()
    assert "parse" in text or "parsable" in text or "well-formed" in text, (
        "clause must state the acceptance property: the field must remain "
        "parseable as the required type/format"
    )


def test_clause_is_a_requirement_not_a_suggestion():
    text = _prompt_text()
    idx = text.upper().find("BEST GUESS")
    assert idx != -1
    window = text[idx : idx + 1400]
    # must be phrased as an obligation
    assert re.search(r"\b(must|always|never|has to|is required)\b", window, re.I), (
        "the clause must be phrased as an obligation, not an option"
    )


def test_no_task_identifying_words_added():
    text = _prompt_text().lower()
    # the clause must generalise; naming the motivating tasks would leak
    for leak in ("gcode", "fibonacci", "polyglot", "doom", "mips", "elf",
                 "terminal-bench", "harbor"):
        assert re.search(rf"\b{re.escape(leak)}\b", text) is None, (
            f"prompt leaks task-identifying word: {leak}"
        )
