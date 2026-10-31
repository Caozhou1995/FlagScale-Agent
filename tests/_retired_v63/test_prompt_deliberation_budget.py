"""Regression tests: the prompt must impose a hard budget on deliberation prose
between tool calls.

Motivation (measured, jobs/2026-10-21__08-10-31, 48 winning trials):
  - Pearson r(total_chars, elapsed_sec) = 0.836 -- generation volume explains
    ~70% of the time variance.
  - Non-tool (reasoning) lines are 92-99% of every trajectory.
  - Tool-call COUNT is nearly identical between trials that fit their budget
    (mean 56.2) and trials that blew it (mean 54.0) -- overtime trials do not
    do more work, they THINK longer between the same number of calls.
  - 3 AgentTimeoutError trials sat at rank 13/18/16 of 65 by chars (not
    extremes), while the 1.86M-char regex-chess trial WON -- so raw size alone
    is not fatal, but per-call deliberation is what burns the clock.

The clause must be domain-neutral (no task names, no harness internals).
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
MARKER = "DELIBERATION IS BILLED BY THE TOKEN"


def _text() -> str:
    return PROMPT.read_text()


def test_marker_present():
    assert MARKER in _text().upper()


def test_clause_requires_action_first_between_calls():
    text = _text()
    idx = text.upper().find(MARKER)
    window = text[idx : idx + 2000].lower()
    # the mechanism: say what you are doing in one line, then CALL
    assert ("one line" in window or "one sentence" in window
            or "single line" in window or "brief" in window), (
        "clause must demand the pre-call statement be one line / brief"
    )
    assert ("call" in window or "invoke" in window or "run the tool" in window), (
        "clause must tie brevity to making the tool call immediately"
    )


def test_clause_forbids_restatement():
    text = _text()
    idx = text.upper().find(MARKER)
    window = text[idx : idx + 2000].lower()
    assert ("restating" in window or "restate" in window or "re-summariz" in window
            or "repeat" in window or "recap" in window), (
        "clause must forbid restating the plan / re-deriving what is already known"
    )


def test_clause_ties_prose_to_the_time_budget():
    """The WHY must be stated: output tokens are the direct cost of wall time."""
    text = _text()
    idx = text.upper().find(MARKER)
    window = text[idx : idx + 2000].lower()
    assert ("wall" in window or "budget" in window or "clock" in window
            or "time" in window), (
        "clause must connect deliberation prose to the time budget"
    )


def test_no_task_identifying_words():
    low = _text().lower()
    for leak in ("cobol", "codegolf", "regex-chess", "fibsqrt", "terminal-bench",
                 "harbor", "glm", "deepseek"):
        assert re.search(rf"\b{re.escape(leak)}\b", low) is None, f"leaks {leak}"
