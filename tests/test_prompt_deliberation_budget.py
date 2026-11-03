"""Regression tests: prompt must stay domain-neutral (no task-identifying words).

History (why this file is short):
  It originally guarded the clause "DELIBERATION IS BILLED BY THE TOKEN":
  say what you are doing in one brief line, then make the tool call; never
  restate the plan; output tokens are the direct cost of wall time.
  Motivation was jobs/2026-10-21__08-10-31 (Pearson r(total_chars,
  elapsed_sec) = 0.836; overtime trials THINK longer between the same number
  of tool calls instead of doing more work).

  The rebuild rebuilt flagscale_agent/react/prompt.py from scratch
  around the S1-S5 loop and the section 1-10 skeleton, and the
  deliberation-per-token capability did NOT survive it. Measured on the
  current bytes: 0 occurrences of "deliberation", "billed", "token",
  "concise", "terse", "preamble", "brief", "recap", "between tool" /
  "between calls". The single "one line" hit is §3 (record an out-of-bounds
  number in one line); the single "restates" hit is §3(c) (a checker that
  only restates its own assumptions). Neither is the retired clause.

  Per the standing ruling ("以现在代码为准, 修改单测" — adapt the tests, never
  the production code) the four frozen-sentence tests for that clause are
  RETIRED with a per-function comment rather than kept green by weakening
  them. If the capability is reintroduced, restore them against the new
  wording; a green assert that no longer means anything is worse than an
  honest retirement.

The surviving test is still a real assertion — it would go red the moment a
task name leaks into the prompt.
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


def _text() -> str:
    return PROMPT.read_text()


# RETIRED test_marker_present -- what: asserted the literal marker
# "DELIBERATION IS BILLED BY THE TOKEN" was in the prompt. why: the marker is
# gone (0 grep hits for "deliberation" and for "billed" in the current bytes).
#
# RETIRED test_clause_requires_action_first_between_calls -- what: asserted the
# prompt demands a one-line/single-line/brief statement before the tool call.
# why: the current prompt never asks for a pre-call statement; "one line"
# occurs once, in §3, about recording an out-of-bounds number.
#
# RETIRED test_clause_forbids_restatement -- what: asserted the prompt forbids
# restating/recapping what is already known. why: no such prohibition exists;
# "restates" occurs once, in §3(c), about a checker restating its own
# assumptions.
#
# RETIRED test_clause_ties_prose_to_the_time_budget -- what: asserted prose is
# tied to wall time. why: the current prompt ties *actions* to the clock
# (§7 "Price before committing: unit cost times units against the budget
# left."), not generation tokens. That surviving budget concept is covered by
# tests/test_prompt_background_budget.py::test_budget_concepts_survive.


def test_no_task_identifying_words():
    low = _text().lower()
    for leak in ("cobol", "codegolf", "regex-chess", "fibsqrt", "terminal-bench",
                 "harbor", "glm", "deepseek"):
        assert re.search(rf"\b{re.escape(leak)}\b", low) is None, f"leaks {leak}"
