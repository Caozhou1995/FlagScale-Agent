"""v22 — an artifact at the delivery path comes before any design work.

Regression guard for the clause in prompt.py (inserted immediately before the
size-ceiling clause).

Motivating failure (regex-chess, batch 2026-10-25__06-48-25, trial
regex-chess__9dVwPEc): the deliverable is a JSON rule file at a fixed path. The
trajectory contains 426 occurrences of "let me write" and NOT ONE write of the
delivery file; every turn refined the regex design instead. With ~39 minutes
left the agent was still reasoning ("Let me write these as rules that match the
marker and the castling field..."), and the verifier's first read was:

    FileNotFoundError: [Errno 2] No such file or directory: '/app/re.json'

Same shape in make-doom-for-mips: the frame artifact never appeared, so the
checker timed out waiting for it. The missing invariant is not "plan better" —
it is that the first write to the delivery path must happen before the design
is settled, and that a long run of planning turns with no mutation of that path
is itself the alarm.

Gates:
  1. ANTI-DECAY: load-bearing sentences present verbatim.
  2. NO TASK LITERAL: no task name, no file name, no path from the task.
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()

HEADER = ("AN ARTIFACT AT THE DELIVERY PATH COMES BEFORE ANY DESIGN WORK, "
          "AND IT IS REPLACED IN PLACE")


def _clause() -> str:
    start = TEXT.find(HEADER)
    assert start != -1, "v22 clause header missing from prompt.py"
    rest = TEXT[start:]
    m = re.search(r"\n- ", rest)
    return rest[: m.start()] if m else rest


def test_clause_present():
    assert _clause().strip()


def test_states_the_first_write_precedes_the_design():
    c = _clause().lower()
    assert "write a syntactically valid, minimally-semantic artifact there" in c
    assert "before any design work" in c
    assert "the first write to the delivery path happens before the design" in c


def test_records_the_planning_only_anti_pattern():
    c = _clause().lower()
    assert "no such file" in c
    assert "reasoning has no natural end" in c
    assert "consists only of planning with no mutation of the delivery path" in c


def test_prescribes_replace_in_place_and_reverify():
    c = _clause().lower()
    assert "in-place replacement" in c
    assert "re-open the delivered file" in c
    assert "build at the path" in c


def test_no_task_literal_leak():
    c = _clause().lower()
    forbidden = ["terminal-bench", "terminal_bench", "regex-chess", "regex_chess",
                 "re.json", "doom", "frame.bmp", "/app/"]
    for word in forbidden:
        assert word not in c, f"task literal leaked into v22 clause: {word!r}"
