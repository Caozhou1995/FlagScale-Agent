"""v21 (anti-decay) — a hard size/count ceiling is enforced at every write.

Regression guard for the clause in prompt.py (inserted just before the
"WHEN THE TASK SAYS REFINE" clause).

Motivating failure (gpt2-codegolf, batch 2026-10-25__04-27-43, trial
gpt2-codegolf__6eQEPiR): the task caps the deliverable at <5000 bytes. The agent
read the cap correctly, measured repeatedly (31 mentions of "5000"), and every
time concluded "need to shrink later — first make it work":

  L2852: "Also size is 6143 > 5000 — need to shrink. But first compile and see
          if it works at all."
  L2944: "Also size 6143 > 5000 — must shrink. Time is nearly out (~1m)."
  L2999: "Compiles. Now run it. Size 6229 > 5000 — need to shrink later."

The verifier then failed on exactly that: "AssertionError: File /app/gpt2.c is
larger than 5000 bytes". Enumerating and measuring the bound was not the
missing piece (v11 already covers constraint enumeration) — the missing piece
is that the bound must gate every growth step, because striking the difference
later requires a structural redesign the agent no longer has time for.

Gates:
  1. ANTI-DECAY: load-bearing sentences present verbatim.
  2. NO TASK LITERAL: no task name, no 5000, no file name from the task.
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()


def _clause() -> str:
    start = TEXT.find("A HARD BOUND ON THE ARTIFACT'S OWN SIZE OR COUNT IS ENFORCED AT EVERY WRITE")
    assert start != -1, "v21 clause header missing from prompt.py"
    rest = TEXT[start:]
    m = re.search(r"\n- ", rest)
    return rest[: m.start()] if m else rest


def test_clause_present():
    assert _clause().strip()


def test_names_bound_as_build_constraint_not_cleanup():
    c = _clause().lower()
    assert "enforced at every write, not compressed at the end" in c
    assert "build constraint of the same rank as correctness" in c
    assert "not a cleanup task you schedule after the logic works" in c


def test_records_the_deferral_anti_pattern():
    c = _clause().lower()
    assert "need to shrink later" in c
    assert "shrinking is never free once the code exists" in c
    assert "which is a design decision" in c


def test_prescribes_measure_and_cut():
    c = _clause().lower()
    assert "measure it against the bound and record the number" in c
    assert "the next action is a cut, not a feature" in c
    assert "decide the size strategy before writing" in c
    # shipping decision at the deadline
    assert "ship the working under-limit version" in c


def test_no_task_literal_leak():
    c = _clause().lower()
    forbidden = [
        "terminal-bench",
        "terminal_bench",
        "gpt2",
        "codegolf",
        "gpt2.c",
        "5000",
        "6143",
        "6229",
        "6737",
    ]
    for word in forbidden:
        assert word not in c, f"task literal leaked into v21 clause: {word!r}"
