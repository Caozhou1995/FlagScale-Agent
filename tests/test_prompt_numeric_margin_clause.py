"""v23 — a numeric bound is met with a MEASURED MARGIN in the verifier's unit.

Regression guard for the clause in prompt.py (inserted before the size-ceiling
clause).

Motivating failures (batch 2026-10-25__06-48-25) — three independent runs all
landed a hair on the wrong side of a stated bound:

  dna-assembly__dKfiFAx  : assert 72.388098 <= 72
                           "Forward primer must have a Tm between 58 and 72"
  write-compressor__VTnidB5: assert 2501 <= 2500
                           "Compressed file size (2501 bytes) exceeds maximum"
  extract-elf__3zsc4VG   : assert 66.66666666666666 >= 75
                           "Only found 66.67% of expected values (required 75%)"

Different domains, one mechanism: the agent treated the bound as a target to
satisfy exactly rather than a limit to stay clear of, and it measured clear of
it in its OWN formula/unit rather than in the checker's. v20 covers only the
artifact's own size ceiling and only says "do not exceed"; it does not say
"leave headroom" nor "measure the margin in the quantity the checker compares".

Gates:
  1. ANTI-DECAY: load-bearing sentences present verbatim.
  2. NO TASK LITERAL: no task names, no numbers from the failing runs.
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()

HEADER = "A NUMERIC BOUND IS MET WITH A MEASURED MARGIN IN THE VERIFIER'S OWN UNIT"


def _clause() -> str:
    start = TEXT.find(HEADER)
    assert start != -1, "v23 clause header missing from prompt.py"
    rest = TEXT[start:]
    m = re.search(r"\n- ", rest)
    return rest[: m.start()] if m else rest


def test_clause_present():
    assert _clause().strip()


def test_forbids_targeting_the_boundary():
    c = _clause().lower()
    assert "do not target the boundary value" in c
    assert "landing exactly on the edge" in c
    assert "fails by a hair" in c


def test_requires_headroom_and_printed_delta():
    c = _clause().lower()
    assert "target the safe side with an explicit headroom" in c
    assert "print \"measured value vs bound, delta\"" in c
    assert "near-miss is visible while it is still fixable" in c


def test_requires_margin_in_the_checkers_unit():
    c = _clause().lower()
    assert "margin in the wrong unit" in c
    assert "measure the margin by reconstructing the checker's quantity" in c
    assert "compute both your value and that reconstructed value" in c


def test_no_task_literal_leak():
    c = _clause().lower()
    forbidden = ["terminal-bench", "terminal_bench", "dna-assembly", "dna_assembly",
                 "write-compressor", "write_compressor", "extract-elf", "extract_elf",
                 "primer", "tm ", "72.388", "2501", "2500", "66.67", "75%"]
    for word in forbidden:
        assert word not in c, f"task literal leaked into v23 clause: {word!r}"
