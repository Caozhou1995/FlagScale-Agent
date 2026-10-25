"""v24 — the verification set is hidden and larger; generate it, do not imagine it.

Regression guard for the clause in prompt.py (inserted before the size-ceiling
clause).

Motivating failures (batch 2026-10-25__06-48-25) — three runs passed their own
samples and failed the hidden set:

  filter-js-from-html__xAEGkJC : "Failed vectors (first 3): [...<iframe/ "
                                 "onload=alert(89)>...</iframe>...]"
  extract-elf__3zsc4VG         : "Only found 66.67% of expected values
                                  (required: 75%)"
  schemelike-metacircular-eval : "Summary: 56 passed, 7 failed out of 63 tests"

The agent enumerated inputs it could think of and treated that set as the whole
gate. v11 covers the constraints the task TEXT states; nothing covered the test
SET the task does not show you — which is generated from the spec and includes
the malformed-edge form a hand-written example omits.

Gates:
  1. ANTI-DECAY: load-bearing sentences present verbatim.
  2. NO TASK LITERAL: no task names, no payloads from the failing runs.
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()

HEADER = "THE VERIFICATION SET IS HIDDEN AND LARGER THAN ANY SAMPLE SET YOU CAN IMAGINE"


def _clause() -> str:
    start = TEXT.find(HEADER)
    assert start != -1, "v24 clause header missing from prompt.py"
    rest = TEXT[start:]
    m = re.search(r"\n- ", rest)
    return rest[: m.start()] if m else rest


def test_clause_present():
    assert _clause().strip()


def test_states_sample_set_is_not_the_gate():
    c = _clause().lower()
    assert "passing the examples you invented is not evidence" in c
    assert "the set you must satisfy is a function of the spec" in c
    assert "generate the set, do not enumerate what you think of" in c


def test_prescribes_spec_derivation_and_adversarial_variants():
    c = _clause().lower()
    assert "at least one case per stated clause" in c
    assert "construct its adversarial variants" in c
    assert "unusual delimiter sequences" in c
    assert "report the generated case count and the rejected count" in c


def test_scopes_itself_against_constraint_enumeration():
    c = _clause().lower()
    assert "it does not replace enumerating the constraints the task does state" in c


def test_no_task_literal_leak():
    c = _clause().lower()
    forbidden = ["terminal-bench", "terminal_bench", "filter-js", "filter_js",
                 "extract-elf", "extract_elf", "schemelike", "scheme",
                 "iframe", "onload", "xss", "alert(", "63 tests", "56 passed"]
    for word in forbidden:
        assert word not in c, f"task literal leaked into v24 clause: {word!r}"
