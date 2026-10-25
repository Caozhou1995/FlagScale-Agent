"""v25 — verified conclusions are not fuel for later rederivation.

Regression guard for the clause in prompt.py (inserted before the
hidden-verification-set clause).

Motivating failures (batch 2026-10-25__12-16-04) — an agent read the correct
value from the source image early, later re-derived it in a deeper pass,
mis-read one character, and delivered the wrong variant; its own earlier
reading was correct and sat unused in its transcript:

  gcode-to-text__jg6xlf2 : verifier expects flag{gc0d3_iz_...}; agent
                           early lines read `gc0d3_iz` (correct), later
                           blocks re-derive `gcod3_iZ` (wrong), and the
                           delivered out.txt carried the LATE variant.
  regex-chess (multiple) : an early reasonable regex design is discarded
                           for a late redesign never validated against
                           the same checks; runs end on timeout.

One mechanism: a VERIFIED early conclusion is evidence; a later
derivation that conflicts with it must be re-verified by the same means
that settled the original before it may replace it. No existing clause
covers this — v22/v23 govern numeric bounds, v24 governs write-first;
none says "re-derive vs earlier-verified".

Gates:
  1. ANTI-DECAY: load-bearing sentences present verbatim.
  2. NO TASK LITERAL: no task names, no flag strings from the failing runs.
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()

HEADER = "VERIFIED CONCLUSIONS ARE NOT FUEL FOR LATER REDERIVATION"


def _clause() -> str:
    start = TEXT.find(HEADER)
    assert start != -1, "v25 clause header missing from prompt.py"
    rest = TEXT[start:]
    m = re.search(r"\n- ", rest)
    return rest[: m.start()] if m else rest


def test_clause_present():
    assert _clause().strip()


def test_requires_rereverify_not_plausibility():
    c = _clause().lower()
    assert "conflicts with an earlier verified one" in c
    assert "do not swap on plausibility" in c
    assert "re-run the verification that settled the original" in c


def test_requires_transcript_diff_before_final_write():
    c = _clause().lower()
    assert "diff it against the earliest verified value" in c
    assert "silent disagreement" in c


def test_default_when_reverify_impossible():
    c = _clause().lower()
    assert "the verified earlier conclusion wins" in c
    assert "stale-but-verified" in c
    assert "fresh-but-unverified" in c


def test_no_task_literals():
    c = _clause()
    for banned in (
        "gcode",
        "regex-chess",
        "gc0d3",
        "gcod3",
        "flag{",
        "re.json",
        "out.txt",
    ):
        assert banned not in c, f"task literal leaked into clause: {banned}"


def test_generalizes_beyond_source_domain():
    c = _clause().lower()
    assert "design" in c
    assert "mis-read" in c or "misread" in c
