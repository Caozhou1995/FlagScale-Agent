"""Regression tests: G1c (WAITING IS NOT WORK) must stay pinned in the prompt.

Motivation (measured, v61 full batch 2026-10-31__02-33-26, settled=49):
  - poll% (wait-class tool calls / total calls) mean: winners 8.4% vs losers
    16.0%; poll%>20 bucket went 5W/4L in the same batch.
  - rstan-to-pystan (v61 loss): 68 tool calls, 23 were sleep/shell_jobs-wait
    polls (34%), 3 guard overrides burned on wait-sleeps.
  - The clause exists in prompt.py (L278+, md5 389c4f32) but NO test anchored
    it -- the same unanchored-clause shape that let the distillation run
    (9632fcf6) silently delete a discipline rule.

The clause must stay domain-neutral (no task names, no harness internals).
"""

import re
from pathlib import Path

PROMPT = (
    Path(__file__).resolve().parents[1]
    / "flagscale_agent"
    / "react"
    / "prompt.py"
)
MARKER = "WAITING IS NOT WORK"


def _text() -> str:
    return PROMPT.read_text()


def test_marker_present():
    assert MARKER in _text().upper()


def test_two_rules_present():
    text = _text()
    # rule 1: bank a crude deliverable BEFORE waiting
    assert "at the required path FIRST" in text
    # rule 2: bound the wait + every poll productive
    assert "BOUND the wait" in text
    assert "never idle in a re-check loop" in text


def test_no_task_names_in_clause():
    # extract the G1c block (from its marker to the next top-level clause)
    text = _text()
    start = text.find(MARKER)
    assert start > 0, "marker missing"
    block = text[start : start + 2500]
    banned = [
        "filter-js", "ocaml", "mteb", "rstan", "pytorch", "tbench",
        "harbor", "v61", "reward", "verifier", "subset3",
    ]
    hits = [b for b in banned if b.lower() in block.lower()]
    assert hits == [], f"task/harness identifiers leaked into G1c clause: {hits}"
