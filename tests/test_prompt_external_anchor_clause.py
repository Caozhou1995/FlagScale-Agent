"""The prompt must require aligning to an EXTERNAL reference that already holds
the answer, not just proving the agent's own pipeline is self-consistent.

Regression anchor: a full-batch trial self-built an extraction pipeline, never
consulted an already-present reference answer, and finished far below the bar
while its own intermediate checks were internally consistent.

The clause is deliberately abstract: it must not name any benchmark's literal
values, file names, commands, or exit tokens.
"""

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC  # noqa: E402


def _flat() -> str:
    return re.sub(r"\s+", " ", SYSTEM_PROMPT_STATIC)


def test_reference_answer_clause_is_present():
    assert "(v) An external reference that already holds the answer" in SYSTEM_PROMPT_STATIC


def test_clause_distinguishes_self_consistency_from_matching():
    flat = _flat()
    assert "self-consistent has proven nothing about matching" in flat


def test_clause_requires_asking_before_building():
    flat = _flat()
    assert "Ask whether such an object exists before you build" in flat


def test_clause_requires_aligning_early_and_reporting_the_gap():
    flat = _flat()
    assert "align to it once early and report the disagreement" in flat


def test_situation_count_banner_matches_the_number_of_situations():
    """(i)..(v) exist, so the closing banner must not say 'all four'."""
    flat = _flat()
    assert "In all five the close is identical" in flat
    for marker in ("(i)", "(ii)", "(iii)", "(iv)", "(v)"):
        assert marker in flat, marker


def test_clause_carries_no_task_literal():
    """Task-agnostic gate: no benchmark constants, filenames, commands, tokens."""
    banned = [
        "extract-moves",
        "terminal-bench",
        "262",
        "11.29",
        "solve.sh",
        "humanity",
        ":wq",
        "ggtctc",
        "annealed",
        "mteb",
        "filter-js",
        "dna-",
    ]
    low = SYSTEM_PROMPT_STATIC.lower()
    for b in banned:
        assert b.lower() not in low, f"task literal leaked into prompt: {b}"


def test_prompt_stays_within_budget():
    # Bound to the ratchet symbol, not a copied literal: two copies of one
    # threshold drift (measured 2026-11-23: this literal still said 45000 after
    # the user raised PROMPT_BUDGET to 50000, reddening a green budget).
    from flagscale_agent.react.prompt import prompt_len, PROMPT_BUDGET
    assert prompt_len() <= PROMPT_BUDGET, f"{prompt_len()} chars vs budget {PROMPT_BUDGET}"
