"""v26 — run your own verifier before delivering.

Regression guard for the clause in prompt.py (inserted immediately before the
"VERIFIED CONCLUSIONS" clause).

Motivating failures (batch 2026-10-25__14-07-59):
  - torch-tensor-pipeline-parallelism: the agent WROTE a correctness test
    (_test_pp.py) itself, left it in the container, and delivered WITHOUT ever
    running it; a grouping-registration bug the test catches in seconds reached
    the verifier -> reward 0.
  - filter-js-from-html: the agent shipped a regex filter; the environment's
    own unchanged-input test (test_clean_html_unchanged roundtrip) failed on
    5 of 12 clean files -> reward 0. The winning trajectory of the same task
    used the parser-normalization path with a byte-identity hedge; the losing
    one never ran the shipped test that distinguishes the two.

Gates:
  1. ANTI-DECAY: load-bearing sentences present verbatim.
  2. NO TASK LITERAL: no task name, no file name, no path from the tasks.
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()

HEADER = "RUN YOUR OWN VERIFIER BEFORE DELIVERING"


def _clause() -> str:
    start = TEXT.find(HEADER)
    assert start != -1, "v26 clause header missing from prompt.py"
    rest = TEXT[start:]
    m = re.search(r"\n- ", rest)
    return rest[: m.start()] if m else rest


def test_clause_present():
    assert _clause().strip()


def test_states_inventory_and_run():
    c = _clause().lower()
    assert "inventory the runnable verifiers" in c
    assert "run each one against the artifact at the delivery path" in c


def test_states_read_fix_rerun():
    c = _clause().lower()
    assert "read the output" in c
    assert "re-run until it exits clean" in c
    assert "a failing self-test left unfixed is a chosen failure" in c


def test_states_subset_fallback_and_evidence_ranking():
    c = _clause().lower()
    assert "run the failing-prone subset" in c
    assert "the verifier's output is the evidence that outranks your reasoning" in c


def test_no_task_literal_leak():
    c = _clause().lower()
    forbidden = ["terminal-bench", "torch-tensor", "_test_pp", "filter-js",
                 "html", "write-compressor", "/app/"]
    for word in forbidden:
        assert word not in c, f"task literal leaked into v26 clause: {word!r}"
