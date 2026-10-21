"""Prompt clause: the "unchanged" arm of a transform needs a corpus sized to the
requirement set, not one convenient sample.

Motivation (measured):
  terminal-bench filter-js-from-html has lost 58 of 66 recorded attempts. The
  2026-10-21 trial __Qrgow5F failed BOTH tests; the decisive failure is the
  preservation arm (verifier/test-stdout.txt:782):
    AssertionError: Filter modified 5 clean HTML files out of 12:
      Test 4 / Test 6 / Test 9 ...
  Dumped diff shows the delivered filter re-serialised clean inputs:
    attribute order  <input id="name" name="name" type="text"/>  ->
                     <input type="text" id="name" name="name">
    void elements    <br/>  ->  <br>
    entities         (c)    ->  &copy;
  The agent's own preservation check used ONE hand-written benign file; the
  grader uses 12. The prompt already forbids hand-rolled lexical transforms
  (prompt.py:109 canonical round-trip) and asks for a deliberately broad corpus
  (prompt.py:110), but nothing ties the SIZE of the preservation corpus to the
  number of clauses the requirement states -- so "1 sample" reads as compliance.
"""
from pathlib import Path

PROMPT = (Path(__file__).resolve().parents[1]
          / "flagscale_agent" / "react" / "prompt.py").read_text()
MARKER = "Size the preservation corpus to the requirement"


def _clause():
    start = PROMPT.index(MARKER)
    end = PROMPT.index("\n", start)
    return PROMPT[start:end]


def test_clause_exists():
    assert MARKER in PROMPT


def test_clause_requires_more_than_one_sample():
    low = _clause().lower()
    assert "one sample" in low or "single sample" in low or "more than one" in low, (
        "must name the single-sample trap explicitly"
    )


def test_clause_counts_clauses_as_the_unit():
    low = _clause().lower()
    assert "clause" in low or "requirement" in low, (
        "must make the stated requirement the unit the corpus is counted against"
    )


def test_clause_demands_one_case_per_clause():
    low = _clause().lower()
    assert "at least one case" in low or "one case per" in low, (
        "must require at least one preservation case per stated clause"
    )


def test_no_task_identifying_words_added():
    banned = ["mteb", "torchmetrics", "sentence-transformers", "primers.fasta",
              "oligotm", "cli_tool", "stolen_A1", "seed5", "anneal",
              "MNIST", "pytorch", "dna", "filter", "html", "xss", "beautifulsoup",
              "bs4", "video", "doom", "mips", "git", "elf"]
    low = _clause().lower()
    import re as _re
    for w in banned:
        # word-boundary match: 'elf' must not fire inside 'self'
        assert not _re.search(r"\b" + _re.escape(w.lower()) + r"\b", low), (
            f"leak word {w!r} in corpus-size clause"
        )
