"""Prompt clause: a deliverable that the grader RE-INVOKES must be re-run on a
fresh, self-constructed input before completion.

Motivation (measured):
  terminal-bench extract-elf keeps losing on a re-invoke contract.
  jobs/2026-10-21__08-10-31/extract-elf__22zNBMF/verifier/test-stdout.txt:136
    AssertionError: Only found 66.67% of expected values (required: 75%)
  The verifier (tests/test_outputs.py test_output_matches_reference) does NOT
  grade the file the agent produced. It writes a brand-new /app/new.c, compiles
  it with gcc into /app/new.o, then runs the agent's /app/extract.js on THAT
  binary and compares against its own REF script in the same run. The agent had
  validated its coverage on /app/a.out only (self-reported 650/698 = 93.12%)
  and its trace contains no attempt to construct an unseen ELF.

  The prompt already carries "Does the grader RUN my script itself, possibly
  after replacing the very inputs it consumes?" (prompt.py:93) and asks for a
  perturbed-input self-check. What is missing is the CONCRETE obligation to
  CONSTRUCT the fresh input and RUN the script on it, and to report the coverage
  number produced by that run rather than by the development sample.
"""
from pathlib import Path

PROMPT = (Path(__file__).resolve().parents[1]
          / "flagscale_agent" / "react" / "prompt.py").read_text()
MARKER = "CONSTRUCT A FRESH INPUT AND RE-RUN"


def _clause():
    start = PROMPT.index(MARKER)
    end = PROMPT.index("\n", start)
    return PROMPT[start:end]


def test_clause_exists():
    assert MARKER in PROMPT


def test_clause_requires_constructing_not_just_imagining():
    low = _clause().lower()
    assert "construct" in low
    assert "new input" in low or "fresh input" in low, (
        "must require building the fresh input, not merely imagining it"
    )


def test_clause_requires_running_the_artifact_on_it():
    low = _clause().lower()
    assert "run" in low, "must require re-running the deliverable on that input"


def test_clause_requires_reporting_that_runs_number():
    low = _clause().lower()
    assert "number" in low or "coverage" in low or "score" in low, (
        "must require quoting the metric produced by the fresh-input run, so a "
        "development-sample number cannot stand in for it"
    )


def test_clause_says_the_dev_sample_number_does_not_transfer():
    low = _clause().lower()
    assert "does not transfer" in low or "not transfer" in low or "no" in low and "evidence" in low, (
        "must say a number measured on the artifact's own development input is "
        "not evidence about the grader's input"
    )


def test_no_task_identifying_words_added():
    banned = ["mteb", "torchmetrics", "sentence-transformers", "primers.fasta",
              "oligotm", "cli_tool", "stolen_A1", "seed5", "anneal",
              "MNIST", "pytorch", "dna", "extract", "elf", "gcc", "new.c",
              "video", "doom", "mips", "git"]
    low = _clause().lower()
    for w in banned:
        assert w.lower() not in low, f"leak word {w!r} in fresh-input clause"
