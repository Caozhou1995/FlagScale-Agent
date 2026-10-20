"""The prompt must state that a check every candidate passes is not a check.

Evidence (m50's real pytorch trajectory, verbatim):
  "div255 gives 2, meanstd gives 2, raw gives 2. So the answer is likely 2.
   Good -- all conventions agree on 2 for the provided image. That's reassuring."
Three mutually exclusive preprocessing conventions agreeing on the ONE visible
sample was read as reassurance; it carried zero information, and the agent then
picked the convention by MNIST accuracy -- the wrong one.
No existing clause covered "an observation all candidates agree on discriminates
nothing". This test pins that clause.
"""
from pathlib import Path

PROMPT = (Path(__file__).resolve().parents[1]
          / "flagscale_agent" / "react" / "prompt.py").read_text()


def test_clause_present():
    assert "A test that every candidate passes is not a test." in PROMPT


def test_states_zero_discriminating_power():
    assert "zero discriminating power" in PROMPT
    assert "confirms none of them" in PROMPT


def test_requires_a_separating_observation():
    assert "Design the check so it can fail" in PROMPT
    assert "SEPARATES the candidates" in PROMPT


def test_forbids_reporting_agreement_as_evidence():
    assert "do not report the agreement as evidence" in PROMPT


def test_single_sample_is_not_a_held_out_set():
    assert "one example is not a held-out test set" in PROMPT


def test_no_task_identifying_words_added():
    """The clause must stay task-generic (no leakage)."""
    banned = ["mteb", "torchmetrics", "sentence-transformers", "primers.fasta",
              "oligotm", "cli_tool", "stolen_A1", "seed5", "anneal",
              "MNIST", "pytorch", "dna"]
    start = PROMPT.index("A test that every candidate passes")
    end = PROMPT.index("\n", PROMPT.index("unanimity about nothing", start))
    clause = PROMPT[start:end].lower()
    for w in banned:
        assert w.lower() not in clause, f"leak word {w!r} in new clause"
