"""Prompt clause: a perturbation corpus must sweep each dimension in BOTH directions.

Motivation (measured, 2026-10-21 batch 08-10-31):
  terminal-bench video-processing trial __MRdBFoo reward=0. The verifier ran the
  agent's script on tests/test_video.mp4 (964534 B) and got returncode -9
  (SIGKILL); the same script passed on /app/example_video.mp4 (551877 B, ~1.75x
  smaller). The agent HAD built "perturbed worlds" and self-reported them as a
  generalization check (agent/flagscale-agent.txt L13448): it trimmed 10 frames,
  brightened +18, darkened -15 -- every perturbation at the SAME length or
  SHORTER. The growth side of the scale dimension was never exercised, and that
  is the only side where the task's own resource envelope (task.toml:
  cpus=1, memory_mb=2048) can kill the process.

The existing prompt already asks for a "deliberately BROAD corpus (perturb
incidental properties, add boundary/adversarial cases, generate from the spec)"
and tells the agent its own examples are not a held-out test set. Neither
constrains the DIRECTION of a perturbation, so a one-sided corpus reads as
compliance.
"""
from pathlib import Path

PROMPT = (Path(__file__).resolve().parents[1]
          / "flagscale_agent" / "react" / "prompt.py").read_text()


def _broad_corpus_clause():
    start = PROMPT.index("Your own examples are not a held-out test set")
    end = PROMPT.index("\n", start)
    return PROMPT[start:end]


def test_clause_requires_both_directions():
    clause = _broad_corpus_clause()
    low = clause.lower()
    assert "both directions" in low, (
        "the corpus clause must require sweeping each dimension in BOTH directions"
    )


def test_clause_names_the_growth_side_as_the_blind_spot():
    clause = _broad_corpus_clause().lower()
    assert "larger" in clause, "must name the growth side explicitly (larger/longer/more)"
    assert "shrink" in clause or "shorter" in clause, (
        "must name the shrinking side as the one that is obvious-but-insufficient"
    )


def test_clause_requires_respecting_the_resource_envelope():
    clause = _broad_corpus_clause().lower()
    assert "memory" in clause or "resource" in clause, (
        "must require re-running under the task's own resource envelope, because "
        "the growth side is where an out-of-memory kill happens"
    )


def test_clause_mentions_the_kill_class_so_the_failure_is_recognisable():
    clause = _broad_corpus_clause().lower()
    assert "kill" in clause or "out-of-memory" in clause or "out of memory" in clause, (
        "a growth-only failure surfaces as a kill, not a wrong answer; the clause "
        "must say so or the agent cannot recognise its own blind spot"
    )


def test_no_task_identifying_words_added():
    """The clause must stay task-generic (no leakage of any benchmark's nouns)."""
    banned = ["mteb", "torchmetrics", "sentence-transformers", "primers.fasta",
              "oligotm", "cli_tool", "stolen_A1", "seed5", "anneal",
              "MNIST", "pytorch", "dna", "video", "hurdle", "jump_analyzer",
              "doom", "mips", "opencv", "cv2", "mp4"]
    clause = _broad_corpus_clause().lower()
    for w in banned:
        assert w.lower() not in clause, f"leak word {w!r} in corpus clause"
