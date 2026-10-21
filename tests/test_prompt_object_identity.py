"""Prompt clause: an object's IDENTITY is part of the delivery contract.

Motivation (measured, 2026-10-21 batch 08-10-31):
  terminal-bench sanitize-git-repo trial __tgTo3tJ reward=0. The verifier's
  three checks split 2 passed / 1 failed:
    PASSED test_removal_of_secret_information
    PASSED test_correct_replacement_of_secret_information
    FAILED test_no_other_files_changed
      -> ValueError: SHA b'd6987af002b122fef54bc0be402062c76488a4d9'
         could not be resolved, git returned: b'd6987af0... missing'
  Every CONTENT requirement was met. The loss was purely that a consumer
  attempted to re-locate a commit by its ORIGINAL hash, and the agent had
  rewritten history so that hash no longer exists.

  The agent was not careless: VcsBackupGuard fired and blocked its
  `git filter-branch` probe (agent/flagscale-agent.txt L1017), and the agent
  then overrode it with "I have a full backup at /tmp/dclm_backup_orig"
  (L1040) and reasoned explicitly that rewriting history is the stronger
  reading of "ensure that the sensitive values are not present in the
  repository" (L1061-1079). So "recoverable from a backup" was treated as
  sufficient, and "still addressable by its old identifier" was never
  considered. The prompt's destructive-action clause asks whether an action
  WIDENS the task; it never asks who else addresses the object by name.
"""
from pathlib import Path

PROMPT = (Path(__file__).resolve().parents[1]
          / "flagscale_agent" / "react" / "prompt.py").read_text()


def _clause():
    start = PROMPT.index("OBJECT IDENTITY IS PART OF THE CONTRACT")
    end = PROMPT.index("\n", start)
    return PROMPT[start:end]


def test_clause_exists():
    assert "OBJECT IDENTITY IS PART OF THE CONTRACT" in PROMPT


def test_clause_names_the_identifier_classes():
    low = _clause().lower()
    for token in ("hash", "path", "uri", "key"):
        assert token in low, f"must name {token!r} as an identifier a consumer may use"


def test_clause_requires_asking_who_addresses_by_old_name():
    low = _clause().lower()
    assert "old" in low and ("address" in low or "references" in low or "refer" in low), (
        "must require asking who still addresses the object by its OLD identifier"
    )


def test_clause_separates_recoverable_from_addressable():
    """The exact confusion that lost this trial: a backup makes it RECOVERABLE,
    not ADDRESSABLE by the original hash."""
    low = _clause().lower()
    assert "resolvable" in low or "addressable" in low, (
        "must say the old identity has to stay resolvable/addressable"
    )
    assert "content" in low, (
        "must say content-correctness is only half the requirement, so the reader "
        "cannot conclude that correct bytes imply a complete delivery"
    )


def test_no_task_identifying_words_added():
    banned = ["mteb", "torchmetrics", "sentence-transformers", "primers.fasta",
              "oligotm", "cli_tool", "stolen_A1", "seed5", "anneal",
              "MNIST", "pytorch", "dna", "git", "commit", "reflog", "dclm",
              "filter-repo", "filter-branch", "video", "doom", "mips"]
    low = _clause().lower()
    for w in banned:
        assert w.lower() not in low, f"leak word {w!r} in identity clause"
