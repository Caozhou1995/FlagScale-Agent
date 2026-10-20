"""The external-verdict clause must exist, stay generic, and name its consequence."""

import re
from pathlib import Path

import pytest

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text(encoding="utf-8")

ANCHOR = '"Am I the one who decides that my own delivery is correct?"'


def _clause() -> str:
    i = TEXT.index(ANCHOR)
    end = TEXT.index("- \"Have I replayed the derivation", i)
    return TEXT[i:end]


def test_clause_present():
    assert ANCHOR in TEXT


def test_clause_demands_a_printed_external_verdict_before_completion():
    c = _clause()
    assert "BEFORE you emit the completion signal" in c
    assert "paste its raw verdict line, verbatim" in c
    assert "unverified claim" in c


def test_clause_forbids_self_check_as_substitute():
    c = _clause()
    assert "self-check shares the premise you already hold" in c


def test_clause_is_generic_no_task_identifiers():
    c = _clause()
    for word in (
        "primers.fasta",
        "oligotm",
        "anneal",
        "stolen_A1",
        "weights.json",
        "cli_tool",
        "predictions.txt",
        "mteb",
        "torchmetrics",
        "mnist",
        "image.png",
        "sequences.fasta",
        "data.txt",
        "result.txt",
    ):
        assert word not in c.lower(), f"task-specific token leaked into clause: {word}"


def test_clause_permits_a_named_non_applicability_statement():
    c = _clause()
    assert "which ones you tried and why neither applies" in c


def test_clause_is_removable_and_its_removal_is_detectable():
    mutant = TEXT.replace(_clause(), "- \"placeholder\"\n")
    assert ANCHOR not in mutant
    assert mutant != TEXT
