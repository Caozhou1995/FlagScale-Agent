"""A GUESS at the consumer's check must never drive REMOVING what the authority requires."""
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
FLAT = " ".join(SRC.read_text(encoding="utf-8").lower().split())


def test_guess_never_justifies_removal_clause_is_present():
    assert "a guess at how a consumer might check never justifies removing what the authority requires" in FLAT


def test_only_a_stated_conflict_permits_removal():
    assert "only a stated constraint that keeping it violates can" in FLAT


def test_clause_is_adjacent_to_the_no_grader_guessing_rule():
    i = FLAT.find("do not close a choice by guessing the consumer")
    j = FLAT.find("a guess at how a consumer might check")
    assert i != -1 and j != -1 and 0 < j - i < 700


def test_stricter_reading_rule_survives():
    # positive control: the pre-existing strictest-standard rule must still be there
    assert "a remembered convention is not defensible and never outranks a sourced reading" in FLAT


def test_no_benchmark_literals_leaked():
    for tok in ("ggtctc", "bsai", "aatt", "primers.fasta", "oligotm"):
        assert tok not in FLAT
