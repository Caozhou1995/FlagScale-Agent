import re
import subprocess
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text(encoding="utf-8")


def test_alignment_anchor_clause_present_and_substantive():
    assert "THE SELF-CHECK'S ALIGNMENT ANCHOR IS THE CHECKER'S" in TEXT
    # Substantive: must name anchor object + task text designation + failing mode.
    for frag in (
        "align, overlap, or attach",
        "the task text names for it",
        "the EXACT object it designates as the anchor",
    ):
        assert frag in TEXT, f"missing substantive fragment: {frag}"


def test_no_task_literal_leakage_in_alignment_clause():
    start = TEXT.index("THE SELF-CHECK'S ALIGNMENT ANCHOR IS THE CHECKER'S")
    end = TEXT.index("Your own examples are not a held-out test set", start)
    clause = TEXT[start:end]
    banned = [
        "primer", "vector", "anneal", "BsaI", "plasmid", "protein", "sequence",
        "gBlock", "FLAG", "donor", "chromophore", "fasta", "raman", "spectral",
        "peak", "wavenumber", "html", "cells", "mask",
    ]
    hits = [b for b in banned if re.search(rf"\b{re.escape(b)}\b", clause, re.IGNORECASE)]
    assert not hits, f"task literal leakage in clause: {hits}"


def test_clause_sits_between_v11_and_examples_blocks():
    i11 = TEXT.index("CONSTRAINT ENUMERATION COMPLETES THE SELF-CHECK")
    i13 = TEXT.index("THE SELF-CHECK'S ALIGNMENT ANCHOR IS THE CHECKER'S")
    iex = TEXT.index("Your own examples are not a held-out test set")
    assert i11 < i13 < iex, "clause ordering drifted"


def test_mutation_softening_turns_red():
    soft = TEXT.replace(
        "your self-check must anchor the alignment on the SAME entity and edge the task text names for it",
        "your self-check should consider the task's alignment context",
    )
    assert soft != TEXT
    for frag in ("the SAME entity and edge the task text names for it",):
        assert frag not in soft
