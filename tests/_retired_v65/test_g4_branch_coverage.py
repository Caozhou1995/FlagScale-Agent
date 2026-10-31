import re, subprocess, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent / "flagscale_agent" / "react" / "prompt.py"
SRC = SRC.read_text()

def test_g4_has_branch_coverage_clause():
    assert "COVER THE IMPLEMENTATION'S OWN BRANCHES" in SRC

def test_g4_branch_anchor_lines():
    for anchor in ["fallback branch, an error path",
                   "EACH branch construct one input that provably reaches",
                   "A byte-level design claim"]:
        assert anchor in SRC, f"missing: {anchor}"

def test_g4_branch_is_inside_G4_section():
    i_g4 = SRC.find("G4 COVER EVERY OUTPUT CLASS")
    i_g5 = SRC.find("G5 VALIDATE THE INSTRUMENT")
    i_clause = SRC.find("COVER THE IMPLEMENTATION'S OWN BRANCHES")
    assert i_g4 < i_clause < i_g5

def test_g2_library_observed_by_running():
    assert "A NAMED LIBRARY IS OBSERVED BY RUNNING IT, NOT BY READING IT" in SRC

def test_g2_library_clause_inside_G2():
    i_g2 = SRC.find("G2 REPLAY THE CONSUMER'S CONSTRUCTION")
    i_next = SRC.find("WHEN THE MEASURED SPANS A COMPOSED OBJECT")
    i_clause = SRC.find("A NAMED LIBRARY IS OBSERVED")
    assert i_g2 < i_clause < i_next
