"""Anti-decay tests for the collapsed-multi-position symbol clause (lever v9).

Guards two failure modes:
1. softening/removal of the clause in react/prompt.py;
2. task-literal leakage (bio-specific words that would turn a general clause
   into a leaked answer for the protein-assembly task).
"""
import pathlib
import re

PROMPT = pathlib.Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()

CLAUSE_ANCHOR = "COLLAPSED MULTI-POSITION FEATURE"
TASK_LITERAL_MARKERS = [
    "DYKDDDDK", "FLAG", "chromophore", "GYG", "MYG", "Clover", "mCherry",
    "7BG1", "8RMO", "rcsb", "fasta", "protein", "amino", "residue",
]


def test_clause_present():
    assert CLAUSE_ANCHOR in TEXT, "v9 clause missing from react/prompt.py"


def test_clause_in_ambiguous_symbol_region():
    # The clause must live adjacent to the existing AMBIGUOUS SYMBOL doctrine,
    # not appended at the tail where no consumer would read it.
    idx_anchor = TEXT.find("RESOLVING AN AMBIGUOUS SYMBOL")
    idx_clause = TEXT.find(CLAUSE_ANCHOR)
    assert idx_anchor != -1 and idx_clause != -1
    assert 0 < idx_clause - idx_anchor < 4000, "v9 clause drifted from its region"


def test_clause_core_rules():
    for required in [
        "GAP PROBE",
        "EXPANDS to those N canonical symbols",
        "length-preserving single-symbol substitution is an INVENTION",
        "consumer's own construction of the expected value",
    ]:
        assert required in TEXT, f"core rule missing: {required!r}"


def test_no_task_literal_leakage():
    window_start = TEXT.find(CLAUSE_ANCHOR)
    window_end = TEXT.find("\n- ", window_start + 1)
    window = TEXT[window_start: window_end if window_end != -1 else window_start + 3000]
    leaked = [m for m in TASK_LITERAL_MARKERS if m.lower() in window.lower()]
    assert not leaked, f"task literals leaked into v9 clause: {leaked}"
