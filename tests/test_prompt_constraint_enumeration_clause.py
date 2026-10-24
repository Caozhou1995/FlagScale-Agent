"""Anti-erosion tests for the CONSTRAINT ENUMERATION clause (lever v11).

The clause was added after dna-assembly failed twice with the same signature:
the agent's functional oracle passed (ASSEMBLY_MATCH True, replay exit 0) but
the hidden grader's explicit structural constraint (clamp >= 1 nt before the
BsaI site) was never in the self-check list. The clause must survive future
edits verbatim and must never name task-specific details (no 'clamp', no
'BsaI', no 'primer').
"""
import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


def _text():
    return PROMPT.read_text(encoding="utf-8")


def test_clause_present_with_core_wording():
    t = _text()
    assert "CONSTRAINT ENUMERATION COMPLETES THE SELF-CHECK" in t
    # the load-bearing mechanism sentences must stay
    assert "ONE executable assertion per constraint" in t
    assert "a functional oracle" in t and "HOW the artifact must be shaped" in t


def test_clause_forbids_softenings():
    t = _text()
    # no hedging that voids the enumeration duty
    assert "when feasible" not in t.split("CONSTRAINT ENUMERATION")[1][:2000]
    assert "if possible" not in t.split("CONSTRAINT ENUMERATION")[1][:2000]


def test_clause_has_no_task_literals():
    t = _text()
    seg = t.split("CONSTRAINT ENUMERATION COMPLETES")[1][:2500]
    for lit in ("clamp", "BsaI", "primer", "dna", "DNA", "overhang"):
        assert lit not in seg, f"task literal leaked into generic clause: {lit}"


def test_clause_lists_constraint_kinds():
    t = _text()
    seg = t.split("CONSTRAINT ENUMERATION COMPLETES")[1][:2500]
    for kind in ("count", "ordering", "naming rule", "unit"):
        assert kind in seg, f"missing constraint kind example: {kind}"
