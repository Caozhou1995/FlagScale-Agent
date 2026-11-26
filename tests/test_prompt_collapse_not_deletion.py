"""The collapse rule reconciles READINGS; it must never license deleting a
required element. Guards the 14:31 dna-assembly failure mode."""
import re
from flagscale_agent.react import prompt as P

FLAT = re.sub(r'\s+', ' ', P.SYSTEM_PROMPT_STATIC).strip()


def test_collapse_boundary_clause_present():
    assert "Collapse reconciles two READINGS of one quantity" in FLAT


def test_collapse_never_licenses_deletion():
    assert "never licensing DELETION of an element an authority requires" in FLAT


def test_collapse_boundary_sits_with_collapse_rule():
    i = FLAT.index("the two readings collapse to a single value")
    j = FLAT.index("never licensing DELETION")
    assert 0 < j - i < 700, (i, j)


def test_keep_the_element_directive_present():
    assert "and no rule forbids" in FLAT


def test_no_benchmark_literals_in_new_clause():
    for bad in ("aatt", "bsei", "ggtctc", "clamp", "primer"):
        assert bad not in FLAT[FLAT.index("never licensing DELETION") - 200:
                               FLAT.index("never licensing DELETION") + 200].lower()
