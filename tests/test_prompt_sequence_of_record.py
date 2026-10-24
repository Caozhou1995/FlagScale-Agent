"""Regression tests for the sequence-of-record and identification-by-content clauses.

Clause source: flagscale_agent/react/prompt.py — the paragraph appended after the
information-conservation tie-break ("A provided sequence or record has ONE source
of truth ...") and the FRAME-audit sentence on recall-based identification.

Guardrails asserted here:
- the clauses exist and carry their decision keys verbatim (softening is a regression);
- zero task-specific literals (no protein/gene/tag names, no database names, no
  sequence motifs) — the clause is a general capability rule, not a task recipe.
"""
import re

PROMPT = (
    "/nfs/caozhou/fa_eval/FlagScale-Agent/flagscale_agent/react/prompt.py"
)
_SRC = open(PROMPT).read()


def _clause_text() -> str:
    assert "ONE source of truth" in _SRC, "sequence-of-record clause missing from prompt"
    m = re.search(r"A provided sequence or record has ONE source of truth.*?(?=\n\n)", _SRC, re.S)
    assert m, "sequence-of-record paragraph not found as a single block"
    return m.group(0)


def _frame_text() -> str:
    m = re.search(r"Audit the FRAME, not the entries.*?(?=\n\n)", _SRC, re.S)
    assert m, "FRAME-audit paragraph not found"
    return m.group(0)


def test_sequence_of_record_names_named_source_as_content():
    text = _clause_text()
    assert "source the task text NAMES" in text
    assert "full form as the named API/file returns it" in text
    # the classic escape is named and closed
    assert "not a cleaner variant" in text
    assert "even when that variant is a substring" in text


def test_containment_direction_is_decided():
    text = _clause_text()
    assert "Deliberately which direction the containment check runs" in text
    assert "locate the REFERENCE record inside YOUR deliverable" in text
    assert "reverse containment and proves nothing" in text


def test_budget_overflow_is_signal_not_permission():
    text = _clause_text()
    assert "A budget or length constraint does not license swapping the source of record" in text
    assert "usually a component picked by recall instead of by the spec chain" in text


def test_identification_requires_external_search_not_recall():
    text = _clause_text()
    assert "recall-from-memory guess is a hypothesis to test" in text
    assert "sequence-similarity/BLAST-style query" in text
    assert "read the hit's own naming record" in text
    assert "If the search yields no hit" in text

    frame = _frame_text()
    assert "recall-from-memory identification" in frame and "(iii) by definition" in frame
    assert "record the concrete hit identifiers" in frame
    assert "A search that returns no hit does not retroactively license the guess" in frame


def test_no_task_literals_in_clauses():
    banned = [
        # task-specific entity/motif literals that must never enter the generic clause
        r"DYKDDDDK", r"FLAG", r"lysozyme", r"HEL\b", r"Clover", r"mCherry",
        r"DHFR", r"SNAP", r"5WJ2", r"2H5Q", r"7BG1", r"8RMO", r"3KZY",
        r"His-?tag", r"TEV", r"RCSB", r"fpbase", r"UniProt", r"P00698",
        r"MNIST", r"0\.1307", r"gblock", r"GS linker",
    ]
    blob = _clause_text() + _frame_text()
    for pat in banned:
        assert re.search(pat, blob) is None, f"task literal leaked into generic clause: {pat!r}"
