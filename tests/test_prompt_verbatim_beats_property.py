"""G10: when two stated constraints collide LITERALLY, the verbatim-preservation one wins.

Evidence (protein-assembly trial, reward=0, batch 2026-10-21__08-10-31): the task
states both "should only contain GS linkers and the five proteins" (a PROPERTY
constraint) and "their protein sequences should match the fasta file returned by
the pdb API" (a VERBATIM constraint). The grader checks the second by literal
substring search (tests/test_outputs.py:32-39, `gblock_aa.find(donor_aa)`); the
agent removed the His-tag/TEV prefix because it read like a non-GS-linker
impurity -- reasoning from biological plausibility -- and the grader's own
signature was `assert 0 < -1`, i.e. donor_idx == -1 (not found). Everything else
in the deliverable passed, including its own three self-checks.

The clause must be domain-neutral: it talks about verbatim vs property
constraints, never about sequences, tags or biology.
"""

from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
HEAD = "A VERBATIM CONSTRAINT BEATS A PROPERTY CONSTRAINT"


def _text() -> str:
    return PROMPT.read_text()


def test_clause_present():
    assert HEAD in _text()


def test_clause_sits_with_the_other_constraint_rules():
    text = _text()
    i_clause = text.index(HEAD)
    i_loyalty = text.index("Constraint loyalty")
    assert abs(i_clause - i_loyalty) < 6000, (i_clause, i_loyalty)


def test_clause_distinguishes_the_two_kinds():
    """It must NAME both kinds, not just say 'prefer preservation'."""
    text = _text()
    i = text.index(HEAD)
    window = text[i : i + 1800]
    assert "verbatim" in window.lower()
    assert "property" in window.lower()


def test_clause_states_the_discriminating_reason():
    """The load-bearing reason: a grader can substring-match verbatim text,
    while a property is not machine-checkable per character."""
    text = _text()
    i = text.index(HEAD)
    window = text[i : i + 1800]
    assert "substring" in window or "literal" in window
    assert "check" in window or "verif" in window


def test_clause_tells_you_what_to_ask_yourself():
    """The actionable test: can the grader still FIND the text literally."""
    text = _text()
    i = text.index(HEAD)
    window = text[i : i + 1800]
    assert "find" in window or "locate" in window


def test_no_task_identifying_words_added():
    """Anti-leak: no project/task words, no domain nouns from this case."""
    text = _text()
    for leak in (
        "protein-assembly",
        "gblock",
        "dhfr",
        "his-tag",
        "fasta",
        "pdb",
        "linker",
        "clover",
        "mcherry",
        "snap-tag",
    ):
        assert leak not in text.lower(), leak
