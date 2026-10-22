"""A constraint acquired from RESEARCH must be compiled into a check, not narrated.

Regression for a live loss (2026-10-22, dna-assembly): the instruction told the
agent to check the enzyme vendor's requirements. It did — and wrote the rule
into its own notes verbatim ("NEB recommends adding an extra base or two 5' of
the recognition site"), then deliberated for ~40 more lines and shipped a primer
that began directly at the recognition site with NO 5' clamp base. The grader
rejected it on exactly that rule ("Primer must have clamp of at least 1
nucleotide before BsaI site").

The existing clause covers constraints the TASK text states. This one covers a
constraint the agent had to go LOOK UP: it was learned, recorded in prose, and
never turned into an executable check. The clause must be generic — no task,
library, enzyme, or vendor names.
"""
import os
import re

PROMPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "flagscale_agent", "react", "prompt.py")


def _text():
    with open(PROMPT) as f:
        return f.read()


def test_research_constraint_clause_present():
    assert "A CONSTRAINT YOU LOOKED UP IS A CONSTRAINT YOU MUST CHECK" in _text()


def test_clause_requires_an_executable_check_not_prose():
    t = _text()
    i = t.index("A CONSTRAINT YOU LOOKED UP IS A CONSTRAINT YOU MUST CHECK")
    body = t[i:i + 1600]
    # must demand a runnable assertion bound to the final bytes
    assert "assertion" in body.lower()
    assert "run" in body.lower()
    # must name the exact failure: quoting is not applying
    assert "quot" in body.lower() or "narrat" in body.lower()


def test_clause_covers_the_looked_up_authority_not_just_the_task_text():
    """The gap this closes is research-derived, so the clause must say so."""
    t = _text()
    i = t.index("A CONSTRAINT YOU LOOKED UP IS A CONSTRAINT YOU MUST CHECK")
    body = t[i:i + 1600].lower()
    assert "research" in body
    assert "binding" in body or "like the ones the task states" in body


def test_clause_has_no_task_or_vendor_leakage():
    """agent-visible prompt must never name the task, its domain, or its vendor."""
    t = _text()
    i = t.index("A CONSTRAINT YOU LOOKED UP IS A CONSTRAINT YOU MUST CHECK")
    body = t[i:i + 1600].lower()
    for banned in ("bsai", "primer", "dna", "neb", "golden gate", "enzyme",
                   "dna-assembly", "mnist", "pytorch-model-cli", "raman"):
        assert banned not in body, f"leaked {banned!r} into agent-visible prompt"


def test_clause_is_a_bullet_at_top_level_style():
    """Match the file's bullet convention so it renders with the other clauses."""
    t = _text()
    assert re.search(r'^- "A CONSTRAINT YOU LOOKED UP IS A CONSTRAINT YOU MUST CHECK"',
                     t, re.M)
