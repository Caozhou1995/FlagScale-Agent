"""Requirements a task DEFERS to an external standard must become obligations.

Measured loss this closes (production trial, dna-assembly): the task stated only
"make sure the cut-sites you design satisfy NEB's requirements" -- the actual
requirement (a protective base 5' of the recognition site) lived in the STANDARD,
not the task text. An agent whose checklist mirrored the task's literal clauses
passed every one of them and still failed the grader. It had read the requirement
and then dropped it on a guess about whether it would be checked.

DESIGN CONSTRAINT (user-stated, and the reason this file has no pattern matching):
semantic judgement is delegated to the judge; regex/keyword detection is not
allowed. So these tests drive a stub judge and assert the MECHANICAL half only --
that deferred requirements join the obligation set, that an uncovered one is
reported, and that a missing judge leaves the gate inert.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flagscale_agent.react.guard.verification import (  # noqa: E402
    _constraint_coverage_missing,
    _stated_constraints,
)

TASK = (
    "Design primers for a one-pot assembly. The primers should respect:\n"
    " * The annealed part should have a length between 15 and 45 nucleotides.\n"
    " * Have a melting temperature between 58 and 72 degrees celsius.\n"
    " * Each pair should be at most 5 degrees celsius apart.\n"
    " * Output the minimum number of primer pairs.\n"
    " * If you aren't familiar with the enzyme make sure to check that the "
    "cut-sites you design satisfy the vendor's requirements.\n"
)

LITERAL = [
    "The annealed part should have a length between 15 and 45 nucleotides.",
    "Have a melting temperature between 58 and 72 degrees celsius.",
    "Each pair should be at most 5 degrees celsius apart.",
    "Output the minimum number of primer pairs.",
]
DEFERRED = [{
    "source": "make sure to check that the cut-sites you design satisfy the "
              "vendor's requirements.",
    "requirement": "At least one protective base must precede the recognition site.",
    "certain": True,
}]


class _Judge:
    """Stub judge: literal constraints + one deferred requirement."""

    def __init__(self, covered_all=True, deferred=None):
        self.covered_all = covered_all
        self._deferred = DEFERRED if deferred is None else deferred
        self.coverage_calls = 0

    def constraints(self, task_text):
        return list(LITERAL)

    def deferred_requirements(self, task_text):
        return list(self._deferred)

    def constraint_coverage(self, quotes, constraints, task_text=""):
        self.coverage_calls += 1
        if self.covered_all:
            return list(range(len(constraints)))
        return [i for i, c in enumerate(constraints) if "protective base" not in c]


ACT = {"constraints": [
    {"quote": "length between 15 and 45", "command": "awk ...", "observed": "len 21"},
    {"quote": "melting temperature between 58 and 72", "command": "oligotm ...",
     "observed": "59.89"},
    {"quote": "at most 5 degrees celsius apart", "command": "oligotm ...",
     "observed": "0.06"},
    {"quote": "minimum number of primer pairs", "command": "grep -c '>'",
     "observed": "pairs 4"},
]}
CORPUS = "@@@ len 21 59.89 0.06 pairs 4 @@@"


def test_deferred_requirement_joins_the_obligation_set():
    """The standard's requirement must be an obligation, not a literal-only set."""
    stated = _stated_constraints(TASK, _Judge())
    assert any("protective base" in s for s in stated), stated
    assert len(stated) == len(LITERAL) + 1


def test_deferred_requirement_deduplicated_against_literal():
    """If the judge returns the same phrase both ways, it is one obligation."""
    dup = [{"source": "x", "requirement": LITERAL[0], "certain": True}]
    stated = _stated_constraints(TASK, _Judge(deferred=dup))
    assert len(stated) == len(LITERAL), stated


def test_uncovered_deferred_requirement_is_reported():
    """The measured loss: everything literal covered, the deferred one missed."""
    missing = _constraint_coverage_missing(ACT, TASK, CORPUS, _Judge(covered_all=False))
    assert len(missing) == 1, missing
    assert "protective base" in missing[0]


def test_covered_deferred_requirement_passes():
    """When the judge says it is covered, no obligation remains."""
    assert _constraint_coverage_missing(ACT, TASK, CORPUS, _Judge(covered_all=True)) == []


def test_no_judge_keeps_the_gate_inert():
    """A dead provider must not manufacture obligations."""
    assert _stated_constraints(TASK, None) == []
    assert _constraint_coverage_missing(ACT, TASK, CORPUS, None) == []


def test_empty_or_blank_task_returns_nothing():
    assert _stated_constraints("", _Judge()) == []
    assert _stated_constraints("   \n\t ", _Judge()) == []


def test_no_deferred_standard_leaves_literal_set_unchanged():
    """A task deferring to nothing must not gain phantom obligations."""
    stated = _stated_constraints(TASK, _Judge(deferred=[]))
    assert stated == LITERAL


def test_judge_exception_fails_open():
    """A raising judge must leave the literal set intact, never crash the gate."""
    class Boom(_Judge):
        def deferred_requirements(self, task_text):
            raise RuntimeError("provider down")

    assert _stated_constraints(TASK, Boom()) == LITERAL
