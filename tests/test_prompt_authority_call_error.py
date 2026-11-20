"""A FAILED AUTHORITY CALL INDICTS THE CALL, NOT THE AUTHORITY (anti-decay pin).

Motivating failure (measured, 2026-11-20, job 2026-11-20__15-33-45): the task
text named an installed library as the thing to use, and the agent DID find the
authority's own contract in it. It then ran the native path, the call ERRORED
(bad task identifier), and instead of repairing the call it reasoned "a grader
writing a simple script would hit that error and would fall back to the shallow
API" -- i.e. it promoted its own failed call into a conclusion about the
AUTHORITY's behaviour, substituted an imagined reference implementation, and
shipped the wrong branch. reward=0, with the verifier's single failing
assertion naming the value the authority's own path produces.

The gap the existing rules left open: E1 ranks a named authority above your own
construction, E2 zeroes an imagined grader, E5 says a named library is observed
by RUNNING it. None of them says what to do when the RUN you attempted FAILS.
The default failure mode is exactly the one measured: "I could not call it, so
its behaviour is undetermined, so I may reason about it freely."

Two gates, both must hold:
  1. ANTI-DECAY: the load-bearing sentences must be present verbatim.
  2. NO TASK LITERAL: the clause must be generic -- no library name, task name,
     benchmark name, or value from the motivating failure.
"""

from pathlib import Path
import pytest

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()

ANCHORS = [
    "A named authority's call that errors indicts YOUR call",
    "not its behavior",
    "the authority is undetermined",
    "an imagined reference implementation",
    "Repair the call",
    "mark it UNCONSULTED",
]

# Task-identifying literals from the motivating failure (and near-neighbours
# that would leak the task class) -- none may appear in the clause.
FORBIDDEN_LITERALS = [
    "mteb",
    "MTEB",
    "HumanEval",
    "bge",
    "prompt_type",
    "task_name",
    "encode(",
    "retrieve",
    "rank5",
    "BEIR",
]


def _clause() -> str:
    """The E7 paragraph: from its header to the end of that paragraph."""
    start = TEXT.find("E7 A named authority's call that errors")
    assert start != -1, "E7 rule header missing from prompt.py"
    end = TEXT.find("\n\n", start)
    assert end != -1, "E7 clause is unterminated (no paragraph break)"
    return TEXT[start:end]


def test_clause_exists_and_is_a_single_paragraph():
    c = _clause()
    assert c.startswith("E7 ")
    assert len(c) > 150, c


def test_anti_decay_all_anchors_present():
    c = _clause()
    missing = [a for a in ANCHORS if a not in c]
    assert not missing, f"E7 anchor sentence(s) decayed/removed: {missing}"


def test_states_the_direction_of_the_indictment():
    """The core judgement: the error is evidence about MY call, not the authority."""
    c = _clause()
    assert "indicts YOUR call" in c
    assert "not its behavior" in c


def test_forbids_promoting_failure_into_undetermined():
    """The exact measured failure mode must be named, not merely implied."""
    c = _clause()
    assert "the authority is undetermined" in c
    assert "imagined reference implementation" in c


def test_gives_exactly_two_outlets():
    """Repair the call, or record the clause UNCONSULTED -- no third path."""
    c = _clause()
    assert "Repair the call" in c
    assert "UNCONSULTED" in c


@pytest.mark.parametrize("lit", FORBIDDEN_LITERALS)
def test_no_task_literal(lit):
    assert lit.lower() not in _clause().lower(), f"task literal leaked into E7: {lit}"


def test_sits_after_e6_and_before_the_rank_summary():
    """Ordering pin: E7 extends the evidence ranking, so it must live inside it."""
    e6 = TEXT.find("E6 When a requirement is only REMEMBERED")
    e7 = TEXT.find("E7 A named authority's call that errors")
    tail = TEXT.find("Each rank exists because ignoring it lost a run.")
    assert e6 != -1 and e7 != -1 and tail != -1
    assert e6 < e7 < tail
