"""WHEN THE TASK TEXT STATES A VALUE, THAT VALUE IS THE ORACLE (anti-decay pin).

Regression guard. Originally pinned as the 2.3b corollary of prompt.py
(originally a numbered corollary; now re-homed). Motivating failure (measured batch,
circuit-fibsqrt): the task text stated two worked examples ("running /app/sim
208 should output 377", "running /app/sim 20000 should give 1407432322"). The
agent ran the example, observed a DIFFERENT value from its own scaffold, and
re-explained the task text to match its artifact — building the entire
deliverable on the wrong index rule. The second stated example appeared ZERO
times in its whole transcript.

The 2026-11 10-section restructure removed the 2.x numbering and folded the
obligation into evidence rule E1 ("The task text's own named authority
outranks anything you built: a stated example value, a named tool, a specified
path"). These pins now target E1's current wording.

Two gates, both must hold:
  1. ANTI-DECAY: E1's load-bearing sentences must be present verbatim.
  2. NO TASK LITERAL: E1 must be generic — no benchmark/task name,
     no stated-example number, no file path from the motivating task.
"""

from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()

ANCHORS = [
    "a stated example value",
    "Turn every such pointer into a runnable check",
    "you are wrong until shown otherwise",
    "EXECUTABLE (bytes that run) or NARRATIVE (prose)",
]

# Task-identifying literals from the motivating failure (and near-neighbors
# that would leak the task class) — none may appear in the clause.
FORBIDDEN_LITERALS = [
    "fibsqrt",
    "gates.txt",
    "sim.c",
    "1407432322",
    "argv[1]/2",
    "377",
    "208",
    "20000",
    "fib",
    "isqrt",
]


def _clause() -> str:
    """The E1 paragraph: from its header to the start of the E2 header."""
    start = TEXT.find("E1 The task text's own named authority")
    assert start != -1, "E1 rule header missing from prompt.py"
    end = TEXT.find("E2 The task text's own wording", start)
    assert end != -1, "E2 header missing after E1 (clause is unterminated)"
    return TEXT[start:end]


def test_v62_clause_anti_decay():
    clause = _clause()
    missing = [a for a in ANCHORS if a not in clause]
    assert not missing, f"E1 anchor sentence(s) decayed/removed: {missing}"


def test_v62_clause_placement_and_containment():
    """E1 sits before E2; the evidence ranking survives as an ordered list."""
    i_e1 = TEXT.find("E1 The task text's own named authority")
    i_e2 = TEXT.find("E2 The task text's own wording")
    assert -1 not in (i_e1, i_e2), "clause or neighbors missing"
    assert i_e1 < i_e2, "E1 must sit before E2 in the evidence ranking"


def test_v62_no_task_literal():
    clause = _clause()
    leaked = [lit for lit in FORBIDDEN_LITERALS if lit in clause]
    assert not leaked, f"E1 leaks task-identifying literal(s): {leaked}"
