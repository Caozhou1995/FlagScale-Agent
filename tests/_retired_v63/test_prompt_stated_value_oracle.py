"""v62 (anti-decay) — WHEN THE TASK TEXT STATES A VALUE, THAT VALUE IS THE ORACLE.

Regression guard for the 2.3b corollary in prompt.py (inserted after 2.3a,
before 2.4). Motivating failure (v61 batch, circuit-fibsqrt): the task text
stated two worked examples ("running /app/sim 208 should output 377",
"running /app/sim 20000 should give 1407432322"). The agent ran the example,
observed a DIFFERENT value from its own scaffold, and re-explained the task
text to match its artifact ("the example gates.txt prints argv[1]/2: 208/2=104
✓") — building the entire deliverable on the wrong index rule. The second
stated example appeared ZERO times in its whole transcript. Verifier FAILED
on every test_sqrt_fib row (six independent rows, one consistent N//2 rule).

Two gates, both must hold:
  1. ANTI-DECAY: the clause's load-bearing sentences must be present verbatim.
  2. NO TASK LITERAL: the clause must be generic — no benchmark/task name,
     no stated-example number, no file path from the motivating task.
"""

from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()

ANCHORS = [
    "WHEN THE TASK TEXT STATES A VALUE, THAT VALUE IS THE ORACLE",
    "RUN IT EXACTLY AS STATED",
    "the DEFAULT hypothesis is that your artifact is wrong",
    "re-explain the task",
    "enumerate EVERY example the text gives",
    "an oracle you refused to",
    "put each stated value in one row",
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
    """The 2.3b paragraph: from its header to the start of the 2.4 header."""
    start = TEXT.find("2.3b WHEN THE TASK TEXT STATES A VALUE")
    assert start != -1, "2.3b clause header missing from prompt.py"
    end = TEXT.find("2.4 MEASURE THE QUANTITY", start)
    assert end != -1, "2.4 header missing after 2.3b (clause is unterminated)"
    return TEXT[start:end]


def test_v62_clause_anti_decay():
    clause = _clause()
    missing = [a for a in ANCHORS if a not in clause]
    assert not missing, f"2.3b anchor sentence(s) decayed/removed: {missing}"


def test_v62_clause_placement_and_containment():
    """2.3b sits between 2.3a and 2.4; no new top-level section was added."""
    i_23a = TEXT.find("2.3a A VERIFIED CONCLUSION")
    i_23b = TEXT.find("2.3b WHEN THE TASK TEXT STATES A VALUE")
    i_24 = TEXT.find("2.4 MEASURE THE QUANTITY")
    assert -1 not in (i_23a, i_23b, i_24), "clause or neighbors missing"
    assert i_23a < i_23b < i_24, "2.3b must sit between 2.3a and 2.4"


def test_v62_no_task_literal():
    clause = _clause()
    leaked = [lit for lit in FORBIDDEN_LITERALS if lit in clause]
    assert not leaked, f"2.3b leaks task-identifying literal(s): {leaked}"
