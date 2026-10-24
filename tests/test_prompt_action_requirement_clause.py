"""v19 (anti-decay) — AN ACTION THE TASK NAMES IS DISCHARGED BY STARTING IT.

Regression guard for the clause in prompt.py's TIME BUDGET section.
Motivating failure (this 8h RSI window, step8): the operator directive was
"8 小时内启动新一轮的全量测试（并发为4）" — LAUNCH a new full run at
concurrency 4. The agent read it as "complete the full run", computed
89 tasks x 45min / 4 = 5.2h > remaining window, and therefore kept DEFERRING
the launch, substituting a cheaper 4-task wave. Deferring produced zero
end-to-end readings where launching would have produced readings on every
wave it reached before the wall.

Two gates, both must hold:
  1. ANTI-DECAY: the clause's load-bearing sentences must be present verbatim.
  2. NO TASK LITERAL: the clause must be stated generically — no benchmark
     name, no task name, no numeric task count (that would be leakage).
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()


def _clause() -> str:
    """The paragraph starting at the load-bearing header, up to the next bullet."""
    start = TEXT.find("AN ACTION THE TASK NAMES IS DISCHARGED BY STARTING IT")
    assert start != -1, "v19 clause header missing from prompt.py"
    # The clause is one bullet; it ends at the next line that starts a new bullet.
    rest = TEXT[start:]
    m = re.search(r"\n- ", rest)
    return rest[: m.start()] if m else rest


def test_clause_present():
    c = _clause()
    assert c, "empty clause"


def test_states_start_not_finish():
    """The core distinction must be stated, not implied."""
    c = _clause().lower()
    assert "discharged by starting it, not by finishing it" in c
    assert "the obligation is that the action happens" in c


def test_rejects_cost_as_reason_to_defer():
    """The exact wrong reasoning that caused the loss must be named as wrong."""
    c = _clause().lower()
    assert "not a reason to defer starting" in c
    assert "cost calculation" in c


def test_covers_substitution_and_truncation_disclosure():
    """The two escape hatches the model used must both be closed."""
    c = _clause().lower()
    # substituting a cheaper subset for the named scope
    assert "replacing the named scope with a cheaper subset" in c
    # silently narrowing without saying so
    assert "say the truncation out loud" in c


def test_no_task_literal_leak():
    """Generic wording only — no benchmark/task names, no task counts."""
    c = _clause()
    forbidden = [
        "terminal-bench",
        "terminal_bench",
        "tb2",
        "terminal bench",
        "concurrency 4",  # the operator's specific number is not part of the rule
        "89",
        "4 tasks",
        "extract-moves",
        "sam-cell",
        "protein-assembly",
    ]
    low = c.lower()
    for word in forbidden:
        assert word not in low, f"task literal leaked into v19 clause: {word!r}"
