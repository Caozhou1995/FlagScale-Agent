"""AN ACTION THE TASK NAMES IS DISCHARGED BY STARTING IT.

Regression guard, adapted to the CURRENT prompt.py bytes (re-homed the
clause into the clock section). RULING: 以现在代码为准，修改单测 — the tests
follow the production wording, never the reverse; every assert below pins a
load-bearing phrase of the CURRENT file and can still go RED.

Motivating failure (this 8h RSI window, step8): the operator directive was
"8 小时内启动新一轮的全量测试（并发为4）" — LAUNCH a new full run at
concurrency 4. The agent read it as "complete the full run", computed
89 tasks x 45min / 4 = 5.2h > remaining window, and therefore kept DEFERRING
the launch, substituting a cheaper 4-task wave. Deferring produced zero
end-to-end readings where launching would have produced readings on every
wave it reached before the wall.

Gates, all must hold against current bytes:
  1. ANTI-DECAY: the clause's load-bearing sentences present verbatim (§7).
  2. ESCAPE HATCHES: silent truncation named a false completion (10.4);
     retargeting to an easier goal named substitution (Section 6).
  3. NO TASK LITERAL: the clause is stated generically — no benchmark name,
     no task name, no numeric task count (that would be leakage).
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text(encoding="utf-8")

# Current wording of the load-bearing header (was ALL-CAPS before the re-home).
_HEADER = "An action the task names is discharged by STARTING it"


def _clause() -> str:
    """The paragraph starting at the load-bearing header, up to the next blank line."""
    start = TEXT.find(_HEADER)
    assert start != -1, "v19 clause header missing from prompt.py"
    return TEXT[start:].split("\n\n")[0]


def test_clause_present():
    c = _clause()
    assert c, "empty clause"


def test_states_start_not_finish():
    """The core distinction must be stated, not implied (current wording)."""
    c = _clause()
    assert "discharged by STARTING it" in c
    # a deferral yields no readings — the old "not by finishing it" phrasing
    # was collapsed into this sentence in the rewrite.
    assert "a deferred action yields nothing and is invisible as failure" in c


def test_rejects_cost_as_reason_to_defer():
    """The wrong reasoning that caused the loss must still be named as wrong.

    Current wording: the full cost is MEASURED, and the action is started
    anyway — cost is an input to how, never a reason to not-start.
    """
    c = _clause()
    assert "measure its full cost and start it" in c
    assert "a deferred action yields nothing" in c


def test_covers_substitution_and_truncation_disclosure():
    """The two escape hatches the model used must both still be closed.

    The re-home moved both: truncation disclosure now lives in 10.4, the
    cheaper-substitute prohibition in Section 6 (the v19 bullet phrasings
    'cheaper subset' / 'say the truncation out loud' are gone — grep 0).
    """
    low = TEXT.lower()
    # silently narrowing without saying so
    assert "a stated truncation is an honest result, a silent one is a false completion" in low
    # substituting a cheaper goal for the named one
    assert "retargeting to an easier goal is substitution" in low


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
