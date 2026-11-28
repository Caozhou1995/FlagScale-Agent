"""A proposal must be NECESSARY and SUFFICIENT — never raised to have one.

Why this clause exists (2026-11-28, user ruling):
    "把所有提案给关掉吧。提案一定是充分且必要的，结合着刚刚进化顺序，可以在
     prompt里强调一下，不要为了提案而提案。"

Nine proposals were open at once. On inspection, three described work that had
already LANDED (the thinking-dump truncation, the provider degrade order, the
image budget's counting unit) — they were never closed, so the queue read as a
backlog while the code had moved on. The queue had stopped being a list of
decisions and become a habit.

The clause encodes the user's own severity order, cheapest layer first:
    an existing harness BUG  ->  PROMPT coverage  ->  a TOOL  ->  core
    mechanics (plan/memory/guard)
so a proposal is admissible only when the cheaper layer it names really showed a
defect. Skipping a layer is how a one-off execution error gets dressed up as a
mechanism requirement.

Honest limit: these tests pin the CLAUSE TEXT and the budget. Whether the agent
actually withholds a proposal is behavioural; the falsifier is reading a
session's proposal list — the same way the stale queue above was falsified.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flagscale_agent.react.prompt import (  # noqa: E402
    PROMPT_BUDGET,
    SYSTEM_PROMPT_STATIC as PROMPT,
    prompt_len,
)

FLAT = " ".join(PROMPT.split())
ANCHOR = "No proposal for its own sake"


def test_the_rule_is_stated() -> None:
    assert ANCHOR in FLAT


def test_the_severity_order_is_named_cheapest_first() -> None:
    """All four layers, in the user's order. A missing layer frees the agent to
    jump straight to the heaviest one — exactly the failure this prevents."""
    i = FLAT.find(ANCHOR)
    assert i >= 0
    window = FLAT[i:i + 90]
    for layer in ("bug", "prompt", "tool", "mechanics"):
        assert layer in window, f"layer {layer!r} missing from the order"
    # ORDER, not just presence: bug before prompt before tool before mechanics.
    assert (
        window.index("bug")
        < window.index("prompt")
        < window.index("tool")
        < window.index("mechanics")
    ), f"order is wrong: {window!r}"


def test_the_prohibition_names_the_habit_not_the_count() -> None:
    """The rule must forbid proposing FOR ITS OWN SAKE. The wording has to name
    the thing prohibited; a vague 'propose less' would not be falsifiable."""
    assert ANCHOR in FLAT


def test_prompt_stays_within_budget() -> None:
    """The clause was funded by compressing existing wording, not by raising the
    ceiling — the user's standing instruction: 不会轻易抬高 prompt 预算."""
    assert prompt_len() <= PROMPT_BUDGET


def test_no_task_literal_leaked() -> None:
    low = FLAT.lower()
    for banned in ("ggtctc", "bsai", "dna-assembly", "gcode", "prop_"):
        assert banned not in low, f"task literal leaked: {banned}"
