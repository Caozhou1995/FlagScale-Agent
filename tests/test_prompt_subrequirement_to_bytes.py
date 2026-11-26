"""Sub-requirements of a NAMED authority must be mapped to byte positions.

Motivating failure (measured, 2026-11-26, dna-assembly standard-budget trial):
the task text named the vendor manual as the authority to satisfy; the agent
read the manual, quoted the rule, and still shipped bytes that omitted one
sub-requirement of it. Reading the rule and placing it in the delivered bytes
had no step between them.

Anti-decay pins: the clause text must survive verbatim. No task literal here:
no vendor name, no enzyme name, no benchmark value.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as PROMPT  # noqa: E402

FLAT = " ".join(PROMPT.split())


def test_clause_is_present() -> None:
    assert "split its rule for this artifact into sub-requirements" in FLAT


def test_mapping_goes_to_delivered_bytes() -> None:
    assert "put each where it must hold in the delivered bytes" in FLAT


def test_unplaceable_subrequirement_is_a_recorded_decision() -> None:
    assert "one you cannot place is a recorded decision, never silent" in FLAT


def test_it_hangs_off_the_existing_pointer_rule() -> None:
    """The new action must attach to E1's existing 'runnable check' sentence,
    not float free."""
    i = FLAT.find("Turn every such pointer into a runnable check")
    j = FLAT.find("split its rule for this artifact into sub-requirements")
    assert 0 <= i < j < i + 400


def test_no_task_literal_leaked() -> None:
    low = FLAT.lower()
    for banned in ("ggtctc", "bsai", "bsa i", "neb", "dna-assembly", "clamp"):
        assert banned not in low, f"task literal leaked: {banned}"
