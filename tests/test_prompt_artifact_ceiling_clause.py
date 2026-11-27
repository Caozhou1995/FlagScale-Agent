"""The artifact's own ceiling is a BUILD constraint; "cannot be met" is a claim
about the ARTIFACT, never about the clock.

Provenance: gpt2-codegolf, job 2026-11-26__23-26-52, trial mexYuMY, reward=0.
The verifier's own bytes (verifier/test-stdout.txt):

    >       assert gpt2_path.stat().st_size < 5000, (
    E       AssertionError: File /app/gpt2.c is larger than 5000 bytes
    E       assert 5695 < 5000

That assertion SHORT-CIRCUITS the test — every later assertion (output equality,
tokenizer behaviour) never ran, so the only axis observed is the ceiling.

The agent was not ignorant of the bound: it measured repeatedly, and its own
numbers GREW while it kept deferring the cut —

    :2698 "Current size 5482 > 5000. I need to be honest: I cannot meet the byte
           bound and correct behavior in remaining time."
    :2805 "That made it bigger (5827->5696, small gain). ... Need ~700 bytes."
    :2887 "...I burned most of the clock on format discovery and had no time left
           to determine the actual tensor order or to golf the source under 5000"

and its close-out names the clock, not the artifact:

    :2882 "My code is not compressed enough ... without genuine code-golfing I
           didn't get to."

The prompt already gave two exits for a task-fixed bound ("fix it or say the
requirement cannot be met") and the agent took the second one — but nothing in
the prompt said what the second exit OWES, so "I ran out of time" discharged it.
This clause closes that seam without removing either exit.
"""

import re

import pytest

from flagscale_agent.react.prompt import PROMPT_BUDGET, SYSTEM_PROMPT_STATIC, prompt_len

FLAT = re.sub(r"\s+", " ", SYSTEM_PROMPT_STATIC)

OPENERS = ("A number you measured outside a stated bound is a verdict",
           "A number, format or literal the task itself fixes is a gate")
CEILING = "A ceiling on the artifact ITSELF"


def test_ceiling_clause_is_present():
    assert CEILING in FLAT


def test_ceiling_follows_the_gate_sentence_it_qualifies():
    """Order is the rule: the ceiling clause extends the exits stated just before it."""
    i_gate = FLAT.index("shipping past it scores zero; the only exits are fixing it")
    i_ceil = FLAT.index(CEILING)
    assert i_gate < i_ceil, "the ceiling clause must follow the gate sentence it qualifies"


def test_both_exits_of_the_gate_sentence_survive():
    """The clause must not close a choice by DELETING an exit the prompt grants."""
    assert "the only exits are fixing it or saying the requirement cannot be met" in FLAT


def test_ceiling_names_build_constraint_not_cleanup():
    seg = FLAT[FLAT.index(CEILING):]
    seg = seg.split("One example is not a held-out test set")[0]
    for face in (
        "is a BUILD constraint, not cleanup",
        "comes out of structure",
        "cannot be deferred past the design you commit to",
        "every write that grows the artifact is measured",
    ):
        assert face in seg, f"lost an executable face: {face!r}"


def test_second_exit_is_scoped_to_the_artifact_not_the_clock():
    seg = FLAT[FLAT.index(CEILING):]
    seg = seg.split("One example is not a held-out test set")[0]
    assert "never the clock" in seg
    assert "owes the construction tried and where it overflowed" in seg
    assert "running out of time is a disclosure, not an exit" in seg


def test_clause_is_task_agnostic():
    """A benchmark literal in the prompt is a leak (six gates scan for it)."""
    seg = FLAT[FLAT.index(CEILING):]
    seg = seg.split("One example is not a held-out test set")[0].lower()
    for word in ("gpt2", "5000", "codegolf", " .c ", "bpe", "transformer"):
        assert word not in seg, f"task literal leaked into the clause: {word!r}"


def test_budget_holds():
    assert prompt_len() <= PROMPT_BUDGET, f"{prompt_len()} > {PROMPT_BUDGET}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
