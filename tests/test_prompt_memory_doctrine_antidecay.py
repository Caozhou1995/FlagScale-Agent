"""Anti-decay guard for §8's MEMORY doctrine (user ruling, 2026-11-23).

WHY THIS TEST EXISTS. §8 is the only section of the prompt that no benchmark
literal pins, so every "compress to pay for a new clause" round reached for it
first. Three successive raises of PROMPT_BUDGET were each paid for out of §8,
and the payment was taken in exactly the currency that made the doctrine
usable — the concrete examples and the reconcile criteria:

  ("first deploy, first launch, first read of an unfamiliar spec")  -> deleted
  "keys truncate AT THE LIMIT and split"                            -> "truncate and split"
  "it persists ON DISK across evictions"                            -> "persists across evictions"
  "(contradiction, durability, generalization, key hygiene)"        -> deleted entirely
  "step notes ARE append-only"                                      -> "step notes append-only"

Measured consequence: tests/test_prompt_memory_read_first.py went 3-failed, and
`contradiction, durability` / `key hygiene` were at 0 occurrences prompt-wide.
The read-half test could not catch this because it only pinned the relations
abb7d41 introduced; it never pinned the write-half substance or a size floor.

This guard pins BOTH:
  (a) the specific lossy deletions above, each as an exact substring, so that
      re-deleting any one of them turns this file red; and
  (b) a FLOOR on the §8 MEMORY paragraph's size, so a compression pass cannot
      quietly shave 200 chars off it while technically keeping every pinned
      phrase. The floor is deliberately set just under the current value: it is
      a ratchet, not a target.

Scope is the MEMORY paragraph inside §8, never the whole prompt — "memory"
appears throughout §2/§9/§10 and a prompt-wide assertion would fire on
sentences this doctine does not own.
"""
from __future__ import annotations

import re

import pytest

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

# The §8 MEMORY paragraph spans from its opener to the first unrelated §8 topic.
_START = "MEMORY pays in BOTH directions."
_END = "Context is finite"


def _memory_paragraph() -> str:
    src = SYSTEM_PROMPT_STATIC
    i = src.find(_START)
    assert i != -1, "§8 lost its MEMORY opener entirely"
    j = src.find(_END, i)
    assert j != -1, "§8 lost the 'Context is finite' boundary — paragraph unreadable"
    return src[i:j]


# ── (a) the specific deletions that were measured, each pinned verbatim ──────

LOSSY_DELETIONS = [
    # Retired 2026-11-26: the concrete trigger examples were compressed into
    # "the KIND of work changes", which is the live text; it cannot be
    # reported as a lossy deletion any more.
    # "truncate and split" states neither WHERE truncation happens nor that a
    # miss must be widened around — both faces were load-bearing.
    ("keys truncate at the limit and split across domains",
     "'at the limit' was shaved off the key-truncation warning"),
    # Eviction durability is the whole reason to write at all.
    ("persists on disk across evictions",
     "'on disk' was shaved off the durability claim"),
    # The four reconcile criteria are the DECISION procedure for a write; with
    # them gone, only 'do not write twice' survives.
    ("contradiction, durability, generalization, key hygiene",
     "the four reconcile criteria were deleted again"),
    # 'are' is trivial-looking and was exactly what a compression pass ate.
    ("step notes are append-only",
     "the append-only rule lost its verb"),
]


@pytest.mark.parametrize("phrase,why", LOSSY_DELETIONS)
def test_a_measured_deletion_has_not_recurred(phrase: str, why: str):
    s = _memory_paragraph()
    assert phrase in s, f"{why} — {phrase!r} is missing from the §8 MEMORY paragraph"


def test_artifact_you_build_clause_is_present():
    """Building a launcher/probe and NOT recording it was the observed failure.

    Measured 2026-11-23: a write-compressor trial was launched with a purpose-built
    script, no memory entry was written for it, and the very next session rebuilt
    an equivalent launcher under a different name. The doctrine said "a path, a
    value, a fix" — it never said "the thing you just built", so the write trigger
    did not fire.
    """
    s = _memory_paragraph()
    assert "AN ARTIFACT YOU BUILD IS WORTH A MEMORY" in s
    # The trigger must name the artifact classes, or it stays an abstraction.
    for noun in ("A launcher", "a probe script", "one-off converter"):
        assert noun in s, f"artifact clause does not name {noun!r}"
    # And it must demand the two facts that make it reusable.
    assert "record its path" in s
    assert "invocation" in s


# ── (b) the size floor: a ratchet, not a target ─────────────────────────────

# Current measured size of the paragraph at the 2026-11-23 restore. The floor is
# set below it so ordinary wording polish is allowed but a bulk shave is not.
_FLOOR = 1900


def test_memory_paragraph_size_floor():
    n = len(_memory_paragraph())
    assert n >= _FLOOR, (
        f"§8 MEMORY paragraph is {n} chars, below the {_FLOOR}-char floor. "
        "It has been compressed again to pay for a clause elsewhere; raise "
        "PROMPT_BUDGET by user ruling instead."
    )


def test_reconcile_criteria_are_a_list_not_prose():
    """The four criteria must stay enumerable — a rewritten sentence dropped them once."""
    s = _memory_paragraph()
    m = re.search(r"in reasoning \(([^)]*)\)", s)
    assert m, "the reconcile criteria are no longer presented as an enumerable list"
    items = [x.strip() for x in m.group(1).split(",")]
    assert len(items) == 4, f"expected 4 reconcile criteria, found {items}"
    assert items == ["contradiction", "durability", "generalization", "key hygiene"]


def test_the_paragraph_stays_inside_the_budget():
    from flagscale_agent.react import prompt as prompt_mod
    assert prompt_mod.prompt_len() <= prompt_mod.PROMPT_BUDGET
