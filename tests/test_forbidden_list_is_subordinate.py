"""A forbidden entry is subordinate to the task text and its named authorities.

Provenance: dna-assembly, job 2026-11-27__00-23-50, trial 9L8TfcV, reward pending
but the DELIVERED BYTES already decide the point:
  /app/primers.fasta -> 8/8 primers have the BsaI site at index 0, clamp = 0,
  byte-identical in SHAPE to the previous loss (8ZXZ9gz). The verifier asserts
  `assert i >= 1, "Primer must have clamp of at least 1 nucleotide..."`.

What made this run different is WHY it lost. The agent's own plan carried:

  forbidden#1 (plan_7d903538.yaml, created 00:32 — before design started):
  "Do not add extra bases 5' of the BsaI site in a way that shifts the site off
   the primer's 5' end WITHOUT VERIFYING a tail parser strips it — instead put
   GGTCTC/GAGACC at a fixed known offset and record the choice."

and it cited that entry twice to overrule the authority:

  :1329 "But adding a clamp shifts the recognition site off the very 5' end. My
         forbidden item #1 says: don't add extra 5' bases"
  :1913 "...But my forbidden list item 1 ..."

So the semantics of the sentence are: an element whose only obstacle is MY OWN
unverified guess is FORBIDDEN. E9 says the opposite ("an element the unread
standard may require and no quoted rule forbids SHIPS"). E9 was in the system
prompt (grep = 1) and was never cited once in 379KB of trajectory
(coin flip = 0, unread standard = 0, SHIPS = 0, no quoted rule = 0), while the
self-authored entry was cited 2x.

The gap is not wording of E9 — it is that nothing told the agent what its own
forbidden list is SUBORDINATE to. This test pins that rule where the agent reads
it when authoring the list.
"""

import re

import pytest

from flagscale_agent.react.tools import plan_create as pc


def _frame_text() -> str:
    return re.sub(r"\s+", " ", " ".join(pc._FRAME_STEP_ACCEPTANCE))


def test_forbidden_entries_are_bounded_by_authority():
    t = _frame_text()
    assert "binds only what YOU control" in t
    assert "subordinate to the task text and to every named authority" in t


def test_forbidden_cannot_override_a_required_element():
    t = _frame_text()
    assert "can never make an element the authority requires 'forbidden'" in t


def test_own_uncertainty_must_not_become_a_prohibition():
    """The exact inversion this run made: 'unverified' -> 'forbidden'."""
    t = _frame_text()
    assert "never converts YOUR OWN unverified guess into a prohibition" in t
    assert "do not ship X *unlabelled*" in t
    assert "never 'do not ship X'" in t


def test_collision_resolution_is_stated():
    t = _frame_text()
    assert "the requirement wins and the entry is rewritten now" in t


def test_the_authoring_instruction_survives():
    """The pre-existing mechanics of writing the list must not be displaced."""
    t = _frame_text()
    assert "write 2-5 entries {action, why, instead}" in t
    assert "add_forbidden" in t
    assert "not virtues ('be careful')" in t


def test_rule_is_task_agnostic():
    t = _frame_text().lower()
    for word in ("clamp", "ggtctc", "primer", "bsai", "dna"):
        assert word not in t, f"task literal leaked into the frame instruction: {word!r}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
