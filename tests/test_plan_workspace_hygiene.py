"""Pin the DELIVERY & WORKSPACE HYGIENE acceptance item of the fixed review step.

Background (2026-11-12, user ruling): the original item covered ONLY the
delivery path. A real trial left /tmp/frame.bmp behind as a self-test
byproduct; the task verifier uses "file exists" as its sync signal and does
not unlink it first, so the stale file was read as a fresh result.  The user's
ruling: hygiene must cover EVERY intermediate/temporary file the agent itself
produced during the work, not just the delivery directory.

These tests are deliberately about SEMANTICS, not about a particular sentence:
they assert that each independently necessary half of the rule is present.
Ablation of either half must turn them red.
"""

import pytest

from flagscale_agent.react.tools.plan_create import _REVIEW_STEP_ACCEPTANCE  # noqa: E402


def _hygiene_items():
    return [a for a in _REVIEW_STEP_ACCEPTANCE if "HYGIENE" in a]


def test_exactly_one_hygiene_item_exists():
    """One item owns hygiene; two competing ones would drift apart."""
    assert len(_hygiene_items()) == 1


def test_hygiene_still_covers_the_delivery_path():
    """The pre-existing half must not be lost in the rewrite."""
    item = _hygiene_items()[0]
    assert "delivery path" in item
    assert "no .bak" in item
    assert "non-empty and well-formed" in item


def test_hygiene_covers_every_self_produced_temp_file():
    """The NEW half: scope is the whole working session, not the delivery dir."""
    item = _hygiene_items()[0]
    # The item must be explicit that it is NOT restricted to the delivery path.
    assert "not only about the delivery path" in item
    # The producer must be the agent itself (so pre-existing task inputs are
    # not wrongly deleted).
    assert "you yourself produced" in item
    # The temporal scope: produced anywhere, during the work.
    assert "intermediate and temporary files" in item
    assert "during the work" in item
    # Scratch / tmp locations must be named, since that is where the failure hit.
    assert "tmp locations" in item


def test_hygiene_names_the_harm_not_just_the_duty():
    """A bare 'clean up' is easy to skip; the failure mode must be stated."""
    item = _hygiene_items()[0]
    assert "mistaken for a fresh result" in item


def test_hygiene_does_not_claim_to_remove_task_provided_inputs():
    """Guard against over-reach: deleting the task's own inputs would be worse."""
    item = _hygiene_items()[0]
    assert "you yourself produced" in item
    # The old wording kept the delivery-path constraint; both must coexist.
    assert "The delivery path holds only what the task requires" in item
