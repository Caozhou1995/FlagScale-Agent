"""The STATED-VS-SHIPPED slot must demand the SHIPPED characters, not a claim.

Measured failure this pins: an agent's own review line read
"<constraint> -> <structure A + B + C> -> agree", which is a *true-sounding
description* of the intended design, while the delivered artifact carried
A + C with the middle element simply absent. The slot passed, the byte-level
gate failed. The wording now requires quoting the characters read off the
shipped artifact, so a description alone cannot fill it.

Two directions, both pinned:
  (1) the instruction text carries the requirement — remove it and this goes red;
  (2) the detector still accepts a real answer AND the explicit no-op, because
      this is a wording change, not a new gate: nothing that used to pass may
      now fail for having said "no constraint I stated".
"""

import pytest

from flagscale_agent.react.guard import system_prompt_remind as m
from flagscale_agent.react.guard.system_prompt_remind import _review_labels_present


# ── 1. The instruction demands a quotation from the artifact ─────────────────
def test_remind_message_requires_quoting_shipped_characters() -> None:
    assert "QUOTED off the SHIPPED file" in m._REMIND_MESSAGE


def test_remind_message_names_the_omission_mechanic_in_the_abstract() -> None:
    """The mechanic must be teachable without this task's vocabulary."""
    low = m._REMIND_MESSAGE.lower()
    assert "a+b+c designs ship as a+c by one omission" in low
    assert "no chars embodying it" in low


def test_instruction_carries_no_task_vocabulary() -> None:
    """Rendered into every tripped context — it must teach the pattern only."""
    low = m._REMIND_MESSAGE.lower()
    for word in ("clamp", "bsai", "ggtctc", "primer", "neb", "spacer",
                 "nucleotide", "enzyme", "dna"):
        assert word not in low, word


# ── 2. The detector still accepts a real answer and a real no-op ────────────
@pytest.mark.parametrize(
    "blob",
    [
        # a full answer that names the axis
        "W2: STATED-VS-SHIPPED: length in range -> read '22,36,16' off the file -> agree",
        # the explicit no-op the slot offers
        "W2: no constraint I stated beyond the filename",
        # a paraphrase that states the conclusion
        "W2: no stated constraint was shipped differently",
    ],
)
def test_detector_still_accepts_answer_and_noop(blob: str) -> None:
    cov = _review_labels_present({"notes": blob})
    assert cov.has_stated_shipped_subslot is True


def test_detector_can_go_red_when_the_axis_is_absent() -> None:
    """Proves the sub-slot is still a gate and not always-true."""
    cov = _review_labels_present({"notes": "W2: I looked at everything, all fine"})
    assert cov.has_stated_shipped_subslot is False
