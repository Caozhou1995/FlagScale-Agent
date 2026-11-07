"""No task-specific answer may live in the shipped system prompt.

Why this file exists: on 2026-11-07 a clause meant to teach the agent "quote the
number you measured" was written with THREE REPLAY ANSWERS inline — a Raman
peak position in both units, a fastText accuracy against its bar, and a
segmentation count. The prompt ships to every task, so those bytes were handed
to the agent as the answer to three real items: a leak, not a lesson. The
clause is worth keeping (19 replays lost exactly there); its evidence set is
not.

The rule this test enforces: an EVALUATION-SPECIFIC literal must not appear in
the prompt. The measurement numbers, target thresholds and item names of any
gradeable task are that item's answer or its bar, and neither belongs in shared
text. Generic mechanism words ("the bar", "the reading") are exactly what should
carry the lesson; a concrete number is the tell that the lesson was written from
a graded item rather than from the mechanism.

Kept deliberately as a DENYLIST so it never blocks a legitimate prompt edit: it
asserts absence of known-bad literals plus a shape check for the specific class
of leak (a numeric literal immediately next to prompt words about grading).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC, prompt_len

# Literals observed in the 2026-11-07 leak. Each is an answer or a bar of one
# specific replay item; none is a mechanism. Extend this list when a new leak is
# caught — that is the maintenance this test asks for.
LEAKED_LITERALS = (
    "1580.5",
    "2670.2",
    "6327",
    "3745",
    "0.6193",
    "0.0007 below",
    "IoU<0.5",
    "IoU 0.5",
    "raman",
    "fasttext",
    "sam-cell-seg",
)


def test_no_leaked_answer_literal_in_prompt() -> None:
    """The exact bytes of the 2026-11-07 leak must never return."""
    low = SYSTEM_PROMPT_STATIC.lower()
    hits = [lit for lit in LEAKED_LITERALS if lit.lower() in low]
    assert not hits, (
        "system prompt carries task-specific answers: "
        f"{hits} — a prompt ships to every item, so these bytes leak the "
        "answer to the item they were taken from. Rewrite the clause from the "
        "mechanism (a reading on the wrong side of the bar is a verdict), not "
        "from the graded numbers."
    )


def test_no_grading_literal_next_to_a_prompt_number() -> None:
    """Shape check: a number sitting on a grading word is the leak's fingerprint.

    The leaked clause read "measured 0.6193 against a 0.62 bar" — a value and a
    threshold from one item. Mechanism text says "the bar the task states" and
    names no digits. This is a lower bound on the rule above: it catches a NEW
    leak whose literal nobody has listed yet.
    """
    pattern = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:%|percent)?\s*(?:vs\.?|against|below|above|under|over)?\s*"
        r"(?:the\s+)?(?:0?\.\d+\s+)?(?:bar|threshold|cutoff|pass\s+mark)\b",
        re.IGNORECASE,
    )
    hits = pattern.findall(SYSTEM_PROMPT_STATIC)
    assert not hits, (
        "system prompt pairs a numeric literal with a grading threshold: "
        f"{hits} — that is the fingerprint of a clause copied from one graded "
        "item instead of written from the mechanism."
    )


def test_prompt_still_carries_the_verdict_mechanism() -> None:
    """Positive half: the lesson the leak was wrapped in must SURVIVE the fix.

    Deleting the leak is only half the gate. A fix that also deletes the rule
    leaves the prompt silent on the failure it was added for, and this test
    would still be green — so assert the mechanism text is still present.
    """
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "w3 is a reading, not a ritual" in low, (
        "the W3 verdict clause is gone; it was added from 19 replay losses where "
        "the agent measured the right number and shipped the wrong side"
    )
    assert "verdict, not a caveat" in low, (
        "the clause no longer states that a self-measured miss is a verdict; "
        "without that sentence it is a description, not a gate"
    )


def test_prompt_within_budget_after_the_fix() -> None:
    """The clause must fit the user-raised 40000 budget, not push past it."""
    n = prompt_len()
    assert n <= 40_000, f"prompt is {n} chars, budget 40000"
