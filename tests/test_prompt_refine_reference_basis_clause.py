"""v20 (anti-decay) — WHEN THE TASK SAYS REFINE, THE PROVIDED REFERENCE IS THE BASIS.

Regression guard for the clause in prompt.py (inserted just before
"MATCH THE FORM THE INPUTS IMPLY").

Motivating failure (sam-cell-seg, 2026-10-25 batch 05-45-50): the task supplied
existing cell annotations and asked to REFINE the masks with a segmentation model.
The agent replaced every annotation with its own SAM output. The grader measured
alignment against the SUPPLIED annotations and reported
`AssertionError: IoU is too low: 0.3615670413394159` — 8 of 9 tests passed, the
only failure was agreement with the provided reference. A more sophisticated
mask that disagrees with the reference is a WORSE answer, not a better one.

Gates:
  1. ANTI-DECAY: the load-bearing sentences must be present verbatim.
  2. NO TASK LITERAL: no task/model/dataset names, no numeric task counts.
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()


def _clause() -> str:
    start = TEXT.find("WHEN THE TASK SAYS REFINE, THE PROVIDED REFERENCE IS THE MEASUREMENT BASIS")
    assert start != -1, "v20 clause header missing from prompt.py"
    rest = TEXT[start:]
    m = re.search(r"\n- ", rest)
    return rest[: m.start()] if m else rest


def test_clause_present():
    assert _clause().strip()


def test_names_the_reference_as_coordinate_system():
    c = _clause().lower()
    assert "the artifact is the coordinate system" in c
    assert "same object, better boundary" in c


def test_states_better_is_not_a_defence():
    """The exact wrong belief that caused the loss."""
    c = _clause().lower()
    assert "the comparison has no clause for \"better\"" in c
    assert "your answer is wrong regardless of how much better it looks" in c


def test_prescribes_quantifying_against_the_reference():
    c = _clause().lower()
    # must instruct measuring against the supplied artifact, not self-confidence
    assert "quantify your output against the reference yourself" in c
    assert "not against your own model's confidence" in c
    # and the tie-break when outside the bound
    assert "the reference wins on that element" in c


def test_requires_minimal_movement_when_no_tolerance():
    c = _clause().lower()
    assert "the minimal-movement reading is the conservative one" in c
    assert "never a wholesale substitution" in c


def test_no_task_literal_leak():
    c = _clause().lower()
    forbidden = [
        "terminal-bench",
        "terminal_bench",
        "sam-cell",
        "sam cell",
        "mobilesam",
        "segment-anything",
        "iou is too low",
        "0.3615",
        "h&e",
        "histopath",
    ]
    for word in forbidden:
        assert word not in c, f"task literal leaked into v20 clause: {word!r}"
