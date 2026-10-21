"""Regression tests: when a convention discriminator reports the two candidate
conventions as NOT SEPARABLE at the final observable, the agent must fall back to
the MINIMAL-UNCERTAINTY default (the convention that adds no transformation /
the one the environment's own reference code executes) — never to a
confidence-style tie-break.

Motivation (measured): pytorch-model-cli__Xs6Wmrf (jobs/2026-10-21__08-10-31)
reward=0, verifier `test_cli_tool_output FAILED 'Predictions do not match
expected values'`. The agent ran convention_discrimination correctly, got
NOT SEPARABLE (both raw/255 and Normalize give argmax=2 on image.png), and then
chose Normalize because its softmax confidence was higher — agent log L378
verbatim:

    "with normalize, p=0.9982 — very confident. With raw/255, p=0.4971 — less
     confident. The high confidence under normalize strongly suggests the model
     was trained with Normalize(0.1307,0.3081)."

The task's own authoritative convention is `solution/solve.sh:135`:
    output[y * 28 + x] = (float)r / 255.0f;
and model.py contains NO preprocessing at all. So the correct fallback was the
identity scaling, and confidence was the wrong discriminator.

The clause must be domain-neutral (no task names, no numeric constants from any
grader).
"""

import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
MARKER = "A CONFIDENCE SCORE IS NOT A TIE-BREAK"


def _text() -> str:
    return PROMPT.read_text()


def test_marker_present():
    assert MARKER in _text().upper(), (
        "prompt must state that a confidence/quality score is not a valid "
        "tie-break between conventions"
    )


def test_clause_gives_the_right_fallback_after_not_separable():
    text = _text()
    idx = text.upper().find(MARKER)
    assert idx != -1
    window = text[idx : idx + 2000].lower()
    assert ("not separable" in window or "no separation" in window
            or "indistinguishable" in window), (
        "clause must be triggered by the NOT-SEPARABLE outcome of a "
        "discrimination probe"
    )
    assert ("minimal" in window or "identity" in window or "no transformation" in window
            or "leaves the value untouched" in window or "reference" in window), (
        "clause must name the correct fallback: the minimal-transformation / "
        "reference-implemented convention"
    )


def test_clause_names_confidence_as_the_wrong_discriminator():
    text = _text()
    idx = text.upper().find(MARKER)
    window = text[idx : idx + 2000].lower()
    assert "confidence" in window, (
        "clause must explicitly forbid confidence as the discriminator"
    )
    assert re.search(r"\b(not|never|cannot|no)\b", window), (
        "clause must be prohibitive, not advisory"
    )


def test_clause_is_obligation():
    text = _text()
    idx = text.upper().find(MARKER)
    window = text[idx : idx + 2000]
    assert re.search(r"\b(must|never|always|take|fall back)\b", window, re.I)


def test_no_task_identifying_words():
    low = _text().lower()
    for leak in ("mnist", "pytorch-model-cli", "0.1307", "0.3081", "255.0",
                 "terminal-bench", "harbor", "softmax"):
        assert re.search(rf"\b{re.escape(leak)}\b", low) is None, f"leaks {leak}"
