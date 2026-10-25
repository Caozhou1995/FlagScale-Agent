"""Char-collapse detection: the fourth filler mode.

Measured on the v28 filter-js trial (jobs/2026-10-25__16-27-04): the thinking
stream emitted ONE line of 119,373 consecutive '0' bytes — 122K of the 145KB
log was a single byte value, with ZERO completed lines, so the line-window
detector (which only evaluates on "\\n") never fired, and the trial burned to
the deadline with /app empty. `_detect_char_collapse` scans the buffered tail
of the current line for a monotone one-byte run.
"""
from flagscale_agent.react.agent import (
    WorkerAgent,
    _detect_char_collapse,
)

MIN_RUN = WorkerAgent.INTRA_STREAM_CHAR_COLLAPSE_MIN_RUN


def test_measured_v28_collapse_is_detected():
    """The exact measured shape: 119,373 consecutive '0' inside one line."""
    assert _detect_char_collapse("0" * 119373, MIN_RUN) is True


def test_fires_early_at_threshold():
    assert _detect_char_collapse("0" * MIN_RUN, MIN_RUN) is True


def test_genuine_reasoning_never_fires():
    # Repetitive but substantive prose: same char diversity as real thinking.
    text = ("I need to verify the grader's construction on the delivered "
            "bytes before I write anything. ") * 60
    assert _detect_char_collapse(text, MIN_RUN) is False


def test_single_byte_chunk_not_falsely_flagged():
    # One small chunk (a delta), not a run — must not fire.
    assert _detect_char_collapse("0" * 256, MIN_RUN) is False


def test_disabled_when_min_run_zero():
    assert _detect_char_collapse("0" * 119373, 0) is False


def test_empty_tail_safe():
    assert _detect_char_collapse("", MIN_RUN) is False


def test_below_threshold_safe():
    assert _detect_char_collapse("0" * (MIN_RUN - 1), MIN_RUN) is False


def test_run_buried_in_text_is_found():
    # The run sits mid-stream: the tail scan must still see it.
    text = "analysis" * 300 + "0" * MIN_RUN
    assert _detect_char_collapse(text, MIN_RUN) is True
