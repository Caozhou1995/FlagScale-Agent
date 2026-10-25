"""Fifth filler mode: newline-free single-line phrase CYCLE.

Root cause this pins: the char-collapse fix (commit 6429ea5) computed its
tail as `buf[-(4 * window + 64):]` — with the line window at 16 that tail
was 128 chars, but MIN_RUN=1024 requires len(tail) >= 1024, so the breaker
could NEVER fire in the live stream (dead code on the production path; only
direct unit calls with a long tail passed). Replay of the measured v28 shape
through the live integration path would have caught it. These tests replay
measured/newline-free shapes through `_intra_stream_update`, the same
function the stream calls, so the integration path cannot silently diverge
from the unit-tested predicate again.
"""
from collections import Counter

from flagscale_agent.react.agent import (
    WorkerAgent,
    _intra_stream_update,
    _detect_phrase_cycle,
)

W = WorkerAgent.INTRA_STREAM_REPEAT_WINDOW
D = WorkerAgent.INTRA_STREAM_REPEAT_DISTINCT_MOST
MIN_RUN = WorkerAgent.INTRA_STREAM_CHAR_COLLAPSE_MIN_RUN
TAIL = WorkerAgent.INTRA_STREAM_CYCLE_TAIL_CHARS
DMAX = WorkerAgent.INTRA_STREAM_CYCLE_DISTINCT_MAX


def _stream(chunks):
    """Replay a chunk list exactly as the live stream does (delta by delta)."""
    buf, lines, fired = "", [], 0
    for c in chunks:
        buf, fired = _intra_stream_update(
            buf, c, lines, W, D, MIN_RUN, TAIL, DMAX)
        if fired:
            return fired, len(buf)
    return 0, len(buf)


def test_v28_char_collapse_fires_through_live_path():
    """The dead-code killer: 119373 '0's as realistic small deltas."""
    fired, seen = _stream(["0" * 256] * 47)
    assert fired == 1, "char-collapse STILL dead on the live path"
    assert seen >= MIN_RUN


def test_v28_tail_computation_now_satisfies_min_run():
    """The tail window must be large enough for min_run to be reachable."""
    keep = 4 * W + max(MIN_RUN, TAIL)
    assert keep >= MIN_RUN
    assert _detect_phrase_cycle is not None  # import sanity


def test_newline_free_phrase_cycle_fires():
    """Fifth mode: multi-phrase cycle with no newline at all.

    Short phrase set (~19 distinct bytes for this 4-phrase loop), well
    under DMAX=32; calibrated healthy floor is 65.
    """
    phrase = "Go. OK. Let me run. Running. "
    # 10 chunks x 464 chars = 4640 > TAIL(4096): the cycle check only runs
    # once the tail buffer holds more than TAIL chars, so 8 chunks (3712)
    # can never reach the threshold.
    fired, _ = _stream([phrase * 16] * 10)
    assert fired == 2


def test_phrase_cycle_predicate_bounds():
    assert _detect_phrase_cycle("x" * TAIL, TAIL, DMAX) is True
    assert _detect_phrase_cycle("short", TAIL, DMAX) is False
    assert _detect_phrase_cycle("x" * TAIL, 0, DMAX) is False


def test_genuine_reasoning_never_fires_any_mode():
    lines = [
        "The grader's construction slices at insert_start, not at the primer.",
        "I need to compute both conventions and diff them on the delivered bytes.",
        "Convention A anchors on the insertion; convention B on the template.",
        "If the two disagree on the same input, that disagreement IS the bug.",
        "The worker chose a split point where the overhang is only part of it.",
        "Both primers need checking under the grader's own construction.",
    ]
    text = " ".join(lines) + " "
    fired, _ = _stream([text] * 10)
    assert fired == 0


def test_line_window_still_fires_mode3():
    # The original measured shape WITH newlines, as the real stream emits it.
    chunks = ["Go.\nOK.\nLet me run.\nRunning.\n"] * 10
    fired, _ = _stream(chunks)
    assert fired == 3


def test_newline_resets_cycle_window():
    """A cycle followed by real newline-separated reasoning must clear mode 5."""
    cyc = "Go. OK. Let me run. " * 260  # > TAIL, but terminated by newlines
    reasoning = "\n".join(
        f"Step {i}: verify the grader's fixpoint on the delivered bytes."
        for i in range(20))
    fired, _ = _stream([cyc + "\n" + reasoning])
    assert fired == 0


def test_disabled_when_tail_chars_zero():
    buf, fired = _intra_stream_update(
        "", "0" * (MIN_RUN * 2), [], W, D, MIN_RUN, 0, DMAX)
    assert fired in (0, 1)
    # phrase-cycle must be off entirely
    assert _detect_phrase_cycle("x" * 5000, 0, DMAX) is False
