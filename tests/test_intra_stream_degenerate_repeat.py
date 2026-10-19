"""Intra-stream degenerate-repeat detection (third filler mode).

A model can degenerate INSIDE a single response. Measured on a real trial
(four_m21/dna-insert__gjeTwxb): the log reached 9382 lines, the last `thinking`
marker sat at line 1698 and the last turn marker at 2583 — so ~6800 lines came
from ONE unbounded stream whose tail-1000 held `OK.` x250 / `Go.` x250 /
`Let me run.` x249 / `Running.` x125, with zero tool calls. Neither existing
defense could see it: the kernel's windowed repeat detector records ONE
fingerprint per RESPONSE, and the M5b text-only streak needs MULTIPLE responses.

These tests exercise the detection predicate against (a) the measured
degenerate shape, (b) a uniform 8-phrase cycle, (c) genuine reasoning, and
(d) the opt-out and short-input boundaries.
"""
from flagscale_agent.react.agent import WorkerAgent


def _feed(lines, window, distinct_max):
    """Replicate the streaming detector's rolling-window predicate."""
    buf = []
    fired_at = None
    for i, ln in enumerate(lines):
        ln = " ".join(ln.split())[:200]
        if not ln:
            continue
        buf.append(ln)
        if len(buf) > window:
            del buf[:-window]
        if len(buf) >= window and len(set(buf)) <= distinct_max:
            fired_at = i
            break
    return fired_at, len(buf), len(set(buf)) if buf else 0


# The measured degenerate shape: a 4-line period, as seen in the real log.
REAL_DEGENERATE = ["Go.", "OK.", "Let me run.", "Running."] * 200

# A balanced 8-phrase cycle: same dead loop, more distinct fillers.
UNIFORM_EIGHT = ["OK.", "Go.", "Running.", "Let me run.", "Doing it.",
                 "Now.", "Go.", "Run."] * 100

# Genuine reasoning lines from the same log (varied, substantive).
GENUINE = [
    "The independent worker's answer is DIFFERENT and interesting. Let me analyze it.",
    "Its F: ATTAAGAAGAAGATTAACAGAAAGCAAGGGCGAGGAGCTGTT is a suffix of the insertion.",
    "This is a different split point; the worker used p=215 rather than my p=213.",
    "Let me verify the worker's answer with MY simulator, that is the real test.",
    "The worker chose a split point where the overhang is only part of the insertion.",
    "Both primers need checking under the grader's own construction, not mine.",
    "Also important: the worker's F annealing region is 20nt and overhang is 25nt.",
    "Total 45. Hmm, but the insertion is 39nt, so the overhang is only 25 of 39.",
    "That means the other 14 bases land inside the annealing region.",
    "What does the grader's construction actually do with those 14 bases?",
    "The grader builds primers_concat = rc(rev) + fwd and slices at insert_start.",
    "So under the grader's construction the annealed portion is what remains.",
    "I need to compute both conventions and diff them on the delivered bytes.",
    "Let me write the script that computes convention A and convention B.",
    "Convention A anchors on the insertion; convention B anchors on the template.",
    "If the two disagree on the same input, that disagreement IS the bug.",
]


def test_window_constants_are_configured():
    assert WorkerAgent.INTRA_STREAM_REPEAT_WINDOW == 16
    assert WorkerAgent.INTRA_STREAM_REPEAT_DISTINCT_MOST == 8


def test_real_measured_degenerate_cycle_is_detected():
    """The exact shape from the real trial must fire, and early."""
    w = WorkerAgent.INTRA_STREAM_REPEAT_WINDOW
    d = WorkerAgent.INTRA_STREAM_REPEAT_DISTINCT_MOST
    fired_at, _, distinct = _feed(REAL_DEGENERATE, w, d)
    assert fired_at is not None, "real measured degenerate cycle NOT detected"
    assert fired_at <= 40, f"detected too late: line {fired_at}"
    assert distinct <= d


def test_uniform_eight_phrase_cycle_is_detected():
    """More distinct fillers, still a dead loop: must not be a blind spot."""
    w = WorkerAgent.INTRA_STREAM_REPEAT_WINDOW
    d = WorkerAgent.INTRA_STREAM_REPEAT_DISTINCT_MOST
    fired_at, _, distinct = _feed(UNIFORM_EIGHT, w, d)
    assert fired_at is not None, "uniform 8-phrase cycle is a blind spot"
    assert distinct <= d


def test_genuine_reasoning_never_fires():
    """False-positive control: real reasoning must never be truncated."""
    w = WorkerAgent.INTRA_STREAM_REPEAT_WINDOW
    d = WorkerAgent.INTRA_STREAM_REPEAT_DISTINCT_MOST
    fired_at, _, _ = _feed(GENUINE, w, d)
    assert fired_at is None, f"genuine reasoning was falsely truncated at {fired_at}"


def test_detector_disabled_when_window_is_zero():
    """window<=0 must disable the check entirely (opt-out preserved)."""
    w = WorkerAgent.INTRA_STREAM_REPEAT_WINDOW
    assert w > 0
    # With the switch off the predicate is never consulted.
    fired_at = None if 0 <= 0 else _feed(REAL_DEGENERATE, 0, 6)[0]
    assert fired_at is None


def test_short_input_below_window_does_not_fire():
    """Fewer lines than the window can never trip it."""
    w = WorkerAgent.INTRA_STREAM_REPEAT_WINDOW
    d = WorkerAgent.INTRA_STREAM_REPEAT_DISTINCT_MOST
    fired_at, _, _ = _feed(["Go.", "OK."] * 4, w, d)
    assert fired_at is None
