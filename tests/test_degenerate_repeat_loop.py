"""Falsifiable tests for the degenerate-repeat loop breaker (window-diversity).

REAL-WORLD ORIGIN (not invented): a live trial (four_m19 dna-insert__mzCAg2n)
regenerated a PERIOD-4 filler cycle — "Let me write the file." / "Let me write
the content." / "Let me write it." / "OK." — 405 times, 1600+ log lines, ZERO
tool calls, and no deliverable ever written to disk.

IMPORTANT LESSON BAKED INTO THIS FILE: the first version of the detector
compared ADJACENT responses. Because a period-4 cycle never repeats back to
back, it never fired — and the replica test (which fed a period-1 repeat) passed
anyway, until the end-to-end test caught the gap. So the rule is now DIVERSITY
based and the tests below feed the real PERIODIC sequence, never a period-1
stand-in.

Two directions matter equally:
  * a real loop MUST be stopped (else the trial burns the wall clock), and
  * distinct work MUST NOT be stopped (else a healthy agent is killed) — this
    is the falsifier for the first.
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.kernel import AgentKernel as Kernel


# The exact filler cycle observed on the real trial.
_FILLER_CYCLE = [
    "Let me write the file.",
    "Let me write the content.",
    "Let me write it.",
    "OK.",
]


class _Window:
    """Replica of the kernel's window-diversity rule (kept in lock-step)."""

    def __init__(self, n: int = Kernel.DEGENERATE_REPEAT_WINDOW_CALLS,
                 distinct_most: int = Kernel.DEGENERATE_REPEAT_DISTINCT_MOST):
        self.n = n
        self.distinct_most = distinct_most
        self.win: list[str] = []

    def feed(self, text: str) -> bool:
        src = (text or "").strip()
        fp = " ".join(src[-Kernel.DEGENERATE_REPEAT_WINDOW:].split())
        if fp:
            self.win.append(fp)
        if len(self.win) > self.n:
            del self.win[:-self.n]
        return self.n > 0 and len(self.win) >= self.n and len(set(self.win)) <= self.distinct_most

    def reset(self):
        self.win = []


def test_thresholds_are_configured():
    assert Kernel.DEGENERATE_REPEAT_WINDOW_CALLS > 0
    assert Kernel.DEGENERATE_REPEAT_DISTINCT_MOST > 0
    assert Kernel.DEGENERATE_REPEAT_DISTINCT_MOST < Kernel.DEGENERATE_REPEAT_WINDOW_CALLS


def test_real_period4_cycle_is_stopped():
    """The EXACT period-4 loop must trip the breaker (adjacency could not)."""
    w = _Window()
    tripped = None
    for i in range(40):
        if w.feed(_FILLER_CYCLE[i % len(_FILLER_CYCLE)]):
            tripped = i + 1
            break
    assert tripped is not None, (
        "the real period-4 filler loop was NOT stopped — the breaker is dead"
    )
    assert tripped == Kernel.DEGENERATE_REPEAT_WINDOW_CALLS


def test_period1_repeat_is_also_stopped():
    w = _Window()
    tripped = None
    for i in range(40):
        if w.feed("OK."):
            tripped = i + 1
            break
    assert tripped is not None


def test_distinct_work_never_trips():
    """FALSIFIER: a healthy agent emitting varied text must never be killed."""
    w = _Window()
    for _ in range(5):
        for i in range(Kernel.DEGENERATE_REPEAT_WINDOW_CALLS + 4):
            assert not w.feed(
                f"Step {i}: inspected artifact {i * 7 % 13}, result {i * 3 % 5}."
            ), "false positive on distinct work"


def test_tool_call_resets_the_window():
    """A real action clears the loop state — progress is not a loop."""
    w = _Window()
    for i in range(Kernel.DEGENERATE_REPEAT_WINDOW_CALLS - 1):
        w.feed(_FILLER_CYCLE[i % len(_FILLER_CYCLE)])
    w.reset()                        # tool executed
    assert not w.feed("Let me write it.")
    assert len(w.win) == 1


def test_whitespace_variation_still_counts_as_the_same_loop():
    fp = _Window()
    variants = ["Let me write it.", "Let  me write\nit.", "Let me write it.  "]
    tripped = False
    for i in range(Kernel.DEGENERATE_REPEAT_WINDOW_CALLS + 2):
        if fp.feed(variants[i % len(variants)]):
            tripped = True
            break
    assert tripped, "whitespace-only variation escaped the breaker"
