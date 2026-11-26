"""Non-blocking blind-spot read + compatibility + in-conversation announce.

MEASURED CONTEXT (2026-11-24, dna-assembly job 15-11-29): the blind-spot read is
one forced-thinking LLM call with HEAVY-TAILED latency — the same prompt on the
same endpoint measured 254.8s once and >510s on another run. Blocking the run's
start on it produced `rounds=1 timed_out=1 lines=0` twice in production, i.e.
the region was silently absent.

Two further requirements from the user, both encoded here:
  (1) COMPATIBILITY — not every run enables the task reader; a run that did not
      opt in must be a byte-identical no-op (the poll never adopts, never
      inserts a message).
  (2) ANNOUNCE IN THE CONVERSATION — a resident system-prompt region is easy to
      scroll past, so when the region lands a short user turn must say so.
"""

from __future__ import annotations

import time

from flagscale_agent.react.agent import WorkerAgent


class _SlowReaders:
    """Stands in for TaskStatementReadersTool with a controllable delay."""

    def __init__(self, delay: float, body: str):
        self.delay = delay
        self.body = body
        self._task_text = ""

    def read_blindspots(self, rounds: int = 1):
        time.sleep(self.delay)
        return [self.body]


def _stub(delay=0.0, body="BOUNDARY: inclusive or exclusive?", enabled=True):
    class _A:
        _BLINDSPOT_ROUNDS = 1

        def _readers_enabled(self):
            return enabled

    a = _A()
    a._task_statement_readers = _SlowReaders(delay, body)
    a._auto_read_blindspots = WorkerAgent._auto_read_blindspots.__get__(a)
    a._render_blindspot_region = WorkerAgent._render_blindspot_region
    a._poll_blindspot_ready = WorkerAgent._poll_blindspot_ready.__get__(a)
    a._blindspot_block = ""
    a._blindspot_thread_result = None
    a._blindspot_thread_done = False
    a.history = []
    return a


def _wait_done(a, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not a._blindspot_thread_done and time.monotonic() < deadline:
        time.sleep(0.05)
    assert a._blindspot_thread_done, "background read never finished"


def test_background_spawn_returns_immediately():
    a = _stub(delay=2.0)
    t0 = time.monotonic()
    region = a._auto_read_blindspots("some task text", background=True)
    elapsed = time.monotonic() - t0
    assert region == "", "background spawn must return an empty region now"
    assert elapsed < 1.0, f"background spawn blocked for {elapsed:.2f}s"
    assert a._blindspot_thread_done is False


def test_result_adopted_and_announced_in_conversation():
    a = _stub(delay=0.2)
    a._auto_read_blindspots("some task text", background=True)
    assert a._poll_blindspot_ready() is False, "not ready yet"

    _wait_done(a)
    assert a._poll_blindspot_ready() is True, "region must be adopted"
    assert "BLIND SPOTS IN THIS TASK" in a._blindspot_block
    # (2) the announcement must be a real user turn mentioning the reader
    msgs = [m for m in a.history if m.get("role") == "user"]
    assert len(msgs) == 1, f"expected one announcement turn, got {len(msgs)}"
    assert "[blind-spot reader]" in msgs[0]["content"]
    assert "BLIND SPOTS IN THIS TASK" in msgs[0]["content"]
    # Idempotent: a second poll must not re-adopt nor re-announce.
    assert a._poll_blindspot_ready() is False
    assert len([m for m in a.history if m.get("role") == "user"]) == 1


def test_reader_disabled_is_a_byte_identical_no_op():
    """(1) COMPATIBILITY: no reader opt-in => never adopt, never insert."""
    a = _stub(enabled=False)
    assert a._auto_read_blindspots("text", background=True) == ""
    assert a._blindspot_thread_done is False, "must not even spawn"
    assert a._poll_blindspot_ready() is False
    assert a.history == [], "a non-opted-in run must get no extra message"
    assert a._blindspot_block == ""


def test_poll_is_noop_when_no_background_read_was_ever_started():
    """Compat: attribute present but read never spawned (e.g. no task text)."""
    a = _stub(enabled=True)
    assert a._poll_blindspot_ready() is False
    assert a.history == []


def test_foreground_mode_still_returns_the_region():
    a = _stub(delay=0.0, body="UNIT: nt vs bp?")
    region = a._auto_read_blindspots("some task text", background=False)
    assert "BLIND SPOTS IN THIS TASK" in region
    assert "UNIT: nt vs bp?" in region
    assert a.history == [], "foreground mode does not announce"


def test_failing_reader_does_not_take_the_run_down():
    class _Boom:
        def __init__(self):
            self._task_text = ""

        def read_blindspots(self, rounds: int = 1):
            raise RuntimeError("provider exploded")

    a = _stub()
    a._task_statement_readers = _Boom()
    a._auto_read_blindspots("task", background=True)
    _wait_done(a)
    assert a._poll_blindspot_ready() is False, "no region from a failed read"
    assert a.history == []
