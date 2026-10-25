# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the in-stream pure-thinking wall-clock cap (agent._call_llm_stream).

Measured failure this closes (v12.1 gpt2__o3NAGa9, Turn1 781.7s): the GLM
Anthropic-compat endpoint serves ~4.6 tok/s, so the 16384-token client-side
cap corresponds to ~2000s+ of wall clock — no TOKEN-based cap can fire inside
a 900s budget. One slow thinking stream burned 405.8s (45% of the turn) and
the turn ended honest-BLOCKED with a 1200-byte placeholder. The cap must be
judged in TIME: a stream that has emitted pure thinking for
IN_STREAM_THINKING_WALL_SEC seconds with no visible output is aborted with
the same "capped" convergence semantics.

Contract:
  - thinking deltas accumulating past the wall cap -> thinking_capped=True,
    stream aborted (remaining events never consumed)
  - below the cap -> stream finishes normally
  - wall cap disabled at 0
  - once text has appeared, the wall cap never fires (pure-thinking only)
"""

import threading
import time
import types
from unittest.mock import MagicMock, patch

from flagscale_agent.react.agent import WorkerAgent


def _fake_agent(wall_sec, thinking_budget=0):
    agent = WorkerAgent.__new__(WorkerAgent)
    agent.config = types.SimpleNamespace(thinking_budget=thinking_budget)
    agent.IN_STREAM_THINKING_WALL_SEC = wall_sec
    return agent


def _stream_events(thinking_deltas, tail=None):
    events = [{"type": "thinking_start"}]
    for d in thinking_deltas:
        events.append({"type": "thinking", "content": d})
    events.append({"type": "signature", "content": "sig"})
    events.extend(tail or [{"type": "text", "content": "answer"},
                           {"type": "done"}])
    return iter(events)


class TestInStreamThinkingWallCap:
    def test_short_thinking_finishes_normally(self):
        """Fast clock (1s/delta, 10s total << 240s) -> no abort, answer kept."""
        agent = _fake_agent(wall_sec=240)
        deltas = ["Reasoning step %d. " % i for i in range(10)]
        agent.provider = MagicMock()
        agent.provider.chat_stream.side_effect = (
            lambda m, s: _stream_events(deltas))

        def fast_time():
            fast_time.t += 1.0
            return fast_time.t
        fast_time.t = 0.0

        with patch("flagscale_agent.react.agent.time.time", fast_time):
            resp, usage = agent._call_llm_stream([], [])
        assert resp["content"] == "answer"
        assert resp["thinking_capped"] is False

    def test_slow_stream_aborted_at_wall_cap(self):
        """100s/delta clock: the 3rd delta crosses the 240s cap -> abort with
        capped semantics; the tail events are never consumed."""
        agent = _fake_agent(wall_sec=240)
        deltas = ["a" * 30, "b" * 30, "c" * 30, "d" * 30]
        agent.provider = MagicMock()
        agent.provider.chat_stream.side_effect = (
            lambda m, s: _stream_events(deltas))

        def slow_time():
            slow_time.t += 100.0
            return slow_time.t
        slow_time.t = 0.0

        with patch("flagscale_agent.react.agent.time.time", slow_time):
            resp, usage = agent._call_llm_stream([], [])
        assert resp["thinking_capped"] is True
        assert resp["reasoning_only"] is True
        assert resp["content"] is None
        # The wall clock now starts at STREAM CREATION, not at the first
        # thinking delta (a stream that never emits a delta, or whose
        # time-to-first-token alone exceeds the cap, must still be aborted —
        # measured: 325s of socket silence with zero deltas). The reference
        # point therefore moves earlier, so with a 100s/read fake clock the cap
        # trips on the first delta rather than the third. What the contract
        # pins is the SEMANTICS (capped + reasoning_only + no content + a
        # non-empty preserved prefix of the thinking), not the delta index.
        absorbed = resp["thinking"]
        full = "".join(deltas)
        assert absorbed, "the tripping stream's reasoning must be preserved"
        assert full.startswith(absorbed), (
            "preserved reasoning must be a prefix of the stream's thinking")

    def test_wall_cap_disabled_when_zero(self):
        """0 disables the wall cap entirely; the stream completes."""
        agent = _fake_agent(wall_sec=0)
        deltas = ["a" * 30, "b" * 30]
        agent.provider = MagicMock()
        agent.provider.chat_stream.side_effect = (
            lambda m, s: _stream_events(deltas))

        def slow_time():
            slow_time.t += 100.0
            return slow_time.t
        slow_time.t = 0.0

        with patch("flagscale_agent.react.agent.time.time", slow_time):
            resp, usage = agent._call_llm_stream([], [])
        assert resp["content"] == "answer"
        assert resp["thinking_capped"] is False

    def test_no_wall_cap_once_text_started(self):
        """Text-first responses are never wall-capped (visible output exists)."""
        agent = _fake_agent(wall_sec=240)
        events = [{"type": "text", "content": "answer"},
                  {"type": "thinking", "content": "a" * 30},
                  {"type": "done"}]
        agent.provider = MagicMock()
        agent.provider.chat_stream.side_effect = lambda m, s: iter(events)

        def slow_time():
            slow_time.t += 100.0
            return slow_time.t
        slow_time.t = 0.0

        with patch("flagscale_agent.react.agent.time.time", slow_time):
            resp, usage = agent._call_llm_stream([], [])
        assert resp["content"] == "answer"
        assert resp["thinking_capped"] is False


class TestSilentStreamWallCap:
    """The measured failure mode: NO thinking delta arrives at all.

    v12.2 write-compressor (2026-10-26): the last delta landed at 20:05:38 and
    the next byte at 20:11:03 — 325s of socket silence. Because the provider's
    `chat_stream` is a blocking read, the per-delta wall-clock check never ran,
    so the 240s cap could not fire and the run burned its budget. The cap is
    therefore also evaluated at the TOP of the event loop, and its clock starts
    at stream creation rather than at the first delta.
    """

    def test_silent_stream_aborts_on_wall_clock(self):
        """Only keep-alive/empty events; the cap must still abort the stream."""
        agent = _fake_agent(wall_sec=240)
        # usage-only heartbeats: no thinking, no text, no tool calls
        events = [{"type": "usage", "input_tokens": 0, "output_tokens": 0}
                  for _ in range(10)]
        events.append({"type": "text", "content": "answer"})
        agent.provider = MagicMock()
        agent.provider.chat_stream.side_effect = lambda m, s: iter(events)

        def slow_time():
            slow_time.t += 100.0
            return slow_time.t
        slow_time.t = 0.0

        with patch("flagscale_agent.react.agent.time.time", slow_time):
            resp, usage = agent._call_llm_stream([], [])
        assert resp["thinking_capped"] is True, (
            "a silent thinking stream must trip the wall-clock cap")
        assert resp["content"] is None, (
            "the stream must be aborted before the trailing answer event")

    def test_silent_stream_below_cap_is_not_aborted(self):
        """Below the cap, silence alone must NOT abort a stream."""
        agent = _fake_agent(wall_sec=240)
        events = [{"type": "usage", "input_tokens": 0, "output_tokens": 0}
                  for _ in range(3)]
        events.append({"type": "text", "content": "answer"})
        agent.provider = MagicMock()
        agent.provider.chat_stream.side_effect = lambda m, s: iter(events)

        def slow_time():
            slow_time.t += 10.0
            return slow_time.t
        slow_time.t = 0.0

        with patch("flagscale_agent.react.agent.time.time", slow_time):
            resp, usage = agent._call_llm_stream([], [])
        assert resp["content"] == "answer"
        assert resp["thinking_capped"] is False


class TestSilentSocketWatchdog:
    """A stream that yields NO events at all must still be aborted.

    This is the shape the event-loop check cannot cover: the SSE decoder drops
    keep-alive/comment lines, so a silent socket produces zero events and the
    `for event in stream` body never executes. A watchdog thread closes the
    stream instead.
    """

    def test_watchdog_closes_a_never_yielding_stream(self):
        agent = _fake_agent(wall_sec=0.3)  # tiny cap keeps the test fast
        closed = {"n": 0}

        class _SilentStream:
            """Emulates a blocking socket read that a close() can interrupt."""

            def __init__(self):
                self._closed = threading.Event()

            def __iter__(self):
                return self

            def __next__(self):
                # Block like a socket waiting for bytes; close() wakes it the
                # way httpx's response.close() raises on a pending read.
                self._closed.wait(5)
                raise StopIteration

            def close(self):
                closed["n"] += 1
                self._closed.set()

        agent.provider = MagicMock()
        agent.provider.chat_stream.side_effect = lambda m, s: _SilentStream()

        started = time.time()
        resp, usage = agent._call_llm_stream([], [])
        elapsed = time.time() - started

        assert closed["n"] >= 1, "watchdog must close the silent stream"
        assert elapsed < 4.5, (
            f"the call must return shortly after the cap, took {elapsed:.1f}s")
        assert resp["thinking_capped"] is True
        assert resp["reasoning_only"] is True
