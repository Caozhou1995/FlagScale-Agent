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

class TestForceToolKeepsABoundedWallCap:
    """2026-11-07 fix: force_tool must NOT disable the wall cap outright.

    Measured loss that motivated this (filter-js-from-html, 2026-11-06, trial
    DBqayX4, reward 0): one saturated step emitted 65536 tokens (===
    max_output_tokens) in 808.4s = 44.9% of the task's 1799s budget, and the
    turn then burned 8 more zero-token steps to the wall clock, because
    `_wd_limit` was 0 for the whole endgame (force_tool armed). The exemption
    exists for a real reason — a forced call still thinks first, and the 240s
    cap aborted it before the mandatory tool call was reached (round 5,
    regex-chess gfpT6ap) — so the fix keeps a REAL limit, just a wider one
    (2x, floor 90s) instead of infinity.

    Every test here is written so it FAILS on the pre-fix source
    (`_wd_limit = 0 if force_tool ...`); a guard that stays green on the bug
    it names is not a guard (verified by reverting the source and re-running).
    """

    def _force_branch_src(self):
        """The source text of the force_tool ELSE-branch of the limit derivation.

        Located structurally (from the `else:` that owns the force_tool term to
        the end of that statement) rather than by a fixed character window: the
        branch carries a long explanatory comment between `else:` and
        `_wd_limit =`, so a byte-offset slice would silently miss the code.
        """
        import inspect
        from flagscale_agent.react.agent import WorkerAgent
        src = inspect.getsource(WorkerAgent._call_llm_stream)
        i = src.find("if force_tool else self.IN_STREAM_THINKING_WALL_SEC")
        assert i > 0, "the force_tool limit branch is missing entirely"
        # Back up to the start of the assignment statement so the derivation
        # (`max(90, ... * 2)`) is inside the slice, not just its tail.
        lo = src.rfind("_wd_limit", 0, i)
        assert lo > 0, "could not locate the start of the _wd_limit statement"
        return src[lo:i + 200]

    def test_force_tool_branch_yields_a_positive_limit(self):
        """The branch must compute a POSITIVE limit when force_tool is armed.

        Pre-fix this literal is `_wd_limit = 0 if force_tool ...` -> 0, which
        the watchdog reads as 'disabled' and returns on immediately.
        """
        branch = self._force_branch_src()
        assert "_wd_limit = 0 if force_tool" not in branch, (
            "force_tool must not disable the wall cap: 0 makes the watchdog "
            "return immediately, which is the filter-js 808.4s loss")
        assert "if force_tool else self.IN_STREAM_THINKING_WALL_SEC" in branch, (
            "force_tool must fall through to a real computed limit")

    def test_force_tool_limit_scales_with_the_normal_cap(self):
        """The force_tool limit is derived FROM the normal cap (2x, floor 90).

        Asserting the derivation (not a hardcoded number) keeps this true if
        IN_STREAM_THINKING_WALL_SEC is retuned, and it fails on the pre-fix
        source because that source contains no `* 2` derivation at all.
        """
        branch = self._force_branch_src()
        assert "IN_STREAM_THINKING_WALL_SEC * 2" in branch, (
            "the force_tool limit must be a wider multiple of the normal cap; "
            "a bare 0 is the bug this test guards")
        assert "max(90" in branch, (
            "the force_tool limit needs a floor so a small configured cap does "
            "not make the exemption narrower than the normal cap")

    def test_watchdog_still_treats_zero_as_disabled(self):
        """Pins the MECHANISM: 0 means 'off', which is why 0 was fatal."""
        import inspect
        from flagscale_agent.react.agent import WorkerAgent
        src = inspect.getsource(WorkerAgent._call_llm_stream)
        i = src.find("def _silence_watchdog")
        assert i > 0
        body = src[i:i + 400]
        assert "if limit <= 0:" in body and "return" in body, (
            "the watchdog returns immediately when the limit is 0 — that is "
            "the whole reason an unbounded exemption let a turn burn the wall")
