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

"""Tests for the cumulative idle think-interval gate (kernel.run_turn).

Measured failure this closes (write-compressor, v11 job 02-33-28 and
v12 03-12-08): 73% of the log / ~620s of wall clock were spent in SIX
separate think round-trips, each under the 16384-token cap, with ZERO tool
calls between them — first write at 76% of the budget. The convergence
machinery is keyed on one stream's TOKEN size; the guards only evaluate on
tool calls. So six individually-legal streams bypass every threshold.

Behavior contract:
  - the 3rd+ consecutive think round-trip once >= IDLE_THINK_INTERVAL_SEC of
    wall-clock has accumulated since the last tool call -> idle-interval
    branch: an "Idle think-interval" convergence nudge is injected and the
    cap-retry counter (not the 3-strike empty budget) is charged
  - below either bound (fewer rounds OR less time) -> gate silent, resume
    nudge unchanged
  - a tool call resets both the clock and the round-trip count
  - 50 consecutive idle trips stop the turn with the gate's OWN stop reason
"""

import types
from unittest.mock import MagicMock, patch

from flagscale_agent.react.kernel import AgentKernel, KernelDeps
from flagscale_agent.react.history import HistoryManager
from flagscale_agent.react.guard import GuardRegistry


class _Harness:
    """Real HistoryManager driven through AgentKernel.run_turn."""

    def __init__(self):
        self.history = HistoryManager(max_context_tokens=64000)
        self.history.append({"role": "user", "content": "do the task"})
        self.registry = GuardRegistry()
        self.config = types.SimpleNamespace(
            max_iterations=20, max_continuations=5, mode="auto", _turn_count=0,
        )
        self.call_count = {"n": 0}

    def build(self, llm_responses):
        def call_llm_fn(messages, schemas):
            idx = min(self.call_count["n"], len(llm_responses) - 1)
            self.call_count["n"] += 1
            return llm_responses[idx], {"input_tokens": 1, "output_tokens": 1}

        provider = MagicMock()
        provider.format_assistant_message.side_effect = lambda resp: {
            "role": "assistant", "content": resp.get("content") or "",
        }

        deps = KernelDeps(
            provider=provider,
            history=self.history,
            tool_registry=MagicMock(),
            judge=MagicMock(),
            guard_registry=self.registry,
            config=self.config,
            display=MagicMock(),
            get_schemas_fn=lambda: [],
            inject_message_fn=lambda msg: self.history.append(
                {"role": "user", "content": msg}),
            append_tool_results_fn=lambda results: None,
            format_tool_result_fn=lambda tid, r: {},
            execute_tools_fn=lambda tcs: ["ok"] * len(tcs),
            is_context_limit_error_fn=lambda e: False,
            call_llm_fn=call_llm_fn,
        )
        return AgentKernel(deps)

    def last_user_message(self):
        for m in reversed(self.history.messages):
            if m.get("role") == "user":
                return m["content"]
        return None


_THINK = {"content": None, "tool_calls": None, "thinking": "still thinking",
          "reasoning_only": True}
_DONE = {"content": "[TASK_COMPLETE]", "tool_calls": []}


class _SteppingClock:
    """time.time() replacement that advances `step` seconds per call, so the
    gate's elapsed-time arithmetic sees real passage (a constant mock yields
    _idle_secs == 0 forever and the gate can never trip)."""

    def __init__(self, start=0.0, step=100.0):
        self.t = start
        self.step = step

    def __call__(self):
        self.t += self.step
        return self.t


class TestIdleThinkIntervalGate:
    def test_three_think_roundtrips_trip_the_gate(self):
        """3 consecutive think round-trips (fake clock past the threshold) ->
        idle branch: 'Idle think-interval' nudge injected, cap counter charged,
        empty-output budget untouched, turn completes on the canned answer."""
        h = _Harness()
        kernel = h.build([_THINK] * 3 + [_DONE])
        kernel.IDLE_THINK_INTERVAL_SEC = 1
        clock = _SteppingClock(step=100.0)
        with patch("flagscale_agent.react.kernel.time.time", clock):
            result = kernel.run_turn()
        assert result.stop_reason == "explicit_signal"
        assert kernel._cap_retries == 0  # reset on final success
        # The first two legal reasoning-only retries consumed the empty budget
        # (2/3), but the idle trip itself must NOT touch it.
        assert getattr(kernel, "_empty_output_retries", 0) < 3
        nudges = [m for m in h.history.messages
                  if m.get("role") == "user"
                  and "Idle think-interval" in (m.get("content") or "")]
        assert len(nudges) == 1  # trips exactly on the 3rd round-trip
        assert "CUMULATIVE TIME" in nudges[0]["content"]
        assert "Land the next concrete step NOW" in nudges[0]["content"]
        assert "<previous_reasoning>" in nudges[0]["content"]

    def test_first_two_roundtrips_stay_silent(self):
        """Below 3 round-trips -> no idle nudge; plain resume nudge instead."""
        h = _Harness()
        kernel = h.build([_THINK] * 2 + [_DONE])
        kernel.IDLE_THINK_INTERVAL_SEC = 1
        clock = _SteppingClock(step=100.0)
        with patch("flagscale_agent.react.kernel.time.time", clock):
            result = kernel.run_turn()
        assert result.stop_reason == "explicit_signal"
        assert all("Idle think-interval" not in (m.get("content") or "")
                   for m in h.history.messages)
        # Resume nudge still present (uncapped reasoning-only path unchanged)
        assert any("resume from where it left off" in (m.get("content") or "")
                   for m in h.history.messages if m.get("role") == "user")

    def test_tool_call_resets_clock_and_rounds(self):
        """A think burst + tool run + think burst: the tool restarts both
        counters, so the second burst never reaches 3 consecutive rounds."""
        h = _Harness()
        kernel = h.build([
            _THINK, _THINK,
            {"content": None,
             "tool_calls": [{"id": "t1", "name": "shell",
                             "arguments": {"command": "ls"}}]},
            _THINK,
        ] + [_DONE])
        kernel.IDLE_THINK_INTERVAL_SEC = 1
        clock = _SteppingClock(step=100.0)
        with patch("flagscale_agent.react.kernel.time.time", clock):
            result = kernel.run_turn()
        assert result.stop_reason == "explicit_signal"
        # After the tool ran, the trailing single burst is rounds=1 -> silent
        assert all("Idle think-interval" not in (m.get("content") or "")
                   for m in h.history.messages)
        assert kernel._idle_think_rounds == 1

    def test_idle_exhaustion_stops_turn_with_own_reason(self):
        """White-box (per test_thinking_cap's cap_retry_sanity pattern):
        pre-set the cap counter to its exhaustion bound and one more idle
        trip -> the turn stops with the gate's OWN stop reason, never the
        empty-output budget."""
        h = _Harness()
        kernel = h.build([_THINK])
        kernel.IDLE_THINK_INTERVAL_SEC = 1
        clock = _SteppingClock(step=100.0)
        with patch("flagscale_agent.react.kernel.time.time", clock):
            pass  # construct inside the patched clock so _turn_start_ts uses it
        with patch("flagscale_agent.react.kernel.time.time", clock):
            kernel._cap_retries = 50  # white-box: 50 prior idle trips
            result = kernel.run_turn()
        assert result.stop_reason == "idle_interval_max_retries"
        assert kernel._cap_retries == 0  # reset on exhaust (next turn fresh)
        # The gate path must not consume the empty-output budget; whatever the
        # canned reasoning-only retries consumed before the white-box trip is
        # pre-existing behavior, not gate consumption (< 3 = turn not ended by
        # that budget).
        assert getattr(kernel, "_empty_output_retries", 0) < 3
