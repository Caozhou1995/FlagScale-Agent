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

"""Tests for the kernel's hard per-turn wall-clock timeout.

The kernel stops the turn BEFORE the next LLM call when the injected budget
stats report remaining <= 0, prints a timeout notice, and sets
stop_reason="time_budget_exhausted". When no stats fn is injected (or it
returns None / raises), the loop runs normally — the hard stop must never
fire on a run that has no real deadline.
"""

from unittest.mock import MagicMock
import types

from flagscale_agent.react.kernel import AgentKernel, KernelDeps
from flagscale_agent.react.guard import GuardRegistry


class _History:
    def __init__(self):
        self.messages = [{"role": "user", "content": "do the task"}]

    def append(self, msg):
        self.messages.append(msg)

    def get_messages(self):
        return self.messages

    def get_context_pressure(self):
        return 0.1

    def get_evictable_indexes(self):
        return []

    def report_actual_tokens(self, n):
        pass


def _make_kernel(stats_fn):
    """Build a kernel with an injectable budget stats fn.

    Returns (kernel, call_count) where call_count["n"] counts LLM calls so a
    test can assert the loop stopped BEFORE calling the LLM.
    """
    history = _History()
    config = types.SimpleNamespace(max_iterations=50, max_continuations=200, mode="auto")
    call_count = {"n": 0}

    def call_llm_fn(messages, schemas):
        call_count["n"] += 1
        return (
            {"content": "working", "tool_calls": [{"id": "1", "name": "shell", "arguments": {}}]},
            {"input_tokens": 1, "output_tokens": 1},
        )

    provider = MagicMock()
    provider.format_assistant_message.side_effect = (
        lambda r: {"role": "assistant", "content": r.get("content", "")}
    )

    deps = KernelDeps(
        provider=provider,
        history=history,
        tool_registry=MagicMock(),
        judge=MagicMock(),
        guard_registry=GuardRegistry(),
        config=config,
        display=MagicMock(),
        get_schemas_fn=lambda: [],
        inject_message_fn=lambda m: None,
        append_tool_results_fn=lambda r: None,
        format_tool_result_fn=lambda t, r: {},
        execute_tools_fn=lambda tcs: ["ok"] * len(tcs),
        is_context_limit_error_fn=lambda e: False,
        call_llm_fn=call_llm_fn,
        time_budget_stats_fn=stats_fn,
    )
    return AgentKernel(deps), call_count


def test_budget_exhausted_stops_before_llm():
    """remaining <= 0 -> break with time_budget_exhausted, zero LLM calls."""
    kernel, cc = _make_kernel(
        lambda: {"elapsed": 1800, "budget": 1800, "remaining": 0.0, "pct": 100.0}
    )
    result = kernel.run_turn()
    assert result.stop_reason == "time_budget_exhausted"
    assert cc["n"] == 0


def test_budget_negative_remaining_stops():
    kernel, cc = _make_kernel(
        lambda: {"elapsed": 1900, "budget": 1800, "remaining": -100.0, "pct": 105.0}
    )
    result = kernel.run_turn()
    assert result.stop_reason == "time_budget_exhausted"
    assert cc["n"] == 0


def test_budget_exhausted_prints_timeout_notice():
    """The timeout path must emit a visible timeout message via display.warn."""
    kernel, _ = _make_kernel(
        lambda: {"elapsed": 1800, "budget": 1800, "remaining": 0.0, "pct": 100.0}
    )
    # capture display.warn calls (display is the real module on deps)
    import flagscale_agent.react.display as display_mod
    calls = []
    orig = display_mod.warn
    display_mod.warn = lambda msg: calls.append(msg)
    try:
        kernel.run_turn()
    finally:
        display_mod.warn = orig
    assert any("TIME BUDGET EXHAUSTED" in c for c in calls), calls


def test_no_stats_fn_runs_normally():
    """No injected budget -> the loop runs (LLM is called), no hard stop."""
    kernel, cc = _make_kernel(None)
    result = kernel.run_turn()
    assert cc["n"] >= 1
    assert result.stop_reason != "time_budget_exhausted"


def test_remaining_positive_runs_normally():
    kernel, cc = _make_kernel(
        lambda: {"elapsed": 100, "budget": 1800, "remaining": 1700.0, "pct": 5.0}
    )
    result = kernel.run_turn()
    assert cc["n"] >= 1
    assert result.stop_reason != "time_budget_exhausted"


def test_stats_fn_raises_is_silent_no_stop():
    """A raising stats fn must be swallowed -> normal run, never a false stop."""
    def boom():
        raise RuntimeError("stats unavailable")
    kernel, cc = _make_kernel(boom)
    result = kernel.run_turn()
    assert cc["n"] >= 1
    assert result.stop_reason != "time_budget_exhausted"


def test_non_dict_stats_return_is_ignored():
    """A non-dict return (bad injection) must not trigger the hard stop."""
    kernel, cc = _make_kernel(lambda: "not-a-dict")
    result = kernel.run_turn()
    assert cc["n"] >= 1
    assert result.stop_reason != "time_budget_exhausted"


def test_dict_with_none_remaining_does_not_raise_or_stop():
    """A dict whose remaining is None must NOT raise and must NOT hard-stop.

    Regression: the comparison `remaining <= 0` cannot run on a non-numeric
    value; a None/str remaining means "no deadline known", so the loop runs
    normally (same contract as a None return or a raising fn).
    """
    kernel, cc = _make_kernel(lambda: {"elapsed": 10, "budget": 1800, "remaining": None})
    result = kernel.run_turn()  # must not raise TypeError
    assert cc["n"] >= 1
    assert result.stop_reason != "time_budget_exhausted"


def test_dict_with_str_remaining_does_not_raise_or_stop():
    kernel, cc = _make_kernel(lambda: {"elapsed": 10, "budget": 1800, "remaining": "0"})
    result = kernel.run_turn()
    assert cc["n"] >= 1
    assert result.stop_reason != "time_budget_exhausted"


def test_dict_missing_remaining_runs_normally():
    """A dict without a 'remaining' key is 'no deadline known', not a stop."""
    kernel, cc = _make_kernel(lambda: {"elapsed": 10, "budget": 1800})
    result = kernel.run_turn()
    assert cc["n"] >= 1
    assert result.stop_reason != "time_budget_exhausted"
