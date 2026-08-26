# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression tests for the completion-gate livelock breaker.

Root cause (observed in sparql-university single-shot run):
  A single-shot agent that repeatedly emits a bare [TASK_COMPLETE] with no
  plan and no _override_reason hits the PlanGuard completion gate every time.
  The gate correctly blocks, the kernel `continue`s WITHOUT incrementing
  _continuation_count, so the loop spins all the way to max_iterations (2000)
  producing ~2000 identical block messages.

The breaker: AgentKernel counts consecutive completion-gate blocks with no
intervening progress and breaks out with stop_reason 'completion_gate_livelock'
after MAX_CONSECUTIVE_COMPLETION_BLOCKS.
"""

from unittest.mock import MagicMock
import types

from flagscale_agent.react.kernel import AgentKernel, KernelDeps
from flagscale_agent.react.guard import GuardRegistry
from flagscale_agent.react.guard.plan import PlanGuard


class _History:
    """Minimal history that supports the methods run_turn touches."""

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


def _make_kernel(llm_responses, single_shot=True, max_iter=2000):
    """Build an AgentKernel driving a real PlanGuard through run_turn.

    llm_responses: list of dicts shaped like provider responses:
        {"content": "<text>", "tool_calls": [...]}  (tool_calls may be [])
    The last response is repeated once the list is exhausted.
    """
    history = _History()

    registry = GuardRegistry()
    registry.register(PlanGuard(task_plan=None, single_shot=single_shot))

    config = types.SimpleNamespace(
        max_iterations=max_iter, max_continuations=200, mode="auto",
        _turn_count=0,
    )

    call_count = {"n": 0}

    def call_llm_fn(messages, schemas):
        idx = min(call_count["n"], len(llm_responses) - 1)
        call_count["n"] += 1
        resp = llm_responses[idx]
        usage = {"input_tokens": 1, "output_tokens": 1}
        return resp, usage

    def format_assistant_message(resp):
        return {"role": "assistant", "content": resp.get("content", "")}

    provider = MagicMock()
    provider.format_assistant_message.side_effect = format_assistant_message

    deps = KernelDeps(
        provider=provider,
        history=history,
        tool_registry=MagicMock(),
        judge=MagicMock(),
        guard_registry=registry,
        config=config,
        display=MagicMock(),
        get_schemas_fn=lambda: [],
        inject_message_fn=lambda msg: history.append({"role": "user", "content": msg}),
        append_tool_results_fn=lambda results: None,
        format_tool_result_fn=lambda tid, r: {},
        execute_tools_fn=lambda tcs: [],
        is_context_limit_error_fn=lambda e: False,
        call_llm_fn=call_llm_fn,
    )
    kernel = AgentKernel(deps)
    kernel._call_count = call_count
    return kernel


def test_bare_task_complete_breaks_before_max_iter():
    """Repeated bare [TASK_COMPLETE] with no plan must not spin to max_iter."""
    kernel = _make_kernel(
        [{"content": "[TASK_COMPLETE]", "tool_calls": []}],
        single_shot=True,
        max_iter=2000,
    )
    result = kernel.run_turn()
    assert result.stop_reason.startswith("completion_gate_livelock"), result.stop_reason
    # Broke out fast: LLM was called ~= threshold times, nowhere near 2000.
    assert kernel._call_count["n"] <= AgentKernel.MAX_CONSECUTIVE_COMPLETION_BLOCKS + 1
    assert kernel._call_count["n"] < 50


def test_inline_override_releases_without_livelock():
    """A [TASK_COMPLETE] carrying an inline _override_reason releases normally."""
    kernel = _make_kernel(
        [{"content": "[TASK_COMPLETE]\n_override_reason: trivial one-line lookup, no plan needed",
          "tool_calls": []}],
        single_shot=True,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal", result.stop_reason
    # Released on the first completion attempt — no repeated blocks.
    assert kernel._call_count["n"] == 1
    assert kernel._consecutive_completion_blocks == 0


def test_interactive_mode_never_blocks_completion():
    """In interactive (non-single-shot) mode the completion gate never fires."""
    kernel = _make_kernel(
        [{"content": "[TASK_COMPLETE]", "tool_calls": []}],
        single_shot=False,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal", result.stop_reason
    assert kernel._call_count["n"] == 1


def test_breaker_counts_only_consecutive_blocks():
    """Threshold is on CONSECUTIVE blocks; the counter starts at zero."""
    kernel = _make_kernel([{"content": "[TASK_COMPLETE]", "tool_calls": []}])
    assert kernel._consecutive_completion_blocks == 0
    kernel.run_turn()
    # After livelock break the counter has reached the threshold.
    assert (kernel._consecutive_completion_blocks
            >= AgentKernel.MAX_CONSECUTIVE_COMPLETION_BLOCKS)


def test_blocked_completion_never_prints_marker():
    """A gate-blocked bare [TASK_COMPLETE] must NOT print the authoritative marker.

    Display-ordering regression: the raw sentinel is stripped from the live
    stream, and the kernel only prints the marker AFTER the gate passes. So a
    completion that only ever gets blocked must never call completion_signal().
    """
    kernel = _make_kernel([{"content": "[TASK_COMPLETE]", "tool_calls": []}])
    result = kernel.run_turn()
    assert result.stop_reason.startswith("completion_gate_livelock")
    kernel.deps.display.completion_signal.assert_not_called()


def test_accepted_completion_prints_marker_once():
    """A gate-approved completion prints the authoritative marker exactly once."""
    kernel = _make_kernel(
        [{"content": "[TASK_COMPLETE]\n_override_reason: trivial lookup, no plan needed",
          "tool_calls": []}],
        single_shot=True,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal"
    kernel.deps.display.completion_signal.assert_called_once_with("[TASK_COMPLETE]")


def test_interactive_completion_prints_marker():
    """Interactive-mode completion (gate never fires) still prints the marker."""
    kernel = _make_kernel(
        [{"content": "all done [TASK_COMPLETE]", "tool_calls": []}],
        single_shot=False,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal"
    kernel.deps.display.completion_signal.assert_called_once_with("[TASK_COMPLETE]")


def test_need_user_input_prints_that_marker():
    """A completion ending in [NEED_USER_INPUT] prints THAT marker, not the other."""
    kernel = _make_kernel(
        [{"content": "here is my question [NEED_USER_INPUT]", "tool_calls": []}],
        single_shot=False,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal"
    kernel.deps.display.completion_signal.assert_called_once_with("[NEED_USER_INPUT]")


def test_bare_completion_shows_text_override_hint():
    """A blocked text-only [TASK_COMPLETE] must show the TEXT override hint,
    not the tool-arg hint.

    Regression (polyglot-c-py single-shot run): the guard appended the
    tool-arg _OVERRIDE_HINT ("Add _override_reason to your next tool call"),
    but [TASK_COMPLETE] has no tool call — the agent could not follow the
    instruction and re-emitted bare [TASK_COMPLETE] until livelock. The fix
    selects _TEXT_OVERRIDE_HINT when tool_name is empty, telling the agent
    to use the inline `_override_reason: <reason>` form instead.
    """
    from flagscale_agent.react.guard import _OVERRIDE_HINT, _TEXT_OVERRIDE_HINT
    kernel = _make_kernel([{"content": "[TASK_COMPLETE]", "tool_calls": []}])
    result = kernel.run_turn()
    assert result.stop_reason.startswith("completion_gate_livelock")
    # The guard message injected into history should contain the text hint,
    # not the tool-arg hint.
    guard_msgs = [m for m in kernel.deps.history.messages
                  if isinstance(m.get("content"), str)
                  and "_override_reason" in m["content"].lower()]
    assert len(guard_msgs) > 0, "Expected at least one guard message with override hint"
    # Tool-arg hint says "to your next tool call" — text hint does NOT.
    combined = " ".join(m["content"] for m in guard_msgs)
    assert "to your next tool call" not in combined, (
        "Should show text-inline override hint, not tool-arg hint"
    )
    assert "_override_reason:" in combined or "_override_reason =" in combined, (
        "Should show inline _override_reason format"
    )


def test_text_override_hint_shows_format_example():
    """The _TEXT_OVERRIDE_HINT must include a concrete format example so the
    agent knows exactly how to write the override."""
    from flagscale_agent.react.guard import _TEXT_OVERRIDE_HINT
    assert "[TASK_COMPLETE]" in _TEXT_OVERRIDE_HINT
    assert "_override_reason:" in _TEXT_OVERRIDE_HINT
    assert "bare" in _TEXT_OVERRIDE_HINT.lower() or "blocked again" in _TEXT_OVERRIDE_HINT.lower(), (
        "Should warn that bare re-emit will be blocked again"
    )


def test_trailing_sentinel_wins_over_earlier_mention():
    """If the text QUOTES one sentinel earlier but ENDS with the other, the
    trailing (last-occurring) sentinel is the authoritative one displayed.

    Regression: the fixed tuple-order loop always printed [TASK_COMPLETE] first,
    so a response quoting "[TASK_COMPLETE]" while explaining code but actually
    ending with [NEED_USER_INPUT] showed the wrong marker on screen.
    """
    kernel = _make_kernel(
        [{"content": "the marker [TASK_COMPLETE] is printed here; "
                     "but I still need input [NEED_USER_INPUT]",
          "tool_calls": []}],
        single_shot=False,
    )
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal"
    kernel.deps.display.completion_signal.assert_called_once_with("[NEED_USER_INPUT]")
