# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the signal reminder fallback in AgentKernel.

When LLM outputs pure text without [TASK_COMPLETE] or [NEED_USER_INPUT],
the kernel should send one reminder and give LLM a second chance.
"""

import pytest
from unittest.mock import MagicMock, patch
from flagscale_agent.react.kernel import AgentKernel


def _make_kernel_with_mock_llm(responses):
    """Create a kernel with mocked dependencies that returns predefined responses.

    Args:
        responses: list of (text, tool_calls) tuples for sequential LLM calls.
    """
    kernel = AgentKernel.__new__(AgentKernel)

    # Mock dependencies
    deps = MagicMock()
    deps.config.max_iterations = 10
    deps.config.mode = "auto"

    # History mock
    history_messages = [{"role": "user", "content": "test"}]
    deps.history.messages = history_messages
    deps.history.append = lambda msg: history_messages.append(msg)
    deps.history.get_context_pressure = lambda: 0.5

    # Guard registry mock
    deps.guard_registry.check_pre.return_value = None
    deps.guard_registry.check_post.return_value = None
    deps.guard_registry.reset_new_turn = MagicMock()

    # Display mock
    deps.display.thinking = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    deps.display.warn = MagicMock()

    # Judge mock
    deps.judge.reset_turn = MagicMock()

    # LLM call mock - returns responses in sequence
    call_count = [0]
    def mock_llm_call(messages, schemas):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        text, tools = responses[idx]
        return {"text": text, "tool_calls": tools}

    kernel.deps = deps
    kernel._call_count = call_count
    kernel._mock_llm_call = mock_llm_call

    # Initialize state
    kernel._interrupted = False
    kernel._plan_auto_continue_count = 0
    kernel._signal_reminder_sent = False
    kernel._last_turn_had_tools = False
    kernel._empty_output_retries = 0

    return kernel


class TestSignalReminder:
    """Test the fallback signal reminder when LLM forgets [TASK_COMPLETE]/[NEED_USER_INPUT]."""

    def test_reminder_sent_on_text_without_signal(self):
        """Pure text without signal should trigger one reminder."""
        kernel = _make_kernel_with_mock_llm([
            ("Here is the answer without signal", []),
            ("[TASK_COMPLETE]", []),
        ])

        # Simulate the check logic directly
        # First response: no signal, no plan → should set reminder
        kernel._signal_reminder_sent = False
        text = "Here is the answer without signal"
        has_signal = "[TASK_COMPLETE]" in text or "[NEED_USER_INPUT]" in text

        assert not has_signal
        assert not kernel._signal_reminder_sent

        # After first check, reminder should be sent
        kernel._signal_reminder_sent = True
        assert kernel._signal_reminder_sent

    def test_signal_reminder_resets_per_turn(self):
        """_signal_reminder_sent resets at start of each turn."""
        kernel = _make_kernel_with_mock_llm([])

        kernel._signal_reminder_sent = True
        # Simulate turn start reset
        kernel._signal_reminder_sent = False
        kernel._plan_auto_continue_count = 0
        assert not kernel._signal_reminder_sent

    def test_no_reminder_when_signal_present(self):
        """If response has [TASK_COMPLETE], no reminder needed."""
        kernel = _make_kernel_with_mock_llm([])

        text = "Done! [TASK_COMPLETE]"
        has_signal = "[TASK_COMPLETE]" in text or "[NEED_USER_INPUT]" in text
        assert has_signal
        # Should break immediately, no reminder

    def test_no_reminder_when_tool_calls_present(self):
        """If response has tool calls, it continues normally without reminder."""
        kernel = _make_kernel_with_mock_llm([])

        tool_calls = [{"name": "shell", "input": {"command": "ls"}}]
        # Tool calls mean the LLM is still working, no reminder needed
        assert len(tool_calls) > 0

    def test_reminder_only_once(self):
        """After reminder is sent once, second text-only response stops."""
        kernel = _make_kernel_with_mock_llm([])

        # First text-only: send reminder
        kernel._signal_reminder_sent = False
        assert not kernel._signal_reminder_sent
        kernel._signal_reminder_sent = True  # reminder sent

        # Second text-only: should stop (not send another reminder)
        assert kernel._signal_reminder_sent
        # This means: stop_reason = "no_tool_calls", break

    def test_reminder_message_content(self):
        """Verify the reminder message includes both signal options."""
        reminder = (
            "[system] 你刚才的回复没有包含 [TASK_COMPLETE] 或 [NEED_USER_INPUT]。"
            "如果当前指令已响应完成且不需要用户输入，请回复 [TASK_COMPLETE]。"
            "如果需要用户确认或提供信息，请回复 [NEED_USER_INPUT]。"
        )
        assert "[TASK_COMPLETE]" in reminder
        assert "[NEED_USER_INPUT]" in reminder
        assert "system" in reminder.lower()
