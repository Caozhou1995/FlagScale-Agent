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

"""Tests for salvaging a tool call emitted as TEXT (wrong channel).

Mechanism: some models write the Anthropic function-call block into the
assistant's TEXT content instead of the native tool_use channel, so
response["tool_calls"] is empty and the kernel loops it as a plain-text turn.
The salvage re-routes a TRAILING, KNOWN-TOOL invoke block into tool_calls.
"""

import types
from unittest.mock import MagicMock

from flagscale_agent.react.text_tool_call import salvage_tool_calls
from flagscale_agent.react.kernel import AgentKernel, KernelDeps
from flagscale_agent.react.guard import GuardRegistry


SCHEMAS = [
    {"name": "shell", "input_schema": {"properties": {
        "command": {"type": "string"},
        "background": {"type": "boolean"}}}},
    {"function": {"name": "read_file", "parameters": {"properties": {
        "path": {"type": "string"},
        "start_line": {"type": "integer"}}}}},
]


# ── module unit tests ──────────────────────────────────────────────────────

def test_trailing_single_invoke_salvaged():
    text = 'I will check.\n<invoke name="shell">\n<parameter name="command">echo hi</parameter>\n</invoke>'
    tc, cleaned = salvage_tool_calls(text, SCHEMAS)
    assert tc and len(tc) == 1
    assert tc[0]["name"] == "shell"
    assert tc[0]["arguments"]["command"] == "echo hi"
    assert cleaned == "I will check."


def test_quoted_block_in_code_fence_not_salvaged():
    # The model is QUOTING invoke syntax mid-message — must never execute it.
    text = ('Here is what I wrote:\n```\n<invoke name="shell">\n'
            '<parameter name="command">rm -rf /</parameter>\n</invoke>\n```\n'
            'That was my earlier message.')
    tc, cleaned = salvage_tool_calls(text, SCHEMAS)
    assert tc is None


def test_unknown_tool_not_salvaged():
    text = ('<invoke name="definitely_not_a_tool">\n'
            '<parameter name="x">1</parameter>\n</invoke>')
    tc, _ = salvage_tool_calls(text, SCHEMAS)
    assert tc is None


def test_unescaped_angle_and_amp_survive_with_bool_coercion():
    text = ('<invoke name="shell">\n'
            '<parameter name="command">echo "a<b && c>d"</parameter>\n'
            '<parameter name="background">true</parameter>\n</invoke>')
    tc, _ = salvage_tool_calls(text, SCHEMAS)
    assert tc[0]["arguments"]["command"] == 'echo "a<b && c>d"'
    assert tc[0]["arguments"]["background"] is True


def test_integer_coercion_openai_schema_shape():
    text = ('<invoke name="read_file">\n<parameter name="path">/x</parameter>\n'
            '<parameter name="start_line">5</parameter>\n</invoke>')
    tc, _ = salvage_tool_calls(text, SCHEMAS)
    assert tc[0]["arguments"]["start_line"] == 5


def test_multiple_trailing_blocks_salvaged_in_order():
    text = ('doing two:\n<invoke name="shell">\n<parameter name="command">ls</parameter>\n</invoke>\n'
            '<invoke name="shell">\n<parameter name="command">pwd</parameter>\n</invoke>')
    tc, cleaned = salvage_tool_calls(text, SCHEMAS)
    assert len(tc) == 2
    assert tc[0]["arguments"]["command"] == "ls"
    assert tc[1]["arguments"]["command"] == "pwd"
    assert cleaned == "doing two:"


def test_incomplete_block_not_salvaged():
    text = '<invoke name="shell">\n<parameter name="command">echo hi</parameter>'
    tc, _ = salvage_tool_calls(text, SCHEMAS)
    assert tc is None


def test_no_params_salvaged_empty_args():
    text = '<invoke name="shell">\n</invoke>'
    tc, _ = salvage_tool_calls(text, SCHEMAS)
    assert tc and tc[0]["arguments"] == {}


def test_function_calls_wrapper_trailing():
    text = ('<function_calls>\n<invoke name="shell">\n'
            '<parameter name="command">ls</parameter>\n</invoke>\n</function_calls>')
    tc, _ = salvage_tool_calls(text, SCHEMAS)
    assert tc and tc[0]["arguments"]["command"] == "ls"


def test_no_schemas_no_salvage():
    text = '<invoke name="shell">\n<parameter name="command">ls</parameter>\n</invoke>'
    tc, _ = salvage_tool_calls(text, [])
    assert tc is None


def test_plain_text_no_invoke_returns_none():
    tc, cleaned = salvage_tool_calls("just a normal message", SCHEMAS)
    assert tc is None and cleaned is None


# ── kernel integration: text-emitted invoke must reach execute_tools_fn ─────

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

    def pop_last_assistant(self):
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                self.messages.pop(i)
                return


def _make_kernel(responses):
    history = _History()
    config = types.SimpleNamespace(
        max_iterations=50, max_continuations=200, mode="auto",
        max_output_tokens=100,
    )
    state = {"i": 0, "executed": []}

    def call_llm_fn(messages, schemas):
        i = state["i"]
        state["i"] += 1
        if i >= len(responses):
            return ({"content": "done [TASK_COMPLETE]", "tool_calls": None},
                    {"input_tokens": 1, "output_tokens": 5})
        return dict(responses[i]), {"input_tokens": 1, "output_tokens": 5}

    def execute_tools_fn(tool_calls):
        state["executed"].extend(tool_calls)
        return ["ok"] * len(tool_calls)

    provider = MagicMock()
    provider.format_assistant_message.side_effect = (
        lambda r: {"role": "assistant", "content": r.get("content") or "",
                   "tool_calls": r.get("tool_calls")}
    )
    provider.format_tool_result.side_effect = (
        lambda tid, content: {"role": "tool", "tool_call_id": tid, "content": content}
    )

    deps = KernelDeps(
        provider=provider,
        history=history,
        tool_registry=MagicMock(),
        judge=MagicMock(),
        guard_registry=GuardRegistry(),
        config=config,
        display=MagicMock(),
        get_schemas_fn=lambda: SCHEMAS,
        inject_message_fn=lambda m: None,
        append_tool_results_fn=lambda r: None,
        format_tool_result_fn=lambda t, r: {"role": "tool", "content": r},
        execute_tools_fn=execute_tools_fn,
        is_context_limit_error_fn=lambda e: False,
        call_llm_fn=call_llm_fn,
    )
    return AgentKernel(deps), history, state


def test_kernel_salvages_text_emitted_invoke_to_executor():
    """A trailing text invoke is routed to execute_tools_fn, not the text loop."""
    text_call = ('I will run it.\n<invoke name="shell">\n'
                 '<parameter name="command">echo hi</parameter>\n</invoke>')
    resp = [
        {"content": text_call, "tool_calls": None},
        {"content": "done [TASK_COMPLETE]", "tool_calls": None},
    ]
    k, history, state = _make_kernel(resp)
    result = k.run_turn()
    # The salvaged call actually executed.
    assert len(state["executed"]) == 1
    assert state["executed"][0]["name"] == "shell"
    assert state["executed"][0]["arguments"]["command"] == "echo hi"
    # It did NOT end as a dead text loop.
    assert result.stop_reason == "explicit_signal"


def test_kernel_does_not_salvage_quoted_block():
    """A quoted invoke (mid-text, in a code fence) must NOT execute."""
    quoted = ('Earlier I wrote:\n```\n<invoke name="shell">\n'
              '<parameter name="command">rm -rf /</parameter>\n</invoke>\n```\n'
              'but that was just an example. [TASK_COMPLETE]')
    resp = [{"content": quoted, "tool_calls": None}]
    k, history, state = _make_kernel(resp)
    result = k.run_turn()
    assert state["executed"] == []
    assert result.stop_reason == "explicit_signal"
