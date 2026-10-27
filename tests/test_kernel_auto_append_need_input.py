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

"""Auto-append [NEED_USER_INPUT] fallback for endless text-only streams.

The escalation ladder (saturated directive, streak hard directive, 3+ nudge)
only ever re-prompts. A model that keeps producing text without tool calls
and without a stop signal therefore burns the whole wall: every round ends
"no stop signal" and the loop continues. Past AUTO_APPEND_NEED_INPUT_AT
consecutive tool-free, signal-free responses the kernel ends the turn
ITSELF by appending a harness-side [NEED_USER_INPUT] — never a fabricated
[TASK_COMPLETE], which would claim a verified deliverable the model never
asserted.
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

    def pop_last_assistant(self):
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                self.messages.pop(i)
                return


def _make_kernel(responses, max_output_tokens=100):
    history = _History()
    config = types.SimpleNamespace(
        max_iterations=50, max_continuations=200, mode="auto",
        max_output_tokens=max_output_tokens,
    )
    state = {"i": 0}

    def call_llm_fn(messages, schemas):
        i = state["i"]
        state["i"] += 1
        if i >= len(responses):
            return ({"content": "done [TASK_COMPLETE]", "tool_calls": None},
                    {"input_tokens": 1, "output_tokens": 5})
        r = dict(responses[i])
        r.pop("_out_tokens", None)
        return r, {"input_tokens": 1, "output_tokens": 5}

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
    )
    return AgentKernel(deps), history, state


def _text_only(text="narrating the next step"):
    return {"content": text, "tool_calls": None}


# ── the fallback itself ───────────────────────────────────────────────────────

def test_auto_append_fires_at_threshold():
    """A long tool-free, signal-free stream ends the turn harness-side."""
    resp = [_text_only() for _ in range(AgentKernel.AUTO_APPEND_NEED_INPUT_AT + 2)]
    k, history, _ = _make_kernel(resp)
    result = k.run_turn()
    assert result.stop_reason == "auto_appended_need_input"
    joined = "\n".join(m.get("content", "") for m in history.messages
                       if isinstance(m.get("content"), str))
    assert "[NEED_USER_INPUT]" in joined
    k.deps.display.completion_signal.assert_called_with("[NEED_USER_INPUT]")


def test_auto_append_not_before_threshold():
    """Below the threshold the model's own signal still wins the turn."""
    resp = [_text_only() for _ in range(AgentKernel.AUTO_APPEND_NEED_INPUT_AT - 1)]
    resp.append({"content": "all done [TASK_COMPLETE]", "tool_calls": None})
    k, _, _ = _make_kernel(resp)
    result = k.run_turn()
    assert result.stop_reason == "explicit_signal"


def test_tool_turn_resets_the_counter():
    """Real work in between keeps the streak from accumulating across it."""
    resp = [_text_only() for _ in range(AgentKernel.AUTO_APPEND_NEED_INPUT_AT - 2)]
    resp.append({"content": "", "tool_calls": [
        {"id": "1", "name": "shell", "arguments": {"command": "ls"}}]})
    resp += [_text_only() for _ in range(AgentKernel.AUTO_APPEND_NEED_INPUT_AT - 2)]
    resp.append({"content": "done [TASK_COMPLETE]", "tool_calls": None})
    k, _, _ = _make_kernel(resp)
    result = k.run_turn()
    assert result.stop_reason == "explicit_signal"
    assert k._consecutive_text_only == 0


def test_auto_append_disabled_when_zero():
    """AUTO_APPEND_NEED_INPUT_AT = 0 disables the mechanism entirely."""
    resp = [_text_only() for _ in range(30)]
    k, _, _ = _make_kernel(resp)
    saved = k.AUTO_APPEND_NEED_INPUT_AT
    try:
        k.AUTO_APPEND_NEED_INPUT_AT = 0
        result = k.run_turn()
        assert result.stop_reason != "auto_appended_need_input"
    finally:
        k.AUTO_APPEND_NEED_INPUT_AT = saved


def test_auto_append_does_not_rewrite_model_text():
    """The harness appends its OWN message; assistant text stays untouched."""
    resp = [_text_only("my narration") for _ in range(AgentKernel.AUTO_APPEND_NEED_INPUT_AT + 1)]
    k, history, _ = _make_kernel(resp)
    k.run_turn()
    assistant_msgs = [m for m in history.messages if m.get("role") == "assistant"]
    for m in assistant_msgs:
        assert not (m.get("content") or "").rstrip().endswith("[NEED_USER_INPUT]")
    last = history.messages[-1]
    assert last["role"] == "user"
    assert "[NEED_USER_INPUT]" in last["content"]


def test_auto_append_never_fabricates_task_complete():
    """The fallback is an honest hand-back, never a claimed completion."""
    resp = [_text_only() for _ in range(AgentKernel.AUTO_APPEND_NEED_INPUT_AT + 1)]
    k, history, _ = _make_kernel(resp)
    result = k.run_turn()
    assert result.stop_reason == "auto_appended_need_input"
    joined = "\n".join(m.get("content", "") for m in history.messages
                       if isinstance(m.get("content"), str))
    assert "auto_appended [TASK_COMPLETE]" not in joined
    # The kernel's own appended message must be an honest NEED_USER_INPUT,
    # never a completion claim. (The escalation-ladder nudges legitimately
    # MENTION [TASK_COMPLETE] as an option the model may still take, so the
    # assertion is scoped to the final hand-back message itself.)
    assert history.messages[-1]["role"] == "user"
    assert "auto_appended [NEED_USER_INPUT]" in history.messages[-1]["content"]
    assert "[TASK_COMPLETE]" not in history.messages[-1]["content"]
