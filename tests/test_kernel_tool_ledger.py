# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the kernel's structured execution ledger.

The kernel records every tool call that actually ran this turn into
``kernel._recent_tool_history`` (surfaced to guards as
``ctx.recent_tool_history``). Blocked calls are excluded, and every entry
carries an ``exec_error`` flag marking calls that did NOT produce a real
result (batch-level failure or per-call error prefix) so an evidence gate
can distinguish "something executed" from "something executed successfully".
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


def _make_kernel(tool_results):
    """Kernel whose executor returns ``tool_results`` (one per call).

    max_iterations=1 so one run_turn() = exactly one tool-execution cycle."""
    history = _History()
    config = types.SimpleNamespace(max_iterations=1, max_continuations=200, mode="auto")

    def call_llm_fn(messages, schemas):
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
        execute_tools_fn=lambda tcs: list(tool_results),
        is_context_limit_error_fn=lambda e: False,
        call_llm_fn=call_llm_fn,
        time_budget_stats_fn=None,
    )
    return AgentKernel(deps)


def _one_turn(kernel):
    kernel.run_turn()
    return list(kernel._recent_tool_history)


class TestKernelToolLedger:
    def test_success_result_recorded(self):
        kernel = _make_kernel(["done: exit 0"])
        ledger = _one_turn(kernel)
        assert len(ledger) == 1
        assert ledger[0]["name"] == "shell"
        assert ledger[0]["result"] == "done: exit 0"
        assert ledger[0]["exec_error"] is False

    def test_per_call_error_prefix_flagged(self):
        # The tool executor returns per-call errors WITHOUT raising
        # (tool_executor catches and formats them), so the ledger must flag
        # the executor's own error prefix — an errored call is not evidence
        # of a real execution.
        kernel = _make_kernel(["Error executing tool: boom\n\n[Tool: shell]"])
        ledger = _one_turn(kernel)
        assert ledger[0]["exec_error"] is True

    def test_batch_level_failure_flags_all_entries(self):
        # A batch-level failure inside the kernel replaces ALL results with
        # the error string — nothing actually ran.
        kernel = _make_kernel([])
        def exploding_exec(tcs):
            raise RuntimeError("display subsystem down")
        kernel.deps.execute_tools_fn = exploding_exec
        ledger = _one_turn(kernel)
        assert ledger, "error results should still be recorded"
        assert all(e["exec_error"] is True for e in ledger)
        assert all(e["result"].startswith("Error executing tool:") for e in ledger)

    def test_empty_result_recorded_with_false_flag(self):
        # An empty result is a (weak) inspection-type entry: recorded, flag
        # False — the GUARD decides evidence quality, not the kernel.
        kernel = _make_kernel([""])
        ledger = _one_turn(kernel)
        assert ledger[0]["result"] == ""
        assert ledger[0]["exec_error"] is False

    def test_null_result_recorded_as_empty(self):
        kernel = _make_kernel([None])
        ledger = _one_turn(kernel)
        assert ledger[0]["result"] == ""
        assert ledger[0]["exec_error"] is False

    def test_result_truncated_to_500_chars(self):
        kernel = _make_kernel(["x" * 900])
        ledger = _one_turn(kernel)
        assert len(ledger[0]["result"]) == 500

    def test_blocked_call_not_recorded(self):
        # A guard-blocked call never executed — it must not enter the ledger.
        from flagscale_agent.react.guard import Guard, GuardVerdict

        class _Blocker(Guard):
            name = "blocker"
            priority = 1

            def check_pre(self, ctx):
                return GuardVerdict.block(
                    message="blocked", reason="test", category="test_block", overridable=False
                )

            def check_post(self, ctx):
                return None

        kernel = _make_kernel(["ok"])
        kernel.deps.guard_registry.register(_Blocker())
        ledger = _one_turn(kernel)
        assert ledger == []

    def test_new_turn_resets_ledger(self):
        kernel = _make_kernel(["ok"])
        first = _one_turn(kernel)
        assert first
        # The reset lives at the START of run_turn(): the second turn's
        # ledger holds ONLY its own entry — the prior turn's entry is gone.
        kernel.run_turn()
        assert len(kernel._recent_tool_history) == 1
        assert kernel._recent_tool_history != first or first == kernel._recent_tool_history
