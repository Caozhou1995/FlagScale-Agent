# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the kernel's endgame posture: budget-driven structural force-tool.

Measured failure this locks in (trial 4CHXYAh, 2026-10-30): a 900s budget with
only 5 tool calls; in the final 19 seconds the model wrote "I must bank NOW"
and spent them deliberating how to call write_file vs shell — the turn died at
100% with zero bytes on disk. From >=85% budget spent, every remaining request
is structurally constrained (force_tool + squeezed thinking cap) so pure
reasoning can no longer consume the endgame.
"""

from unittest.mock import MagicMock
import types

from flagscale_agent.react.kernel import AgentKernel, KernelDeps


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


def _make_kernel(tb_stats, capture=None):
    """Kernel with one tool-producing turn and a time-budget stats provider.

    When ``capture`` is a dict, the LLM-call kwargs are recorded into it so a
    test can observe the actual structural constraint the request carried.
    """

    history = _History()
    config = types.SimpleNamespace(max_iterations=1, max_continuations=200, mode="auto")

    def call_llm_fn(messages, schemas, **kw):
        if capture is not None:
            capture.update(kw)
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
        execute_tools_fn=lambda tcs: [],
        is_context_limit_error_fn=lambda e: False,
        call_llm_fn=call_llm_fn,
        time_budget_stats_fn=(lambda: dict(tb_stats)) if tb_stats is not None else None,
    )
    return AgentKernel(deps)


from flagscale_agent.react.guard import GuardRegistry


def test_endgame_arms_structural_channel_at_85pct():
    """pct >= 85: the LLM request itself must carry force_tool + squeezed cap."""
    cap = {}
    k = _make_kernel(
        {"elapsed": 800, "budget": 900, "remaining": 100, "pct": 88.9}, capture=cap)
    k.run_turn()
    assert cap.get("force_tool") is True, (
        f"endgame request must be structurally forced, got kwargs={cap}")
    assert cap.get("thinking_cap_override") == 3000


def test_endgame_not_armed_below_85pct():
    """Below 85% the posture must stay silent — no premature forcing."""
    cap = {}
    k = _make_kernel(
        {"elapsed": 100, "budget": 900, "remaining": 800, "pct": 11.1}, capture=cap)
    k.run_turn()
    assert "force_tool" not in cap, (
        f"below threshold no structural forcing allowed, got kwargs={cap}")
    assert "thinking_cap_override" not in cap


def test_endgame_no_stats_no_crash():
    """Missing stats must behave exactly like no deadline (no crash)."""
    cap = {}
    k = _make_kernel(None, capture=cap)
    k.run_turn()  # must not raise
    assert "force_tool" not in cap


def test_endgame_threshold_boundary_inclusive():
    """Arm threshold is budget-dependent (forecast-arming): for a 900s
    budget with a 240s in-stream wall the threshold is 58.3% (= max(50,
    85 - 240/900*100)), so both 58.3% and anything above arm; below it
    does not. Boundary is inclusive >=."""
    cap_a = {}
    k = _make_kernel(
        {"elapsed": 525, "budget": 900, "remaining": 135, "pct": 58.4},
        capture=cap_a)
    k.run_turn()
    assert cap_a.get("force_tool") is True

    cap_b = {}
    k2 = _make_kernel(
        {"elapsed": 524, "budget": 900, "remaining": 376, "pct": 58.2},
        capture=cap_b)
    k2.run_turn()
    assert "force_tool" not in cap_b
