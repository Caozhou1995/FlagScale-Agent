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

"""Edit-churn stop-rule: the FINAL ladder rung before the text-only fallback.

Measured signature (analysis/anti_pattern.py over the 2026-10-27 baseline):
65% of losses show runs of >=8 edit-family calls with no successful
non-edit execution interleaved, vs 22% of wins. The winning 22% is a
legitimate TDD cadence — a hard stop would misfire on it. Therefore the
detector requires BOTH a long edit run AND no interleaved successful
execution (verify/test/shell commands that exit without exec_error).

The rule never hard-kills and never fabricates [TASK_COMPLETE]. It joins
the existing ladder as a final rung: a directive naming the churned file
and, when a plan with a forbidden list is active, quoting the actual
prohibition(s) — then requiring a binary choice: hand back with
[NEED_USER_INPUT] or state what NEW evidence changed. Fires once per
streak (counter resets), so a model that answers it honestly keeps working.
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


def _make_kernel(responses, executed=None, plan=None, max_output_tokens=100):
    """executed: optional list of per-response tool results. Each entry is
    either a string (non-error tool result) or the literal executor error
    prefix start 'Error executing tool:' (the ledger marks exec_error=True).
    plan: optional fake active-plan dict (with 'forbidden') exposed through
    deps.task_plan.get_active()."""
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

    exec_list = list(executed or [])

    def execute_tools_fn(tcs):
        i = state["i"] - 1
        if i < len(exec_list):
            return [exec_list[i]]
        return ["ok"]

    plan_holder = {"plan": plan}

    class _FakePlan:
        def get_active(self):
            return plan_holder["plan"]

    deps = KernelDeps(
        provider=MagicMock(),
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
        execute_tools_fn=execute_tools_fn,
        is_context_limit_error_fn=lambda e: False,
        call_llm_fn=call_llm_fn,
        task_plan=_FakePlan() if plan else None,
    )
    return AgentKernel(deps), history, state, plan_holder


class _FakePlan:
    def __init__(self, holder):
        self._h = holder

    def get_active(self):
        return self._h["plan"]


def _edit(name="edit_file"):
    return {"content": "", "tool_calls": [
        {"id": "1", "name": name, "arguments": {}}]}


ERR = "Error executing tool: boom"


# ── the rule itself ───────────────────────────────────────────────────────────

def test_fires_after_six_failed_edits():
    """6 edit calls, each followed by an exec error -> directive once."""
    n = AgentKernel.FORBIDDEN_STOP_AFTER
    resp = [_edit() for _ in range(n + 1)]
    executed = [ERR] * (n + 1)
    k, history, _, _ = _make_kernel(resp, executed=executed)
    result = k.run_turn()
    joined = "\n".join(str(m.get("content", "")) for m in history.messages)
    assert "FORBIDDEN" in joined
    # fires ONCE, not every subsequent turn
    assert joined.count("FORBIDDEN") == 1
    # the kernel ended the turn harness-side (honest hand-back semantics)
    assert result.stop_reason == "forbidden_edit_churn_stop"
    k.deps.display.completion_signal.assert_called_with("[NEED_USER_INPUT]")


def test_tdd_cadence_never_fires():
    """edit, passing-test, edit, passing-test: legit TDD, no directive."""
    n = AgentKernel.FORBIDDEN_STOP_AFTER
    resp = []
    executed = []
    for i in range(n + 2):
        resp.append(_edit())
        executed.append("ok")
        resp.append({"content": "", "tool_calls": [
            {"id": str(i), "name": "shell", "arguments": {"command": "pytest"}}]})
        executed.append("ok")
    k, _, _, _ = _make_kernel(resp, executed=executed)
    result = k.run_turn()
    assert result.stop_reason != "forbidden_edit_churn_stop"


def test_successful_verify_resets_the_streak():
    """A passing execution in the middle of edits breaks the run."""
    n = AgentKernel.FORBIDDEN_STOP_AFTER
    resp = [_edit() for _ in range(n - 2)]
    executed = [ERR] * (n - 2)
    resp.append({"content": "", "tool_calls": [
        {"id": "v", "name": "shell", "arguments": {"command": "pytest"}}]})
    executed.append("ok")
    resp += [_edit() for _ in range(n - 2)]
    executed += [ERR] * (n - 2)
    resp.append({"content": "done [TASK_COMPLETE]", "tool_calls": None})
    k, _, _, _ = _make_kernel(resp, executed=executed)
    result = k.run_turn()
    assert result.stop_reason == "explicit_signal"


def test_directive_quotes_the_forbidden_item():
    """With an active plan carrying a relevant prohibition, name it."""
    plan = {"title": "t", "forbidden": [
        {"action": "Do not edit vm.js by hand",
         "why": "it is generated", "instead": "regenerate it"}]}
    resp = [_edit() for _ in range(AgentKernel.FORBIDDEN_STOP_AFTER + 1)]
    executed = [ERR] * (AgentKernel.FORBIDDEN_STOP_AFTER + 1)
    k, history, _, _ = _make_kernel(resp, executed=executed, plan=plan)
    k.run_turn()
    joined = "\n".join(str(m.get("content", "")) for m in history.messages)
    assert "Do not edit vm.js by hand" in joined


def test_disabled_when_zero():
    """FORBIDDEN_STOP_AFTER = 0 disables the rule entirely."""
    resp = [_edit() for _ in range(20)]
    executed = [ERR] * 20
    k, _, _, _ = _make_kernel(resp, executed=executed)
    saved = AgentKernel.FORBIDDEN_STOP_AFTER
    try:
        AgentKernel.FORBIDDEN_STOP_AFTER = 0
        result = k.run_turn()
        assert result.stop_reason != "forbidden_edit_churn_stop"
    finally:
        AgentKernel.FORBIDDEN_STOP_AFTER = saved
