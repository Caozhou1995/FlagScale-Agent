"""Regression tests: system prompt must NOT be rebuilt on every tool round.

Cache regression root cause (2026-10-07): the dashboard lives at the tail of
the system prompt, AFTER the anthropic provider's cache breakpoint on the
static section (split at DASHBOARD_TEMPLATE's "\\n---\\n[" separator). An
intra-turn rebuild changed those bytes on every request, invalidating the
whole cached conversation prefix (cache_read collapsed to the static section
only, ~23040 tokens). The rebuild must fire only on plan-tool usage; turn
boundaries already refresh it via _inject_context.
"""
import pytest
from unittest.mock import Mock

from flagscale_agent.react.agent import WorkerAgent
from flagscale_agent.react.config import AgentConfig
from flagscale_agent.react.memory import Memory
from flagscale_agent.react.plan import TaskPlan


PLAN_TOOLS = ("plan_create", "plan_update", "plan_status")


@pytest.fixture
def agent(tmp_path, monkeypatch):
    mock_provider = Mock()
    mock_provider.count_tokens.return_value = 100
    mock_memory = Mock(spec=Memory)
    mock_task_plan = Mock(spec=TaskPlan)
    # No active plan: _build_plan_context() returns "" instead of choking on a
    # truthy Mock inside the hook's silent except.
    mock_task_plan.get_active.return_value = None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")
    config = AgentConfig(
        session_dir=str(tmp_path / "test_session"),
        api_key="test-key-12345",
        provider="anthropic",
        max_context_tokens=50000,
    )
    return WorkerAgent(
        config,
        _provider=mock_provider,
        _memory=mock_memory,
        _task_plan=mock_task_plan,
    )


def _tool_call(name):
    return {"name": name, "arguments": {}}


def _system_prompt_of(agent):
    """System prompt bytes as the provider sees them (messages[0], role=system)."""
    msgs = agent.history._messages
    if msgs and msgs[0].get("role") == "system":
        return msgs[0]["content"]
    return None


def test_non_plan_tool_round_keeps_system_prompt_bytes(agent):
    """A shell/read tool round must leave the system prompt byte-identical."""
    before = _system_prompt_of(agent)

    agent._on_kernel_tool_results([_tool_call("shell")], ["ok"])

    assert _system_prompt_of(agent) == before


def test_plan_tool_round_refreshes_system_prompt(agent, monkeypatch):
    """Plan tools still trigger the dashboard rebuild (rebuild call fires)."""
    calls = []
    orig = agent._refresh_system_prompt

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(agent, "_refresh_system_prompt", spy)

    agent._on_kernel_tool_results([_tool_call("plan_status")], ["ok"])

    assert len(calls) == 1


def test_mixed_round_with_shell_only_keeps_prompt(agent, monkeypatch):
    """A round containing only non-plan tools must not rebuild, even in bulk."""
    calls = []
    orig = agent._refresh_system_prompt

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(agent, "_refresh_system_prompt", spy)
    before = _system_prompt_of(agent)

    agent._on_kernel_tool_results(
        [_tool_call("shell"), _tool_call("read_file"), _tool_call("evict")],
        ["ok", "ok", "ok"],
    )

    assert _system_prompt_of(agent) == before
    assert len(calls) == 0


def test_refresh_signature_preserves_turn_elapsed_gauge(agent):
    """_refresh_system_prompt still feeds the turn_elapsed stopwatch (gauge
    semantics from the dashboard-live work are retained; only the per-tool-round
    REBUILD TIMING was reverted)."""
    agent._prompt_builder.runtime_stats = {}
    agent._turn_start = 12345.0
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("time.time", lambda: 12375.0)
        agent._refresh_system_prompt()
    stats = agent._prompt_builder.runtime_stats
    assert stats.get("turn_elapsed") == pytest.approx(30.0)
