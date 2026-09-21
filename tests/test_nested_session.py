# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the nested subagent session home.

A spawned worker's own session dir must become a CHILD of its parent's:
<parent_session_dir>/subagents/<task_id>. Only the session dir nests — memory
and proposals stay global (they key on FLAGSCALE_HOME, which never moves).
"""

import subprocess

import pytest

from flagscale_agent.react.multi_agent.ledger import TaskLedger
from flagscale_agent.react.multi_agent.spawn import SpawnWorkerTool


class _FakeProc:
    def __init__(self, pid=7777):
        self.pid = pid


@pytest.fixture
def led(tmp_path):
    return TaskLedger(str(tmp_path / "tasks"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("FLAGSCALE_TASK_ID", "FLAGSCALE_TASK_DEPTH",
              "FLAGSCALE_MAX_DEPTH", "FLAGSCALE_PARENT_TRACE",
              "FLAGSCALE_CONTRACT_PATH", "FLAGSCALE_OUTPUT_DIR",
              "FLAGSCALE_SESSION_ROOT", "FLAGSCALE_SESSION_ID"):
        monkeypatch.delenv(k, raising=False)


def _args(tmp_path, goal="write a 3-line report"):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return dict(
        goal=goal,
        constraints={"writable": [str(work)]},
        acceptance=[{"kind": "check_command", "check": "test -f out.md"}],
        output_ptr=str(work / "out.md"),
        deadline_minutes=10,
    )


def _capture_spawn(monkeypatch, tool, tmp_path):
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    out = tool.execute(**_args(tmp_path))
    assert out.startswith("spawned"), out
    return captured, out


class TestBuildEnvNestsSession:
    def test_session_root_and_id_injected(self, tmp_path, led, monkeypatch):
        parent = str(tmp_path / "parent_sess")
        tool = SpawnWorkerTool(ledger=led, agent_bin="x", session_dir=parent)
        captured, out = _capture_spawn(monkeypatch, tool, tmp_path)
        env = captured["kwargs"]["env"]
        tid = env["FLAGSCALE_TASK_ID"]
        assert env["FLAGSCALE_SESSION_ROOT"] == parent + "/subagents"
        assert env["FLAGSCALE_SESSION_ID"] == tid

    def test_recursive_nesting(self, tmp_path, led, monkeypatch):
        # A worker (which is itself nested) spawning a child nests one level
        # deeper: <parent>/subagents/<child>/subagents/<grandchild>.
        child_sess = str(tmp_path / "parent" / "subagents" / "child")
        tool = SpawnWorkerTool(ledger=led, agent_bin="x", session_dir=child_sess)
        captured, _ = _capture_spawn(monkeypatch, tool, tmp_path)
        env = captured["kwargs"]["env"]
        assert env["FLAGSCALE_SESSION_ROOT"] == child_sess + "/subagents"

    def test_absent_session_dir_omits_vars(self, tmp_path, led, monkeypatch):
        # Backward-compatible: no session_dir -> no nesting vars (child falls
        # back to the global default sessions root).
        tool = SpawnWorkerTool(ledger=led, agent_bin="x")
        captured, _ = _capture_spawn(monkeypatch, tool, tmp_path)
        env = captured["kwargs"]["env"]
        assert "FLAGSCALE_SESSION_ROOT" not in env
        assert "FLAGSCALE_SESSION_ID" not in env


class TestAgentHonorsSessionEnv:
    def _mk_agent(self, monkeypatch, tmp_path, **cfg_kw):
        from unittest.mock import Mock
        from flagscale_agent.react.agent import WorkerAgent
        from flagscale_agent.react.config import AgentConfig
        from flagscale_agent.react.memory import Memory
        from flagscale_agent.react.plan import TaskPlan

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")
        prov = Mock()
        prov.count_tokens.return_value = 100
        cfg = AgentConfig(
            api_key="test-key-12345", provider="anthropic",
            max_context_tokens=50000, **cfg_kw,
        )
        return WorkerAgent(cfg, _provider=prov, _memory=Mock(spec=Memory),
                           _task_plan=Mock(spec=TaskPlan))

    def test_env_root_and_id_used(self, tmp_path, monkeypatch):
        root = str(tmp_path / "nested_root")
        monkeypatch.setenv("FLAGSCALE_SESSION_ROOT", root)
        monkeypatch.setenv("FLAGSCALE_SESSION_ID", "abc12345")
        agent = self._mk_agent(monkeypatch, tmp_path)
        assert agent._session_id == "abc12345"
        assert agent._sessions_root == root
        assert agent._session_dir == root + "/abc12345"

    def test_config_session_dir_beats_env(self, tmp_path, monkeypatch):
        cfg_dir = str(tmp_path / "cfg_sess")
        monkeypatch.setenv("FLAGSCALE_SESSION_ROOT", str(tmp_path / "env_root"))
        monkeypatch.setenv("FLAGSCALE_SESSION_ID", "envid")
        agent = self._mk_agent(monkeypatch, tmp_path, session_dir=cfg_dir)
        assert agent._sessions_root == cfg_dir
        assert agent._session_id == "envid"  # id still honored from env

    def test_uuid_when_no_env(self, tmp_path, monkeypatch):
        agent = self._mk_agent(monkeypatch, tmp_path)
        assert len(agent._session_id) == 8
        assert agent._session_dir.endswith(agent._session_id)

    def test_memory_and_proposals_stay_global(self, tmp_path, monkeypatch):
        # Nesting the session dir must NOT shift memory/proposals.
        from flagscale_agent.react.paths import get_memory_dir, get_proposals_dir
        before_mem = get_memory_dir()
        before_prop = get_proposals_dir()
        monkeypatch.setenv("FLAGSCALE_SESSION_ROOT", str(tmp_path / "root"))
        monkeypatch.setenv("FLAGSCALE_SESSION_ID", "abc12345")
        self._mk_agent(monkeypatch, tmp_path)
        assert get_memory_dir() == before_mem
        assert get_proposals_dir() == before_prop


class TestDispatchThreadsSessionDir:
    def test_dispatch_builds_spawn_with_session_dir(self, tmp_path):
        from flagscale_agent.react.multi_agent.dispatch import DispatchManyTool

        parent = str(tmp_path / "parent_sess")
        tool = DispatchManyTool(ledger=TaskLedger(str(tmp_path / "t")),
                                session_dir=parent)
        assert tool._spawn is not None
        assert tool._spawn._session_dir == parent
