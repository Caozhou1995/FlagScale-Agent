# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for parent-side spawn."""

import json
import os
import signal
import subprocess
import time

import pytest

from flagscale_agent.react.multi_agent.ledger import (
    DEADLINE_MISSED,
    RUNNING,
    TaskLedger,
)
from flagscale_agent.react.multi_agent.spawn import (
    MAX_CONCURRENT,
    SpawnWorkerTool,
    _Watchdog,
)


class _FakeProc:
    def __init__(self, pid=4321):
        self.pid = pid


@pytest.fixture
def led(tmp_path):
    return TaskLedger(str(tmp_path / "tasks"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("FLAGSCALE_TASK_ID", "FLAGSCALE_TASK_DEPTH",
              "FLAGSCALE_MAX_DEPTH", "FLAGSCALE_PARENT_TRACE",
              "FLAGSCALE_CONTRACT_PATH", "FLAGSCALE_OUTPUT_DIR"):
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


class TestRefusals:
    def test_worker_can_spawn_under_cap(self, tmp_path, led, monkeypatch):
        # A worker (env carries a task id) is NO LONGER refused outright; the
        # depth cap governs. At depth 0 (< cap) the spawn is allowed.
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
        monkeypatch.setenv("FLAGSCALE_TASK_ID", "deadbeef")
        tool = SpawnWorkerTool(ledger=led)
        out = tool.execute(**_args(tmp_path))
        assert out.startswith("spawned")

    def test_depth_limit(self, tmp_path, led, monkeypatch):
        # With the cap at 1, a depth-1 process is AT the cap -> explicit refusal.
        monkeypatch.setenv("FLAGSCALE_TASK_DEPTH", "1")
        monkeypatch.setenv("FLAGSCALE_MAX_DEPTH", "1")
        tool = SpawnWorkerTool(ledger=led)
        out = tool.execute(**_args(tmp_path))
        assert out.startswith("ERROR")
        assert "depth" in out.lower()
        assert "D10" in out

    def test_concurrency_cap(self, tmp_path, led):
        # Fill the slots with active tasks.
        from flagscale_agent.react.multi_agent.contract import Contract
        work = tmp_path / "work"
        work.mkdir()
        for i in range(MAX_CONCURRENT):
            c = Contract.build(
                goal=f"noop {i}", constraints={"writable": [str(work)]},
                acceptance=[{"check": "true"}],
                output_ptr=str(work / f"o{i}.md"))
            led.create(c)
        tool = SpawnWorkerTool(ledger=led)
        out = tool.execute(**_args(tmp_path))
        assert out.startswith("ERROR")
        assert "MAX_CONCURRENT" in out

    def test_duplicate_contract_rejected(self, tmp_path, led, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
        tool = SpawnWorkerTool(ledger=led)
        first = tool.execute(**_args(tmp_path))
        assert first.startswith("spawned")
        second = tool.execute(**_args(tmp_path))
        assert second.startswith("ERROR")
        assert "duplicate" in second.lower()

    def test_bad_deadline(self, tmp_path, led):
        tool = SpawnWorkerTool(ledger=led)
        a = _args(tmp_path)
        a["deadline_minutes"] = 0
        assert tool.execute(**a).startswith("ERROR")

    def test_contract_validation_error(self, tmp_path, led):
        tool = SpawnWorkerTool(ledger=led)
        a = _args(tmp_path)
        a["output_ptr"] = "/etc/passwd"  # outside writable (INV4)
        out = tool.execute(**a)
        assert out.startswith("ERROR")


class TestRecursion:
    def test_default_cap_allows_depth1_spawn(self, tmp_path, led, monkeypatch):
        # Default cap is 2: a depth-1 worker may spawn a depth-2 child.
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
        monkeypatch.setenv("FLAGSCALE_TASK_DEPTH", "1")
        tool = SpawnWorkerTool(ledger=led)
        out = tool.execute(**_args(tmp_path))
        assert out.startswith("spawned")
        assert "depth=2" in out

    def test_default_cap_refuses_depth2_spawn(self, tmp_path, led, monkeypatch):
        # A depth-2 worker is AT the default cap and must be refused EXPLICITLY.
        monkeypatch.setenv("FLAGSCALE_TASK_DEPTH", "2")
        tool = SpawnWorkerTool(ledger=led)
        out = tool.execute(**_args(tmp_path))
        assert out.startswith("ERROR")
        assert "depth" in out.lower()
        assert "D10" in out

    def test_env_cap_override_permits_deeper(self, tmp_path, led, monkeypatch):
        # FLAGSCALE_MAX_DEPTH=3 permits a depth-2 worker to spawn a depth-3 child.
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
        monkeypatch.setenv("FLAGSCALE_TASK_DEPTH", "2")
        monkeypatch.setenv("FLAGSCALE_MAX_DEPTH", "3")
        tool = SpawnWorkerTool(ledger=led)
        out = tool.execute(**_args(tmp_path))
        assert out.startswith("spawned")
        assert "depth=3" in out

    def test_env_cap_clamped(self, monkeypatch):
        # A corrupt/absurd env value cannot disable the invariant: it is clamped.
        from flagscale_agent.react.multi_agent.spawn import (
            _effective_max_depth, DEFAULT_MAX_DEPTH, MIN_MAX_DEPTH, HARD_MAX_DEPTH,
        )
        monkeypatch.setenv("FLAGSCALE_MAX_DEPTH", "9999")
        assert _effective_max_depth() == HARD_MAX_DEPTH
        monkeypatch.setenv("FLAGSCALE_MAX_DEPTH", "0")
        assert _effective_max_depth() == MIN_MAX_DEPTH
        monkeypatch.setenv("FLAGSCALE_MAX_DEPTH", "not-a-number")
        assert _effective_max_depth() == DEFAULT_MAX_DEPTH
        monkeypatch.delenv("FLAGSCALE_MAX_DEPTH", raising=False)
        assert _effective_max_depth() == DEFAULT_MAX_DEPTH

    def test_parent_trace_recorded_in_contract(self, tmp_path, led, monkeypatch):
        # The child's contract carries the full ancestry chain (auditable tree).
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
        monkeypatch.setenv("FLAGSCALE_TASK_ID", "worker1")
        monkeypatch.setenv("FLAGSCALE_TASK_DEPTH", "1")
        monkeypatch.setenv("FLAGSCALE_PARENT_TRACE", json.dumps(["root"]))
        tool = SpawnWorkerTool(ledger=led)
        out = tool.execute(**_args(tmp_path))
        tid = out.split()[2]
        rec = led.get(tid)
        assert rec.contract.depth == 2
        assert rec.contract.parent["task_id"] == "worker1"
        assert rec.contract.parent["depth"] == 1
        assert rec.contract.parent["parent_trace"] == ["root", "worker1"]


class TestSpawnEnvAndTty:
    def test_env_injection_and_tty_safety(self, tmp_path, led, monkeypatch):
        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return _FakeProc(pid=999)

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        tool = SpawnWorkerTool(ledger=led, agent_bin="flagscale-agent")
        out = tool.execute(**_args(tmp_path))
        assert out.startswith("spawned")

        env = captured["kwargs"]["env"]
        assert env["FLAGSCALE_TASK_ID"]
        assert env["FLAGSCALE_TASK_DEPTH"] == "1"
        assert env["FLAGSCALE_CONTRACT_PATH"].endswith("contract.prompt")
        assert env["FLAGSCALE_OUTPUT_DIR"] == str(tmp_path / "work")
        # The child must resolve the SAME ledger dir as the parent, else
        # report_result in the worker cannot find its own task.
        assert env["FLAGSCALE_TASKS_DIR"] == str(led._dir)
        # Orchestrator has no ancestry -> the child inherits an empty trace.
        assert env["FLAGSCALE_PARENT_TRACE"] == "[]"

        # THE load-bearing constraint: child must not inherit the parent tty.
        assert captured["kwargs"]["stdin"] == subprocess.DEVNULL
        assert captured["kwargs"]["start_new_session"] is True
        assert captured["kwargs"]["stderr"] == subprocess.STDOUT

        # R1: argv carries ONLY the contract path + time budget, not the text.
        # Order matters: typer requires OPTIONS BEFORE the positional `query`,
        # else it errors "No such command '--time-budget-sec'".
        assert captured["argv"][0] == "flagscale-agent"
        assert captured["argv"][1] == "--time-budget-sec"
        assert captured["argv"][-1].endswith("contract.prompt")
        assert "--time-budget-sec" in captured["argv"]

    def test_contract_file_written(self, tmp_path, led, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
        tool = SpawnWorkerTool(ledger=led)
        out = tool.execute(**_args(tmp_path))
        tid = out.split()[2]
        cpath = led.task_dir(tid) / "contract.prompt"
        assert cpath.exists()
        text = cpath.read_text(encoding="utf-8")
        assert "write a 3-line report" in text
        assert str(tmp_path / "work" / "out.md") in text

    def test_running_state_and_pid_recorded(self, tmp_path, led, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc(pid=555))
        tool = SpawnWorkerTool(ledger=led)
        out = tool.execute(**_args(tmp_path))
        tid = out.split()[2]
        rec = led.get(tid)
        assert rec.status == RUNNING
        assert rec.pid == 555


class TestWatchdog:
    def test_deadline_kills_and_marks(self, led, tmp_path):
        from flagscale_agent.react.multi_agent.contract import Contract
        work = tmp_path
        c = Contract.build(goal="sleep", constraints={"writable": [str(work)]},
                           acceptance=[{"check": "true"}],
                           output_ptr=str(work / "o.md"))
        led.create(c)
        # a real, long-lived process so killpg has something to reap
        p = subprocess.Popen(["sleep", "60"], start_new_session=True)
        led.transition(c.id, RUNNING, pid=p.pid)
        try:
            wd = _Watchdog(led, c.id, p.pid, deadline_epoch=int(time.time()) - 1,
                           interval=0.2)
            wd.run()  # runs one cycle synchronously
            assert led.get(c.id).status == DEADLINE_MISSED
            time.sleep(0.3)
            assert p.poll() is not None  # was killed
        finally:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except OSError:
                pass

    def test_process_gone_marks_failed(self, led, tmp_path):
        from flagscale_agent.react.multi_agent.contract import Contract
        c = Contract.build(goal="gone", constraints={"writable": [str(tmp_path)]},
                           acceptance=[{"check": "true"}],
                           output_ptr=str(tmp_path / "o.md"))
        led.create(c)
        # pid that does not exist
        led.transition(c.id, RUNNING, pid=2 ** 22 + 12345)
        wd = _Watchdog(led, c.id, 2 ** 22 + 12345,
                       deadline_epoch=int(time.time()) + 10_000, interval=0.1)
        wd.run()
        assert led.get(c.id).status == "FAILED"
