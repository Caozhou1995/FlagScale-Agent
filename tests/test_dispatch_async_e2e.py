# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""REAL-subprocess E2E for ASYNC dispatch_many.

Unit tests (test_dispatch.py) use a FakeSpawn; this module runs the full
process chain: dispatch -> SpawnWorkerTool Popen -> real child process that
does the work and reports through the REAL ledger -> background thread judges
-> poll returns bounded pointers.

The "async" property is proven OBSERVABLY: the dispatch call returns while the
workers are still running (first poll shows state='running'), and the elapsed
time of the dispatch call is far below the workers' total work time.
"""

import os
import stat
import sys
import textwrap
import time

import pytest

from flagscale_agent.react.multi_agent.dispatch import (
    DispatchManyTool,
    dispatch_many,
    poll_dispatch,
    start_dispatch_many,
)
from flagscale_agent.react.multi_agent.ledger import TaskLedger
from flagscale_agent.react.multi_agent.spawn import SpawnWorkerTool

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

WORKER_SLEEP_S = 1.0

_FAKE_AGENT = textwrap.dedent("""\
    #!/usr/bin/env python3
    import os, sys, time
    sys.path.insert(0, os.environ["FAKE_REPO_PATH"])
    from flagscale_agent.react.multi_agent.ledger import TaskLedger
    # argv: [--time-budget-sec, N, <contract_path>] — the contract is ALSO
    # reachable via env, exactly as the real worker reads it.
    tid = os.environ["FLAGSCALE_TASK_ID"]
    led = TaskLedger(os.environ["FLAGSCALE_TASKS_DIR"])
    time.sleep(float(os.environ.get("FAKE_WORK_S", "0.3")))
    out_dir = os.environ["FLAGSCALE_OUTPUT_DIR"]
    contract = open(os.environ["FLAGSCALE_CONTRACT_PATH"]).read()
    if "FAIL" not in contract:  # spec marker: goal containing 'FAIL' fails
        with open(os.path.join(out_dir, "out.md"), "w") as f:
            f.write("worker done\\n")
    led.write_result(tid, {"summary": "fake worker done",
                           "claim": {"goal_met": True}})
""")


@pytest.fixture
def tasks_dir(tmp_path):
    return str(tmp_path / "tasks")


@pytest.fixture
def led(tasks_dir):
    return TaskLedger(tasks_dir)


@pytest.fixture
def fake_agent_bin(tmp_path, monkeypatch):
    """A real, executable fake agent bin: does the 'work', reports via ledger."""
    script = tmp_path / "fake_agent.py"
    script.write_text(_FAKE_AGENT, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
                 | stat.S_IXOTH)
    monkeypatch.setenv("FAKE_REPO_PATH", REPO)
    monkeypatch.setenv("FAKE_WORK_S", str(WORKER_SLEEP_S))
    monkeypatch.setenv("FLAGSCALE_AGENT_BIN", str(script))
    for k in ("FLAGSCALE_TASK_ID", "FLAGSCALE_TASK_DEPTH",
              "FLAGSCALE_MAX_DEPTH", "FLAGSCALE_PARENT_TRACE",
              "FLAGSCALE_CONTRACT_PATH", "FLAGSCALE_OUTPUT_DIR",
              "FLAGSCALE_SESSION_ROOT", "FLAGSCALE_SESSION_ID"):
        monkeypatch.delenv(k, raising=False)
    return str(script)


def _spec(tmp_path, i, ok=True):
    work = tmp_path / f"w{i}"
    work.mkdir(exist_ok=True)
    out = work / "out.md"
    check = [{"kind": "check_command", "check": f"test -f {out}"}]
    goal = f"e2e async task {i}"
    if not ok:
        goal += " FAIL"  # fake agent fails when the goal contains 'FAIL'
    return {
        "goal": goal,
        "constraints": {"writable": [str(work)]},
        "acceptance": check,
        "output_ptr": str(out),
        "deadline_minutes": 5,
    }


def test_async_dispatch_returns_control_immediately(
        tmp_path, led, fake_agent_bin, monkeypatch):
    """dispatch returns while workers are STILL RUNNING — control is returned."""
    spawn = SpawnWorkerTool(ledger=led)
    specs = [_spec(tmp_path, 0), _spec(tmp_path, 1),
             _spec(tmp_path, 2, ok=False)]

    t0 = time.time()
    handle = start_dispatch_many(specs, degree=2, spawn=spawn, ledger=led)
    t_dispatch = time.time() - t0

    # (1) handle returned immediately — well before any worker could finish
    # (each worker sleeps WORKER_SLEEP_S).
    assert t_dispatch < WORKER_SLEEP_S / 2, (
        f"dispatch blocked for {t_dispatch:.2f}s — not async!")
    assert handle["dispatch_id"].startswith("dsp_")

    # (2) first poll, right now: state must be 'running' — the fan-out is
    # still in flight. THE observable proof of non-blocking.
    info = poll_dispatch(handle["dispatch_id"], ledger=led)
    assert info["state"] == "running", info

    # (3) meanwhile the parent does its own work (the whole point of async).
    time.sleep(0.2)
    info = poll_dispatch(handle["dispatch_id"], ledger=led)
    assert info["state"] == "running"

    # (4) eventually the background thread reunites everything.
    deadline = time.time() + 60
    while time.time() < deadline:
        info = poll_dispatch(handle["dispatch_id"], ledger=led)
        if info["state"] == "complete":
            break
        time.sleep(0.2)
    assert info["state"] == "complete", "fan-out never completed"
    recs = info["records"]
    assert len(recs) == 3
    by_goal_ok = {r["task_id"]: r for r in recs}
    passed = [r for r in recs if r["passed"]]
    failed = [r for r in recs if not r["passed"]]
    assert len(passed) == 2 and all(r["status"] == "DONE" for r in passed)
    assert len(failed) == 1 and failed[0]["status"] == "REJECTED"
    # Bounded note, pointer-shaped output.
    assert len(failed[0]["note"]) <= 200
    assert all(r["output_ptr"].endswith("out.md") for r in recs)
    # All D9 slots freed by the background judge loop.
    assert led.active_ids() == []


def test_dispatch_tool_async_surface_real_workers(
        tmp_path, tasks_dir, fake_agent_bin, monkeypatch):
    """Tool surface E2E: action='dispatch' → id; action='poll' → pointers."""
    spawn = SpawnWorkerTool(ledger=TaskLedger(tasks_dir))
    tool = DispatchManyTool(ledger=TaskLedger(tasks_dir), spawn=spawn)
    specs = [_spec(tmp_path, 0), _spec(tmp_path, 1)]
    out = tool.execute(specs=specs, degree=2)
    assert "dispatch_id:" in out, out
    did = out.split("dispatch_id:")[1].split("\n")[0].strip()
    deadline = time.time() + 60
    polled = ""
    while time.time() < deadline:
        polled = tool.execute(action="poll", dispatch_id=did)
        if "state=complete" in polled:
            break
        time.sleep(0.2)
    assert "state=complete" in polled, polled
    assert polled.count("PASS") == 2
    assert polled.count("FAIL") == 0


def test_real_worker_log_lands_in_nested_session(
        tmp_path, led, fake_agent_bin, monkeypatch):
    """REAL subprocess: with a bound session dir, worker.log AND the ledger
    record both exist, and the log pointer names the nested-session path —
    proving the process-boundary write matches the parent-side pointer."""
    sdir = tmp_path / "sess"
    spawn = SpawnWorkerTool(ledger=led, session_dir=str(sdir))
    specs = [_spec(tmp_path, 0)]
    recs = dispatch_many_blocking_e2e(specs, spawn=spawn, ledger=led)
    assert len(recs) == 1
    r = recs[0]
    tid = r.task_id
    nested_log = sdir / "subagents" / tid / "worker.log"
    assert nested_log.exists(), f"nested worker.log missing: {nested_log}"
    # The pointer handed to the parent names the SAME file that was written.
    assert r.log_path == str(nested_log)
    # The ledger record still lives under the global tasks dir (by design).
    assert led.task_dir(tid).is_dir()
    # And the worker really ran (reported through the real ledger).
    assert r.passed is True and r.status == "DONE"


def dispatch_many_blocking_e2e(specs, spawn, ledger):
    from flagscale_agent.react.multi_agent.dispatch import dispatch_many_blocking
    return dispatch_many_blocking(specs, degree=2, spawn=spawn, ledger=ledger,
                                  poll_interval=0.1, max_wait_s=60)

