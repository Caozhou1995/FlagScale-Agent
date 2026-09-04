"""Tests for background shell jobs: shell(background=true) + shell_jobs tool."""

import time

import pytest

from flagscale_agent.react.tools.shell import (
    ShellTool,
    ShellJobsTool,
    _JOB_REGISTRY,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    # Ensure a clean registry before and after each test.
    _JOB_REGISTRY.cleanup_all()
    _JOB_REGISTRY._counter = 0
    yield
    _JOB_REGISTRY.cleanup_all()
    _JOB_REGISTRY._counter = 0


def _wait_until(pred, timeout=10.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def test_background_launch_returns_immediately():
    sh = ShellTool()
    t0 = time.time()
    # Command sleeps 3s; background launch must return well under that.
    out = sh.execute(command="sleep 3; echo done", background=True)
    elapsed = time.time() - t0
    assert elapsed < 1.5, f"background launch blocked for {elapsed:.2f}s"
    assert "BACKGROUNDED as job" in out
    assert "job1" in out


def test_sync_still_blocks_by_default():
    sh = ShellTool()
    out = sh.execute(command="echo hello")
    assert "hello" in out
    # No job should have been registered for a synchronous run.
    assert _JOB_REGISTRY.all() == []


def test_poll_incremental_then_finish():
    sh = ShellTool()
    jobs = ShellJobsTool()
    out = sh.execute(
        command="echo first; sleep 1; echo second; sleep 1; echo third",
        background=True,
    )
    job_id = "job1"
    assert job_id in out

    # First chunk should appear quickly.
    assert _wait_until(
        lambda: "first" in "".join(_JOB_REGISTRY.get(job_id).stdout_chunks),
        timeout=5,
    )
    p1 = jobs.execute(action="poll", job_id=job_id)
    # Body is everything after the first (header) line, which echoes the command.
    body1 = p1.split("\n", 1)[1] if "\n" in p1 else ""
    assert "first" in body1
    # A second poll should NOT repeat "first" in the body (cursor advanced).
    p2 = jobs.execute(action="poll", job_id=job_id)
    body2 = p2.split("\n", 1)[1] if "\n" in p2 else ""
    assert "first" not in body2

    # Wait for completion.
    w = jobs.execute(action="wait", job_id=job_id, timeout=10)
    assert "third" in w or "finished" in w
    # After finishing, job is removed from registry.
    assert _JOB_REGISTRY.get(job_id) is None


def test_list_shows_running_job():
    sh = ShellTool()
    jobs = ShellJobsTool()
    sh.execute(command="sleep 5", background=True)
    listing = jobs.execute(action="list")
    assert "job1" in listing
    assert "running" in listing


def test_kill_terminates_job():
    sh = ShellTool()
    jobs = ShellJobsTool()
    sh.execute(command="sleep 30", background=True)
    job = _JOB_REGISTRY.get("job1")
    assert job.running
    res = jobs.execute(action="kill", job_id="job1")
    assert "Killed job1" in res
    assert _wait_until(lambda: not job.running, timeout=5)
    assert _JOB_REGISTRY.get("job1") is None


def test_wait_timeout_returns_control():
    sh = ShellTool()
    jobs = ShellJobsTool()
    sh.execute(command="sleep 10", background=True)
    t0 = time.time()
    res = jobs.execute(action="wait", job_id="job1", timeout=1)
    elapsed = time.time() - t0
    assert elapsed < 3, f"wait blocked {elapsed:.2f}s past its 1s timeout"
    assert "still running" in res


def test_poll_unknown_job():
    jobs = ShellJobsTool()
    res = jobs.execute(action="poll", job_id="nope")
    assert "no such job" in res


def test_wait_requires_job_id():
    jobs = ShellJobsTool()
    res = jobs.execute(action="wait")
    assert "requires a job_id" in res


# --- Monitor-driven detach: health judge action="background" ---


def test_monitor_detach_backgrounds_instead_of_kill():
    """When the health judge returns action='background', the monitor must
    detach the still-running process to a job (not kill it) and return a
    DETACHED handle immediately."""
    calls = {"n": 0}

    def judge(command, recent_output, elapsed, **kwargs):
        calls["n"] += 1
        return {"action": "background", "reason": "healthy but long"}

    # remind_interval=1 makes the monitor loop consult the judge quickly.
    sh = ShellTool(remind_interval=1, health_judge_fn=judge)
    t0 = time.time()
    # A long-running command that would otherwise block the monitor.
    out = sh.execute(command="sleep 30; echo done")
    elapsed = time.time() - t0

    assert "DETACHED as job" in out, out
    assert calls["n"] >= 1, "health judge was never consulted"
    # It detached, so it must have returned well before the 30s sleep finished.
    assert elapsed < 15, f"detach blocked for {elapsed:.2f}s"

    # The process must STILL be running under the registry (not killed).
    job = _JOB_REGISTRY.get("job1")
    assert job is not None, "detached job was not registered"
    assert job.running, "detached process was killed instead of backgrounded"
    # Health sampler must be stopped for a backgrounded job.
    assert job.health_sampler is None


def test_monitor_kill_still_kills():
    """action='kill' (and legacy kill=True) must still terminate the process."""
    def judge(command, recent_output, elapsed, **kwargs):
        return {"action": "kill", "reason": "stalled loop"}

    sh = ShellTool(remind_interval=1, health_judge_fn=judge)
    out = sh.execute(command="sleep 30")
    assert "TERMINATED" in out, out
    # Nothing left running in the registry.
    assert _JOB_REGISTRY.get("job1") is None


def test_monitor_legacy_kill_flag_still_kills():
    """Backward compat: a judge returning the old {'kill': True} shape kills."""
    def judge(command, recent_output, elapsed, **kwargs):
        return {"kill": True, "reason": "network hang"}

    sh = ShellTool(remind_interval=1, health_judge_fn=judge)
    out = sh.execute(command="sleep 30")
    assert "TERMINATED" in out, out


def test_monitor_continue_lets_it_run():
    """action='continue' (and legacy kill=False) must not detach or kill."""
    def judge(command, recent_output, elapsed, **kwargs):
        return {"action": "continue", "reason": ""}

    sh = ShellTool(remind_interval=1, health_judge_fn=judge)
    out = sh.execute(command="echo hi; sleep 0.2")
    assert "TERMINATED" not in out
    assert "DETACHED" not in out
    assert "hi" in out
