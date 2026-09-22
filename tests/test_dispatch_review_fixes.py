"""Regression tests for the blind-review findings on async dispatch_many.

Each test pins ONE confirmed finding so it cannot silently regress. Findings
that were REFUTED (F4/F5/F7/F9) are covered by the existing suite.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flagscale_agent.react.multi_agent.dispatch import (  # noqa: E402
    DispatchManyTool, _format_poll, _persist_job, _load_jobs,
)


def test_f1_job_error_is_surfaced_not_silent():
    """A background crash must render as JOB ERROR, never a clean 0-task complete."""
    info = {
        "dispatch_id": "dsp_x", "state": "complete",
        "records": [], "error": "dispatch job error: boom",
    }
    out = _format_poll(info)
    assert "JOB ERROR" in out and "boom" in out
    assert "dispatched 0 task(s):" not in out.replace("before the failure", "")


def test_f1_clean_complete_has_no_error_marker():
    info = {
        "dispatch_id": "dsp_x", "state": "complete",
        "records": [{"passed": True, "status": "DONE",
                     "task_id": "t1", "output_ptr": "/o.md"}],
        "error": "",
    }
    out = _format_poll(info)
    assert "JOB ERROR" not in out
    assert "[PASS]" in out


def test_f2_partial_poll_marks_pass_as_provisional():
    """A running poll must NOT present a status-derived PASS as authoritative."""
    info = {
        "dispatch_id": "dsp_x", "state": "running", "n_specs": 2,
        "records": [{"passed": True, "status": "DONE",
                     "task_id": "t1", "output_ptr": "/o.md"}],
    }
    out = _format_poll(info)
    assert "PROVISIONAL" in out
    # no authoritative [PASS] token in the partial view
    assert "[PASS]" not in out


def test_f10_persist_tmp_is_unique_and_leaves_no_stray(tmp_path):
    """Concurrent writers must not collide on a fixed tmp name; no .tmp left over."""
    td = str(tmp_path)
    _persist_job(td, {"dispatch_id": "dsp_a", "state": "running"})
    _persist_job(td, {"dispatch_id": "dsp_b", "state": "running"})
    loaded = _load_jobs(td)
    assert set(loaded) == {"dsp_a", "dsp_b"}
    strays = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert strays == []


def test_f6_recorder_persists_task_id_under_lock(tmp_path):
    """A recorded task id must survive into the persisted registry (no drop)."""
    from flagscale_agent.react.multi_agent.dispatch import (
        _RecordingSpawn, _DISPATCH_JOBS,
    )

    class _FakeSpawn:
        def execute(self, **kw):
            return "spawned task deadbeef pid=1"

    class _Led:
        _dir = str(tmp_path)

    did = "dsp_rec"
    _DISPATCH_JOBS[did] = {"dispatch_id": did, "state": "running", "task_ids": []}
    rec = _RecordingSpawn(_FakeSpawn(), did, _Led())
    rec.execute(goal="g")
    assert "deadbeef" in _DISPATCH_JOBS[did]["task_ids"]
    on_disk = _load_jobs(str(tmp_path)).get(did, {})
    assert "deadbeef" in on_disk.get("task_ids", [])


def test_f3_recorded_worker_log_path_prefers_frozen_task_path(tmp_path):
    """A resume under a DIFFERENT session dir must append to the frozen path."""
    from flagscale_agent.react.multi_agent.spawn import SpawnWorkerTool
    from flagscale_agent.react.multi_agent.ledger import TaskLedger
    from flagscale_agent.react.multi_agent.contract import Contract

    led = TaskLedger(str(tmp_path / "tasks"))
    c = Contract.build(goal="g", constraints={"writable": [str(tmp_path)]},
                       acceptance=[{"check": "true"}],
                       output_ptr=str(tmp_path / "o.md"))
    led.create(c)
    tid = c.id
    frozen = tmp_path / "S_orig" / "subagents" / tid / "worker.log"
    tdir = Path(led.task_dir(tid))
    (tdir / "worker_log_path").write_text(str(frozen), encoding="utf-8")
    # A DIFFERENT resuming session dir must NOT override the frozen path.
    spawn = SpawnWorkerTool(ledger=led, session_dir=str(tmp_path / "S_other"))
    assert spawn.recorded_worker_log_path(tid) == frozen
    # Without a recorded path, it falls back to the caller's session derivation.
    other = SpawnWorkerTool(ledger=led, session_dir=str(tmp_path / "S_other"))
    assert str(other.recorded_worker_log_path("nonexistent")) == str(
        other.worker_log_path("nonexistent"))
