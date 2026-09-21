# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the fan-out dispatcher + bounded reunite.

Two things the dispatcher must prove:
  (a) fan-out runs up to `degree` workers concurrently and reunites them all;
  (b) the reunite is BOUNDED — the parent-facing output carries POINTERS
      (task_id/status/output_ptr), never a worker's text body.

These unit tests use a FAKE spawn (no real subprocess) so scheduling and the
pointer/handoff contract are deterministic. The real-subprocess wall-clock
speedup is proven separately by scripts/e2e_m4.py (design §14.4 smoke).
"""

import os

import pytest

from flagscale_agent.react.multi_agent.dispatch import (
    DispatchManyTool,
    PointerRecord,
    dispatch_many,
    format_pointers,
)
from flagscale_agent.react.multi_agent.ledger import (
    DONE,
    REJECTED,
    REPORTED,
    RUNNING,
    TaskLedger,
)
from flagscale_agent.react.multi_agent.spawn import MAX_CONCURRENT


@pytest.fixture
def tasks_dir(tmp_path):
    return str(tmp_path / "tasks")


@pytest.fixture
def led(tasks_dir):
    return TaskLedger(tasks_dir)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FLAGSCALE_TASK_ID", raising=False)


class FakeSpawn:
    """Spawns nothing; creates a ledger entry + reports RUNNING→REPORTED.

    Mimics SpawnWorkerTool.execute's success string and its ledger effects
    (create → RUNNING → REPORTED) so the dispatcher's poll/judge path is
    exercised without a real process. Records max concurrency observed.
    """

    def __init__(self, ledger, report_ok=True, concurrency_log=None):
        self._ledger = ledger
        self._report_ok = report_ok
        self.max_live = 0
        self._concurrency_log = concurrency_log

    def execute(self, goal="", constraints=None, acceptance=None, inputs=None,
                output_ptr="", deadline_minutes=0, **kw):
        from flagscale_agent.react.multi_agent.contract import Contract
        cons = dict(constraints or {})
        cons.setdefault("max_minutes", deadline_minutes or 5)
        import time
        c = Contract.build(
            goal=goal, constraints=cons, acceptance=list(acceptance or []),
            output_ptr=output_ptr, inputs=list(inputs or []),
            deadline_epoch=int(time.time()) + 300, depth=1, parent={},
        )
        try:
            self._ledger.create(c)
        except Exception as e:
            return f"ERROR: {e}"
        self._ledger.transition(c.id, RUNNING, pid=1)
        self._ledger.transition(c.id, REPORTED, note="fake reported")
        # True concurrency = tasks the dispatcher has spawned but not yet judged
        # (every one of them is still in an ACTIVE status until check_result).
        live = len(self._ledger.active_ids())
        self.max_live = max(self.max_live, live)
        if self._concurrency_log is not None:
            self._concurrency_log.append(live)
        return f"spawned task {c.id} pid=1 depth=1 deadline=5min log=x"


def _spec(i, tmp_path, ok=True):
    work = tmp_path / f"w{i}"
    work.mkdir(exist_ok=True)
    out = work / "out.md"
    # acceptance passes iff the artifact exists; FakeSpawn won't write it.
    check = [{"kind": "check_command", "check": f"test -f {out}"}]
    if ok:
        out.write_text("done\n", encoding="utf-8")
    return {
        "goal": f"task {i}",
        "constraints": {"writable": [str(work)]},
        "acceptance": check,
        "output_ptr": str(out),
        "deadline_minutes": 5,
    }


# ── bounded pointer records ──────────────────────────────────────────────────
def test_pointer_record_has_pointer_not_body(tmp_path):
    r = PointerRecord(task_id="abc", status=DONE, passed=True,
                      output_ptr="/x/out.md", log_path="/x/worker.log",
                      note="all 1 acceptance checks passed")
    d = r.to_dict()
    assert d["output_ptr"] == "/x/out.md"
    assert "worker.log" in d["log_path"]
    # No field is meant to carry a worker's prose body.
    assert set(d) == {"task_id", "status", "passed", "output_ptr",
                      "log_path", "note"}


def test_format_pointers_is_bounded_no_text_wall(tmp_path):
    # A worker note longer than the cap must be clipped, not inlined whole.
    huge = "X" * 5000
    recs = [PointerRecord(task_id="t1", status=REJECTED, passed=False,
                          output_ptr="/o", note=huge[:180] + "…")]
    out = format_pointers(recs)
    assert huge not in out            # the 5000-char body never appears
    assert "t1" in out and "ptr=" in out


def test_dispatch_many_done_and_pointer_only(led, tmp_path):
    spawn = FakeSpawn(led, report_ok=True)
    specs = [_spec(i, tmp_path) for i in range(3)]
    recs = dispatch_many(specs, degree=2, spawn=spawn, ledger=led,
                         poll_interval=0.01, max_wait_s=30)
    assert len(recs) == 3
    for r in recs:
        assert r.passed is True
        assert r.status == DONE
        assert r.output_ptr.endswith("out.md")
        assert r.log_path.endswith("worker.log")
    # Parent-facing text stays small and pointer-shaped.
    text = format_pointers(recs)
    assert text.count("\n") <= 4
    assert "task=" in text and "ptr=" in text


def test_dispatch_many_rejected_when_artifact_missing(led, tmp_path):
    spawn = FakeSpawn(led, report_ok=True)
    specs = [_spec(0, tmp_path, ok=True), _spec(1, tmp_path, ok=False)]
    recs = dispatch_many(specs, degree=2, spawn=spawn, ledger=led,
                         poll_interval=0.01, max_wait_s=30)
    assert recs[0].passed is True and recs[0].status == DONE
    assert recs[1].passed is False and recs[1].status == REJECTED
    # The failure note is capped (bounded), not a raw dump.
    assert len(recs[1].note) <= 200


def test_degree_is_clamped_to_cap(led, tmp_path):
    log = []
    spawn = FakeSpawn(led, report_ok=True, concurrency_log=log)
    specs = [_spec(i, tmp_path) for i in range(4)]
    dispatch_many(specs, degree=99, spawn=spawn, ledger=led,
                  poll_interval=0.01, max_wait_s=30)
    # Never exceeds the D9 hard cap, even when asked for more.
    assert spawn.max_live <= MAX_CONCURRENT


def test_no_active_tasks_left_after_dispatch(led, tmp_path):
    spawn = FakeSpawn(led, report_ok=True)
    specs = [_spec(i, tmp_path) for i in range(3)]
    dispatch_many(specs, degree=2, spawn=spawn, ledger=led,
                  poll_interval=0.01, max_wait_s=30)
    # Judging each REPORTED task freed every D9 slot.
    assert led.active_ids() == []


# ── tool surface ─────────────────────────────────────────────────────────────
def test_dispatch_tool_refuses_in_worker(monkeypatch, led, tmp_path):
    monkeypatch.setenv("FLAGSCALE_TASK_ID", "x")
    tool = DispatchManyTool(ledger=led)
    out = tool.execute(specs=[_spec(0, tmp_path)])
    assert out.startswith("ERROR") and "parent-only" in out


def test_dispatch_tool_empty_specs(led):
    tool = DispatchManyTool(ledger=led)
    assert tool.execute(specs=[]).startswith("ERROR")


def test_dispatch_tool_runs(led, tmp_path):
    tool = DispatchManyTool(ledger=led, spawn=FakeSpawn(led))
    out = tool.execute(specs=[_spec(0, tmp_path)], degree=1)
    assert "dispatched 1 task" in out and "PASS" in out
