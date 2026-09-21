# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the multi-agent TaskLedger (design §1.2)."""

import os
import shutil
import tempfile
import time

import pytest

from flagscale_agent.react.multi_agent.contract import Contract
from flagscale_agent.react.multi_agent.ledger import (
    TaskLedger,
    LedgerError,
    DuplicateTask,
    SPAWNING,
    RUNNING,
    REPORTED,
    DONE,
    REJECTED,
    FAILED,
    DEADLINE_MISSED,
    TERMINATED,
    ACTIVE_STATUSES,
)


def _contract(tmp_path, goal="Count files and write a report"):
    writable = str(tmp_path)
    out = os.path.join(writable, "out.json")
    return Contract.build(
        goal=goal,
        constraints={"writable": [writable], "max_minutes": 10},
        acceptance=[{"check": "test -f out.json", "kind": "check_command"}],
        output_ptr=out,
        inputs=[],
    )


@pytest.fixture
def led():
    d = tempfile.mkdtemp()
    yield TaskLedger(d)
    shutil.rmtree(d, ignore_errors=True)


class TestCreate:
    def test_create_writes_contract_and_state(self, led, tmp_path):
        c = _contract(tmp_path)
        tdir = led.create(c)
        assert (tdir / "contract.json").exists()
        assert (tdir / "state.json").exists()
        rec = led.get(c.id)
        assert rec.status == SPAWNING
        assert rec.output_ptr == c.output_ptr

    def test_create_duplicate_active_rejected(self, led, tmp_path):
        c = _contract(tmp_path)
        led.create(c)
        with pytest.raises(DuplicateTask):
            led.create(c)

    def test_content_addressed_same_goal_same_id(self, led, tmp_path):
        c1 = _contract(tmp_path, goal="same goal")
        c2 = _contract(tmp_path, goal="same goal")
        assert c1.id == c2.id
        led.create(c1)
        with pytest.raises(DuplicateTask):
            led.create(c2)  # id identical → duplicate detected without a registry

    def test_create_validates_contract(self, led, tmp_path):
        bad = Contract.build(
            goal="", constraints={"writable": [str(tmp_path)]},
            acceptance=[{"check": "true"}],
            output_ptr=os.path.join(str(tmp_path), "o.json"),
        )
        with pytest.raises(Exception):
            led.create(bad)


class TestTransitions:
    def test_legal_lifecycle(self, led, tmp_path):
        c = _contract(tmp_path)
        led.create(c)
        led.transition(c.id, RUNNING, pid=1234)
        assert led.get(c.id).status == RUNNING
        assert led.get(c.id).pid == 1234
        led.transition(c.id, REPORTED)
        led.transition(c.id, DONE)
        assert led.get(c.id).status == DONE

    def test_illegal_transition_rejected(self, led, tmp_path):
        c = _contract(tmp_path)
        led.create(c)
        led.transition(c.id, RUNNING)
        led.transition(c.id, DONE)
        # DONE is terminal
        with pytest.raises(LedgerError):
            led.transition(c.id, RUNNING)

    def test_spawning_cannot_jump_to_done(self, led, tmp_path):
        c = _contract(tmp_path)
        led.create(c)
        with pytest.raises(LedgerError):
            led.transition(c.id, DONE)

    def test_history_appended(self, led, tmp_path):
        c = _contract(tmp_path)
        led.create(c)
        led.transition(c.id, RUNNING, note="spawned")
        led.transition(c.id, DEADLINE_MISSED, note="watchdog")
        hist = [h["status"] for h in led.get(c.id).history]
        assert hist == [SPAWNING, RUNNING, DEADLINE_MISSED]
        # DEADLINE_MISSED → TERMINATED is legal
        led.transition(c.id, TERMINATED)
        assert led.get(c.id).status == TERMINATED

    def test_unknown_task(self, led):
        with pytest.raises(LedgerError):
            led.transition("nope", RUNNING)

    def test_unknown_status(self, led, tmp_path):
        c = _contract(tmp_path)
        led.create(c)
        with pytest.raises(LedgerError):
            led.transition(c.id, "BOGUS")


class TestActiveAndResult:
    def test_active_ids(self, led, tmp_path):
        c1 = _contract(tmp_path, goal="task one")
        c2 = _contract(tmp_path, goal="task two")
        led.create(c1)
        led.create(c2)
        led.transition(c2.id, RUNNING)
        led.transition(c2.id, DONE)
        assert led.active_ids() == [c1.id]

    def test_write_result_moves_to_reported(self, led, tmp_path):
        c = _contract(tmp_path)
        led.create(c)
        led.transition(c.id, RUNNING)
        led.write_result(c.id, {"output_ptr": c.output_ptr,
                                "self_report": "done!", "exit_code": 0})
        assert led.get(c.id).status == REPORTED
        res = led.read_result(c.id)
        assert res["self_report"] == "done!"

    def test_write_result_missing_task(self, led):
        with pytest.raises(LedgerError):
            led.write_result("ghost", {"a": 1})


class TestPrune:
    def test_prune_removes_old_terminal(self, led, tmp_path):
        c = _contract(tmp_path)
        led.create(c)
        led.transition(c.id, RUNNING)
        led.transition(c.id, DONE)
        # age the state file into the past
        sp = led.task_dir(c.id) / "state.json"
        old = time.time() - 10 * 86400
        os.utime(sp, (old, old))
        removed = led.prune(older_than_days=7)
        assert c.id in removed
        assert led.get(c.id) is None

    def test_prune_skips_active(self, led, tmp_path):
        c = _contract(tmp_path)
        led.create(c)  # SPAWNING = active
        sp = led.task_dir(c.id) / "state.json"
        old = time.time() - 10 * 86400
        os.utime(sp, (old, old))
        assert led.prune(older_than_days=7) == []
        assert led.get(c.id) is not None


class TestRenderForest:
    def test_render_forest_reconstructs_chain(self, led, tmp_path):
        from flagscale_agent.react.multi_agent.ledger import render_forest
        # root (depth1) -> child (depth2): the child records root as its parent.
        root = Contract.build(
            goal="root task", constraints={"writable": [str(tmp_path)]},
            acceptance=[{"check": "true"}],
            output_ptr=os.path.join(str(tmp_path), "r.json"),
            depth=1, parent={"task_id": None, "depth": 0, "parent_trace": []},
        )
        led.create(root)
        child = Contract.build(
            goal="child task", constraints={"writable": [str(tmp_path)]},
            acceptance=[{"check": "true"}],
            output_ptr=os.path.join(str(tmp_path), "c.json"),
            depth=2, parent={"task_id": root.id, "depth": 1,
                             "parent_trace": [root.id]},
        )
        led.create(child)
        nodes = {n["task_id"]: n for n in render_forest(led)}
        assert set(nodes) == {root.id, child.id}
        assert nodes[root.id]["parent_id"] is None
        assert nodes[root.id]["parent_trace"] == []
        assert nodes[root.id]["depth"] == 1
        assert nodes[root.id]["status"] == SPAWNING
        assert nodes[child.id]["parent_id"] == root.id
        assert nodes[child.id]["parent_trace"] == [root.id]
        assert nodes[child.id]["depth"] == 2

    def test_render_forest_empty(self, led):
        from flagscale_agent.react.multi_agent.ledger import render_forest
        assert render_forest(led) == []


class TestConcurrency:
    def test_flock_serializes_multiprocess_writes(self, led, tmp_path):
        """Two processes transitioning the same task must not lose a history entry.

        Each process does N SPAWNING→... no-op-safe transitions under the lock;
        the history length must equal the number of accepted transitions.
        """
        import multiprocessing as mp

        c = _contract(tmp_path)
        led.create(c)
        led.transition(c.id, RUNNING)
        state_path = str(led.task_dir(c.id) / "state.json")

        def worker(n):
            import fcntl, json
            for _ in range(n):
                with open(state_path, "r+", encoding="utf-8") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        st = json.load(f)
                        st["history"].append({"status": "PING", "ts": "", "note": "", "pid": None})
                        f.seek(0); f.truncate()
                        json.dump(st, f)
                        f.flush(); os.fsync(f.fileno())
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        procs = [mp.Process(target=worker, args=(20,)) for _ in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        import json as _json
        with open(state_path, encoding="utf-8") as f:
            st = _json.load(f)
        # 1 created + 1 RUNNING + 4*20 PING = 82 (no lost updates)
        assert len(st["history"]) == 1 + 1 + 80

    def test_concurrent_reader_never_sees_partial_state(self, led, tmp_path):
        """A reader must never hit the truncate->write window.

        Regression: get() originally read state.json with no lock while
        transition() rewrote it in place under LOCK_EX, so a concurrent reader
        could catch an empty file -> JSONDecodeError. get() now takes LOCK_SH.
        """
        import threading

        c = _contract(tmp_path)
        led.create(c)
        errors = []
        stop = {"v": False}

        def reader():
            while not stop["v"]:
                try:
                    led.get(c.id)
                except Exception as e:  # noqa: BLE001
                    errors.append(repr(e))
                    return

        def writer():
            for i in range(2000):
                try:
                    led.transition(c.id, RUNNING if i % 2 == 0 else DEADLINE_MISSED)
                except LedgerError:
                    pass

        t = threading.Thread(target=reader)
        w = threading.Thread(target=writer)
        t.start(); w.start(); w.join(); stop["v"] = True; t.join()
        assert errors == [], f"reader observed partial state: {errors[:3]}"
