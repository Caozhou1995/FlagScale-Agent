# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for resume-with-message (the parent<->child dialogue channel)."""

import json
import os
import subprocess
import time

import pytest

from flagscale_agent.react.multi_agent.contract import Contract
from flagscale_agent.react.multi_agent.ledger import (
    DONE,
    FAILED,
    REJECTED,
    RUNNING,
    TaskLedger,
)
from flagscale_agent.react.multi_agent.resume import (
    ResumeChildTool,
    authorize,
    child_session_dir,
    resolve_target,
    write_adoption_audit,
)
from flagscale_agent.react.multi_agent.wiring import resolve_resume_query


class _FakeProc:
    def __init__(self, pid=9100):
        self.pid = pid


@pytest.fixture
def led(tmp_path):
    return TaskLedger(str(tmp_path / "tasks"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("FLAGSCALE_TASK_ID", "FLAGSCALE_TASK_DEPTH",
              "FLAGSCALE_MAX_DEPTH", "FLAGSCALE_PARENT_TRACE",
              "FLAGSCALE_CONTRACT_PATH", "FLAGSCALE_OUTPUT_DIR",
              "FLAGSCALE_SESSION_ROOT", "FLAGSCALE_SESSION_ID",
              "FLAGSCALE_RESUME_PATH"):
        monkeypatch.delenv(k, raising=False)


def _make_task(led, tmp_path, task_id_parent, parent_trace, status=REJECTED):
    """Create a task whose contract carries a parent chain, then reach `status`."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    c = Contract.build(
        goal=f"task {task_id_parent}",
        constraints={"writable": [str(work)], "max_minutes": 5},
        acceptance=[{"kind": "check_command", "check": "test -f out.md"}],
        output_ptr=str(work / "out.md"),
        depth=len(parent_trace) + 1,
        parent={"task_id": task_id_parent, "parent_trace": list(parent_trace)},
    )
    led.create(c, check_inputs_exist=False)
    # SPAWNING -> RUNNING -> target terminal
    led.transition(c.id, RUNNING, note="spawned")
    if status != RUNNING:
        led.transition(c.id, status, note="test setup")
    return c.id


class TestChildSessionDir:
    def test_layout(self):
        assert child_session_dir("/p/sess", "abc") == "/p/sess/subagents/abc"


class TestResolveTarget:
    def test_reads_contract(self, tmp_path, led):
        tid = _make_task(led, tmp_path, "P1", ["GP"])
        tgt = resolve_target(led, tid, str(tmp_path / "sess"))
        assert tgt.task_id == tid
        assert tgt.status == REJECTED
        assert tgt.parent_task_id == "P1"
        assert tgt.parent_trace == ["GP"]
        assert tgt.session_dir == str(tmp_path / "sess" / "subagents" / tid)

    def test_missing_task_is_none(self, led):
        assert resolve_target(led, "nope", "/s") is None


class TestAuthorize:
    def test_direct_parent_allowed(self, tmp_path, led):
        tid = _make_task(led, tmp_path, "P1", ["GP"])
        tgt = resolve_target(led, tid, "/s")
        ok, via, _ = authorize("P1", tgt, led, "/s")
        assert ok and via == "parent"

    def test_stranger_refused(self, tmp_path, led):
        tid = _make_task(led, tmp_path, "P1", ["GP"])
        tgt = resolve_target(led, tid, "/s")
        ok, _via, reason = authorize("X9", tgt, led, "/s")
        assert not ok and "not the direct parent" in reason

    def test_ancestor_adoption_allowed_when_parent_dead(self, tmp_path, led):
        # parent P1 has NO ledger entry and holds NO lock -> dead.
        tid = _make_task(led, tmp_path, "P1", ["GP"])
        tgt = resolve_target(led, tid, "/s")
        ok, via, _ = authorize("GP", tgt, led, "/s")
        assert ok and via == "adoption"

    def test_ancestor_refused_when_parent_alive(self, tmp_path, led):
        # Make P1 a live ACTIVE task in the ledger.
        work = tmp_path / "workp"
        work.mkdir(exist_ok=True)
        c = Contract.build(
            goal="parent task",
            constraints={"writable": [str(work)], "max_minutes": 5},
            acceptance=[{"kind": "check_command", "check": "test -f out.md"}],
            output_ptr=str(work / "out.md"),
            depth=1, parent={"task_id": None, "parent_trace": []},
        )
        led.create(c, check_inputs_exist=False)
        led.transition(c.id, RUNNING, note="alive")
        tid = _make_task(led, tmp_path, c.id, ["GP"])
        tgt = resolve_target(led, tid, "/s")
        ok, _via, reason = authorize("GP", tgt, led, "/s")
        assert not ok and "still alive" in reason


class TestResumeChildToolExecute:
    def _tool(self, led, parent_sess):
        return ResumeChildTool(ledger=led, session_dir=parent_sess)

    def test_parent_resumes_rejected_child(self, tmp_path, led, monkeypatch):
        parent_sess = str(tmp_path / "sess")
        tid = _make_task(led, tmp_path, "P1", ["GP"])
        monkeypatch.setenv("FLAGSCALE_TASK_ID", "P1")
        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["env"] = kwargs["env"]
            return _FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        out = self._tool(led, parent_sess).execute(task_id=tid, message="fix it")
        assert out.startswith("resumed"), out
        # Reopened to RUNNING with the new pid.
        rec = led.get(tid)
        assert rec.status == RUNNING
        assert rec.pid == 9100
        # Resume message written into the CHILD's nested session dir.
        msg_path = os.path.join(parent_sess, "subagents", tid, "resume.prompt")
        assert open(msg_path).read() == "fix it"
        # Env carries the resume pointer + the nested session binding.
        assert captured["env"]["FLAGSCALE_RESUME_PATH"] == msg_path
        assert captured["env"]["FLAGSCALE_SESSION_ROOT"] == parent_sess + "/subagents"
        assert captured["env"]["FLAGSCALE_SESSION_ID"] == tid
        # argv is the normal spawn form (options before positional).
        assert captured["argv"][1] == "--time-budget-sec"

    def test_done_is_not_resumable(self, tmp_path, led, monkeypatch):
        parent_sess = str(tmp_path / "sess")
        tid = _make_task(led, tmp_path, "P1", ["GP"], status=DONE)
        monkeypatch.setenv("FLAGSCALE_TASK_ID", "P1")
        out = self._tool(led, parent_sess).execute(task_id=tid, message="go")
        assert out.startswith("ERROR")
        assert "not resumable" in out

    def test_worker_cannot_resume_sibling(self, tmp_path, led, monkeypatch):
        parent_sess = str(tmp_path / "sess")
        tid = _make_task(led, tmp_path, "P1", ["GP"])
        monkeypatch.setenv("FLAGSCALE_TASK_ID", "SIBLING")
        out = self._tool(led, parent_sess).execute(task_id=tid, message="go")
        assert out.startswith("ERROR")
        assert led.get(tid).status == REJECTED  # untouched

    def test_empty_message_refused(self, tmp_path, led, monkeypatch):
        tid = _make_task(led, tmp_path, "P1", ["GP"])
        monkeypatch.setenv("FLAGSCALE_TASK_ID", "P1")
        out = self._tool(led, str(tmp_path / "s")).execute(task_id=tid, message="  ")
        assert out.startswith("ERROR")

    def test_adoption_rewrites_parent_and_writes_audit(self, tmp_path, led, monkeypatch):
        parent_sess = str(tmp_path / "sess")
        tid = _make_task(led, tmp_path, "P1", ["GP"])
        cpath = led.task_dir(tid) / "contract.json"
        before = json.loads(cpath.read_text())
        monkeypatch.setenv("FLAGSCALE_TASK_ID", "GP")
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
        out = self._tool(led, parent_sess).execute(task_id=tid, message="resume")
        assert out.startswith("resumed") and "adoption" in out
        # contract.parent is EXPLICITLY rewritten to the adopter...
        after = json.loads(cpath.read_text())
        assert after["parent"]["task_id"] == "GP"
        # ...but the contract IDENTITY is unchanged (id excludes parent).
        assert after["id"] == before["id"]
        # ...and the immutable promise fields are otherwise preserved.
        assert after["goal"] == before["goal"]
        assert after["acceptance"] == before["acceptance"]
        # The adoption audit sidecar is written too.
        audit = led.task_dir(tid) / "adoption.json"
        payload = json.loads(audit.read_text())
        assert payload["events"][-1]["adopted_by"] == "GP"
        assert payload["events"][-1]["prior_parent"] == "P1"


class TestWriteAdoptionAudit:
    def test_appends_events(self, tmp_path, led):
        tid = _make_task(led, tmp_path, "P1", ["GP"])
        write_adoption_audit(led, tid, "GP", "P1")
        write_adoption_audit(led, tid, "GGP", "GP")
        payload = json.loads((led.task_dir(tid) / "adoption.json").read_text())
        assert [e["adopted_by"] for e in payload["events"]] == ["GP", "GGP"]


class TestResolveResumeQuery:
    def test_reads_message(self, tmp_path, monkeypatch):
        p = tmp_path / "resume.prompt"
        p.write_text("  continue please  \n")
        monkeypatch.setenv("FLAGSCALE_RESUME_PATH", str(p))
        assert resolve_resume_query() == "continue please"

    def test_none_without_env(self, monkeypatch):
        assert resolve_resume_query() is None

    def test_none_when_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FLAGSCALE_RESUME_PATH", str(tmp_path / "nope"))
        assert resolve_resume_query() is None


class TestAgentResumePath:
    def _agent(self, monkeypatch, session_root, session_id):
        from unittest.mock import Mock
        from flagscale_agent.react.agent import WorkerAgent
        from flagscale_agent.react.config import AgentConfig
        from flagscale_agent.react.memory import Memory
        from flagscale_agent.react.plan import TaskPlan

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")
        prov = Mock()
        prov.count_tokens.return_value = 100
        cfg = AgentConfig(api_key="test-key-12345", provider="anthropic",
                          max_context_tokens=50000)
        return WorkerAgent(cfg, _provider=prov, _memory=Mock(spec=Memory),
                           _task_plan=Mock(spec=TaskPlan))

    def test_resume_loads_history_and_appends_message(self, tmp_path, monkeypatch):
        root = str(tmp_path / "root")
        sid = "child1"
        monkeypatch.setenv("FLAGSCALE_SESSION_ROOT", root)
        monkeypatch.setenv("FLAGSCALE_SESSION_ID", sid)
        session_dir = os.path.join(root, sid)
        os.makedirs(session_dir, exist_ok=True)
        # Pre-existing conversation from the child's first run.
        with open(os.path.join(session_dir, "conversation.json"), "w") as f:
            json.dump({"session_id": sid, "messages": [
                {"role": "user", "content": "original task"},
                {"role": "assistant", "content": "I got rejected"},
            ]}, f)
        rp = tmp_path / "resume.prompt"
        rp.write_text("please fix the edge case")
        monkeypatch.setenv("FLAGSCALE_RESUME_PATH", str(rp))

        agent = self._agent(monkeypatch, root, sid)
        assert agent._session_dir == session_dir
        agent._inject_context = lambda: None
        agent._react_loop = lambda: None  # stub; we assert state, not LLM

        agent._run_single_shot("ignored-contract-path")

        contents = [str(m.get("content", "")) for m in agent.history.messages]
        # Prior history is present (in-context continuation, not a fresh run)...
        assert any("original task" in c for c in contents)
        assert any("I got rejected" in c for c in contents)
        # ...and the parent's resume message is the newest user turn.
        assert contents[-1] == "please fix the edge case"

    def test_no_resume_path_is_normal_run(self, tmp_path, monkeypatch):
        root = str(tmp_path / "root")
        monkeypatch.setenv("FLAGSCALE_SESSION_ROOT", root)
        monkeypatch.setenv("FLAGSCALE_SESSION_ID", "child2")
        agent = self._agent(monkeypatch, root, "child2")
        agent._inject_context = lambda: None
        agent._react_loop = lambda: None
        agent._run_single_shot("a normal contract")
        contents = [str(m.get("content", "")) for m in agent.history.messages]
        assert "a normal contract" in contents


class _RecordingWatchdog:
    """Stand-in for _Watchdog that records construction and start()."""

    instances = []

    def __init__(self, ledger, task_id, pid, deadline_epoch, interval=5.0):
        self.ledger = ledger
        self.task_id = task_id
        self.pid = pid
        self.deadline_epoch = deadline_epoch
        self.started = False
        _RecordingWatchdog.instances.append(self)

    def start(self):
        self.started = True


class TestResumeStartsWatchdog:
    """A resumed child must get the SAME parent-side watchdog a spawned child
    gets; otherwise a hung resumed worker is never reaped and its RUNNING task
    (an ACTIVE status) permanently occupies a global concurrency slot."""

    def _tool(self, led, parent_sess):
        return ResumeChildTool(ledger=led, session_dir=parent_sess)

    def test_watchdog_started_for_resumed_child(self, led, tmp_path, monkeypatch):
        import flagscale_agent.react.multi_agent.resume as resume_mod

        parent_sess = str(tmp_path / "sess")
        tid = _make_task(led, tmp_path, "P1", ["GP"])
        monkeypatch.setenv("FLAGSCALE_TASK_ID", "P1")
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc(pid=4242))
        _RecordingWatchdog.instances = []
        monkeypatch.setattr(resume_mod, "_Watchdog", _RecordingWatchdog)

        out = self._tool(led, parent_sess).execute(
            task_id=tid, message="fix it", deadline_minutes=3)
        assert out.startswith("resumed"), out
        assert len(_RecordingWatchdog.instances) == 1
        wd = _RecordingWatchdog.instances[0]
        assert wd.started is True
        assert wd.task_id == tid
        assert wd.pid == 4242
        # Deadline is per-run: now + 3 minutes (allow a little slack).
        assert wd.deadline_epoch - time.time() > 2.5 * 60

    def test_resumed_child_watchdog_kills_and_expires(self, led, tmp_path, monkeypatch):
        """End-to-end of the watchdog itself: a live process past its deadline
        is killed and its RUNNING task moves to DEADLINE_MISSED."""
        from flagscale_agent.react.multi_agent.ledger import DEADLINE_MISSED
        from flagscale_agent.react.multi_agent.spawn import _Watchdog

        # A real, long-lived child so os.killpg(pid,0) reports it alive.
        proc = subprocess.Popen(["sleep", "30"],
                                stdin=subprocess.DEVNULL,
                                start_new_session=True)
        try:
            tid = _make_task(led, tmp_path, "P1", ["GP"], status=RUNNING)
            # Deadline already in the past -> the first poll expires it.
            wd = _Watchdog(led, tid, proc.pid, int(time.time()) - 1, interval=0.05)
            wd.start()
            # Poll until the watchdog reaps + transitions (bounded wait).
            deadline = time.time() + 5
            while time.time() < deadline:
                if led.get(tid).status == DEADLINE_MISSED:
                    break
                time.sleep(0.05)
            assert led.get(tid).status == DEADLINE_MISSED
            # The child process group was SIGKILLed.
            assert proc.poll() is not None
        finally:
            try:
                proc.kill()
            except Exception:
                pass

    def test_parent_log_fd_closed_after_success(self, led, tmp_path, monkeypatch):
        """The parent's copy of worker.log must be closed on the success path
        (the child holds its own dup), or the long-lived parent leaks an fd."""
        import flagscale_agent.react.multi_agent.resume as resume_mod

        parent_sess = str(tmp_path / "sess")
        tid = _make_task(led, tmp_path, "P1", ["GP"])
        monkeypatch.setenv("FLAGSCALE_TASK_ID", "P1")
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
        monkeypatch.setattr(resume_mod, "_Watchdog", _RecordingWatchdog)
        _RecordingWatchdog.instances = []

        opened = []
        real_open = open

        def tracking_open(path, *a, **k):
            fh = real_open(path, *a, **k)
            opened.append(fh)
            return fh

        monkeypatch.setattr("builtins.open", tracking_open)
        out = self._tool(led, parent_sess).execute(task_id=tid, message="fix it")
        assert out.startswith("resumed"), out
        log_handles = [fh for fh in opened if str(getattr(fh, "name", "")).endswith("worker.log")]
        assert log_handles, "worker.log was never opened"
        assert all(fh.closed for fh in log_handles), "parent leaked a worker.log fd"
