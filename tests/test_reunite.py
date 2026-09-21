# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for parent-side reunite: check_result + PollTasksTool (design §2.4, M3).

Covers the four M3 acceptance cases plus the adversarial one:
  * check passes  → REPORTED → DONE
  * check fails   → REPORTED → REJECTED, note carries stdout_tail
  * not ready     → pending, NO ledger mutation
  * terminal      → idempotent, returned as-is
  * ADVERSARIAL (§1.4 rule 5 / §8 trust collapse): a worker that forges a
    result.json claiming success BUT left the product missing must STILL be
    REJECTED — because the parent only runs the acceptance predicate itself
    (independent VERIFICATION, not re-execution) and never reads result.json.
    This is the anti-forgery safety valve.
"""

import os

import pytest

from flagscale_agent.react.multi_agent.contract import Contract
from flagscale_agent.react.multi_agent.ledger import (
    DONE,
    REJECTED,
    REPORTED,
    RUNNING,
    TaskLedger,
)
from flagscale_agent.react.multi_agent.reunite import (
    PollTasksTool,
    Verdict,
    check_result,
)


@pytest.fixture
def tasks_dir(tmp_path):
    return str(tmp_path / "tasks")


@pytest.fixture
def led(tasks_dir):
    return TaskLedger(tasks_dir)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FLAGSCALE_TASK_ID", raising=False)


def _make_reported(led, tmp_path, checks, goal="produce out.md", products=()):
    """Create a task, mark it RUNNING→REPORTED, optionally write products."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    for rel in products:
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("hello\n", encoding="utf-8")
    c = Contract.build(
        goal=goal,
        constraints={"writable": [str(work)]},
        acceptance=checks,
        output_ptr=str(work / "out.md"),
    )
    led.create(c)
    led.transition(c.id, RUNNING, pid=222)
    led.transition(c.id, REPORTED, note="worker reported")
    return c, work


class TestCheckPass:
    def test_all_checks_pass_transitions_done(self, led, tmp_path):
        c, work = _make_reported(
            led, tmp_path,
            [{"check": "test -f out.md"}],
            products=["out.md"],
        )
        v = check_result(led, c.id, default_cwd=str(work))
        assert isinstance(v, Verdict)
        assert v.passed is True
        assert v.status == DONE
        assert v.pending is False
        assert len(v.checks) == 1
        assert v.checks[0].exit_code == 0
        # ledger actually moved
        assert led.get(c.id).status == DONE

    def test_relative_cwd_falls_back_to_default(self, led, tmp_path):
        c, work = _make_reported(
            led, tmp_path,
            [{"check": "test -f out.md", "cwd": "relative/ignored"}],
            products=["out.md"],
        )
        v = check_result(led, c.id, default_cwd=str(work))
        assert v.passed is True
        assert v.checks[0].cwd == str(work)


class TestCheckFail:
    def test_failing_check_transitions_rejected(self, led, tmp_path):
        c, work = _make_reported(
            led, tmp_path,
            [{"check": "test -f out.md"}],
            products=[],  # product never written
        )
        v = check_result(led, c.id, default_cwd=str(work))
        assert v.passed is False
        assert v.status == REJECTED
        assert led.get(c.id).status == REJECTED

    def test_rejected_note_carries_stdout_tail(self, led, tmp_path):
        c, work = _make_reported(
            led, tmp_path,
            [{"check": "echo MARKER_XYZ && exit 3"}],
        )
        v = check_result(led, c.id, default_cwd=str(work))
        assert v.passed is False
        assert "MARKER_XYZ" in v.note
        assert v.checks[0].stdout_tail.strip().endswith("MARKER_XYZ")
        assert v.checks[0].exit_code == 3

    def test_timeout_counts_as_failure(self, led, tmp_path):
        c, work = _make_reported(
            led, tmp_path,
            [{"check": "sleep 5"}],
        )
        v = check_result(led, c.id, timeout_s=1, default_cwd=str(work))
        assert v.passed is False
        assert v.checks[0].timed_out is True
        assert v.status == REJECTED


class TestNotReady:
    def test_running_is_pending_no_mutation(self, led, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        c = Contract.build(
            goal="g", constraints={"writable": [str(work)]},
            acceptance=[{"check": "true"}], output_ptr=str(work / "out.md"),
        )
        led.create(c)
        led.transition(c.id, RUNNING, pid=1)
        v = check_result(led, c.id)
        assert v.pending is True
        assert v.passed is False
        # ledger NOT changed
        assert led.get(c.id).status == RUNNING


class TestTerminalIdempotent:
    def test_terminal_returned_as_is(self, led, tmp_path):
        c, work = _make_reported(
            led, tmp_path,
            [{"check": "true"}],
            products=[],
        )
        # force terminal directly
        led.transition(c.id, REJECTED, note="pre")
        v = check_result(led, c.id)
        assert v.pending is False
        assert v.status == REJECTED
        assert v.passed is False

    def test_missing_task(self, led):
        v = check_result(led, "deadbeef")
        assert v.pending is False
        assert v.passed is False
        assert "no such task" in v.note


class TestAdversarial:
    """§1.4 rule 5 / §8 trust collapse — forge a self-report, skip the product."""

    def test_forged_result_json_still_rejected(self, led, tmp_path):
        c, work = _make_reported(
            led, tmp_path,
            [{"check": "test -f out.md"}],
            products=[],  # no product on disk
        )
        # Worker lies: writes a result.json claiming it did the work.
        led.write_result(c.id, {
            "summary": "DONE! wrote out.md successfully",
            "files_written": [str(work / "out.md")],
            "output_ptr": str(work / "out.md"),
        })
        # Even though result.json claims success, the parent VERIFIES the product
        # (runs the acceptance predicate) and rejects — it never redoes the task.
        v = check_result(led, c.id, default_cwd=str(work))
        assert v.passed is False
        assert v.status == REJECTED
        assert v.self_report is not None          # carried for reference only
        assert "DONE" in v.self_report["summary"]  # the lie is visible
        assert led.get(c.id).status == REJECTED


class TestPollTasksTool:
    def test_list(self, led, tmp_path):
        _make_reported(led, tmp_path, [{"check": "true"}])
        out = PollTasksTool(ledger=led).execute(action="list")
        assert "task list:" in out
        assert "REPORTED" in out

    def test_check_action_pass(self, led, tmp_path):
        c, work = _make_reported(
            led, tmp_path,
            [{"check": f"test -f {tmp_path}/work/out.md"}],
            products=["out.md"],
        )
        out = PollTasksTool(ledger=led).execute(action="check", task_id=c.id)
        assert "PASSED" in out
        assert led.get(c.id).status == DONE

    def test_check_action_pending(self, led, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        c = Contract.build(
            goal="g", constraints={"writable": [str(work)]},
            acceptance=[{"check": "true"}], output_ptr=str(work / "out.md"),
        )
        led.create(c)
        led.transition(c.id, RUNNING, pid=1)
        out = PollTasksTool(ledger=led).execute(action="check", task_id=c.id)
        assert "not ready" in out

    def test_result_action(self, led, tmp_path):
        c, work = _make_reported(
            led, tmp_path, [{"check": "true"}], products=[],
        )
        led.write_result(c.id, {"summary": "ref only"})
        out = PollTasksTool(ledger=led).execute(action="result", task_id=c.id)
        assert "self-report" in out and "NOT acceptance evidence" in out
        assert "ref only" in out

    def test_bad_action(self, led):
        out = PollTasksTool(ledger=led).execute(action="bogus")
        assert out.startswith("ERROR")

    def test_check_requires_task_id(self, led):
        out = PollTasksTool(ledger=led).execute(action="check")
        assert out.startswith("ERROR")
