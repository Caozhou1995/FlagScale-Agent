# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for worker-side ReportResultTool (design §2.5, M2)."""

import os

import pytest

from flagscale_agent.react.multi_agent.contract import Contract
from flagscale_agent.react.multi_agent.ledger import (
    REPORTED,
    RUNNING,
    DONE,
    TaskLedger,
)
from flagscale_agent.react.multi_agent.report_result import ReportResultTool


@pytest.fixture
def tasks_dir(tmp_path):
    return str(tmp_path / "tasks")


@pytest.fixture
def led(tasks_dir):
    return TaskLedger(tasks_dir)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FLAGSCALE_TASK_ID", raising=False)


def _make_running(led, tmp_path, goal="do work"):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    c = Contract.build(
        goal=goal,
        constraints={"writable": [str(work)]},
        acceptance=[{"check": "test -f out.md"}],
        output_ptr=str(work / "out.md"),
    )
    led.create(c)
    led.transition(c.id, RUNNING, pid=111)
    return c


class TestNonWorker:
    def test_absent_env_returns_error(self, led):
        tool = ReportResultTool(ledger=led)
        out = tool.execute(summary="done")
        assert out.startswith("ERROR")
        assert "FLAGSCALE_TASK_ID" in out or "worker" in out

    def test_empty_summary_rejected(self, led, monkeypatch):
        monkeypatch.setenv("FLAGSCALE_TASK_ID", "abc")
        tool = ReportResultTool(ledger=led)
        out = tool.execute(summary="   ")
        assert out.startswith("ERROR")


class TestINV4Writable:
    def test_file_outside_writable_rejected(self, led, tmp_path, monkeypatch):
        c = _make_running(led, tmp_path)
        monkeypatch.setenv("FLAGSCALE_TASK_ID", c.id)
        tool = ReportResultTool(ledger=led)
        out = tool.execute(summary="done", files_written=["/etc/hosts"])
        assert out.startswith("ERROR")
        assert "writable" in out or "INV4" in out
        # state must NOT have moved
        assert led.get(c.id).status == RUNNING

    def test_file_inside_writable_ok(self, led, tmp_path, monkeypatch):
        c = _make_running(led, tmp_path)
        monkeypatch.setenv("FLAGSCALE_TASK_ID", c.id)
        tool = ReportResultTool(ledger=led)
        out = tool.execute(summary="done",
                           files_written=[str(tmp_path / "work" / "out.md")])
        assert out.startswith("reported")
        assert led.get(c.id).status == REPORTED

    def test_output_ptr_inside_writable_ok(self, led, tmp_path, monkeypatch):
        c = _make_running(led, tmp_path)
        monkeypatch.setenv("FLAGSCALE_TASK_ID", c.id)
        tool = ReportResultTool(ledger=led)
        # output_ptr is inside writable by construction → passes
        out = tool.execute(summary="done")
        assert out.startswith("reported")


class TestReportedTransition:
    def test_result_written_and_reported(self, led, tmp_path, monkeypatch):
        c = _make_running(led, tmp_path)
        monkeypatch.setenv("FLAGSCALE_TASK_ID", c.id)
        tool = ReportResultTool(ledger=led)
        tool.execute(summary="all good")
        res = led.read_result(c.id)
        assert res["summary"] == "all good"
        assert led.get(c.id).status == REPORTED

    def test_reported_is_not_done(self, led, tmp_path, monkeypatch):
        """Self-report ≠ acceptance (INV3): still REPORTED, not DONE."""
        c = _make_running(led, tmp_path)
        monkeypatch.setenv("FLAGSCALE_TASK_ID", c.id)
        ReportResultTool(ledger=led).execute(summary="trust me")
        assert led.get(c.id).status == REPORTED
        assert led.get(c.id).status != DONE

    def test_missing_task_error(self, led, monkeypatch):
        monkeypatch.setenv("FLAGSCALE_TASK_ID", "ghost000")
        out = ReportResultTool(ledger=led).execute(summary="x")
        assert out.startswith("ERROR")


class TestClaim:
    """The worker's structured, UNTRUSTED claim is stored for the parent to
    compare against — never trusted as acceptance."""

    def test_claim_stored_in_result_json(self, led, tmp_path, monkeypatch):
        c = _make_running(led, tmp_path)
        monkeypatch.setenv("FLAGSCALE_TASK_ID", c.id)
        tool = ReportResultTool(ledger=led)
        out = tool.execute(
            summary="did it",
            claim={"goal_met": True,
                   "checks_self_run": [{"check": "test -f out.md", "exit_code": 0}]},
        )
        assert out.startswith("reported")
        res = led.read_result(c.id)
        assert res["claim"]["goal_met"] is True
        assert res["claim"]["checks_self_run"] == [
            {"check": "test -f out.md", "exit_code": 0}]

    def test_absent_claim_normalizes_to_empty(self, led, tmp_path, monkeypatch):
        c = _make_running(led, tmp_path)
        monkeypatch.setenv("FLAGSCALE_TASK_ID", c.id)
        ReportResultTool(ledger=led).execute(summary="--")
        res = led.read_result(c.id)
        assert res["claim"] == {"goal_met": None, "checks_self_run": []}

    def test_invalid_claim_shape_rejected(self, led, tmp_path, monkeypatch):
        c = _make_running(led, tmp_path)
        monkeypatch.setenv("FLAGSCALE_TASK_ID", c.id)
        tool = ReportResultTool(ledger=led)
        out = tool.execute(summary="x", claim={"goal_met": "yes"})  # not a bool
        assert out.startswith("ERROR")
        assert led.get(c.id).status == RUNNING  # state untouched

    def test_claim_with_bad_self_run_check_rejected(self, led, tmp_path, monkeypatch):
        c = _make_running(led, tmp_path)
        monkeypatch.setenv("FLAGSCALE_TASK_ID", c.id)
        tool = ReportResultTool(ledger=led)
        out = tool.execute(
            summary="x",
            claim={"goal_met": True, "checks_self_run": [{"check": "true"}]},
        )  # missing exit_code
        assert out.startswith("ERROR")

    def test_false_claim_still_reports(self, led, tmp_path, monkeypatch):
        """A worker may honestly claim failure — that is still a valid report."""
        c = _make_running(led, tmp_path)
        monkeypatch.setenv("FLAGSCALE_TASK_ID", c.id)
        ReportResultTool(ledger=led).execute(
            summary="could not finish", claim={"goal_met": False})
        assert led.get(c.id).status == REPORTED
        assert led.read_result(c.id)["claim"]["goal_met"] is False
