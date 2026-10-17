# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for worker-side wiring helpers."""

import os

import pytest

from flagscale_agent.react.multi_agent.contract import Contract
from flagscale_agent.react.multi_agent.ledger import (
    FAILED, REPORTED, RUNNING, TaskLedger,
)
from flagscale_agent.react.multi_agent.wiring import (
    CONTRACT_PATH_ENV, MULTI_AGENT_NUDGE_SUFFIX, RESUME_PATH_ENV,
    TASK_ID_ENV, TASK_ROLE_ENV, WORKER_ROLE_PREFIX,
    finalize_worker_if_no_report, is_worker, resolve_worker_query,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Clear TASK_ROLE_ENV too: _role_prefix_for_env() reads it to pick the
    # leaf vs ordinary prefix. If the suite runs inside a reviewer/diverger
    # worker (FLAGSCALE_TASK_ROLE=reviewer), a leaked role flips the prefix to
    # the leaf one and the WORKER_ROLE_PREFIX assertions below fail. Tests must
    # be hermetic regardless of the parent's role env.
    for k in (TASK_ID_ENV, TASK_ROLE_ENV, CONTRACT_PATH_ENV, "FLAGSCALE_OUTPUT_DIR"):
        monkeypatch.delenv(k, raising=False)


class TestRoleDetection:
    def test_not_worker_without_env(self):
        assert is_worker() is False

    def test_is_worker_with_env(self, monkeypatch):
        monkeypatch.setenv(TASK_ID_ENV, "abc")
        assert is_worker() is True


class TestResolveQuery:
    def test_passthrough_when_not_worker(self, monkeypatch):
        assert resolve_worker_query("hi") == "hi"

    def test_worker_reads_contract_and_prefixes(self, tmp_path, monkeypatch):
        cpath = tmp_path / "contract.prompt"
        cpath.write_text("## 目标\n写文件", encoding="utf-8")
        monkeypatch.setenv(TASK_ID_ENV, "abc")
        monkeypatch.setenv(CONTRACT_PATH_ENV, str(cpath))
        out = resolve_worker_query(str(cpath))
        assert out.startswith(WORKER_ROLE_PREFIX)
        assert "写文件" in out

    def test_worker_no_contract_still_prefixes(self, monkeypatch):
        monkeypatch.setenv(TASK_ID_ENV, "abc")
        out = resolve_worker_query("raw")
        assert out.startswith(WORKER_ROLE_PREFIX)
        assert "raw" in out

    def test_nudge_constant_is_english_and_explains_why(self):
        """The appended nudge is a single English sentence explaining the ask."""
        text = MULTI_AGENT_NUDGE_SUFFIX
        assert "multi-agent" in text
        assert "concurrent" in text
        assert "efficiency" in text and "quality" in text
        assert "\n" not in text  # one line, appended after a blank line

    def test_user_task_nudge_helper_appends_once(self):
        """The wiring-level helper appends exactly one blank-line-separated nudge."""
        from flagscale_agent.react.agent import _run_single_shot_append_nudge

        q = _run_single_shot_append_nudge("analyze logs")
        assert q.startswith("analyze logs\n\n")
        assert q.count(MULTI_AGENT_NUDGE_SUFFIX) == 1
        # Idempotent: a query already carrying the nudge is not re-suffixed.
        q2 = _run_single_shot_append_nudge(q)
        assert q2 == q


class TestFinalizeNoReport:
    def _mk(self, tmp_path):
        led = TaskLedger(str(tmp_path / "tasks"))
        work = tmp_path / "work"
        work.mkdir()
        c = Contract.build(goal="g", constraints={"writable": [str(work)]},
                           acceptance=[{"check": "true"}],
                           output_ptr=str(work / "o.md"))
        led.create(c)
        led.transition(c.id, RUNNING, pid=1)
        return led, c

    def test_noop_when_not_worker(self, tmp_path):
        led, c = self._mk(tmp_path)
        assert finalize_worker_if_no_report(led) is None
        assert led.get(c.id).status == RUNNING

    def test_closes_dangling_running(self, tmp_path, monkeypatch):
        led, c = self._mk(tmp_path)
        monkeypatch.setenv(TASK_ID_ENV, c.id)
        res = finalize_worker_if_no_report(led)
        assert res == c.id
        assert led.get(c.id).status == FAILED

    def test_noop_when_already_reported(self, tmp_path, monkeypatch):
        led, c = self._mk(tmp_path)
        led.write_result(c.id, {"summary": "ok"})  # → REPORTED
        monkeypatch.setenv(TASK_ID_ENV, c.id)
        assert finalize_worker_if_no_report(led) is None
        assert led.get(c.id).status == REPORTED
