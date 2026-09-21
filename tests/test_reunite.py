# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for parent-side reunite: check_result + PollTasksTool.

Covers the four acceptance cases plus the adversarial one:
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
    DEADLINE_MISSED,
    DONE,
    REJECTED,
    REPORTED,
    RUNNING,
    TERMINATED,
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


class TestSettledWithoutReport:
    """Regression for F1: DEADLINE_MISSED / TERMINATED must be settled, not
    pending-forever. The watchdog writes DEADLINE_MISSED and the worker can
    never reach REPORTED, so polling must terminate with passed=False."""

    def test_deadline_missed_is_settled_not_pending(self, led, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        c = Contract.build(
            goal="g", constraints={"writable": [str(work)]},
            acceptance=[{"check": "true"}], output_ptr=str(work / "out.md"),
        )
        led.create(c)
        led.transition(c.id, RUNNING, pid=1)
        led.transition(c.id, DEADLINE_MISSED, note="watchdog deadline")
        v = check_result(led, c.id)
        assert v.pending is False
        assert v.passed is False
        assert v.status == DEADLINE_MISSED
        assert "settled without report" in v.note
        # idempotent: a second poll stays settled
        v2 = check_result(led, c.id)
        assert v2.pending is False
        assert v2.status == DEADLINE_MISSED

    def test_terminated_is_settled_not_pending(self, led, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        c = Contract.build(
            goal="g", constraints={"writable": [str(work)]},
            acceptance=[{"check": "true"}], output_ptr=str(work / "out.md"),
        )
        led.create(c)
        led.transition(c.id, RUNNING, pid=1)
        led.transition(c.id, TERMINATED, note="reaped")
        v = check_result(led, c.id)
        assert v.pending is False
        assert v.passed is False
        assert v.status == TERMINATED


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


class TestClaimDivergence:
    """M5 acceptance closure: the parent surfaces a worker claim that disagrees
    with the parent's own measured result — the lie is named, not silent."""

    def test_claim_says_pass_but_checks_fail_flags_divergence(
            self, led, tmp_path):
        """The M5 exit smoke: a fake worker returns a false conclusion WITH
        evidence; the parent rejects on evidence and flags the divergence."""
        c, work = _make_reported(
            led, tmp_path,
            [{"check": "test -f out.md"}],
            products=[],  # worker produced nothing
        )
        # Worker asserts success, and attaches "evidence" (a self-run check
        # that it claims exited 0) — but the product is absent.
        led.write_result(c.id, {
            "summary": "DONE — wrote out.md",
            "claim": {"goal_met": True,
                      "checks_self_run": [{"check": "test -f out.md",
                                           "exit_code": 0}]},
            "files_written": [str(work / "out.md")],
            "output_ptr": str(work / "out.md"),
        })
        v = check_result(led, c.id, default_cwd=str(work))
        assert v.passed is False
        assert v.status == REJECTED
        assert v.claim is not None
        assert v.claim["goal_met"] is True
        assert v.claim_divergence is True          # mismatch made explicit
        assert "DIVERGENCE" in v.note
        assert led.get(c.id).status == REJECTED

    def test_claim_says_fail_but_checks_pass_flags_divergence(
            self, led, tmp_path):
        """The reverse lie: a worker under-claims while the product is fine."""
        c, work = _make_reported(
            led, tmp_path,
            [{"check": "test -f out.md"}],
            products=["out.md"],  # product IS present
        )
        led.write_result(c.id, {
            "summary": "I could not finish",
            "claim": {"goal_met": False},
            "output_ptr": str(work / "out.md"),
        })
        v = check_result(led, c.id, default_cwd=str(work))
        assert v.passed is True
        assert v.status == DONE
        assert v.claim_divergence is True
        assert "DIVERGENCE" in v.note

    def test_agreeing_claim_no_divergence(self, led, tmp_path):
        """When the claim agrees with the measured result, no flag is raised."""
        c, work = _make_reported(
            led, tmp_path,
            [{"check": "test -f out.md"}],
            products=["out.md"],
        )
        led.write_result(c.id, {
            "summary": "done",
            "claim": {"goal_met": True},
            "output_ptr": str(work / "out.md"),
        })
        v = check_result(led, c.id, default_cwd=str(work))
        assert v.passed is True
        assert v.claim_divergence is False
        assert "DIVERGENCE" not in v.note

    def test_absent_claim_no_divergence(self, led, tmp_path):
        """A worker that reported no claim cannot diverge (nothing asserted)."""
        c, work = _make_reported(
            led, tmp_path, [{"check": "test -f out.md"}], products=[])
        led.write_result(c.id, {"summary": "done without a claim"})
        v = check_result(led, c.id, default_cwd=str(work))
        assert v.claim is None
        assert v.claim_divergence is False

    def test_claim_bool_none_not_a_divergence(self, led, tmp_path):
        """goal_met=None asserts nothing definite → not a divergence."""
        c, work = _make_reported(
            led, tmp_path, [{"check": "test -f out.md"}], products=[])
        led.write_result(c.id, {
            "summary": "not sure",
            "claim": {"goal_met": None, "checks_self_run": []},
        })
        v = check_result(led, c.id, default_cwd=str(work))
        assert v.status == REJECTED
        assert v.claim_divergence is False

    def test_divergence_surfaced_in_poll_tool_output(self, led, tmp_path):
        """The model-visible poll output names the divergence."""
        c, work = _make_reported(
            led, tmp_path, [{"check": "test -f out.md"}], products=[])
        led.write_result(c.id, {
            "summary": "DONE",
            "claim": {"goal_met": True},
        })
        out = PollTasksTool(ledger=led).execute(action="check", task_id=c.id)
        assert "DIVERGENCE" in out
        assert "goal_met=True" in out

    def test_divergence_flagged_on_settled_without_report(self, led, tmp_path):
        """A reported-then-terminated task that left a false claim is flagged.

        REPORTED → TERMINATED is legal, so a stale claim can coexist with a
        settled-without-report status; the parent still surfaces the mismatch."""
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        c = Contract.build(
            goal="produce out.md",
            constraints={"writable": [str(work)]},
            acceptance=[{"check": "test -f out.md"}],
            output_ptr=str(work / "out.md"),
        )
        led.create(c)
        led.transition(c.id, RUNNING, pid=333)
        led.write_result(c.id, {
            "summary": "DONE!", "claim": {"goal_met": True}})
        led.transition(c.id, TERMINATED, note="cancelled by parent")
        v = check_result(led, c.id, default_cwd=str(work))
        assert v.status == TERMINATED
        assert v.passed is False
        assert v.claim_divergence is True


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
