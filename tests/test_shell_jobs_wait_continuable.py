# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for making a long `wait` CONTINUABLE rather than merely blocking.

Two mechanisms, tested separately because they fail differently:

  * the guard stops blocking a long wait that carries `return_early_after`
    (a long wait WITHOUT it is still blocked — that is the behaviour it exists
    to prevent, and it must survive this change untouched);
  * the tool's `return_early_after` / `wait_any` actually return early and
    KEEP the job handle, so nothing is cancelled and the agent can come back.

The registry-survival assertions are the point: a wait that quietly dropped the
job on an early return would look identical to the caller and lose the job.
"""

import time

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.shell_jobs_wait import ShellJobsWaitGuard
from flagscale_agent.react.tools.shell import (
    ShellTool,
    ShellJobsTool,
    _JOB_REGISTRY,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    _JOB_REGISTRY.cleanup_all()
    _JOB_REGISTRY._counter = 0
    yield
    _JOB_REGISTRY.cleanup_all()
    _JOB_REGISTRY._counter = 0


def _sj(**args):
    return GuardContext(tool_name="shell_jobs", tool_args=args)


# ── Guard: a long wait WITH an early cap is allowed ─────────────────────


class TestGuardAllowsEarlyCap:
    def test_long_wait_with_early_cap_allowed(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj(action="wait", timeout=600,
                               return_early_after=30)) is None

    def test_long_wait_with_early_cap_as_string_allowed(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj(action="wait", timeout="600",
                               return_early_after="30")) is None

    def test_wait_any_with_early_cap_allowed(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj(action="wait_any", job_ids=["job1", "job2"],
                               timeout=600, return_early_after=30)) is None


class TestGuardStillBlocksIdleWaits:
    """The original contract must survive byte for byte."""

    def test_long_wait_without_early_cap_still_blocks(self):
        g = ShellJobsWaitGuard()
        v = g.check_pre(_sj(action="wait", timeout=600))
        assert v is not None and v.action == "block"
        assert v.reason == "long_blocking_wait"

    def test_early_cap_at_or_above_timeout_still_blocks(self):
        # return_early_after >= timeout is a no-op cap: it blocks the whole
        # time under another name, so it must NOT buy an exemption.
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj(action="wait", timeout=600,
                               return_early_after=600)).action == "block"
        assert g.check_pre(_sj(action="wait", timeout=600,
                               return_early_after=900)).action == "block"

    def test_zero_or_negative_early_cap_still_blocks(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj(action="wait", timeout=600,
                               return_early_after=0)).action == "block"
        assert g.check_pre(_sj(action="wait", timeout=600,
                               return_early_after=-5)).action == "block"

    def test_long_wait_any_without_early_cap_blocks(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj(action="wait_any", job_ids=["job1"],
                               timeout=600)).action == "block"

    def test_short_wait_with_early_cap_allowed(self):
        g = ShellJobsWaitGuard()
        assert g.check_pre(_sj(action="wait", timeout=10,
                               return_early_after=5)) is None


class TestGuardMessageGivesCopyableCommand:
    """The block message must hand over a runnable alternative, not a linter."""

    def test_message_contains_return_early_after_form(self):
        g = ShellJobsWaitGuard()
        msg = g.check_pre(_sj(action="wait", timeout=600)).message
        assert "return_early_after" in msg

    def test_message_contains_wait_any_form(self):
        g = ShellJobsWaitGuard()
        msg = g.check_pre(_sj(action="wait", timeout=600)).message
        assert "wait_any" in msg
        assert "job_ids" in msg

    def test_message_contains_poll_and_short_wait(self):
        g = ShellJobsWaitGuard()
        msg = g.check_pre(_sj(action="wait", timeout=600)).message
        assert "poll" in msg
        assert "60" in msg
        assert "_override_reason" in msg

    def test_message_has_no_unrendered_placeholder(self):
        g = ShellJobsWaitGuard()
        msg = g.check_pre(_sj(action="wait", timeout=600)).message
        assert "{" not in msg and "}" not in msg


# ── Tool: return_early_after ────────────────────────────────────────────


class TestReturnEarlyAfter:
    def test_returns_early_and_keeps_handle(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 20", background=True)
        t0 = time.time()
        res = jobs.execute(action="wait", job_id="job1", timeout=120,
                           return_early_after=1)
        elapsed = time.time() - t0
        assert elapsed < 5, f"return_early_after=1 blocked for {elapsed:.2f}s"
        # The handle SURVIVED: this is what makes the early return safe.
        assert _JOB_REGISTRY.get("job1") is not None
        assert "early" in res
        assert "job1" in res
        assert "finished" not in res.split("\n")[0].lower()

    def test_early_return_text_says_not_cancelled_and_offers_empty_work(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 20", background=True)
        res = jobs.execute(action="wait", job_id="job1", timeout=120,
                           return_early_after=1)
        assert "NOT cancelled" in res
        assert "poll" in res

    def test_without_early_cap_old_timeout_text_is_unchanged(self):
        # Backward compatibility: no return_early_after -> the historical
        # "still running after Ns wait" wording, byte for byte.
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 20", background=True)
        res = jobs.execute(action="wait", job_id="job1", timeout=1)
        assert "still running after 1s wait" in res
        assert "early" not in res.split("\n")[0]

    def test_without_early_cap_still_blocks_the_FULL_timeout(self):
        """LOWER bound: the default wait must not start returning early.

        The repo suite only asserts upper bounds on wait (`elapsed < 3`), so a
        regression that made the DEFAULT path return early would pass every
        existing test while silently changing the tool's contract for every
        caller that never asked for it. This is the other side of that bound.
        """
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 30", background=True)
        t0 = time.time()
        jobs.execute(action="wait", job_id="job1", timeout=3)
        elapsed = time.time() - t0
        assert elapsed >= 2.9, (
            f"default wait on a still-running job returned after {elapsed:.2f}s "
            f"for timeout=3 — it must block the full timeout"
        )

    def test_early_cap_above_timeout_uses_timeout(self):
        # cap = min(timeout, early): an early cap larger than the timeout must
        # not extend the block past the timeout.
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 20", background=True)
        t0 = time.time()
        jobs.execute(action="wait", job_id="job1", timeout=1,
                     return_early_after=999)
        assert time.time() - t0 < 5
        assert _JOB_REGISTRY.get("job1") is not None

    def test_early_return_is_not_inflated_by_stream_joins(self):
        """An early return must hand control back AT the cap, not cap+1s.

        The two output-stream joins cost a fixed extra second when the job is
        still running (both reader threads are alive). Paying that on the
        historical path is fine — that path is byte-for-byte frozen — but on an
        early return it defeats the purpose: the agent asked to be let go early
        and would still lose a second per check-in, which over a supervision
        loop is the same idle tax under a different name.
        """
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 30", background=True)
        t0 = time.time()
        jobs.execute(action="wait", job_id="job1", timeout=300,
                     return_early_after=2)
        elapsed = time.time() - t0
        assert elapsed < 2.8, f"early return took {elapsed:.2f}s for a 2s cap"

    def test_early_return_still_reports_output_produced_so_far(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="echo partial; sleep 20", background=True)
        # Give the echo a moment to land in the buffers.
        deadline = time.time() + 5
        while time.time() < deadline:
            job = _JOB_REGISTRY.get("job1")
            if job and "partial" in "".join(job.stdout_chunks):
                break
            time.sleep(0.05)
        res = jobs.execute(action="wait", job_id="job1", timeout=120,
                           return_early_after=1)
        assert "partial" in res

    def test_finished_job_returns_finished_text_not_early_text(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="echo done", background=True)
        res = jobs.execute(action="wait", job_id="job1", timeout=30,
                           return_early_after=1)
        assert "finished" in res
        assert _JOB_REGISTRY.get("job1") is None


# ── Tool: wait_any ──────────────────────────────────────────────────────


class TestWaitAny:
    def test_returns_when_first_of_two_finishes(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 0.3", background=True)   # job1, finishes
        sh.execute(command="sleep 30", background=True)    # job2, keeps running
        t0 = time.time()
        res = jobs.execute(action="wait_any", job_ids=["job1", "job2"],
                           timeout=20)
        elapsed = time.time() - t0
        assert elapsed < 10, f"wait_any waited {elapsed:.2f}s for a 0.3s job"
        # The finished one is gone; the running one KEEPS its handle.
        assert _JOB_REGISTRY.get("job1") is None
        assert _JOB_REGISTRY.get("job2") is not None
        assert "FINISHED" in res
        assert "still running" in res

    def test_does_not_wait_for_the_slow_one(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 0.2", background=True)
        sh.execute(command="sleep 40", background=True)
        t0 = time.time()
        jobs.execute(action="wait_any", job_ids=["job1", "job2"], timeout=35)
        assert time.time() - t0 < 10

    def test_unknown_job_is_an_error_not_a_hang(self):
        jobs = ShellJobsTool()
        res = jobs.execute(action="wait_any", job_ids=["nope"], timeout=5)
        assert "no such job" in res.lower()

    def test_missing_job_ids_is_an_error(self):
        jobs = ShellJobsTool()
        res = jobs.execute(action="wait_any", timeout=5)
        assert "job_ids" in res

    def test_single_string_job_id_is_coerced(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="echo quick", background=True)
        res = jobs.execute(action="wait_any", job_ids="job1", timeout=20)
        assert "FINISHED" in res

    def test_timeout_reports_all_still_running_and_keeps_handles(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 20", background=True)
        sh.execute(command="sleep 20", background=True)
        res = jobs.execute(action="wait_any", job_ids=["job1", "job2"], timeout=1)
        assert "No job finished" in res
        assert _JOB_REGISTRY.get("job1") is not None
        assert _JOB_REGISTRY.get("job2") is not None
        assert "still running" in res

    def test_early_cap_on_wait_any_keeps_every_handle(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 20", background=True)
        sh.execute(command="sleep 20", background=True)
        t0 = time.time()
        res = jobs.execute(action="wait_any", job_ids=["job1", "job2"],
                           timeout=120, return_early_after=1)
        assert time.time() - t0 < 5
        assert "NOTHING was cancelled" in res
        assert _JOB_REGISTRY.get("job1") is not None
        assert _JOB_REGISTRY.get("job2") is not None

    def test_all_finished_returns_every_output(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="echo one", background=True)
        sh.execute(command="echo two", background=True)
        res = jobs.execute(action="wait_any", job_ids=["job1", "job2"], timeout=20)
        assert "one" in res
        # job2 may still have been running when wait_any returned on job1; only
        # assert the finished one was removed and the call returned promptly.
        assert _JOB_REGISTRY.get("job1") is None


# ── Health supervision must survive the refactor ────────────────────────


class _StubEvaluator:
    def __init__(self, should_kill):
        self._should_kill = should_kill

    def evaluate(self, sampler, out, err, elapsed):
        return {
            "status": "ok",
            "should_kill": self._should_kill,
            "kill_reason": "stalled loop" if self._should_kill else "",
            "activity": "running",
            "recent_text": "",
            "output_changed": True,
            "stall_count": 0,
        }


class _StubSampler:
    def stop(self):
        pass


def _attach_stub_health(job, should_kill):
    job.evaluator = _StubEvaluator(should_kill)
    job.health_sampler = _StubSampler()


class TestHealthSupervisionSurvives:
    """A refactor that keeps the blocking loop must keep the health tick.

    The in-wait tick is the ONLY thing that can catch a backgrounded job that
    hangs while the agent is blocked on it. If `return_early_after` moved the
    loop somewhere the tick no longer runs, a hang would go unnoticed — a
    silent capability loss that no timing assertion would reveal.
    """

    def test_hang_is_killed_inside_an_early_capped_wait(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 20", background=True)
        job = _JOB_REGISTRY.get("job1")
        _attach_stub_health(job, should_kill=True)
        res = jobs.execute(action="wait", job_id="job1", timeout=120,
                           return_early_after=30)
        assert "TERMINATED" in res
        assert _JOB_REGISTRY.get("job1") is None

    def test_hang_is_killed_inside_wait_any(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 20", background=True)
        sh.execute(command="sleep 20", background=True)
        _attach_stub_health(_JOB_REGISTRY.get("job1"), should_kill=True)
        res = jobs.execute(action="wait_any", job_ids=["job1", "job2"],
                           timeout=30)
        assert "TERMINATED" in res

    def test_healthy_job_is_not_killed_by_the_tick(self):
        sh, jobs = ShellTool(), ShellJobsTool()
        sh.execute(command="sleep 3", background=True)
        _attach_stub_health(_JOB_REGISTRY.get("job1"), should_kill=False)
        res = jobs.execute(action="wait", job_id="job1", timeout=15,
                           return_early_after=10)
        assert "TERMINATED" not in res
        assert "finished" in res
