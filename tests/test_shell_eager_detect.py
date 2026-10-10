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

"""Tests for the eager background-hang detector (_EagerDetector).

A detached job is health-checked continuously by a detector that rides on the
job's own long-lived sampler thread, queueing a one-line event when the job
looks hung. The event is drained and surfaced the next time the agent calls
shell_jobs(poll/wait), so a job the agent forgot about still warns.
"""

import time

import pytest

from flagscale_agent.react.tools.shell import (
    ShellTool,
    ShellJobsTool,
    _JOB_REGISTRY,
    _BackgroundJob,
    _EagerDetector,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    _JOB_REGISTRY.cleanup_all()
    _JOB_REGISTRY._counter = 0
    yield
    _JOB_REGISTRY.cleanup_all()
    _JOB_REGISTRY._counter = 0


class _FakeJob:
    """Minimal stand-in for a _BackgroundJob exposing only what the detector
    touches: queue_event, job_id, start, stdout/stderr chunk lists."""

    def __init__(self, job_id="job1", start=None):
        self.job_id = job_id
        self.start = start if start is not None else time.time()
        self.stdout_chunks = []
        self.stderr_chunks = []
        self.events = []

    def queue_event(self, msg):
        self.events.append(msg)


class _StubEvaluator:
    """Returns a scripted response sequence; records call count."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def evaluate(self, sampler, stdout_chunks, stderr_chunks, elapsed):
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return {'status': 'skip'}


def _ok_kill(reason="no progress"):
    return {'status': 'ok', 'should_kill': True, 'kill_reason': reason,
            'activity': 'stalled', 'output_changed': False, 'stall_count': 5,
            'recent_text': ''}


def _ok_alive():
    return {'status': 'ok', 'should_kill': False, 'kill_reason': '',
            'activity': 'working', 'output_changed': True, 'stall_count': 0,
            'recent_text': ''}


def test_detector_queues_event_on_hang():
    job = _FakeJob()
    det = _EagerDetector(job)
    det._evaluator = _StubEvaluator([_ok_kill("tree idle")])
    det(sampler=None)
    assert len(job.events) == 1
    assert "job1" in job.events[0]
    assert "looks hung" in job.events[0]
    assert "tree idle" in job.events[0]


def test_detector_does_not_repeat_while_still_hung():
    job = _FakeJob()
    det = _EagerDetector(job)
    det._evaluator = _StubEvaluator([_ok_kill(), _ok_kill(), _ok_kill()])
    det(sampler=None)
    det(sampler=None)
    det(sampler=None)
    # Only the FIRST hang is reported; the flag suppresses the rest.
    assert len(job.events) == 1
    assert det._evaluator.calls == 3


def test_detector_rearms_after_recovery():
    job = _FakeJob()
    det = _EagerDetector(job)
    # hang -> healthy -> hang again
    det._evaluator = _StubEvaluator([_ok_kill(), _ok_alive(), _ok_kill()])
    det(sampler=None)
    det(sampler=None)   # recovery re-arms
    det(sampler=None)
    assert len(job.events) == 2


def test_detector_ignores_skip_and_healthy():
    job = _FakeJob()
    det = _EagerDetector(job)
    det._evaluator = _StubEvaluator([
        {'status': 'skip'}, _ok_alive(), _ok_alive(),
    ])
    det(sampler=None)
    det(sampler=None)
    det(sampler=None)
    assert job.events == []


def test_detector_survives_evaluator_exception():
    job = _FakeJob()
    det = _EagerDetector(job)

    class _Boom:
        def evaluate(self, *a, **k):
            raise RuntimeError("boom")

    det._evaluator = _Boom()
    # Must not raise; the sampler thread would otherwise die.
    det(sampler=None)
    assert job.events == []


def test_queue_and_drain_events_roundtrip():
    # Real _BackgroundJob: queue + drain via the registry path.
    job = _BackgroundJob(
        "job9", "echo hi", None, [], [], None, None, time.time(),
        health_sampler=None,
    )
    assert job.drain_events() == []
    job.queue_event("a")
    job.queue_event("b")
    assert job.drain_events() == ["a", "b"]
    # Drain clears.
    assert job.drain_events() == []


def test_registry_add_binds_detector_to_sampler():
    # A background launch with a live sampler must bind an eager detector.
    sh = ShellTool()
    out = sh.execute(command="sleep 5", background=True)
    assert "BACKGROUNDED as job" in out
    job = _JOB_REGISTRY.get("job1")
    assert job is not None
    assert job.health_sampler is not None
    # The sampler carries a bound on_sample callback (the detector).
    assert callable(getattr(job.health_sampler, "_on_sample", None))


def test_poll_surfaces_queued_event():
    sh = ShellTool()
    jobs = ShellJobsTool()
    out = sh.execute(command="sleep 5", background=True)
    job_id = "job1"
    assert job_id in out
    job = _JOB_REGISTRY.get(job_id)
    # Simulate the detector having queued a hang event while unpolled.
    job.queue_event(f"{job_id} looks hung after 42s (no progress)")
    p = jobs.execute(action="poll", job_id=job_id)
    assert "background health" in p
    assert "looks hung after 42s" in p
    # Draining is one-shot: a second poll does NOT repeat it.
    p2 = jobs.execute(action="poll", job_id=job_id)
    assert "looks hung after 42s" not in p2
