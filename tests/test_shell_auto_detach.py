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

"""Tests for the foreground grace-period auto-detach mechanism.

A foreground command that runs past `auto_detach_grace_secs` is DETACHED to a
background job so the agent regains control (it can poll / kill / do other
work) instead of blocking blind. This is independent of the health judge and
of any task budget — the trigger is simply "elapsed > window".
"""

import time

import pytest

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


def test_grace_default_is_30():
    assert ShellTool()._auto_detach_grace == 30


def test_grace_zero_disables_autodetect():
    # With grace=0 the long command runs to completion synchronously.
    sh = ShellTool(auto_detach_grace_secs=0)
    out = sh.execute(command="sleep 1; echo done")
    assert "done" in out
    assert "DETACHED" not in out


def test_short_command_not_detached():
    # A quick command finishes well within the window — normal result.
    sh = ShellTool(auto_detach_grace_secs=5)
    out = sh.execute(command="echo hello")
    assert "hello" in out
    assert "DETACHED" not in out


def test_long_command_is_detached_and_process_survives():
    sh = ShellTool(auto_detach_grace_secs=1)
    t0 = time.time()
    # Command runs 3s — past the 1s grace window.
    out = sh.execute(command="sleep 3; echo finished")
    elapsed = time.time() - t0
    # Detach must fire promptly at the window edge, NOT wait the full 3s and
    # NOT wait for a 30s health tick.
    assert elapsed < 2.5, f"detach did not preempt promptly: {elapsed:.2f}s"
    assert "DETACHED as job" in out
    assert "grace window" in out.lower()
    # Lossless: nothing was killed, the job is registered and still running.
    assert len(_JOB_REGISTRY.all()) == 1
    job_id = _JOB_REGISTRY.all()[0].job_id
    # The process must still be alive right after detach.
    job = _JOB_REGISTRY.get(job_id)
    assert job.proc.poll() is None

    # The agent can keep waiting via shell_jobs and still get the final output.
    jobs = ShellJobsTool()
    res = jobs.execute(action="wait", job_id=job_id, timeout=10)
    assert "finished" in res


def test_detached_job_completes_in_background():
    sh = ShellTool(auto_detach_grace_secs=1)
    out = sh.execute(command="sleep 2; echo late-output")
    assert "DETACHED as job" in out
    job_id = _JOB_REGISTRY.all()[0].job_id
    jobs = ShellJobsTool()
    # Poll until it finishes; final output must be recoverable.
    deadline = time.time() + 10
    seen = ""
    while time.time() < deadline:
        seen = jobs.execute(action="poll", job_id=job_id)
        if "late-output" in seen:
            break
        time.sleep(0.2)
    assert "late-output" in seen
