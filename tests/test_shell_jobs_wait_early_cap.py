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

"""The `return_early_after` cap must itself be SHORT.

The escape hatch used to accept any cap strictly smaller than the timeout, which
made the guard a no-op: the cheapest way to satisfy that test is a cap only
marginally smaller than the blocking timeout. Measured on train-fasttext
(2026-11-12) every wait carried exactly such a cap and every one blocked for
roughly the full timeout -- three completed waits totalled 446.4s against 168.9s
of real LLM generation, i.e. 72.5% of the agent's active time sitting idle.

These tests pin the fix in BOTH directions: the hole stays closed, and the
legitimate short-cap pattern still passes (a fix that refuses every early cap
would break the documented escape hatch instead of repairing it).
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.shell_jobs_wait import (
    _MAX_WAIT_SEC,
    ShellJobsWaitGuard,
)


def _wait(action: str = "wait", **kwargs):
    ctx = GuardContext(
        messages=[],
        tool_name="shell_jobs",
        tool_args={"action": action, "job_id": "job1", **kwargs},
    )
    return ShellJobsWaitGuard().check_pre(ctx)


@pytest.mark.parametrize(
    "timeout,early",
    [
        (120, 100),
        (200, 180),
        (260, 240),
        (320, 300),
        (100, 90),
        (600, 599),
    ],
)
def test_marginal_early_cap_is_blocked(timeout, early):
    v = _wait(timeout=timeout, return_early_after=early)
    assert v is not None, (
        "a cap of {}s on a {}s timeout blocks for ~{}s of idle time and must be "
        "blocked: a cap merely SMALLER than the timeout is the same idling "
        "under a different name".format(early, timeout, early)
    )
    assert v.action == "block"
    assert v.reason == "long_blocking_wait"


def test_long_wait_without_any_cap_is_blocked():
    assert _wait(timeout=600) is not None
    assert _wait(timeout=320) is not None


def test_short_early_cap_on_a_long_wait_still_passes():
    assert _wait(timeout=600, return_early_after=30) is None
    assert _wait(timeout=600, return_early_after=60) is None
    assert _wait(timeout=320, return_early_after=20) is None


def test_cap_boundary_is_inclusive():
    assert _wait(timeout=600, return_early_after=_MAX_WAIT_SEC) is None
    assert _wait(timeout=600, return_early_after=_MAX_WAIT_SEC + 1) is not None


def test_short_waits_are_never_blocked():
    assert _wait(timeout=60) is None
    assert _wait(timeout=30) is None
    assert _wait() is None


def test_poll_is_never_blocked():
    assert _wait(action="poll", timeout=600) is None
    assert _wait(action="poll", timeout=600, return_early_after=300) is None
