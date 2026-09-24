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

"""Shared test configuration.

Worker-identity isolation
-------------------------
Several modules branch on the process environment to distinguish a *parent*
agent from a *worker* (spawned subagent):

  - ``is_worker()`` (multi_agent/wiring) reads ``FLAGSCALE_TASK_ID``; when set,
    ``WorkerAgent.__init__`` registers only the worker tool surface (no
    ``dispatch_many``) and ``_run_single_shot`` enters the resume branch.
  - ``resolve_resume_query()`` reads ``FLAGSCALE_RESUME_PATH``.

The suite must yield the SAME result whether pytest is launched from a plain
shell OR from inside a worker process (the harness runs its own tests via
``spawn_worker``, whose env carries these vars). Without isolation, a run
inheriting a worker env leaks the worker branch into the unit tests and they
fail for reasons unrelated to the code under test.

The fixture below clears the worker-identity vars for every test by default, so
tests are hermetic. A test that WANTS to exercise the worker branch still does
so explicitly via its own ``monkeypatch.setenv`` (which overrides this).
"""
import pytest

# Env keys that flip a process into (or out of) the worker branch. Keep this in
# sync with flagscale_agent/react/multi_agent/wiring.py + spawn.py._build_env.
_WORKER_IDENTITY_ENV = (
    "FLAGSCALE_TASK_ID",
    "FLAGSCALE_TASK_DEPTH",
    "FLAGSCALE_MAX_DEPTH",
    "FLAGSCALE_PARENT_TRACE",
    "FLAGSCALE_CONTRACT_PATH",
    "FLAGSCALE_OUTPUT_DIR",
    "FLAGSCALE_RESUME_PATH",
)


@pytest.fixture(autouse=True)
def _isolate_worker_identity_env(monkeypatch):
    """Default every test to a NON-worker environment.

    worker-identity vars (FLAGSCALE_TASK_ID / FLAGSCALE_RESUME_PATH / ...) are
    removed for the duration of each test, so pytest is hermetic to the
    launching shell. Tests that need the worker branch set them back explicitly.
    """
    for key in _WORKER_IDENTITY_ENV:
        monkeypatch.delenv(key, raising=False)
    yield
