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

"""Guard: the test suite must never write into the operator's real store.

Measured 2026-11-01: the real sessions root held 278 empty shells out of 374
dirs — each a conversation.json containing only the system prompt. Every
dir's ``.session.lock`` ``cmd`` field recorded a pytest command line as the
launcher: the suite was constructing WorkerAgent instances without an
explicit session_dir, and FLAGSCALE_HOME pointed at the live store.

conftest's autouse ``_isolate_flagscale_home`` fixture fixes the mechanism.
These tests keep it fixed: without them the fixture can be deleted or
shadowed by a future refactor and nobody would notice until a real operator's
store filled up again (which is exactly how the original bug survived).

The assertion is deliberately about the RESOLVED path, not about the fixture:
a test that asserts "the fixture exists" passes while the leak returns.
"""

import os

import pytest


def _sessions_root_under(home: str) -> str:
    return os.path.join(home, "sessions")


def test_flagscale_home_is_sandboxed_during_tests():
    """The resolved store must not be the operator's real ~/.flagscale store."""
    home = os.environ.get("FLAGSCALE_HOME")
    assert home, (
        "FLAGSCALE_HOME is unset during a test run. conftest's autouse "
        "_isolate_flagscale_home fixture must set it, otherwise any agent "
        "constructed without an explicit session_dir writes into the real store."
    )

    # The sandbox home must be a pytest tmp dir, not a shared/global location.
    assert os.path.isabs(home)
    assert "pytest-" in home or "/tmp/" in home or home.startswith(
        os.path.expanduser("~/.cache")
    ), f"FLAGSCALE_HOME={home!r} does not look like a per-test sandbox"


def test_paths_module_resolves_into_the_sandbox():
    """Every path helper must derive from the sandboxed home.

    This is the load-bearing check: it exercises the real consumer
    (paths.get_sessions_root) instead of trusting the env var alone. If a
    future refactor caches the root at import time or bypasses
    get_dot_flagscale_root(), this fails even though FLAGSCALE_HOME is set.
    """
    from flagscale_agent.react.paths import (
        get_dot_flagscale_root,
        get_memory_dir,
        get_proposals_dir,
        get_sessions_root,
    )

    home = os.environ["FLAGSCALE_HOME"]
    assert get_dot_flagscale_root() == os.path.abspath(home)
    for helper, expected in (
        (get_sessions_root, _sessions_root_under(home)),
        (get_memory_dir, os.path.join(home, "agent_memory")),
        (get_proposals_dir, os.path.join(home, "proposals")),
    ):
        resolved = helper()
        assert resolved.startswith(os.path.abspath(home)), (
            f"{helper.__name__}() resolved to {resolved!r}, outside the "
            f"sandbox {home!r} — the test suite would pollute real state."
        )
        assert resolved == os.path.join(os.path.abspath(home), os.path.basename(expected))


@pytest.mark.parametrize(
    "module_path,exit_ok",
    [
        ("flagscale_agent.react.agent", True),
    ],
)
def test_agent_module_imports_without_touching_real_store(module_path, exit_ok):
    """Importing the agent module must not create anything in the sandbox's
    parent (a coarse but real 'no side effect at import' check)."""
    import importlib

    importlib.import_module(module_path)
    home = os.environ["FLAGSCALE_HOME"]
    # The sandbox home itself may not even exist yet; what must NOT happen is
    # a session dir appearing in the operator's default root.
    real_root = os.path.join(os.path.expanduser("~"), ".flagscale", "sessions")
    if os.path.isdir(real_root):
        # Cheap heuristic: a dir created in the last 5 seconds would be ours.
        import time

        now = time.time()
        fresh = [
            e
            for e in os.listdir(real_root)
            if os.path.isdir(os.path.join(real_root, e))
            and now - os.stat(os.path.join(real_root, e)).st_mtime < 5
        ]
        assert not fresh, (
            f"importing {module_path} just created {fresh} in the real store "
            f"{real_root} — tests are leaking into operator state again."
        )
