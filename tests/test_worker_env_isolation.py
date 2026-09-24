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

"""Regression: the suite must be hermetic to the LAUNCHING shell's env.

Bug (found by an independent reviewer worker running the suite on itself):
when pytest is launched from inside a worker process, the inherited
worker-identity env flips agent-construction and single-shot dispatch into
their worker branches, and two unit tests fail for reasons unrelated to the
code under test:
  - test_restore_repoints_multiagent_tools -> KeyError 'Tool not found:
    dispatch_many'  (is_worker() True => DispatchManyTool not registered)
  - test_single_shot_saves_on_completion -> user query never appended
    (FLAGSCALE_RESUME_PATH present => _run_single_shot resume branch)
Reproduced exactly: FLAGSCALE_TASK_ID + FLAGSCALE_RESUME_PATH in the env gave
"2 failed, 2122 passed"; a plain shell gave "2124 passed".
Fix: tests/conftest.py autouse fixture clears worker-identity vars per test.
This test re-runs a representative suite file in a subprocess that carries the
worker env and asserts it is green.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_suite_green_under_worker_env(tmp_path):
    resume_file = tmp_path / "resume.prompt"
    resume_file.write_text("parent resume message\n", encoding="utf-8")

    env = dict(os.environ)
    # The exact worker/resumed-worker identity env a harness worker carries.
    env["FLAGSCALE_TASK_ID"] = "probe123"
    env["FLAGSCALE_RESUME_PATH"] = str(resume_file)
    # A worker also gets these from _build_env; include for completeness.
    env["FLAGSCALE_CONTRACT_PATH"] = str(tmp_path / "contract.prompt")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_session_rebind.py",
         "tests/test_single_shot_persistence.py", "-q", "--no-header"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    summary = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, (
        "suite must pass even when launched from a worker environment "
        f"(env had FLAGSCALE_TASK_ID/RESUME_PATH/CONTRACT_PATH). Output:\n{summary}"
    )
    # Guard against a vacuous pass: the subprocess must have actually run
    # the regression tests, not collected nothing.
    assert "6 passed" in summary or "passed" in summary, summary
