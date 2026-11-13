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

"""EnvProbeGuard: raise environment-exploration awareness on a REAL missing-module reading.

The measured failure (torch-pipeline-parallelism__uLnHEFe, 2026-11-11): the agent
probed ONE interpreter, read `ModuleNotFoundError`, concluded "this container has
no torch", skipped runtime verification, and shipped code that crashed on
`position_embeddings=None`. The environment did have a reachable path (the `uv`
the setup stage installed), and `uv` appears zero times in that log.

These pin both directions -- it fires on the real signal, and it does NOT fire on
the things that would make it noise (a normal command, an unrelated path typo, a
second time in the same session).
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.env_probe import EnvProbeGuard, _looks_like_interpreter_probe


def _ctx(tool_name="shell", tool_result=None, **args):
    return GuardContext(
        messages=[],
        tool_name=tool_name,
        tool_args=dict(args),
        tool_result=tool_result,
    )


# --------------------------------------------------------------------------
# the real signal: the interpreter says it cannot resolve a module
# --------------------------------------------------------------------------


def test_module_not_found_fires():
    g = EnvProbeGuard()
    v = g.check_post(_ctx(tool_result="Traceback...\nModuleNotFoundError: No module named 'torch'"))
    assert v is not None
    assert v.action == "inject"
    assert v.reason == "env_missing_module"
    assert "one entry point" in v.message.lower()


def test_no_module_named_alone_fires():
    """Either Python spelling counts -- the token is the tool's own output."""
    g = EnvProbeGuard()
    v = g.check_post(_ctx(tool_result="No module named 'transformers'"))
    assert v is not None
    assert v.action == "inject"


def test_message_is_principle_not_a_command_checklist():
    """The user rejected checklist-style design. The reminder must teach a way of
    reading the environment, not hand over a copy-paste command list."""
    g = EnvProbeGuard()
    v = g.check_post(_ctx(tool_result="ModuleNotFoundError: No module named 'x'"))
    msg = v.message
    # principle present
    assert "LAST conclusion" in msg
    assert "have not found the entrance" in msg
    # no copy-pasteable probe commands of the kind a checklist would carry
    for bad in ("curl -s", "pip install ", "uv run ", "conda activate", "PYTHONPATH="):
        assert bad not in msg, f"message leaked a checklist command: {bad!r}"


# --------------------------------------------------------------------------
# does NOT fire on noise (a guard that fires on everything teaches nothing)
# --------------------------------------------------------------------------


def test_normal_result_does_not_fire():
    g = EnvProbeGuard()
    assert g.check_post(_ctx(tool_result="COMPILE_OK\n")) is None


def test_empty_or_none_result_does_not_fire():
    g = EnvProbeGuard()
    assert g.check_post(_ctx(tool_result=None)) is None
    assert g.check_post(_ctx(tool_result="")) is None


def test_unrelated_path_error_on_a_non_interpreter_command_does_not_fire():
    """`ls /nonexistent` prints 'No such file or directory' -- an ordinary typo.
    Firing there would dilute the reminder into noise."""
    g = EnvProbeGuard()
    v = g.check_post(
        _ctx(tool_result="ls: cannot access '/nope': No such file or directory",
             command="ls /nope")
    )
    assert v is None


def test_no_such_file_on_an_interpreter_command_DOES_fire():
    """Same token, but the thing missing is an INTERPRETER -> that is the class
    of event this guard is about (an entry point that was not on PATH)."""
    g = EnvProbeGuard()
    v = g.check_post(
        _ctx(tool_result="bash: python3.13: command not found", command="python3.13 -V")
    )
    assert v is not None
    assert v.reason == "env_missing_interpreter"


def test_non_shell_tool_result_is_ignored():
    g = EnvProbeGuard()
    assert g.check_post(_ctx(tool_name="plan_update",
                             tool_result="ModuleNotFoundError")) is None


# --------------------------------------------------------------------------
# single-shot latch
# --------------------------------------------------------------------------


def test_fires_only_once_per_session():
    g = EnvProbeGuard()
    first = g.check_post(_ctx(tool_result="ModuleNotFoundError: No module named 'a'"))
    assert first is not None
    second = g.check_post(_ctx(tool_result="ModuleNotFoundError: No module named 'b'"))
    assert second is None, "a repeated reminder is nagging, not teaching"


def test_latch_survives_reset_turn():
    """The lesson is about the environment, which does not change between turns."""
    g = EnvProbeGuard()
    assert g.check_post(_ctx(tool_result="No module named 'a'")) is not None
    g.reset_turn()
    assert g.check_post(_ctx(tool_result="No module named 'a'")) is None


def test_never_blocks():
    """check_pre is a no-op: this guard must not gate any call."""
    g = EnvProbeGuard()
    assert g.check_pre(_ctx(tool_result=None, command="python3 -c 'import torch'")) is None


# --------------------------------------------------------------------------
# the narrow interpreter heuristic (unit of the secondary signal)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd,expected",
    [
        ("python3 -c 'import torch'", True),
        ("python3.13 -V", True),
        ("uv run --python 3.13 python -c 'import torch'", True),
        ("pip install torch", True),
        ("conda env list", True),
        ("node -e 'console.log(1)'", True),
        ("npm ls", True),
        ("ls /app", False),
        ("cat /etc/hosts", False),
        ("grep -rn TODO ./src", False),
    ],
)
def test_interpreter_heuristic(cmd, expected):
    assert _looks_like_interpreter_probe(cmd) is expected
