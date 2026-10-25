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

"""Tests for FirstArtifactGuard.

The guard exists because a measured failure shape (a whole run spent in design
reasoning, zero writes, verifier read "no such file") was invisible to every
existing text-only nudge. These tests pin both directions:

  * it BLOCKS the read-only forever-agent once the turn budget is spent
  * it RELEASES an agent that writes (so it can never livelock a healthy run)
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.first_artifact import (
    FirstArtifactGuard,
    _shell_writes_bytes,
)


def _ctx(tool_name="shell", tool_args=None, turn_count=0, messages=None):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        turn_count=turn_count,
        messages=messages or [],
    )


def _history_with(tool: str, args: dict):
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": tool, "input": args},
            ],
        }
    ]


# ── the failure shape: read-only forever ─────────────────────────────────


def test_blocks_read_only_shell_after_turn_budget():
    """The measured failure: 4+ turns of `ls`/`cat`, no write. Must block."""
    guard = FirstArtifactGuard()
    ctx = _ctx(
        tool_name="shell",
        tool_args={"command": "cd /app && ls -la && cat check.py"},
        turn_count=FirstArtifactGuard.TURN_BUDGET,
        messages=_history_with("shell", {"command": "cat /app/check.py"}),
    )
    verdict = guard.check_pre(ctx)
    assert verdict is not None, "a design-only run past the budget must be blocked"
    assert verdict.action == "block"
    assert verdict.reason == "no_artifact_write_yet"


def test_does_not_block_before_budget():
    """Orientation is legitimate — the guard must not fire on turn 1."""
    guard = FirstArtifactGuard()
    ctx = _ctx(tool_args={"command": "ls -la /app"}, turn_count=0)
    assert guard.check_pre(ctx) is None


def test_does_not_block_non_artifact_tools():
    """read_file/plan_update are not delivery-surface mutations."""
    guard = FirstArtifactGuard()
    ctx = _ctx(tool_name="read_file", turn_count=99)
    assert guard.check_pre(ctx) is None


# ── the healthy shape: must never livelock ────────────────────────────────


def test_write_file_through_after_budget():
    """A write_file IS the fix, so it must pass even past the budget."""
    guard = FirstArtifactGuard()
    ctx = _ctx(
        tool_name="write_file",
        tool_args={"path": "/app/re.json", "content": "[]"},
        turn_count=99,
    )
    assert guard.check_pre(ctx) is None


def test_shell_write_through_after_budget():
    """A heredoc write is how most deliverables are actually produced."""
    guard = FirstArtifactGuard()
    ctx = _ctx(
        tool_args={"command": "cat > /app/re.json <<'EOF'\n[]\nEOF"},
        turn_count=99,
    )
    assert guard.check_pre(ctx) is None


def test_earlier_write_disarms_guard():
    """A write in an EARLIER turn must disarm — history is authoritative."""
    guard = FirstArtifactGuard()
    ctx = _ctx(
        tool_args={"command": "ls -la /app"},
        turn_count=99,
        messages=_history_with("write_file", {"path": "/app/re.json"}),
    )
    assert guard.check_pre(ctx) is None


def test_override_disarms_for_session():
    """Overriding must stop the guard firing again (no livelock)."""
    guard = FirstArtifactGuard()
    ctx = _ctx(tool_args={"command": "cat /app/spec.md"}, turn_count=99)
    verdict = guard.check_pre(ctx)
    assert verdict is not None
    assert guard.accept_override("I must read the spec format first", ctx) is True
    assert guard.check_pre(ctx) is None


# ── the shell-write classifier ───────────────────────────────────────────


@pytest.mark.parametrize(
    "command",
    [
        "ls -la /app",
        "cat /app/check.py",
        "grep -rn foo /app",
        "cd /app && ls -la && cat check.py",
        "ls -la /app 2>&1",
        "cat /app/x >/dev/null",
    ],
)
def test_read_only_shell_does_not_count_as_write(command):
    assert _shell_writes_bytes(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "cat > /app/re.json <<'EOF'",
        "echo '[]' > /app/re.json",
        "cp /tmp/x /app/re.json",
        "python3 -c \"open('/app/re.json','w').write('[]')\"",
        "sed -i 's/a/b/' /app/re.json",
        "tee /app/re.json",
    ],
)
def test_write_shell_counts_as_write(command):
    assert _shell_writes_bytes(command) is True
