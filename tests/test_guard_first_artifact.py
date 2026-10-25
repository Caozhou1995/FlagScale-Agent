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


def _ctx(tool_name="shell", tool_args=None, messages=None):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        messages=messages or [],
    )


def _turns(n: int, command: str = "cat /app/check.py"):
    """n assistant turns, each reading something and writing nothing."""
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "shell", "input": {"command": command}},
            ],
        }
        for _ in range(n)
    ]


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
    """The measured failure: N+ turns of `ls`/`cat`, no write. Must block."""
    guard = FirstArtifactGuard()
    ctx = _ctx(
        tool_name="shell",
        tool_args={"command": "cd /app && ls -la && cat check.py"},
        messages=_turns(FirstArtifactGuard.TURN_BUDGET),
    )
    verdict = guard.check_pre(ctx)
    assert verdict is not None, "a design-only run past the budget must be blocked"
    assert verdict.action == "block"
    assert verdict.reason == "no_artifact_write_yet"


def test_does_not_block_before_budget():
    """Orientation is legitimate — the guard must not fire on turn 1."""
    guard = FirstArtifactGuard()
    ctx = _ctx(tool_args={"command": "ls -la /app"}, messages=_turns(1))
    assert guard.check_pre(ctx) is None


def test_does_not_block_non_artifact_tools():
    """read_file/plan_update are not delivery-surface mutations."""
    guard = FirstArtifactGuard()
    ctx = _ctx(tool_name="read_file", messages=_turns(99))
    assert guard.check_pre(ctx) is None


# ── regression: ctx.turn_count is always 0 ───────────────────────────────


def test_turn_count_field_is_not_used_as_the_clock():
    """REGRESSION: the budget must come from history, not ctx.turn_count.

    kernel.py builds GuardContext.turn_count from
    `getattr(d.config, "_turn_count", 0)`, but nothing in the package ever
    assigns that attribute (the only `_turn_count` belongs to PromptBuilder).
    A guard that trusted it could never fire. This test drives exactly that
    world: turn_count stays 0 while the history clearly holds enough turns.
    """
    guard = FirstArtifactGuard()
    ctx = _ctx(
        tool_args={"command": "ls -la /app"},
        messages=_turns(FirstArtifactGuard.TURN_BUDGET + 3),
    )
    assert ctx.turn_count == 0, "precondition: the context field really is 0"
    verdict = guard.check_pre(ctx)
    assert verdict is not None, (
        "guard must count turns from history; relying on ctx.turn_count (always "
        "0) makes it unreachable"
    )


# ── the healthy shape: must never livelock ────────────────────────────────


def test_write_file_through_after_budget():
    """A write_file IS the fix, so it must pass even past the budget."""
    guard = FirstArtifactGuard()
    ctx = _ctx(
        tool_name="write_file",
        tool_args={"path": "/app/re.json", "content": "[]"},
        messages=_turns(99),
    )
    assert guard.check_pre(ctx) is None


def test_shell_write_through_after_budget():
    """A heredoc write is how most deliverables are actually produced."""
    guard = FirstArtifactGuard()
    ctx = _ctx(
        tool_args={"command": "cat > /app/re.json <<'EOF'\n[]\nEOF"},
        messages=_turns(99),
    )
    assert guard.check_pre(ctx) is None


def test_earlier_write_disarms_guard():
    """A write in an EARLIER turn must disarm — history is authoritative."""
    guard = FirstArtifactGuard()
    msgs = _turns(99) + _history_with("write_file", {"path": "/app/re.json"})
    ctx = _ctx(tool_args={"command": "ls -la /app"}, messages=msgs)
    assert guard.check_pre(ctx) is None


def test_override_disarms_for_session():
    """Overriding must stop the guard firing again (no livelock)."""
    guard = FirstArtifactGuard()
    ctx = _ctx(tool_args={"command": "cat /app/spec.md"}, messages=_turns(99))
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
        # REGRESSION: this exact command disarmed the guard in a real run.
        # Pure introspection, no file created, yet inline python used to be a
        # write marker by itself.
        'ls -la /app; echo ---; cat /app/check.py 2>/dev/null; '
        'python3 -c "import chess; print(chess.__version__, chess.__file__)"',
        'python3 -c "import json; print(json.load(open(\'/app/x.json\')))"',
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
