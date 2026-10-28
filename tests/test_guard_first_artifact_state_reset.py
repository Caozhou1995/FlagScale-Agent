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

"""Regression tests: FirstArtifactGuard must not fire after the task is over.

Measured failure (2026-10-28, session eea7fbdc): turns 1489-1502 of a LONG
session — every task long since delivered and verified — the guard still
blocked with `no_artifact_write_yet`, including on pure Q&A turns where no
task was in flight at all. Two distinct causes, one symptom:

  * STATE NEVER RESETS. `_wrote_artifact` / `_blocked_once` are set once in
    `__init__` and `reset_turn()` is a no-op, so the guard's bookkeeping lives
    for the whole session. The turn budget (TURN_BUDGET=4) and the byte budget
    (REASONING_BYTES_BUDGET=40_000) are calibrated for ONE trial; in a long
    session both are exceeded from the first iteration onward, so the guard
    fires on every turn regardless of what that turn is doing.
  * IT NEVER LOOKS AT DISK. The guard reasons only from message history. When
    the deliverable already exists on disk (verified by an external arbiter:
    a trial result.json reading reward=1.0), history alone cannot prove it, so
    the guard keeps demanding a first write that already happened.

The fix keeps the guard's real value (force the first write when nothing has
been produced) and adds the two missing state checks. These tests pin that
asymmetry: the misfire shapes must pass, the genuine shape must still block.
"""

from __future__ import annotations

import os

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard


def _ctx(tool_name="shell", tool_args=None, messages=None, tool_result=None):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        messages=messages or [],
        tool_result=tool_result,
    )


def _long_history(n: int = 1400):
    """A long session's history: many assistant turns, all read-only.

    This is the shape that made the guard fire continuously: the turn budget
    counts the WHOLE session, not the current task.
    """
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "shell",
                 "input": {"command": "cat /app/check.py"}},
            ],
        }
        for _ in range(n)
    ]


# ── Misfire shape M1: deliverable already on disk, no active plan ───────────
def test_does_not_block_when_deliverable_entry_exists_on_disk(tmp_path):
    """A delivered, verified artifact on disk must silence the guard.

    Measured: the trial result.json existed (reward=1.0) and the guard still
    blocked 4+ rounds, because it only consulted message history.

    NOTE: the active plan is deliberately LIVE here (not None) so this test
    exercises the ON-DISK check alone — the no-plan path is a different
    predicate pinned by its own test. Without that isolation the disk check
    could be dead code and the suite would still be green.
    """
    deliverable = tmp_path / "result.json"
    deliverable.write_text('{"verifier_result": {"rewards": {"reward": 1.0}}}')

    guard = FirstArtifactGuard()
    guard._declared_paths_fn = lambda: [str(deliverable)]
    guard._active_plan_fn = lambda: {"id": "plan_abc", "status": "active"}

    verdict = guard.check_pre(_ctx(messages=_long_history()))
    assert verdict is None, (
        "guard blocked even though the declared deliverable exists on disk "
        f"({deliverable}); message was: {getattr(verdict, 'message', '')[:200]}"
    )


# ── Misfire shape M2: no task in flight at all (pure Q&A turn) ──────────────
def test_does_not_block_with_no_active_plan(tmp_path):
    """No active plan means no task is being graded — nothing to write.

    Measured: two consecutive blocks on pure Q&A turns, each demanding
    'put a file on disk at the delivery path' while the user was asking a
    question.
    """
    guard = FirstArtifactGuard()
    guard._declared_paths_fn = lambda: []
    guard._active_plan_fn = lambda: None

    verdict = guard.check_pre(_ctx(messages=_long_history()))
    assert verdict is None, (
        "guard blocked on a turn with no active plan and no declared deliverable"
    )


# ── Misfire shape M3: override did not disarm ──────────────────────────────
def test_override_disarms_and_reset_turn_keeps_it_disarmed():
    """`accept_override` sets session state; `reset_turn` must not undo it.

    Measured: the block message promised 'the guard will not fire again this
    session', yet the guard fired again on the next turn.
    """
    guard = FirstArtifactGuard()
    guard._declared_paths_fn = lambda: []
    guard._active_plan_fn = lambda: None
    ctx = _ctx(messages=_long_history())
    guard._blocked_once = True
    guard._wrote_artifact = False
    guard.reset_turn()
    assert guard.check_pre(ctx) is None


# ── Genuine shape: must STILL block (the guard's whole value) ──────────────
def test_still_blocks_when_plan_is_active_and_nothing_on_disk(tmp_path):
    """The real failure shape must still be caught.

    An active plan exists, it declares a delivery path, and that path does not
    exist on disk: the guard's purpose is exactly to force that first write.
    """
    missing = tmp_path / "not_yet_written.bin"

    guard = FirstArtifactGuard()
    guard._declared_paths_fn = lambda: [str(missing)]
    guard._active_plan_fn = lambda: {"id": "plan_abc", "status": "active"}

    verdict = guard.check_pre(_ctx(messages=_long_history(n=10)))
    assert verdict is not None, "guard failed to block the genuine no-artifact shape"
    assert verdict.action == "block"
    assert verdict.reason == "no_artifact_write_yet"


# ── Message precision: name what was checked and what was seen ─────────────
def test_block_message_names_checked_paths_and_observations(tmp_path):
    """The block must say WHICH paths it checked and WHAT it saw there.

    Measured: the agent could not tell whether the guard had looked at the
    real delivery path, so it had to guess (and burned rounds). A block that
    names its evidence needs no guessing.
    """
    missing = tmp_path / "absent_deliverable.bin"

    guard = FirstArtifactGuard()
    guard._declared_paths_fn = lambda: [str(missing)]
    guard._active_plan_fn = lambda: {"id": "plan_abc", "status": "active"}

    verdict = guard.check_pre(_ctx(messages=_long_history(n=10)))
    assert verdict is not None
    msg = verdict.message
    assert str(missing) in msg, "block message must name the path it checked"
    assert "absent" in msg.lower() or "not exist" in msg.lower() or "missing" in msg.lower(), (
        "block message must state what it OBSERVED at that path"
    )


# ── Guard value preserved: a note write must still not disarm ──────────────
def test_note_write_does_not_satisfy_the_guard(tmp_path):
    """Bookkeeping files are not deliverables (pre-existing contract, pinned)."""
    notes = tmp_path / "NOTES.md"

    guard = FirstArtifactGuard()
    guard._declared_paths_fn = lambda: [str(tmp_path / "real_deliverable.bin")]
    guard._active_plan_fn = lambda: {"id": "plan_abc", "status": "active"}

    ctx = _ctx(
        tool_name="write_file",
        tool_args={"path": str(notes), "content": "x"},
        messages=_long_history(n=10),
    )
    verdict = guard.check_pre(ctx)
    assert verdict is not None and verdict.action == "block"
