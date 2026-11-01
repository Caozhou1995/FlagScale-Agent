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

"""prop_8754e12a — a PREVENTED call must survive as its own record.

Measured cost of NOT having this: Audit #83's central finding rested entirely on
the fact that the only shell call in the audited window was swallowed by a
guard block, and that was visible only because the guard banner happened to be
the record. An auditor could not otherwise distinguish "never attempted" from
"attempted and blocked" — so a run that TRIED to measure something and was
stopped looks identical to one that never tried.

Two invariants, and the SECOND is the subtle one:
1. A blocked call leaves a row in the audit snapshot (role=guard_block).
2. It must NOT enter the execution ledger (`recent_tool_history`) — guards like
   first_e2e_run read that list as evidence of real progress, so a blocked entry
   there would forge a rung release.
"""

from __future__ import annotations

import json

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.reviewer_audit import (
    AUDIT_DIRNAME, SNAPSHOT_FILENAME, ReviewerAuditGuard)


def _ctx(blocked=None):
    return GuardContext(
        tool_name="plan_update",
        tool_args={"action": "step_done", "step_id": 1},
        tool_result="Plan updated.",
        messages=[{"role": "system", "content": "doctrine"},
                  {"role": "assistant", "content": "did things"}],
        blocked_tool_calls=blocked or [],
    )


def _guard(tmp_path):
    return ReviewerAuditGuard(session_dir_fn=lambda: str(tmp_path),
                              spawn_fn=lambda **k: "ok",
                              stats_fn=None, enabled_fn=lambda: True)


def _rows(tmp_path):
    text = (tmp_path / AUDIT_DIRNAME / SNAPSHOT_FILENAME).read_text()
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _blocked(name="shell", guard="context_pressure",
             reason="context_pressure_block", action="block"):
    return {"name": name, "guard": guard, "reason": reason, "action": action}


# ── 1. the record exists ────────────────────────────────────────────────────

def test_prevented_call_appears_in_the_snapshot(tmp_path):
    _guard(tmp_path).check_post(_ctx([_blocked()]))
    rows = _rows(tmp_path)
    gb = [r for r in rows if r.get("blocked")]
    assert len(gb) == 1
    assert gb[0]["role"] == "guard_block"
    content = gb[0]["content"]
    assert "shell" in content
    assert "context_pressure" in content
    assert "NEVER EXECUTED" in content, (
        "the row must state the call did not run, otherwise it is just more "
        "transcript noise")
    assert "not evidence the agent did not attempt it" in content


def test_no_blocked_calls_means_no_extra_rows(tmp_path):
    _guard(tmp_path).check_post(_ctx([]))
    assert not [r for r in _rows(tmp_path) if r.get("blocked")]


def test_several_blocks_are_each_recorded(tmp_path):
    blocks = [_blocked("shell"), _blocked("write_file", "post_edit_far_end",
                                          "measure_failed", "escalate")]
    _guard(tmp_path).check_post(_ctx(blocks))
    gb = [r for r in _rows(tmp_path) if r.get("blocked")]
    assert len(gb) == 2
    assert any("write_file" in r["content"] for r in gb)


def test_snapshot_stays_bounded_with_a_block_flood(tmp_path):
    # Same size discipline as the transcript rows: an auditor reads intent, not
    # megabytes. Measured 200 unrendered rows -> 67 KB, which would have blown
    # the snapshot contract on its own.
    _guard(tmp_path).check_post(_ctx([_blocked(f"tool{i}") for i in range(200)]))
    text = (tmp_path / AUDIT_DIRNAME / SNAPSHOT_FILENAME).read_text()
    assert len(text) < 60000, f"blocked-call rows blew up the snapshot: {len(text)}"
    # Nothing is lost silently: the overflow is a count.
    assert "further blocked calls omitted" in text
    assert "200 total this turn" in text


# ── 2. it must NOT forge execution evidence ─────────────────────────────────

def test_guard_context_keeps_the_two_lists_separate():
    # GuardContext must expose blocked_tool_calls as its OWN field: if a caller
    # passed them through recent_tool_history, a progress gate reading that list
    # would see a fake execution.
    c = _ctx([_blocked()])
    assert c.blocked_tool_calls and not c.recent_tool_history
    assert c.blocked_tool_calls[0]["name"] == "shell"


def test_blocked_rows_are_marked_and_not_shaped_like_executed_rows(tmp_path):
    _guard(tmp_path).check_post(_ctx([_blocked()]))
    rows = _rows(tmp_path)
    executed = [r for r in rows if r["role"] in ("assistant", "system")]
    blocked = [r for r in rows if r.get("blocked")]
    assert blocked and executed
    for r in blocked:
        assert r.get("blocked") is True
    for r in executed:
        assert not r.get("blocked"), "an executed row must not carry the blocked flag"
