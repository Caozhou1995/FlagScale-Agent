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

"""prop_136fce9c — the spawn registry must not fail OPEN silently.

Both per-session registries used one `except Exception: return []`. For
read_adjudications that is fail-SAFE (a reviewer stays gated). For the SPAWN
registry it is not: an unreadable/corrupt spawned.jsonl empties
ReviewerFeedbackGuard's view of the session, so the completion gate passes and a
spawned reviewer is never adjudicated — indistinguishable from "this session
spawned nobody".

These tests pin the corrected contract:
- MISSING file is the legitimate empty case -> no warning, gate silent.
- UNREADABLE / partially-corrupt file -> a reason is reported AND the gate
  injects an advisory instead of passing silently. It must never BLOCK (a
  corrupt file must not hold completion hostage).
- The warning does not go stale: after the registry becomes readable the gate
  is silent again.
"""

import os

import pytest

from flagscale_agent.react.guard.reviewer_feedback import ReviewerFeedbackGuard
from flagscale_agent.react.multi_agent.ledger import TaskLedger
from flagscale_agent.react.multi_agent.spawn import (
    SUBAGENTS_DIRNAME, SPAWN_REGISTRY_FILENAME,
    record_spawn, read_session_spawns, read_session_spawns_ex)
from tests.test_reviewer_feedback_guard import _ctx, _guard  # noqa: F401


def _registry_path(session_dir):
    return os.path.join(str(session_dir), SUBAGENTS_DIRNAME,
                        SPAWN_REGISTRY_FILENAME)


@pytest.fixture
def env(tmp_path):
    return {"ledger": TaskLedger(str(tmp_path / "tasks")),
            "session": str(tmp_path / "sess")}


# ── primitive: read_session_spawns_ex ────────────────────────────────────────

def test_missing_registry_is_the_legitimate_empty_case(tmp_path):
    # "No spawns yet" is NOT a fault: no warning, empty list.
    assert read_session_spawns_ex(str(tmp_path / "nope")) == ([], None)
    assert read_session_spawns_ex(None) == ([], None)


def test_readable_registry_reports_no_error(tmp_path):
    sd = str(tmp_path / "s")
    record_spawn(sd, "t1", "reviewer")
    recs, err = read_session_spawns_ex(sd)
    assert [r["task_id"] for r in recs] == ["t1"]
    assert err is None


def test_unreadable_registry_reports_a_reason(tmp_path):
    sd = str(tmp_path / "s")
    p = _registry_path(sd)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    # A directory where a file is expected: read_text() raises OSError.
    os.makedirs(p, exist_ok=True)
    recs, err = read_session_spawns_ex(sd)
    assert recs == []
    assert err and "cannot read" in err


def test_partially_corrupt_registry_keeps_the_parsed_records(tmp_path):
    # A partial view beats no view: the good line is still returned, and the
    # bad line is reported rather than swallowed.
    sd = str(tmp_path / "s")
    p = _registry_path(sd)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write('{"task_id": "good", "role": "reviewer"}\n')
        fh.write("{not json at all\n")
    recs, err = read_session_spawns_ex(sd)
    assert [r["task_id"] for r in recs] == ["good"]
    assert err and "unparseable" in err


def test_backward_compatible_wrapper_still_returns_a_plain_list(tmp_path):
    sd = str(tmp_path / "s")
    record_spawn(sd, "t1", "reviewer")
    assert [r["task_id"] for r in read_session_spawns(sd)] == ["t1"]
    assert read_session_spawns(str(tmp_path / "nope")) == []


# ── gate behaviour ───────────────────────────────────────────────────────────

def test_gate_is_silent_when_no_registry_exists(env):
    # Fail-open is only a defect when it HIDES something; an absent registry
    # genuinely means no spawns, so the gate must stay quiet.
    assert _guard(env).check_pre(_ctx()) is None


def test_gate_injects_advisory_when_registry_unreadable(env):
    p = _registry_path(env["session"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    os.makedirs(p, exist_ok=True)  # unreadable-as-file
    v = _guard(env).check_pre(_ctx())
    assert v is not None, "an unreadable registry must NOT pass silently"
    assert v.action == "inject", "a corrupt registry must never BLOCK completion"
    assert v.reason == "reviewer_feedback_registry_unreadable"
    assert "BLIND" in v.message


def test_warning_does_not_go_stale_across_resolves(env):
    # Same guard instance: unreadable first, then repaired. The stale warning
    # must not resurface once the condition is gone.
    p = _registry_path(env["session"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    os.makedirs(p, exist_ok=True)
    g = _guard(env)
    assert g.check_pre(_ctx()) is not None
    os.rmdir(p)
    assert g.check_pre(_ctx()) is None
