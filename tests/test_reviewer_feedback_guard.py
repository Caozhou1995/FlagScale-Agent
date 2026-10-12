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

"""ReviewerFeedbackGuard — a reviewer must be adjudicated before completion.

Contract:
- Fires ONLY at the text-only completion path (tool_name=="" AND llm_responded).
  A tool call, or a top-of-loop scan (llm_responded=False), is never gated.
- Blocks when a reviewer-class child THIS session spawned is still ACTIVE
  (SPAWNING/RUNNING/REPORTED), read from <session_dir>/subagents/spawned.jsonl.
- Silent when: no session_dir, registry empty/missing, ledger unreadable, the
  only children are NON-reviewer (plain/diverger), or the reviewer is terminal.
- Override requires the reason to NAME at least one pending reviewer task_id;
  a bare/no reason and a generic "done" never release it.
- Silent in a worker subprocess (FLAGSCALE_TASK_ID set).
"""

import os

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.reviewer_feedback import ReviewerFeedbackGuard
from flagscale_agent.react.multi_agent.contract import Contract
from flagscale_agent.react.multi_agent.ledger import TaskLedger, DONE
from flagscale_agent.react.multi_agent.spawn import record_spawn, read_session_spawns


def _contract(tmp_path, name, reviewer=False, role=None):
    writable = str(tmp_path)
    out = os.path.join(writable, name + ".json")
    cons = {"writable": [writable], "max_minutes": 10}
    if reviewer:
        cons["reviewer"] = True
    if role:
        cons["role"] = role
    return Contract.build(
        goal=name, constraints=cons,
        acceptance=[{"check": "test -f out.json", "kind": "check_command"}],
        output_ptr=out, inputs=[],
    )


@pytest.fixture
def env(tmp_path):
    led = TaskLedger(str(tmp_path / "tasks"))
    return {"ledger": led, "session": str(tmp_path / "sess")}


def _active(led, c, status="REPORTED"):
    """Drive a freshly-created task (SPAWNING) to an ACTIVE status legally.

    Legal SPAWNING path is SPAWNING->RUNNING->REPORTED (never SPAWNING->REPORTED).
    """
    if status in ("RUNNING", "REPORTED"):
        led.transition(c.id, "RUNNING")
    if status == "REPORTED":
        led.transition(c.id, "REPORTED")


def _ctx(name="", responded=True, args=None):
    return GuardContext(tool_name=name, tool_args=args or {},
                        llm_responded=responded, messages=[])


def _guard(env):
    return ReviewerFeedbackGuard(
        session_dir_fn=lambda: env["session"], ledger=env["ledger"])


# ── registry primitive ───────────────────────────────────────────────────────

def test_record_and_read_spawn_roundtrip(tmp_path):
    sd = str(tmp_path / "s")
    record_spawn(sd, "t1", "reviewer")
    record_spawn(sd, "t2", "")
    recs = read_session_spawns(sd)
    assert [r["task_id"] for r in recs] == ["t1", "t2"]
    assert recs[0]["role"] == "reviewer" and recs[1]["role"] == ""


def test_read_spawns_missing_dir_is_empty(tmp_path):
    assert read_session_spawns(str(tmp_path / "nope")) == []
    assert read_session_spawns(None) == []


# ── block path ───────────────────────────────────────────────────────────────

def test_blocks_when_reviewer_still_reported(env):
    led = env["ledger"]
    c = _contract(env["session"], "rev", reviewer=True)
    led.create(c)
    _active(led, c, "REPORTED")
    record_spawn(env["session"], c.id, "reviewer")
    v = _guard(env).check_pre(_ctx())
    assert v is not None and v.action == "block" and v.overridable
    assert c.id in v.message


def test_blocks_only_at_completion_path(env):
    led = env["ledger"]
    c = _contract(env["session"], "rev", reviewer=True)
    led.create(c)
    _active(led, c, "RUNNING")
    record_spawn(env["session"], c.id, "reviewer")
    g = _guard(env)
    # A tool call is never gated.
    assert g.check_pre(_ctx(name="poll_tasks")) is None
    # A top-of-loop scan (stale sentinel in history) is never gated.
    assert g.check_pre(_ctx(responded=False)) is None


def test_silent_when_reviewer_terminal(env):
    led = env["ledger"]
    c = _contract(env["session"], "rev", reviewer=True)
    led.create(c)
    _active(led, c, "REPORTED")
    led.transition(c.id, DONE)
    record_spawn(env["session"], c.id, "reviewer")
    assert _guard(env).check_pre(_ctx()) is None


def test_silent_for_non_reviewer_children(env):
    led = env["ledger"]
    plain = _contract(env["session"], "plain")
    div = _contract(env["session"], "div", role="diverger")
    led.create(plain)
    _active(led, plain, "REPORTED")
    led.create(div)
    _active(led, div, "RUNNING")
    record_spawn(env["session"], plain.id, "")
    record_spawn(env["session"], div.id, "diverger")
    assert _guard(env).check_pre(_ctx()) is None


def test_silent_without_session_dir(env):
    led = env["ledger"]
    c = _contract(env["session"], "rev", reviewer=True)
    led.create(c)
    _active(led, c, "REPORTED")
    g = ReviewerFeedbackGuard(session_dir_fn=lambda: None, ledger=led)
    assert g.check_pre(_ctx()) is None


def test_silent_in_worker_subprocess(env, monkeypatch):
    led = env["ledger"]
    c = _contract(env["session"], "rev", reviewer=True)
    led.create(c)
    _active(led, c, "REPORTED")
    record_spawn(env["session"], c.id, "reviewer")
    monkeypatch.setenv("FLAGSCALE_TASK_ID", "someid")
    assert _guard(env).check_pre(_ctx()) is None


# ── override contract ────────────────────────────────────────────────────────

def test_override_rejects_bare_and_generic(env):
    led = env["ledger"]
    c = _contract(env["session"], "rev", reviewer=True)
    led.create(c)
    _active(led, c, "REPORTED")
    record_spawn(env["session"], c.id, "reviewer")
    g = _guard(env)
    assert g.accept_override("", _ctx()) is False
    assert g.accept_override("done", _ctx()) is False
    assert g.accept_override("all checks pass, shipping now", _ctx()) is False


def test_override_accepts_reason_naming_task_id(env):
    led = env["ledger"]
    c = _contract(env["session"], "rev", reviewer=True)
    led.create(c)
    _active(led, c, "REPORTED")
    record_spawn(env["session"], c.id, "reviewer")
    g = _guard(env)
    assert g.accept_override(
        f"abandoning reviewer {c.id}, its process hung and I hold the evidence",
        _ctx()) is True


def test_override_accepts_when_nothing_pending(env):
    # Nothing live anymore (parent already finalized between block and override).
    assert _guard(env).accept_override("reviewer handled already", _ctx()) is True
