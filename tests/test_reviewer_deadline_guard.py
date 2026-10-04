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

"""ReviewerDeadlineGuard — bound review waiting by the enforced time budget.

Contract:
- No external wall (stats_fn None / raises / remaining None) → fully silent.
- Budget far from the wall (remaining > max(300s, 15%)) → silent, even for
  reviewer polls (generic discipline stays PollSpacingGuard's job).
- Near the deadline + poll of a reviewer-class task:
  * fewer than 2 prior not-ready polls → soft inject (adjudicate from evidence)
  * 2+ prior not-ready polls of the SAME reviewer task → block (overridable,
    override must point at the reviewer's report file)
- Non-reviewer poll targets are never touched, near deadline or not.
- Settled poll rounds (PASSED/FAILED) reset the per-task counter.
- Blocked calls advance nothing (the [BLOCKED BY GUARD] marker is judged).
- The guard ignores everything when FLAGSCALE_TASK_ID is set (a worker/reviewer
  polling its own children has no parent budget to protect).
"""

import os

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.reviewer_deadline import (
    BLOCK_THRESHOLD,
    ReviewerDeadlineGuard,
    _is_reviewer_target,
    _is_reviewer_worker,
)
from flagscale_agent.react.multi_agent.contract import Contract
from flagscale_agent.react.multi_agent.ledger import TaskLedger


def _contract(tmp_path, reviewer=False):
    writable = str(tmp_path)
    out = os.path.join(writable, "out.json")
    cons = {"writable": [writable], "max_minutes": 10}
    if reviewer:
        cons["reviewer"] = True
    return Contract.build(
        goal="review the deliverable" if reviewer else "do the work",
        constraints=cons,
        acceptance=[{"check": "test -f out.json", "kind": "check_command"}],
        output_ptr=out,
        inputs=[],
    )


@pytest.fixture
def ledger_env(tmp_path, monkeypatch):
    led = TaskLedger(str(tmp_path / "tasks"))
    monkeypatch.setattr(
        "flagscale_agent.react.multi_agent.wiring.get_tasks_dir",
        lambda: str(tmp_path / "tasks"),
    )
    reviewer = _contract(tmp_path, reviewer=True)
    plain = _contract(tmp_path / "sub", reviewer=False)
    led.create(reviewer)
    led.create(plain)
    return {"ledger": led, "reviewer_id": reviewer.id, "plain_id": plain.id}


def _ctx(name="poll_tasks", result=None, args=None):
    return GuardContext(tool_name=name, tool_result=result,
                        tool_args=args or {}, messages=[])


def _stats(remaining, budget=1800.0):
    return {"elapsed": budget - remaining, "budget": budget,
            "remaining": remaining, "pct": 100.0 * (budget - remaining) / budget}


# ── F1: override contract enforcement (report file + quote) ──────────────────

def test_override_accepts_real_report_and_quote(ledger_env, tmp_path):
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(250))
    rp = tmp_path / "review.md"
    rp.write_text("Finding: the parse loop drops the last chunk when the "
                  "stream ends mid-token.\n")
    reason = (f"Report {rp} states: 'the parse loop drops the last chunk "
              "when the stream ends mid-token.'")
    assert g.accept_override(reason, None) is True


def test_override_rejects_missing_report(ledger_env, tmp_path):
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(250))
    reason = ("Report /tmp/definitely_not_here_zz9.md states: 'the parse "
              "loop drops the last chunk when the stream ends mid-token.'")
    assert g.accept_override(reason, None) is False


def test_override_accepts_path_with_trailing_punctuation(ledger_env, tmp_path):
    """A reason may cite the path in prose (`/tmp/x.md,`, `/tmp/x.md.`, or in
    backticks) — trailing punctuation must not defeat the file lookup."""
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(250))
    rp = tmp_path / "review.md"
    rp.write_text("the parse loop drops the last chunk.\n")
    q = "the parse loop drops the last chunk"
    assert g.accept_override(f"Report {rp}, states '{q}'", None) is True
    assert g.accept_override(f"Report {rp}. It states '{q}'", None) is True
    assert g.accept_override(f"Report `{rp}` states '{q}'", None) is True


def test_override_rejects_file_without_quote(ledger_env, tmp_path):
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(250))
    rp = tmp_path / "review.md"
    rp.write_text("Finding: the parse loop drops the last chunk.\n")
    assert g.accept_override(f"Report {rp} exists and looks great, ship it",
                             None) is False


def test_override_rejects_text_only_reason(ledger_env):
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(250))
    assert g.accept_override("review matters a lot to me", None) is False


def test_blocked_l2_poll_cannot_self_release(ledger_env, tmp_path):
    """2 armed not-ready polls -> L2 block -> a text-only override reason is
    refused by accept_override (the only gate accept_override guards)."""
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(250))
    nr = f"task {ledger_env['reviewer_id']}: not ready (still RUNNING)"
    g.check_post(_ctx(result=nr, args={"task_id": ledger_env["reviewer_id"]}))
    g.check_post(_ctx(result=nr, args={"task_id": ledger_env["reviewer_id"]}))
    blk = g.check_pre(_ctx(args={"task_id": ledger_env["reviewer_id"]}))
    assert blk is not None and blk.action == "block" and blk.overridable
    # A bare text-only reason must NOT release it; the contract path must.
    assert g.accept_override("review matters", None) is False
    rp = tmp_path / "review.md"
    rp.write_text("the wait produced real findings: 3 verified defects.\n")
    assert g.accept_override(
        f"Report {rp} says: 'the wait produced real findings: 3 verified defects.'",
        None) is True


# ── F2: streak counts only polls taken while the budget is thin ──────────────

def test_unarmed_polls_do_not_accumulate_streak(ledger_env):
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(900))  # floor=300: not armed
    nr = f"task {ledger_env['reviewer_id']}: not ready (still RUNNING)"
    for _ in range(3):
        g.check_post(_ctx(result=nr, args={"task_id": ledger_env["reviewer_id"]}))
    assert g._not_ready.get(ledger_env["reviewer_id"], 0) == 0
    # First armed poll is soft, second blocks only after armed not-ready rounds.
    g._stats_fn = lambda: _stats(250)
    r1 = g.check_pre(_ctx(args={"task_id": ledger_env["reviewer_id"]}))
    assert r1.action == "inject"
    g.check_post(_ctx(result=nr, args={"task_id": ledger_env["reviewer_id"]}))
    r2 = g.check_pre(_ctx(args={"task_id": ledger_env["reviewer_id"]}))
    assert r2.action == "inject"
    g.check_post(_ctx(result=nr, args={"task_id": ledger_env["reviewer_id"]}))
    r3 = g.check_pre(_ctx(args={"task_id": ledger_env["reviewer_id"]}))
    assert r3.action == "block"


def test_armed_to_unarmed_transition_resets_streak(ledger_env):
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(250))
    tid = ledger_env["reviewer_id"]
    nr = f"task {tid}: not ready (still RUNNING)"
    g.check_post(_ctx(result=nr, args={"task_id": tid}))
    g.check_post(_ctx(result=nr, args={"task_id": tid}))
    assert g._not_ready.get(tid, 0) == 2
    g._stats_fn = lambda: _stats(1200)  # budget refilled / not armed
    g.check_post(_ctx(result=nr, args={"task_id": tid}))
    assert g._not_ready.get(tid, 0) == 0


# ── silence conditions ───────────────────────────────────────────────────────

def test_no_stats_fn_silent(ledger_env):
    g = ReviewerDeadlineGuard()
    assert g.check_pre(_ctx(args={"task_id": ledger_env["reviewer_id"]})) is None
    assert g.check_post(_ctx(result="task x: not ready")) is None


def test_stats_fn_raising_silent(ledger_env):
    def boom():
        raise RuntimeError("stats hiccup")
    g = ReviewerDeadlineGuard(stats_fn=boom)
    assert g.check_pre(_ctx(args={"task_id": ledger_env["reviewer_id"]})) is None


def test_remaining_none_silent(ledger_env):
    g = ReviewerDeadlineGuard(stats_fn=lambda: {"pct": 99.0})
    assert g.check_pre(_ctx(args={"task_id": ledger_env["reviewer_id"]})) is None


def test_far_from_deadline_silent(ledger_env):
    # remaining 1000s of a 1800s budget = 55% left > max(300, 270) → armed=False
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(1000.0))
    assert g.check_pre(_ctx(args={"task_id": ledger_env["reviewer_id"]})) is None


# ── target classification ────────────────────────────────────────────────────

def test_is_reviewer_target(ledger_env):
    assert _is_reviewer_target(ledger_env["reviewer_id"])
    assert not _is_reviewer_target(ledger_env["plain_id"])
    assert not _is_reviewer_target("no-such-task")
    assert not _is_reviewer_target("")


def test_non_reviewer_poll_untouched_near_deadline(ledger_env):
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(100.0))
    # not-ready streaks on the plain task first — still no verdict
    for _ in range(BLOCK_THRESHOLD + 2):
        g.check_post(_ctx(args={"task_id": ledger_env["plain_id"]},
                          result=f"task {ledger_env['plain_id']}: not ready"))
    assert g.check_pre(_ctx(args={"task_id": ledger_env["plain_id"]})) is None


# ── L1 soft inject / L2 hard block ──────────────────────────────────────────

def test_near_deadline_soft_inject_then_block(ledger_env):
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(100.0))
    tid = ledger_env["reviewer_id"]
    # no not-ready rounds yet → soft inject
    v = g.check_pre(_ctx(args={"task_id": tid}))
    assert v is not None and v.action == "inject"
    assert "adjudicate" in v.message and tid in v.message
    # 1 not-ready round → still soft
    g.check_post(_ctx(args={"task_id": tid}, result=f"task {tid}: not ready"))
    v = g.check_pre(_ctx(args={"task_id": tid}))
    assert v is not None and v.action == "inject"
    # 2 not-ready rounds → block on the next poll
    g.check_post(_ctx(args={"task_id": tid}, result=f"task {tid}: not ready"))
    v = g.check_pre(_ctx(args={"task_id": tid}))
    assert v is not None and v.action == "block" and v.overridable
    assert "Hard stop" in v.message and "report file" in v.message


def test_settled_poll_resets_counter(ledger_env):
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(100.0))
    tid = ledger_env["reviewer_id"]
    for _ in range(BLOCK_THRESHOLD):
        g.check_post(_ctx(args={"task_id": tid}, result=f"task {tid}: not ready"))
    g.check_post(_ctx(args={"task_id": tid}, result=f"task {tid}: PASSED"))
    v = g.check_pre(_ctx(args={"task_id": tid}))
    assert v is not None and v.action == "inject"  # back to soft, not block


def test_blocked_call_advances_nothing(ledger_env):
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(100.0))
    tid = ledger_env["reviewer_id"]
    before = g._not_ready.get(tid, 0)
    g.check_post(_ctx(args={"task_id": tid},
                      result="[BLOCKED BY GUARD] ..."))
    assert g._not_ready.get(tid, 0) == before


def test_other_tools_never_policed(ledger_env):
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(100.0))
    tid = ledger_env["reviewer_id"]
    assert g.check_pre(_ctx(name="read_file", args={"path": "/x"})) is None
    g.check_post(_ctx(name="read_file", result="ok"))
    assert g._not_ready.get(tid, 0) == 0


# ── worker-side exemption ────────────────────────────────────────────────────

def test_worker_side_exemption(ledger_env, monkeypatch):
    monkeypatch.setenv("FLAGSCALE_TASK_ID", "w1")
    assert _is_reviewer_worker()
    g = ReviewerDeadlineGuard(stats_fn=lambda: _stats(100.0))
    tid = ledger_env["reviewer_id"]
    for _ in range(BLOCK_THRESHOLD + 1):
        g.check_post(_ctx(args={"task_id": tid}, result=f"task {tid}: not ready"))
    assert g.check_pre(_ctx(args={"task_id": tid})) is None
