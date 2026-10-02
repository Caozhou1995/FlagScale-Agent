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

"""PollSpacingGuard — nudge on back-to-back no-progress poll rounds.

Contract:
- post-only: check_pre never fires.
- round 1 (first not-ready poll) stays silent — one wait is legitimate.
- streak >= 2 → inject; >= 3 → harder "you are spinning" message.
- a settled poll (PASSED/FAILED/state=complete), any other tool, or a blocked
  call resets the streak.
- reunite not-ready text no longer says "poll again later" (the old wording
  actively invited bare re-polling) but still contains "not ready".
"""

import pytest

from flagscale_agent.react.guard.poll_spacing import (
    PollSpacingGuard,
    _poll_round_not_ready,
)
from flagscale_agent.react.multi_agent.reunite import PollTasksTool


def _ctx(name, result, messages=None, **kw):
    from flagscale_agent.react.guard import GuardContext
    return GuardContext(tool_name=name, tool_result=result,
                        messages=messages or [], **kw)


def _iter(n):
    """Fake history snapshot of length n — stands in for one LLM iteration."""
    return [f"msg{i}" for i in range(n)]


# ── classification helper ────────────────────────────────────────────────────

def test_not_ready_poll_detected():
    assert _poll_round_not_ready("poll_tasks", "task t1: not ready (status=RUNNING)")
    assert _poll_round_not_ready(
        "dispatch_many", "dispatch d1: state=running 2/5 settled")

def test_progress_rounds_not_flagged():
    assert not _poll_round_not_ready(
        "poll_tasks", "task t1: PASSED (status=DONE)")
    # REAL running format carries the literal substring "state=complete" inside
    # its trailing hint ("poll again until state=complete") — a bare substring
    # match would misclassify it as settled. Line-anchored check required.
    running = ("dispatch d1: state=running (settled 0/2 so far)\n"
               "- [~REPORTED] task=t ptr=/x\n"
               "these are PROVISIONAL (status only); poll again until "
               "state=complete for acceptance-checked PASS/FAIL.")
    assert _poll_round_not_ready("dispatch_many", running)
    assert not _poll_round_not_ready(
        "dispatch_many", "dispatch d1: state=complete — 5/5")
    assert not _poll_round_not_ready(
        "dispatch_many", "dispatch d1: state=complete — JOB ERROR: boom")
    assert not _poll_round_not_ready(
        "poll_tasks", "task t1: FAILED (status=REPORTED)")


def test_settled_failed_with_noisy_stdout_tail_not_flagged():
    # A settled FAILED round's stdout_tail routinely embeds "not ready" /
    # "state=running" (service probes, log dumps). The verdict lives on the
    # FIRST line; the payload lines must never reclassify a settled round.
    real = ("task abc: FAILED (status=REJECTED)\n"
            "note: acceptance FAILED (1/1): [exit=1] false\n"
            "  [FAIL] (exit=1) test -f out\n"
            "      stdout_tail: wait_for_svc: service not ready after 120s")
    assert not _poll_round_not_ready("poll_tasks", real)
    assert not _poll_round_not_ready(
        "poll_tasks", "task abc: PASSED (status=DONE)\n      stdout_tail: svc not ready")


def test_dispatch_start_is_not_a_poll_round():
    # dispatch_many(action='dispatch') output must not count as a not-ready
    # poll round (it is real work — starting the fan-out).
    start = ("dispatched 2 task(s) in the background (degree=2).\n"
             "dispatch_id: dsp_1\n"
             "Poll with: dispatch_many(action='poll', dispatch_id='dsp_1').")
    assert not _poll_round_not_ready("dispatch_many", start)


def test_double_poll_in_one_iteration_counts_once():
    # The kernel zips per tool_call; two poll calls emitted in the SAME
    # assistant message must advance the streak by one, not two (the advisory
    # claims per-iteration cost). check_post distinguishes iterations via the
    # history length (tool results are appended after the per-call guard loop,
    # so all calls in one iteration share it; it grows between iterations).
    g = PollSpacingGuard()
    r = "task t: not ready (status=RUNNING); no ledger changes made."
    assert g.check_post(_ctx("poll_tasks", r, messages=_iter(10))) is None
    v = g.check_post(
        _ctx("dispatch_many", "dispatch d: state=running (settled 0/2)",
             messages=_iter(10)))
    assert v is None and g._streak == 1  # one iteration, one advance

def test_other_tools_ignored():
    assert not _poll_round_not_ready("shell", "not ready")
    assert not _poll_round_not_ready("poll_tasks", "")
    assert not _poll_round_not_ready("poll_tasks", None)


# ── guard verdicts ───────────────────────────────────────────────────────────

def test_first_not_ready_round_is_silent():
    g = PollSpacingGuard()
    v = g.check_post(_ctx("poll_tasks", "task t: not ready (status=RUNNING)"))
    assert v is None  # one wait is legitimate
    assert g._streak == 1

def test_second_round_injects_bounded_wait_hint():
    g = PollSpacingGuard()
    r = "task t: not ready (status=RUNNING)"
    assert g.check_post(_ctx("poll_tasks", r, messages=_iter(10))) is None
    v = g.check_post(_ctx("poll_tasks", r, messages=_iter(11)))
    assert v is not None and v.action == "inject"
    assert "sleep <=30s" in v.message
    assert "2 not-ready polls" in v.message

def test_third_round_escalates_to_spinning():
    g = PollSpacingGuard()
    r = "task t: not ready (status=RUNNING)"
    for n in (10, 11):
        g.check_post(_ctx("poll_tasks", r, messages=_iter(n)))
    v = g.check_post(_ctx("poll_tasks", r, messages=_iter(12)))
    assert v is not None and "spinning" in v.message
    assert "3 consecutive" in v.message

def test_settled_poll_resets_streak():
    g = PollSpacingGuard()
    r = "task t: not ready (status=RUNNING)"
    g.check_post(_ctx("poll_tasks", r))
    v = g.check_post(_ctx("poll_tasks", "task t: PASSED (status=DONE)"))
    assert v is None and g._streak == 0

def test_other_tool_resets_streak():
    g = PollSpacingGuard()
    r = "task t: not ready (status=RUNNING)"
    g.check_post(_ctx("poll_tasks", r))
    g.check_post(_ctx("shell", "some output"))
    v = g.check_post(_ctx("poll_tasks", r))
    assert v is None  # streak was reset by the intervening tool

def test_blocked_call_does_not_advance_streak():
    g = PollSpacingGuard()
    blocked = "[BLOCKED BY GUARD] This tool call was prevented."
    g.check_post(_ctx("poll_tasks", blocked))
    assert g._streak == 0

def test_reset_turn_clears_streak():
    g = PollSpacingGuard()
    r = "task t: not ready (status=RUNNING)"
    g.check_post(_ctx("poll_tasks", r))
    g.reset_turn()
    assert g._streak == 0


# ── reunite not-ready wording ────────────────────────────────────────────────

@pytest.fixture
def led(tmp_path):
    from flagscale_agent.react.multi_agent.ledger import TaskLedger
    return TaskLedger(str(tmp_path / "tasks"))


def test_reunite_not_ready_no_longer_says_poll_again_later(led, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    c = _make_running(led, work)
    out = PollTasksTool(ledger=led).execute(action="check", task_id=c.id)
    assert "not ready" in out
    assert "poll again later" not in out
    assert "sleep <=30s" in out


def _make_running(led, work):
    from flagscale_agent.react.multi_agent.contract import Contract
    from flagscale_agent.react.multi_agent.ledger import RUNNING
    c = Contract.build(
        goal="g", constraints={"writable": [str(work)]},
        acceptance=[{"check": "true"}], output_ptr=str(work / "out.md"),
    )
    led.create(c)
    led.transition(c.id, RUNNING, pid=1)
    return c
