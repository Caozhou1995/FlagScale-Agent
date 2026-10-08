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

"""Tests for the decision-type TIME reminder and early judge consult.

Two behaviors added on top of the escalate-block loop diagnosis:

1. The FIRST time-gated reminder (inject, not block) already consults the
   LLM judge and appends the loop escape note when the recent trace looks
   like a sideways loop — so the agent gets a decision-type nudge while
   budget remains, not only at the escalate stop.
2. The long-hold hint text asks the falsifiable-prediction questions
   (expected NUMBER, what it proves, which route to switch to).
"""

import tempfile

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.guard.plan_update import (
    PlanUpdateGuard,
    _extract_recent_activity,
)
from flagscale_agent.react.guard import GuardContext


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _fire_first_inject(guard, clock, messages, classify_fn):
    """Cross two time windows so the first reminder fires (inject path).

    The first window only ANCHORS the clock (anchor is None on guard start),
    so the earliest possible reminder is the second crossed window.
    """
    for _ in range(2):
        clock.advance(PlanUpdateGuard.TIME_REMIND_SECONDS + 1)
        ctx = GuardContext(
            tool_name="shell",
            turn_count=1,
            messages=messages,
            classify_fn=classify_fn,
        )
        v = guard.check_post(ctx)
        if v is not None:
            return v
    raise AssertionError("no reminder fired after two windows")


def _guard_with_doing_step(tmpdir):
    tp = TaskPlan(tmpdir)
    tp.create("Test", ["Step 1"])
    tp.update_step(1, "doing")
    clock = _FakeClock()
    return PlanUpdateGuard(tp, time_fn=clock), clock


_LOOP_MESSAGES = [
    {"role": "assistant", "content": "retrying the same edit on vm.js"},
]


class TestDecisionTypeReminder:
    def test_first_inject_carries_loop_note_when_judged_looping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, clock = _guard_with_doing_step(tmpdir)

            def classify_fn(category, context, default=False):
                assert category == "agent_stuck_in_sideways_loop"
                return True  # judge: yes, looping

            v = _fire_first_inject(guard, clock, _LOOP_MESSAGES, classify_fn)
            assert v.action == "inject"
            # Decision-type nudge present on the FIRST advisory.
            assert "The recent trace looks like a LOOP" in v.message
            assert "DOWNWARD" in v.message and "UPWARD" in v.message

    def test_first_inject_no_note_when_judge_says_not_looping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, clock = _guard_with_doing_step(tmpdir)

            def classify_fn(category, context, default=False):
                return False

            v = _fire_first_inject(guard, clock, _LOOP_MESSAGES, classify_fn)
            assert v.action == "inject"
            assert "The recent trace looks like a LOOP" not in v.message
            # Base time body still present.
            assert "TIME signal" in v.message

    def test_first_inject_no_judge_no_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, clock = _guard_with_doing_step(tmpdir)
            v = _fire_first_inject(guard, clock, _LOOP_MESSAGES, None)
            assert v.action == "inject"
            assert "The recent trace looks like a LOOP" not in v.message

    def test_judge_exception_on_inject_degrades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, clock = _guard_with_doing_step(tmpdir)

            def classify_fn(category, context, default=False):
                raise RuntimeError("judge unavailable")

            v = _fire_first_inject(guard, clock, _LOOP_MESSAGES, classify_fn)
            assert v.action == "inject"
            assert "The recent trace looks like a LOOP" not in v.message

    def test_no_activity_skips_judge_on_inject(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, clock = _guard_with_doing_step(tmpdir)
            called = {"n": 0}

            def classify_fn(category, context, default=False):
                called["n"] += 1
                return True

            v = _fire_first_inject(guard, clock, [], classify_fn)
            assert v.action == "inject"
            assert called["n"] == 0

    def test_long_hold_hint_asks_prediction_questions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, clock = _guard_with_doing_step(tmpdir)
            v = _fire_first_inject(guard, clock, [], None)
            assert v.action == "inject"
            # Falsifiable-prediction questions in the long-hold hint.
            assert "NUMBER" in v.message
            assert "PROVE" in v.message
            assert "switch" in v.message.lower()
