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

"""Tests for PlanUpdateGuard — iteration-based reminder logic."""

import tempfile
import pytest

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.guard.plan_update import PlanUpdateGuard
from flagscale_agent.react.guard import GuardContext


class TestPlanUpdateGuardIterationCounting:
    """Test that PlanUpdateGuard uses iteration counting, not turn counting."""

    def test_no_reminder_without_active_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPlan(tmpdir)
            guard = PlanUpdateGuard(tp)
            
            ctx = GuardContext(tool_name="shell", turn_count=50)
            verdict = guard.check_post(ctx)
            assert verdict is None

    def test_first_reminder_at_10_iterations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPlan(tmpdir)
            tp.create("Test", ["Step 1"])
            tp.update_step(1, "doing")
            
            guard = PlanUpdateGuard(tp)
            
            # No reminder before the first threshold.
            for i in range(9):
                ctx = GuardContext(tool_name="shell", turn_count=i)
                assert guard.check_post(ctx) is None
            
            # Fires exactly at FIRST_REMIND (10).
            verdict = guard.check_post(GuardContext(tool_name="shell", turn_count=9))
            assert verdict is not None
            assert verdict.action == "inject"
            assert "10 iterations" in verdict.message
    
    def test_stall_framing_in_message(self):
        """Message frames the situation as a stall and prescribes down/up, not sideways."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPlan(tmpdir)
            tp.create("Test", ["Step 1"])
            tp.update_step(1, "doing")
            
            guard = PlanUpdateGuard(tp)
            
            verdict = None
            for i in range(10):
                verdict = guard.check_post(GuardContext(tool_name="shell", turn_count=i))
            
            assert verdict is not None
            msg = verdict.message.lower()
            assert "stall" in msg
            assert "information gain" in msg
            assert "downward" in msg
            assert "upward" in msg
            assert "sideways" in msg
            assert "variant" in msg
            # Third escape (path-tracing case): on optimization steps, oscillating
            # around a plateau is not a stall to escape — it means gains have stopped
            # and you should ship the best measured version, not tune one more turn.
            assert "third escape" in msg
            assert "oscillating" in msg
            assert "variance" in msg
            assert "deliver the best version you already measured" in msg
            # Fourth escape (budget-order case): when a required output path exists in
            # the task but no artifact is on disk yet, stop refining and write the
            # crudest complete-but-valid version to the path now.
            assert "fourth escape" in msg
            assert "budget order" in msg
            assert "crudest" in msg
            assert "never got written" in msg
            assert verdict.reason == "possible_stall"
            # Reconstructed as answer-first self-check: the stall message must OPEN
            # by forcing a concrete answer to "what did the last round tell you that
            # you did not already know" — an externally-referring question whose
            # "nothing new" answer is itself the proof of a loop. This must appear
            # before the escape guidance, not be buried after the framing.
            assert "what did the last round tell you" in msg
            assert "nothing new" in msg
            # the question comes early (before the down/up escape prescription)
            assert msg.index("what did the last round tell you") < msg.index("downward")
    
    def test_periodic_reminders_10_then_every_20(self):
        """Reminders fire at 10, then every 20 after: 10, 30, 50, 70, 90.

        Timing is independent of whether a given reminder injects or blocks —
        collect any verdict so this stays a pure timing test (escalation is
        covered separately in TestPlanUpdateGuardEscalation).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPlan(tmpdir)
            tp.create("Test", ["Step 1"])
            tp.update_step(1, "doing")
            
            guard = PlanUpdateGuard(tp)
            
            reminders = []
            for i in range(95):
                ctx = GuardContext(tool_name="shell", turn_count=i)
                verdict = guard.check_post(ctx)
                if verdict and verdict.action in ("inject", "block"):
                    reminders.append(guard._iters_since_update)
            
            assert reminders == [10, 30, 50, 70, 90]

    def test_meta_tools_dont_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPlan(tmpdir)
            tp.create("Test", ["Step 1"])
            tp.update_step(1, "doing")
            
            guard = PlanUpdateGuard(tp)
            
            meta_tools = ["plan_status", "evict", "memory_read"]
            
            for i, tool in enumerate(meta_tools * 15):
                ctx = GuardContext(tool_name=tool, turn_count=i)
                verdict = guard.check_post(ctx)
                assert verdict is None
            
            assert guard._iters_since_update == 0

    def test_counter_resets_on_plan_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPlan(tmpdir)
            tp.create("Test", ["Step 1"])
            tp.update_step(1, "doing")
            
            guard = PlanUpdateGuard(tp)
            
            for i in range(20):
                ctx = GuardContext(tool_name="shell", turn_count=i)
                guard.check_post(ctx)
            
            assert guard._iters_since_update == 20
            
            ctx_update = GuardContext(tool_name="plan_update", turn_count=20)
            guard.check_post(ctx_update)
            assert guard._iters_since_update == 0


class _FakeClock:
    """Controllable monotonic clock for time-signal tests."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class TestPlanUpdateGuardTimeSignal:
    """Test the wall-clock stall signal, orthogonal to the count signal."""

    def _guard_with_doing_step(self, tmpdir, clock):
        tp = TaskPlan(tmpdir)
        tp.create("Test", ["Step 1"])
        tp.update_step(1, "doing")
        return PlanUpdateGuard(tp, time_fn=clock)

    def test_time_fires_before_count_when_calls_are_slow(self):
        """Few tool calls but long thinks between them → time signal fires first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard = self._guard_with_doing_step(tmpdir, clock)

            # First call anchors the clock, no fire.
            assert guard.check_post(GuardContext(tool_name="shell")) is None
            # A single long think (>180s) precedes the second call.
            clock.advance(200)
            verdict = guard.check_post(GuardContext(tool_name="shell"))
            assert verdict is not None
            assert verdict.action == "inject"
            # Only 2 tool calls — count signal (10) has NOT fired.
            assert guard._iters_since_update == 2
            assert "min elapsed" in verdict.message
            assert verdict.reason == "possible_stall"

    def test_time_reminder_is_periodic(self):
        """Time signal re-anchors on fire → one reminder per 180s window."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard = self._guard_with_doing_step(tmpdir, clock)

            fires = 0
            guard.check_post(GuardContext(tool_name="shell"))  # anchor
            for _ in range(6):
                clock.advance(100)  # 100s per step
                v = guard.check_post(GuardContext(tool_name="shell"))
                if v is not None:
                    fires += 1
            # Re-anchor on each fire: fires at cumulative 200, 400(from re-anchor
            # 200→ +200), 600 → 3 windows.
            assert fires == 3

    def test_plan_update_resets_time_anchor(self):
        """A plan_update resets the clock; elapsed restarts from there."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard = self._guard_with_doing_step(tmpdir, clock)

            guard.check_post(GuardContext(tool_name="shell"))  # anchor at t=0
            clock.advance(170)
            # plan_update resets anchor to t=170.
            guard.check_post(GuardContext(tool_name="plan_update"))
            clock.advance(170)  # 170s since reset → below 180, no fire
            assert guard.check_post(GuardContext(tool_name="shell")) is None
            clock.advance(20)  # now 190s since reset → fire
            assert guard.check_post(GuardContext(tool_name="shell")) is not None

    def test_meta_tools_dont_tick_time(self):
        """Meta tools return before the time check → they never fire a reminder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard = self._guard_with_doing_step(tmpdir, clock)

            guard.check_post(GuardContext(tool_name="shell"))  # anchor
            clock.advance(500)
            # A meta tool, despite huge elapsed, must not fire.
            assert guard.check_post(GuardContext(tool_name="memory_read")) is None

    def test_displayed_minutes_are_cumulative_not_per_window(self):
        """Regression: the reminder must display TOTAL time stuck on the step,
        cumulative from _stall_start — NOT the periodic-window elapsed, which
        resets on every fire and would under-report the real stall duration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard = self._guard_with_doing_step(tmpdir, clock)

            guard.check_post(GuardContext(tool_name="shell"))  # anchor at t=0
            mins_seen = []
            for _ in range(3):
                clock.advance(190)  # each window > 180s → fire once
                v = guard.check_post(GuardContext(tool_name="shell"))
                assert v is not None
                # Extract the "~N min elapsed" figure from the message.
                import re
                m = re.search(r"~(\d+) min elapsed", v.message)
                assert m is not None, v.message
                mins_seen.append(int(m.group(1)))
            # Cumulative: ~3min, ~6min, ~9min (total from stall start), NOT
            # 3,3,3 (which is the per-window elapsed the old code showed).
            assert mins_seen == [3, 6, 9], mins_seen

    def test_stall_start_resets_on_plan_update(self):
        """plan_update resets _stall_start, so the cumulative total restarts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard = self._guard_with_doing_step(tmpdir, clock)

            guard.check_post(GuardContext(tool_name="shell"))  # anchor at t=0
            clock.advance(600)
            guard.check_post(GuardContext(tool_name="plan_update"))  # reset at t=600
            clock.advance(190)
            v = guard.check_post(GuardContext(tool_name="shell"))
            assert v is not None
            import re
            m = re.search(r"~(\d+) min elapsed", v.message)
            assert m is not None, v.message
            # Only ~3 min since the reset, not ~13 min from the very start.
            assert int(m.group(1)) == 3


class TestPlanUpdateGuardEscalation:
    """Every ESCALATE_AFTER-th reminder blocks; then resets and repeats."""

    def _guard_with_doing_step(self, tmpdir, clock=None):
        tp = TaskPlan(tmpdir)
        tp.create("Test", ["Step 1"])
        tp.update_step(1, "doing")
        if clock is not None:
            return PlanUpdateGuard(tp, time_fn=clock), tp
        return PlanUpdateGuard(tp), tp

    def _fire_count_reminder(self, guard, start=0):
        """Drive enough shell calls to trigger one COUNT reminder; return it."""
        v = None
        # FIRST_REMIND then every REMIND_INTERVAL — just call until one fires.
        for i in range(start, start + 200):
            v = guard.check_post(GuardContext(tool_name="shell", turn_count=i))
            if v is not None:
                return v
        raise AssertionError("no reminder fired")

    def test_third_reminder_escalates_to_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._guard_with_doing_step(tmpdir)

            v1 = self._fire_count_reminder(guard)
            assert v1.action == "inject"
            v2 = self._fire_count_reminder(guard)
            assert v2.action == "inject"
            v3 = self._fire_count_reminder(guard)
            assert v3.action == "block"
            assert v3.reason == "repeated_stall_ignored"
            # Block carries the same diagnostic body plus the escalation note.
            assert "BLOCKS" in v3.message
            assert "_override_reason" in v3.message
            # Block must also offer the "ship best on a plateau" exit, not only
            # "name a refuted assumption + next method-class".
            assert "oscillating around a plateau" in v3.message
            assert "shipping best" in v3.message
            # The block demands a concrete, falsifiable fact — not a vague
            # "I'm progressing" defence (guards against defensive-note escape).
            assert "falsifiable" in v3.message
            assert "vague" in v3.message.lower()
            assert (
                "refuted" in v3.message.lower()
                or "eliminated" in v3.message.lower()
            )
            # Side-channel clause: on deliverable+threshold tasks, a fact only
            # clears the block if OBSERVED AT THE DELIVERABLE (edit + re-run +
            # measure), not derived in a scratch script / prototype / on paper.
            # Guards against path-tracing-style stall where each block is cleared
            # by a new paper derivation that never touches the graded artifact.
            msg = v3.message.lower()
            assert "deliverable" in msg
            assert "side-channel" in msg or "side channel" in msg
            assert "before/after" in msg or "re-ran" in msg or "re-measure" in msg
            # Must name the wrong-medium symptom: derivation without a score move.
            assert "paper" in msg or "prototype" in msg or "scratch" in msg

    def test_cycle_repeats_after_block(self):
        """inject, inject, block, inject, inject, block."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._guard_with_doing_step(tmpdir)
            actions = [self._fire_count_reminder(guard).action for _ in range(6)]
            assert actions == [
                "inject", "inject", "block",
                "inject", "inject", "block",
            ]

    def test_substantive_plan_update_resets_escalation_count(self):
        """A SUBSTANTIVE response earns a clean slate — no accumulation to block.

        Substantive = a progress action (step_done/skip/...) or any update
        carrying non-empty notes. Here we use notes.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._guard_with_doing_step(tmpdir)

            assert self._fire_count_reminder(guard).action == "inject"
            assert self._fire_count_reminder(guard).action == "inject"
            # Agent responds substantively (records a concrete fact) → reset.
            guard.check_post(GuardContext(
                tool_name="plan_update", turn_count=999,
                tool_args={"action": "step_doing",
                           "notes": "measured X=3, ruled out linear approx"},
            ))
            # Next reminder is the FIRST again → inject, not block.
            assert self._fire_count_reminder(guard).action == "inject"

    def test_empty_ping_does_not_reset_escalation_count(self):
        """A bare plan_update (no notes, non-progress action) must NOT buy a
        clean escalation slate — it is an empty guard-clearing ping, exactly the
        gaming move the block exists to stop. Escalation count is preserved, so
        the next reminder still escalates to block."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._guard_with_doing_step(tmpdir)

            assert self._fire_count_reminder(guard).action == "inject"
            assert self._fire_count_reminder(guard).action == "inject"
            # Empty ping: step_doing with no notes → non-substantive.
            guard.check_post(GuardContext(
                tool_name="plan_update", turn_count=999,
                tool_args={"action": "step_doing"},
            ))
            # Escalation count preserved (still at 2) → next reminder is the
            # 3rd → BLOCK, not a fresh inject.
            assert self._fire_count_reminder(guard).action == "block"

    def test_empty_ping_reblocked_while_block_pending(self):
        """Once a block fires, an empty ping cannot clear it — it is re-blocked
        with a distinct reason until a substantive response arrives."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._guard_with_doing_step(tmpdir)

            self._fire_count_reminder(guard)  # inject 1
            self._fire_count_reminder(guard)  # inject 2
            assert self._fire_count_reminder(guard).action == "block"  # block

            # Empty ping tries to clear the block → re-blocked.
            v = guard.check_post(GuardContext(
                tool_name="plan_update", turn_count=999,
                tool_args={"action": "step_doing"},
            ))
            assert v is not None and v.action == "block"
            assert v.reason == "empty_ping_does_not_clear_block"

    def test_substantive_update_clears_pending_block(self):
        """A substantive response (progress action or notes) clears the pending
        block; afterwards the guard is back to a clean slate (next → inject)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._guard_with_doing_step(tmpdir)

            self._fire_count_reminder(guard)  # inject 1
            self._fire_count_reminder(guard)  # inject 2
            assert self._fire_count_reminder(guard).action == "block"  # block

            # Substantive response: mark a concrete fact in notes → clears block.
            v = guard.check_post(GuardContext(
                tool_name="plan_update", turn_count=999,
                tool_args={"action": "step_doing",
                           "notes": "isolated bug to tokenizer offset, switching"},
            ))
            assert v is None
            # Clean slate: next reminder is a fresh inject, not a block.
            assert self._fire_count_reminder(guard).action == "inject"

    def test_progress_action_clears_pending_block_without_notes(self):
        """A progress action (step_done) is substantive even with no notes —
        it mutates real plan state — so it clears a pending block."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, tp = self._guard_with_doing_step(tmpdir)

            self._fire_count_reminder(guard)
            self._fire_count_reminder(guard)
            assert self._fire_count_reminder(guard).action == "block"

            v = guard.check_post(GuardContext(
                tool_name="plan_update", turn_count=999,
                tool_args={"action": "step_done"},
            ))
            assert v is None

    def test_time_and_count_share_one_escalation_counter(self):
        """A time reminder and a count reminder both advance the SAME counter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clock = _FakeClock()
            guard, _ = self._guard_with_doing_step(tmpdir, clock)

            # Reminder 1: TIME signal.
            guard.check_post(GuardContext(tool_name="shell"))  # anchor at t=0
            clock.advance(PlanUpdateGuard.TIME_REMIND_SECONDS + 1)
            v1 = guard.check_post(GuardContext(tool_name="shell"))
            assert v1 is not None and v1.action == "inject"

            # Reminder 2: TIME signal again.
            clock.advance(PlanUpdateGuard.TIME_REMIND_SECONDS + 1)
            v2 = guard.check_post(GuardContext(tool_name="shell"))
            assert v2 is not None and v2.action == "inject"

            # Reminder 3: TIME signal — shared counter hits ESCALATE_AFTER → block.
            clock.advance(PlanUpdateGuard.TIME_REMIND_SECONDS + 1)
            v3 = guard.check_post(GuardContext(tool_name="shell"))
            assert v3 is not None and v3.action == "block"
