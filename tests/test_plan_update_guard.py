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

    def test_reminder_after_30_iterations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPlan(tmpdir)
            tp.create("Test", ["Step 1"])
            tp.update_step(1, "doing")
            
            guard = PlanUpdateGuard(tp)
            
            verdict = None
            for i in range(30):
                ctx = GuardContext(tool_name="shell", turn_count=i)
                verdict = guard.check_post(ctx)
            
            assert verdict is not None
            assert verdict.action == "inject"
            assert "30 iterations" in verdict.message
    
    def test_periodic_reminder_at_60_90(self):
        """Test that reminders fire at 30, 60, 90, ... iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tp = TaskPlan(tmpdir)
            tp.create("Test", ["Step 1"])
            tp.update_step(1, "doing")
            
            guard = PlanUpdateGuard(tp)
            
            # Should remind at 30, 60, 90
            reminders = []
            for i in range(95):
                ctx = GuardContext(tool_name="shell", turn_count=i)
                verdict = guard.check_post(ctx)
                if verdict and verdict.action == "inject":
                    reminders.append(guard._iters_since_update)
            
            assert reminders == [30, 60, 90]

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
