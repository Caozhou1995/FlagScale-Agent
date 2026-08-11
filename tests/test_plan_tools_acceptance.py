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

"""Test plan tools with acceptance and verification."""

import pytest
import tempfile
import shutil

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.tools.plan_create import PlanCreateTool
from flagscale_agent.react.tools.plan_update import PlanUpdateTool


@pytest.fixture
def plan_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class TestPlanCreateToolStructuredSteps:
    def test_dict_steps_with_acceptance(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="Test",
            steps=[
                {"title": "A", "acceptance": ["A1", "A2"]},
                {"title": "B", "acceptance": ["B1"]},
            ]
        )
        assert "Plan created" in result
        plan = tp.get_active()
        assert plan["steps"][0]["acceptance"] == ["A1", "A2"]
        assert plan["steps"][1]["acceptance"] == ["B1"]

    def test_mixed_steps(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="Test",
            steps=[
                {"title": "A", "acceptance": ["A1"]},
                "B",
            ]
        )
        assert "Plan created" in result
        plan = tp.get_active()
        assert plan["steps"][0]["acceptance"] == ["A1"]
        assert plan["steps"][1]["acceptance"] == []

    def test_backward_compatible_string_steps(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(title="Test", steps=["A", "B"])
        assert "Plan created" in result
        plan = tp.get_active()
        assert plan["steps"][0]["title"] == "A"
        assert plan["steps"][0]["acceptance"] == []

    def test_dict_without_title_fails(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="Test",
            steps=[{"acceptance": ["A1"]}]
        )
        assert "ERROR" in result
        assert "title" in result.lower()


class TestPlanUpdateToolVerification:
    def test_step_done_with_verification(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tp.create("Test", ["A", "B"])
        tool = PlanUpdateTool(tp)
        
        result = tool.execute(
            action="step_done",
            step_id=1,
            verification=["V1", "V2"]
        )
        assert "✓ V1" in result
        assert "✓ V2" in result
        
        plan = tp.get_active()
        assert plan["steps"][0]["verification"] == ["V1", "V2"]

    def test_step_done_without_verification(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tp.create("Test", ["A"])
        tool = PlanUpdateTool(tp)
        
        result = tool.execute(action="step_done", step_id=1)
        assert "✓" in result or "done" in result.lower()
        
        plan = tp.get_active()
        assert plan["steps"][0]["verification"] == []


class TestPlanUpdateToolUpdateAcceptance:
    def test_update_acceptance(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tp.create("Test", ["A"])
        tool = PlanUpdateTool(tp)
        
        result = tool.execute(
            action="update_acceptance",
            step_id=1,
            acceptance=["A1", "A2"]
        )
        assert "A1" in result
        assert "A2" in result
        
        plan = tp.get_active()
        assert plan["steps"][0]["acceptance"] == ["A1", "A2"]

    def test_update_acceptance_without_list_fails(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tp.create("Test", ["A"])
        tool = PlanUpdateTool(tp)
        
        result = tool.execute(action="update_acceptance", step_id=1)
        assert "ERROR" in result


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
