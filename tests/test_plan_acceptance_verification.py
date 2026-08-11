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

"""Tests for Plan Acceptance & Verification mechanism."""

import os
import shutil
import tempfile

import pytest

from flagscale_agent.react.plan import TaskPlan


@pytest.fixture
def plan_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tp(plan_dir):
    return TaskPlan(plan_dir)


class TestAcceptanceInCreate:
    """Test acceptance field in plan creation."""

    def test_create_with_dict_steps(self, tp):
        """Create plan with structured steps containing acceptance."""
        steps = [
            {
                "title": "Implement feature X",
                "acceptance": ["Code compiles", "Tests pass", "Docs updated"]
            },
            {
                "title": "Deploy to staging",
                "acceptance": ["Service runs", "Health check passes"]
            }
        ]
        plan = tp.create("Test plan", steps)
        assert len(plan["steps"]) == 2
        assert plan["steps"][0]["title"] == "Implement feature X"
        assert plan["steps"][0]["acceptance"] == ["Code compiles", "Tests pass", "Docs updated"]
        assert plan["steps"][0]["verification"] == []
        assert plan["steps"][1]["acceptance"] == ["Service runs", "Health check passes"]

    def test_create_with_string_steps_has_empty_acceptance(self, tp):
        """Backward compatibility: string steps have empty acceptance."""
        plan = tp.create("Test", ["Step A", "Step B"])
        assert plan["steps"][0]["acceptance"] == []
        assert plan["steps"][0]["verification"] == []

    def test_create_with_mixed_steps(self, tp):
        """Mix of string and dict steps."""
        steps = [
            "Simple step",
            {"title": "Complex step", "acceptance": ["criterion 1", "criterion 2"]},
            "Another simple step"
        ]
        plan = tp.create("Mixed", steps)
        assert len(plan["steps"]) == 3
        assert plan["steps"][0]["acceptance"] == []
        assert plan["steps"][1]["acceptance"] == ["criterion 1", "criterion 2"]
        assert plan["steps"][2]["acceptance"] == []


class TestVerificationInUpdate:
    """Test verification field in step updates."""

    def test_update_with_verification(self, tp):
        """Provide verification when marking step done."""
        tp.create("Test", [{"title": "Task", "acceptance": ["A", "B"]}])
        plan = tp.update_step(
            1, "done",
            verification=["Proof of A", "Proof of B"]
        )
        assert plan["steps"][0]["verification"] == ["Proof of A", "Proof of B"]
        assert plan["steps"][0]["status"] == "done"

    def test_update_without_verification(self, tp):
        """Verification remains empty if not provided."""
        tp.create("Test", ["Task"])
        plan = tp.update_step(1, "done")
        assert plan["steps"][0]["verification"] == []

    def test_verification_replaces_previous(self, tp):
        """Verification is replaced, not appended."""
        tp.create("Test", ["Task"])
        tp.update_step(1, "doing", verification=["v1"])
        plan = tp.update_step(1, "done", verification=["v2", "v3"])
        assert plan["steps"][0]["verification"] == ["v2", "v3"]


class TestUpdateAcceptance:
    """Test update_acceptance() method."""

    def test_update_acceptance(self, tp):
        """Change acceptance criteria during execution."""
        tp.create("Test", [{"title": "Task", "acceptance": ["old1", "old2"]}])
        plan = tp.update_acceptance(1, ["new1", "new2", "new3"])
        assert plan["steps"][0]["acceptance"] == ["new1", "new2", "new3"]

    def test_update_acceptance_appends_note(self, tp):
        """Updating acceptance auto-appends a timestamped note."""
        tp.create("Test", [{"title": "Task", "acceptance": ["A"]}])
        tp.update_step(1, "doing", "working on it")
        plan = tp.update_acceptance(1, ["A", "B"])
        notes = plan["steps"][0]["notes"]
        assert "working on it" in notes
        assert "[Acceptance updated at" in notes

    def test_update_acceptance_no_active_plan(self, tp):
        """Error if no active plan."""
        with pytest.raises(ValueError, match="No active plan"):
            tp.update_acceptance(1, ["new"])

    def test_update_acceptance_step_not_found(self, tp):
        """Error if step doesn't exist."""
        tp.create("Test", ["A"])
        with pytest.raises(ValueError, match="Step 99 not found"):
            tp.update_acceptance(99, ["new"])


class TestAddStepsWithAcceptance:
    """Test add_steps() with acceptance."""

    def test_add_dict_steps(self, tp):
        """Add structured steps with acceptance."""
        tp.create("Test", ["A"])
        plan = tp.add_steps([
            {"title": "B", "acceptance": ["b1", "b2"]},
            {"title": "C", "acceptance": ["c1"]}
        ])
        assert len(plan["steps"]) == 3
        assert plan["steps"][1]["acceptance"] == ["b1", "b2"]
        assert plan["steps"][2]["acceptance"] == ["c1"]

    def test_add_mixed_steps(self, tp):
        """Add mix of string and dict."""
        tp.create("Test", ["A"])
        plan = tp.add_steps([
            "B",
            {"title": "C", "acceptance": ["x", "y"]}
        ])
        assert plan["steps"][1]["acceptance"] == []
        assert plan["steps"][2]["acceptance"] == ["x", "y"]


class TestFormatting:
    """Test display of acceptance and verification."""

    def test_format_plan_with_acceptance(self, tp):
        """Summary shows acceptance criteria."""
        tp.create("Test", [{"title": "Task", "acceptance": ["A", "B"]}])
        summary = tp.summary()
        assert "验收:" in summary
        assert "• A" in summary
        assert "• B" in summary

    def test_format_plan_with_verification(self, tp):
        """Summary shows verification evidence."""
        tp.create("Test", [{"title": "Task", "acceptance": ["A"]}])
        tp.update_step(1, "done", verification=["Proof A"])
        summary = tp.summary()
        assert "产出:" in summary
        assert "✓ Proof A" in summary

    def test_context_for_prompt_includes_acceptance(self, tp):
        """Prompt context shows acceptance."""
        tp.create("Test", [{"title": "T", "acceptance": ["X"]}])
        ctx = tp.context_for_prompt()
        assert "验收:" in ctx
        assert "• X" in ctx

    def test_context_for_prompt_includes_verification(self, tp):
        """Prompt context shows verification."""
        tp.create("Test", ["T"])
        tp.update_step(1, "done", verification=["Evidence"])
        ctx = tp.context_for_prompt()
        assert "产出:" in ctx
        assert "✓ Evidence" in ctx


class TestPersistence:
    """Test acceptance/verification persist across reloads."""

    def test_acceptance_persists(self, plan_dir):
        """Acceptance survives reload."""
        tp1 = TaskPlan(plan_dir)
        tp1.create("Test", [{"title": "T", "acceptance": ["A", "B"]}])
        
        tp2 = TaskPlan(plan_dir)
        plan = tp2.get_active()
        assert plan["steps"][0]["acceptance"] == ["A", "B"]

    def test_verification_persists(self, plan_dir):
        """Verification survives reload."""
        tp1 = TaskPlan(plan_dir)
        tp1.create("Test", ["T"])
        tp1.update_step(1, "done", verification=["V1", "V2"])
        
        tp2 = TaskPlan(plan_dir)
        plan = tp2.get_active()
        assert plan["steps"][0]["verification"] == ["V1", "V2"]
