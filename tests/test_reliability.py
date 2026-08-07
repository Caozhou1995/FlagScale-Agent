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

"""Tests for P7 reliability guards: ErrorClassifier, StepCheckpoint."""

import time
import tempfile
import os

import pytest

from flagscale_agent.react.guard import GuardContext, GuardVerdict
from flagscale_agent.react.guard.error_classifier import ErrorClassifierGuard
from flagscale_agent.react.plan import TaskPlan, StepCheckpoint


def _make_ctx(tool_result: str = "", tool_name: str = "shell",
              classify_fn=None) -> GuardContext:
    return GuardContext(
        tool_name=tool_name,
        tool_args={},
        tool_result=tool_result,
        classify_fn=classify_fn,
    )


# ── ErrorClassifierGuard Tests ──────────────────────────────────────────────


def _mock_classify_error(category="env_missing"):
    """Create a classify_fn that returns 'yes' for is_error and a category."""
    def classify(cat, context, **kwargs):
        if cat == "is_error":
            return "yes", "fast"
        return category, "fast"
    return classify


def _mock_classify_no_error():
    """Classify_fn that returns 'no' for is_error."""
    def classify(cat, context, **kwargs):
        return "no", "fast"
    return classify


class TestErrorClassifier:
    def test_no_error_returns_none(self):
        """Non-error output should not trigger (keyword gate)."""
        guard = ErrorClassifierGuard()
        ctx = _make_ctx("Success: file written to /tmp/out.txt")
        assert guard.check_post(ctx) is None

    def test_keyword_gate_triggers_on_error_text(self):
        """Error keywords should pass the gate and call classify_fn."""
        guard = ErrorClassifierGuard()
        classify_fn = _mock_classify_error("env_missing")
        ctx = _make_ctx(
            "Error: ModuleNotFoundError: No module named 'torch'",
            classify_fn=classify_fn,
        )
        result = guard.check_post(ctx)
        # First occurrence: inject with category based on tool_name
        assert result is not None
        assert result.action == "inject_msg"
        assert "shell_error" in result.reason

    def test_no_classify_fn_returns_none(self):
        """Without classify_fn, guard can't classify — returns None."""
        guard = ErrorClassifierGuard()
        ctx = _make_ctx("Error: something failed", classify_fn=None)
        assert guard.check_post(ctx) is None

    def test_llm_says_not_error_returns_none(self):
        """If LLM says it's not an error, guard passes."""
        guard = ErrorClassifierGuard()
        classify_fn = _mock_classify_no_error()
        ctx = _make_ctx("Error in log: this is expected", classify_fn=classify_fn)
        assert guard.check_post(ctx) is None

    def test_escalation_on_consecutive_same_errors(self):
        """2+ consecutive same-category errors trigger escalation messages."""
        guard = ErrorClassifierGuard()
        classify_fn = _mock_classify_error("permission")
        ctx = _make_ctx("Error: Permission denied", classify_fn=classify_fn)

        r1 = guard.check_post(ctx)
        assert r1 is not None  # first hit
        assert guard._consecutive_same == 1

        r2 = guard.check_post(ctx)
        assert r2 is not None
        assert "repeated" in r2.message.lower() or "consider" in r2.message.lower()

    def test_strong_escalation_at_threshold(self):
        """3+ consecutive same-category errors trigger strong warning."""
        guard = ErrorClassifierGuard()
        classify_fn = _mock_classify_error("network")
        ctx = _make_ctx("Error: Connection refused", classify_fn=classify_fn)

        guard.check_post(ctx)
        guard.check_post(ctx)
        r3 = guard.check_post(ctx)
        assert r3 is not None
        assert "stop" in r3.message.lower() or "root cause" in r3.message.lower()

    def test_success_resets_streak(self):
        """Non-error output resets the consecutive counter."""
        guard = ErrorClassifierGuard()
        classify_fn = _mock_classify_error("resource")
        err_ctx = _make_ctx("Error: CUDA out of memory", classify_fn=classify_fn)
        guard.check_post(err_ctx)
        assert guard._consecutive_same == 1

        ok_ctx = _make_ctx("File written successfully.")
        guard.check_post(ok_ctx)
        assert guard._consecutive_same == 0

        # After reset, hitting same error starts from 1 again
        guard.check_post(err_ctx)
        assert guard._consecutive_same == 1


# ── StepCheckpoint Tests ────────────────────────────────────────────────────


class TestStepCheckpoint:
    def _make_plan(self):
        tmpdir = tempfile.mkdtemp()
        tp = TaskPlan(tmpdir)
        plan = tp.create("Test Plan", ["Step 1", "Step 2", "Step 3"])
        # Start step 1
        tp.update_step(1, "doing")
        return tp, plan

    def test_checkpoint_creation(self):
        tp, plan = self._make_plan()
        cp = tp.checkpoint(
            step_id=1,
            files=["src/main.py", "config.yaml"],
            memory_keys=["env_info"],
            summary="Completed environment setup",
        )
        assert cp is not None
        assert cp.step_id == 1
        assert cp.files_modified == ["src/main.py", "config.yaml"]
        assert cp.memory_keys == ["env_info"]
        assert cp.summary == "Completed environment setup"
        assert cp.timestamp > 0

    def test_get_checkpoint(self):
        tp, plan = self._make_plan()
        tp.checkpoint(step_id=1, files=["a.py"], summary="did step 1")
        retrieved = tp.get_checkpoint(1)
        assert retrieved is not None
        assert retrieved.summary == "did step 1"

    def test_get_checkpoint_missing(self):
        tp, plan = self._make_plan()
        assert tp.get_checkpoint(99) is None

    def test_rollback_info(self):
        tp, plan = self._make_plan()
        tp.checkpoint(step_id=1, files=["a.py"], memory_keys=["k1"], summary="step 1 done")
        tp.update_step(1, "done")
        tp.checkpoint(step_id=2, files=["b.py"], memory_keys=["k2"], summary="step 2 done")

        info = tp.get_rollback_info(1)
        assert "step 1 done" in info
        assert "a.py" in info
        assert "b.py" in info
        assert "k1" in info

    def test_rollback_info_no_plan(self):
        tmpdir = tempfile.mkdtemp()
        tp = TaskPlan(tmpdir)
        assert "No active plan" in tp.get_rollback_info(1)

    def test_list_checkpoints(self):
        tp, plan = self._make_plan()
        tp.checkpoint(step_id=1, summary="s1")
        tp.update_step(1, "done")
        tp.checkpoint(step_id=2, summary="s2")

        cps = tp.list_checkpoints()
        assert len(cps) == 2
        assert cps[0]["step_id"] == 1
        assert cps[1]["step_id"] == 2

    def test_checkpoint_to_dict(self):
        cp = StepCheckpoint(
            step_id=1,
            timestamp=1000.0,
            files_modified=["x.py"],
            memory_keys=["mem1"],
            summary="test",
        )
        d = cp.to_dict()
        assert d["step_id"] == 1
        assert d["timestamp"] == 1000.0
        assert d["files_modified"] == ["x.py"]

    def test_checkpoint_from_dict(self):
        data = {
            "step_id": 2,
            "timestamp": 2000.0,
            "files_modified": ["y.py"],
            "memory_keys": ["m2"],
            "summary": "restored",
        }
        cp = StepCheckpoint.from_dict(data)
        assert cp.step_id == 2
        assert cp.summary == "restored"
