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

"""Tests for native Guard implementations (safety, progress, training_runtime, etc.)."""

from types import SimpleNamespace

from flagscale_agent.react.guard import GuardContext, GuardVerdict, GuardRegistry
from flagscale_agent.react.guard.safety import SafetyGuard
from flagscale_agent.react.guard.progress import ProgressGuard
from flagscale_agent.react.guard.loop_detect import LoopDetectGuard
from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
from flagscale_agent.react.guard.plan import PlanGuard
from flagscale_agent.react.guard.training_runtime import TrainingRuntimeGuard
from flagscale_agent.react.guard.utils import _is_flagscale_launch_command
from flagscale_agent.react.guard.training_attempt import TrainingAttemptGuard
from flagscale_agent.react.state_machine import AgentState
from flagscale_agent.react.judge import Judge, JudgeBudget


class MockProvider:
    """Returns controlled JSON responses in sequence."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages[-1]["content"][:100])
        resp = self.responses.pop(0) if self.responses else "{}"
        return {"content": resp}


def _ctx(tool_name="", tool_args=None, tool_result=None,
         classify_fn=None, state=AgentState.EXECUTING, **kwargs):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        current_state=state,
        classify_fn=classify_fn,
        **kwargs,
    )


# ── SafetyGuard ──────────────────────────────────────────────────────────


class TestSafetyGuard:
    def test_blocks_dangerous_command(self):
        provider = MockProvider(responses=['{"real": true, "need_more": null}'])
        judge = Judge(provider)
        g = SafetyGuard()
        ctx = _ctx("shell", {"command": "rm -rf /etc"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"

    def test_allows_safe_command(self):
        provider = MockProvider(responses=['{"real": false, "need_more": null}'])
        judge = Judge(provider)
        g = SafetyGuard()
        ctx = _ctx("shell", {"command": "ls -la"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is None

    def test_skips_non_shell_tools(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        g = SafetyGuard()
        ctx = _ctx("read_file", {"path": "/tmp/test.py"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is None
        assert len(provider.calls) == 0

    def test_blocks_when_no_classify(self):
        g = SafetyGuard()
        ctx = _ctx("shell", {"command": "rm -rf /"})
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"

    def test_error_increments_counter(self):
        provider = MockProvider(responses=[
            '{"real": true, "need_more": null}',   # is_error
            '{"real": false, "need_more": null}',  # is_success
        ])
        judge = Judge(provider)
        g = SafetyGuard()
        ctx = _ctx("shell", {"command": "python broken.py"},
                   "RuntimeError: something failed", classify_fn=judge.classify)
        g.check_post(ctx)
        assert g._consecutive_errors == 1

    def test_escalates_at_hard_threshold(self):
        g = SafetyGuard()
        g._consecutive_errors = 4
        provider = MockProvider(responses=[
            '{"real": true, "need_more": null}',   # is_error
            '{"real": false, "need_more": null}',  # is_success
        ])
        judge = Judge(provider)
        ctx = _ctx("shell", {"command": "fail"}, "RuntimeError",
                   classify_fn=judge.classify)
        result = g.check_post(ctx)
        assert result is not None
        assert result.action == "escalate"
        assert g._consecutive_errors == 5


# ── ProgressGuard ─────────────────────────────────────────────────────────


class TestProgressGuard:
    def test_tracks_reads(self):
        g = ProgressGuard()
        # Without SharedState, ProgressGuard uses ctx.recent_tool_names fallback
        for i in range(5):
            ctx = _ctx("read_file", {"path": f"/tmp/file_{i}.py"}, "content",
                       )
            ctx.recent_tool_names = ["read_file"] * (i + 1)
            g.check_post(ctx)
        # Track unique files read
        assert len(g._read_files) == 5

    def test_resets_on_productive_tool(self):
        g = ProgressGuard()
        g._read_files = {"/tmp/a.py", "/tmp/b.py"}
        g._reread_count = 3
        ctx = _ctx("write_file", {"path": "/tmp/test.py", "content": "x=1"},
                   "File written",
                   )
        g.check_post(ctx)
        assert len(g._read_files) == 0
        assert g._reread_count == 0

    def test_stale_threshold_triggers_inject(self):
        g = ProgressGuard()
        # Pre-populate: file already seen, so re-reads trigger
        g._read_files.add("/tmp/same.py")
        # Need to re-read enough times to hit threshold
        for i in range(4):
            ctx = _ctx("read_file", {"path": "/tmp/same.py"}, "content",
                       )
            ctx.recent_tool_names = ["read_file"] * (i + 6)  # simulate streak
            result = g.check_post(ctx)
        # After multiple re-reads, should trigger inject
        assert result is not None
        assert result.action == "inject_msg"


# ── LoopDetectGuard ───────────────────────────────────────────────────────


class TestLoopDetectGuard:
    def test_detects_repeated_calls(self):
        g = LoopDetectGuard()
        for _ in range(3):
            ctx = _ctx("read_file", {"path": "/tmp/same.py"})
            g.check_pre(ctx)
        # After 3 identical calls, should detect loop
        ctx = _ctx("read_file", {"path": "/tmp/same.py"})
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "inject_msg"

    def test_no_loop_with_different_calls(self):
        g = LoopDetectGuard()
        for i in range(5):
            ctx = _ctx("read_file", {"path": f"/tmp/file_{i}.py"})
            result = g.check_pre(ctx)
        assert result is None


# ── ContextPressureGuard ──────────────────────────────────────────────────


class TestContextPressureGuard:
    def test_no_action_below_threshold(self):
        g = ContextPressureGuard()
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.5)
        result = g.check_post(ctx)
        assert result is None

    def test_inject_at_soft_threshold(self):
        g = ContextPressureGuard()
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.78)
        result = g.check_post(ctx)
        assert result is not None
        assert result.action == "inject_msg"

    def test_block_at_hard_threshold(self):
        g = ContextPressureGuard()
        # v6: pressure > 85% AND evictable < 50 → sets hard_reset flag, injects warning
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.96,
                   evictable_indexes=[1, 2, 3, 4, 5])
        result = g.check_post(ctx)
        assert result is not None
        assert result.action == "inject_msg"  # post always injects
        assert "hard_reset" in (result.category or "")
        assert g._hard_reset_needed is True

        # check_pre then blocks non-save tools
        ctx_pre = _ctx("shell", {"command": "echo hi"}, context_pressure=0.96,
                       evictable_indexes=[1, 2, 3, 4, 5])
        result_pre = g.check_pre(ctx_pre)
        assert result_pre is not None
        assert result_pre.action == "block"


# ── PlanGuard ─────────────────────────────────────────────────────────────


class TestPlanGuard:
    def test_allows_plan_tools(self):
        g = PlanGuard()
        g._complex_task_no_plan = True
        ctx = _ctx("plan_create", {})
        result = g.check_pre(ctx)
        assert result is None

    def test_blocks_after_threshold_when_complex(self):
        g = PlanGuard()
        g._complex_task_no_plan = True
        for i in range(7):
            ctx = _ctx("read_file", {"path": f"/tmp/f{i}.py"},
                       )
            g.check_pre(ctx)
        ctx = _ctx("read_file", {"path": "/tmp/extra.py"},
                   )
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"

    def test_resets_on_plan_create(self):
        g = PlanGuard()
        g._complex_task_no_plan = True
        g._pre_plan_tool_calls = 5
        g._consecutive_reads = 9
        g._block_count = 1
        ctx = _ctx("plan_create", {})
        g.check_post(ctx)
        assert g._complex_task_no_plan is False
        assert g._pre_plan_tool_calls == 0
        assert g._consecutive_reads == 0
        assert g._block_count == 0

    def test_does_not_block_when_plan_exists(self):
        """Regression: once a plan exists, PlanGuard must not block reads."""
        from unittest.mock import MagicMock
        task_plan = MagicMock()
        task_plan.get_active.return_value = {"title": "test", "steps": []}

        g = PlanGuard(task_plan=task_plan)
        g._complex_task_no_plan = True
        # Simulate many consecutive reads — should NOT block because plan exists
        for i in range(20):
            ctx = _ctx("read_file", {"path": f"/tmp/f{i}.py"},
                       )
            result = g.check_pre(ctx)
            assert result is None, f"Should not block at call {i+1} when plan exists"

    def test_independent_mode_does_not_block_when_plan_exists(self):
        """Regression: independent mode (no mark_complex_task) also respects existing plan."""
        from unittest.mock import MagicMock
        task_plan = MagicMock()
        task_plan.get_active.return_value = {"title": "docs plan", "steps": [{"status": "doing"}]}

        g = PlanGuard(task_plan=task_plan)
        # Do NOT call mark_complex_task — this tests independent mode
        for i in range(15):
            ctx = _ctx("read_file", {"path": f"/tmp/f{i}.py"},
                       )
            result = g.check_pre(ctx)
            assert result is None, f"Independent mode should not block at call {i+1} when plan exists"

    def test_independent_warn_still_fires_without_plan(self):
        """Without active plan, independent-mode warn still triggers at threshold."""
        g = PlanGuard(task_plan=None)
        # Use the dynamic property that accounts for TaskMode multiplier
        g._consecutive_reads = g._plan_gate_independent_warn - 1
        ctx = _ctx("read_file", {"path": "/tmp/warn.py"},
                   )
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "inject_msg"


# ── TrainingRuntimeGuard ──────────────────────────────────────────────────


class TestTrainingRuntimeGuard:
    """v6: Simplified — monitor gate only, no heartbeat."""

    def test_detects_training_launch(self):
        g = TrainingRuntimeGuard()
        ctx = _ctx("shell", {"command": "flagscale train qwen3_0_6b"})
        g.check_post(ctx)
        assert g._awaiting_monitor is True

    def test_monitor_gate_injects_after_launch(self):
        g = TrainingRuntimeGuard()
        g._awaiting_monitor = True
        ctx = _ctx("shell", {"command": "pip install pkg"})
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "inject_msg"

    def test_monitor_clears_gate(self):
        g = TrainingRuntimeGuard()
        g._awaiting_monitor = True
        ctx = _ctx("flagscale_train_monitor", {"output_dir": "/tmp/train"})
        result = g.check_pre(ctx)
        assert result is None
        assert g._awaiting_monitor is False

    def test_read_only_diagnostic_allowed(self):
        g = TrainingRuntimeGuard()
        g._awaiting_monitor = True
        ctx = _ctx("shell", {"command": "nvidia-smi"})
        result = g.check_pre(ctx)
        assert result is None

    def test_recording_tools_allowed(self):
        g = TrainingRuntimeGuard()
        g._awaiting_monitor = True
        ctx = _ctx("plan_update", {"action": "step_done", "step_id": 1})
        result = g.check_pre(ctx)
        assert result is None

    def test_reset_new_turn_noop(self):
        """v6: reset_new_turn does nothing (no heartbeat counters)."""
        g = TrainingRuntimeGuard()
        g._awaiting_monitor = True
        g.reset_new_turn()
        assert g._awaiting_monitor is True  # State persists

    def test_reset_state_clears(self):
        g = TrainingRuntimeGuard()
        g._awaiting_monitor = True
        g.reset_state()
        assert g._awaiting_monitor is False


# ── GuardRegistry ─────────────────────────────────────────────────────────


class TestGuardRegistry:
    def test_register_and_priority_order(self):
        reg = GuardRegistry()
        g1 = SafetyGuard()  # priority 10
        g2 = ProgressGuard()  # priority 30
        reg.register(g2)
        reg.register(g1)
        assert reg.guards[0].priority <= reg.guards[1].priority

    def test_check_pre_first_verdict_wins(self):
        reg = GuardRegistry()
        g = SafetyGuard()
        reg.register(g)
        # No classify_fn → blocks
        ctx = _ctx("shell", {"command": "rm -rf /"})
        verdict = reg.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"

    def test_reset_turn(self):
        reg = GuardRegistry()
        g = LoopDetectGuard()
        reg.register(g)
        g._tool_call_cache[("read_file", "path=/tmp/x")] = "content"
        reg.reset_turn()
        assert len(g._tool_call_cache) == 0


# ── GuardContext ──────────────────────────────────────────────────────────


class TestGuardContextPhaseName:
    def test_phase_name_from_executing(self):
        ctx = GuardContext(current_state=AgentState.EXECUTING)
        assert ctx.phase_name == "executing"

    def test_phase_name_from_idle(self):
        ctx = GuardContext(current_state=AgentState.IDLE)
        assert ctx.phase_name == "idle"

    def test_phase_name_from_planning(self):
        ctx = GuardContext(current_state=AgentState.PLANNING)
        assert ctx.phase_name == "planning"


# ── _is_flagscale_launch_command ──────────────────────────────────────────


class TestIsFlagscaleLaunchCommand:
    """Test precise FlagScale launch detection."""

    def test_flagscale_train_basic(self):
        assert _is_flagscale_launch_command("flagscale train qwen3_0_6b") is True

    def test_flagscale_train_with_config(self):
        assert _is_flagscale_launch_command("flagscale train qwen3 --config /path/to/config.yaml") is True

    def test_flagscale_run_with_config(self):
        assert _is_flagscale_launch_command(
            "flagscale run --config-path /workspace --config-name train_config"
        ) is True

    def test_python_run_py(self):
        assert _is_flagscale_launch_command(
            "python run.py --config-path=/workspace --config-name=train action=run"
        ) is True

    def test_python3_run_py(self):
        # v6: Pattern 3 requires action=run
        assert _is_flagscale_launch_command(
            "python3 run.py --config-path=/workspace --config-name=train action=run"
        ) is True
        # Without action=run → not a launch
        assert _is_flagscale_launch_command(
            "python3 run.py --config-path=/workspace --config-name=train"
        ) is False

    def test_flagscale_train_stop_not_launch(self):
        assert _is_flagscale_launch_command("flagscale train qwen3 --stop") is False

    def test_flagscale_train_dryrun_not_launch(self):
        assert _is_flagscale_launch_command("flagscale train qwen3 --dryrun") is False

    def test_flagscale_run_action_stop_not_launch(self):
        assert _is_flagscale_launch_command(
            "flagscale run --config-path /p --config-name c --action stop"
        ) is False

    def test_grep_flagscale_not_launch(self):
        """grep with flagscale keyword should NOT be detected as launch."""
        assert _is_flagscale_launch_command('grep "flagscale train" logs/') is False

    def test_echo_flagscale_not_launch(self):
        assert _is_flagscale_launch_command('echo "flagscale train qwen3"') is False

    def test_git_push_flagscale_not_launch(self):
        assert _is_flagscale_launch_command("git push origin dev_flagscale") is False

    def test_cat_run_py_not_launch(self):
        assert _is_flagscale_launch_command("cat run.py") is False

    def test_cd_flagscale_not_launch(self):
        assert _is_flagscale_launch_command("cd /workspace/FlagScale && ls") is False

    def test_compound_with_launch(self):
        """Compound command where one segment is a real launch."""
        assert _is_flagscale_launch_command(
            "cd /workspace/FlagScale && flagscale train qwen3_0_6b"
        ) is True

    def test_plain_torchrun_not_launch(self):
        """torchrun alone is NOT a FlagScale launch."""
        assert _is_flagscale_launch_command("torchrun --nproc_per_node=8 train.py") is False


# ── TrainingAttemptGuard ──────────────────────────────────────────────────


class TestTrainingAttemptGuard:
    """Tests for 2-Strike periodic block logic."""

    def _make_guard(self):
        return TrainingAttemptGuard()

    def _launch_ctx(self):
        return _ctx("shell", {"command": "flagscale train qwen3_0_6b"})

    def _experiment_ctx(self, action, **kwargs):
        args = {"action": action, **kwargs}
        return _ctx("workspace_experiment", args)

    def test_no_block_without_experiment(self):
        g = self._make_guard()
        result = g.check_pre(self._launch_ctx())
        assert result is None

    def test_no_block_on_first_attempt(self):
        g = self._make_guard()
        # Create experiment
        g.check_post(self._experiment_ctx("create", name="exp1"))
        # Add first attempt
        g.check_post(self._experiment_ctx("add_attempt"))
        # Launch should pass (count=1, 1%2!=0)
        result = g.check_pre(self._launch_ctx())
        assert result is None

    def test_blocks_on_second_attempt(self):
        g = self._make_guard()
        g.check_post(self._experiment_ctx("create", name="exp1"))
        g.check_post(self._experiment_ctx("add_attempt"))  # count=1
        g.check_post(self._experiment_ctx("add_attempt"))  # count=2
        # Launch should be blocked (2%2==0)
        result = g.check_pre(self._launch_ctx())
        assert result is not None
        assert result.action == "block"
        assert "2strike" in result.reason

    def test_no_block_on_third_attempt(self):
        g = self._make_guard()
        g.check_post(self._experiment_ctx("create", name="exp1"))
        g.check_post(self._experiment_ctx("add_attempt"))  # count=1
        g.check_post(self._experiment_ctx("add_attempt"))  # count=2
        g.check_post(self._experiment_ctx("add_attempt"))  # count=3
        # Launch should pass (3%2!=0)
        result = g.check_pre(self._launch_ctx())
        assert result is None

    def test_blocks_on_fourth_attempt(self):
        g = self._make_guard()
        g.check_post(self._experiment_ctx("create", name="exp1"))
        for _ in range(4):
            g.check_post(self._experiment_ctx("add_attempt"))
        # count=4, 4%2==0 → block
        result = g.check_pre(self._launch_ctx())
        assert result is not None
        assert result.action == "block"

    def test_finalize_resets(self):
        g = self._make_guard()
        g.check_post(self._experiment_ctx("create", name="exp1"))
        g.check_post(self._experiment_ctx("add_attempt"))
        g.check_post(self._experiment_ctx("add_attempt"))
        # Finalize resets
        g.check_post(self._experiment_ctx("finalize"))
        # Launch should pass
        result = g.check_pre(self._launch_ctx())
        assert result is None
        assert g._attempt_count == 0

    def test_override_accepted_with_long_reason(self):
        g = self._make_guard()
        ctx = self._launch_ctx()
        reason = "OOM 根因是 batch_size=64 过大，GPU 显存不够。修复：改为 batch_size=32"
        assert g.accept_override(reason, ctx) is True

    def test_override_rejected_short(self):
        g = self._make_guard()
        ctx = self._launch_ctx()
        assert g.accept_override("try again", ctx) is False

    def test_override_rejected_lazy(self):
        g = self._make_guard()
        ctx = self._launch_ctx()
        assert g.accept_override("just do it", ctx) is False

    def test_inject_warning_at_strike_point(self):
        g = self._make_guard()
        g.check_post(self._experiment_ctx("create", name="exp1"))
        g.check_post(self._experiment_ctx("add_attempt"))  # count=1, no warning
        result = g.check_post(self._experiment_ctx("add_attempt"))  # count=2, warning
        assert result is not None
        assert result.action == "inject_msg"
        assert "2-Strike" in result.message

    def test_only_blocks_launch_commands(self):
        """Non-launch shell commands should not be blocked."""
        g = self._make_guard()
        g.check_post(self._experiment_ctx("create", name="exp1"))
        g.check_post(self._experiment_ctx("add_attempt"))
        g.check_post(self._experiment_ctx("add_attempt"))
        # ls command should pass even at strike point
        ctx = _ctx("shell", {"command": "ls -la"})
        result = g.check_pre(ctx)
        assert result is None

    def test_non_shell_tools_not_blocked(self):
        """read_file, write_file etc should never be blocked."""
        g = self._make_guard()
        g.check_post(self._experiment_ctx("create", name="exp1"))
        g.check_post(self._experiment_ctx("add_attempt"))
        g.check_post(self._experiment_ctx("add_attempt"))
        ctx = _ctx("read_file", {"path": "/tmp/foo.py"})
        result = g.check_pre(ctx)
        assert result is None
