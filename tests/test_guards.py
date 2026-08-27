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
from flagscale_agent.react.guard.safety import ShellSafetyGuard

from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
from flagscale_agent.react.guard.training_monitor import TrainingMonitorGuard
from flagscale_agent.react.guard.plan import PlanGuard
from flagscale_agent.react.guard.utils import _is_flagscale_launch_command
from flagscale_agent.react.kernel import AgentKernel
from flagscale_agent.react.judge import Judge


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
         classify_fn=None, **kwargs):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        classify_fn=classify_fn,
        **kwargs,
    )


# ── ShellSafetyGuard ──────────────────────────────────────────────────────────


class TestShellSafetyGuard:
    def test_blocks_dangerous_command(self):
        # First response: is_fatal=false, second: is_dangerous=true
        provider = MockProvider(responses=[
            '{"real": false, "need_more": null}',
            '{"real": true, "need_more": null}',
        ])
        judge = Judge(provider)
        g = ShellSafetyGuard()
        ctx = _ctx("shell", {"command": "rm -rf /etc"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"

    def test_escalates_fatal_command(self):
        # is_fatal=true → escalate (cannot override)
        provider = MockProvider(responses=[
            '{"real": true, "need_more": null}',
        ])
        judge = Judge(provider)
        g = ShellSafetyGuard()
        ctx = _ctx("shell", {"command": "rm -rf /"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "escalate"
        assert "FATAL" in result.message

    def test_allows_safe_command(self):
        # is_fatal=false, is_dangerous=false
        provider = MockProvider(responses=[
            '{"real": false, "need_more": null}',
            '{"real": false, "need_more": null}',
        ])
        judge = Judge(provider)
        g = ShellSafetyGuard()
        ctx = _ctx("shell", {"command": "ls -la"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is None

    def test_skips_non_shell_tools(self):
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        g = ShellSafetyGuard()
        ctx = _ctx("read_file", {"path": "/tmp/test.py"}, classify_fn=judge.classify)
        result = g.check_pre(ctx)
        assert result is None
        assert len(provider.calls) == 0

    def test_check_post_returns_none(self):
        """After refactor, safety check_post does nothing."""
        g = ShellSafetyGuard()
        provider = MockProvider(responses=[])
        judge = Judge(provider)
        ctx = _ctx("shell", {"command": "python broken.py"},
                   "RuntimeError: something failed", classify_fn=judge.classify)
        result = g.check_post(ctx)
        assert result is None



# ── ContextPressureGuard ──────────────────────────────────────────────────


class TestContextPressureGuard:
    def test_no_action_below_threshold(self):
        g = ContextPressureGuard()
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.5)
        result = g.check_pre(ctx)
        assert result is None

    def test_no_action_at_78_percent(self):
        g = ContextPressureGuard()
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.78)
        result = g.check_pre(ctx)
        assert result is None  # below 80% threshold

    def test_block_at_80_percent_with_evictable(self):
        g = ContextPressureGuard()
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.82,
                   evictable_indexes=list(range(80)))
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"
        assert result.category == "context_pressure_evict"

    def test_block_at_hard_reset_threshold(self):
        g = ContextPressureGuard()
        # pressure >= 85% AND evictable < 50 → block non-save tools
        ctx = _ctx("shell", {"command": "ls"}, context_pressure=0.88,
                   evictable_indexes=[1, 2, 3, 4, 5])
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "block"
        assert "hard_reset" in (result.category or "")

    def test_hard_reset_allows_save_tools(self):
        g = ContextPressureGuard()
        ctx = _ctx("memory_write", {"key": "x", "type": "fact", "content": "y"},
                   context_pressure=0.88, evictable_indexes=[1, 2, 3])
        result = g.check_pre(ctx)
        assert result is None


# ── PlanGuard ─────────────────────────────────────────────────────────────


class TestPlanGuard:
    def test_allows_plan_tools(self):
        g = PlanGuard()
        # First plan_create is the plan-framing moment: PlanGuard injects the
        # qualifier-extraction reminder once (never blocks). It is advisory —
        # the tool still proceeds.
        ctx = _ctx("plan_create", {})
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "inject"
        assert result.reason == "qualifier_extraction"
        # Subsequent plan tools pass through cleanly.
        assert g.check_pre(_ctx("plan_create", {})) is None
        assert g.check_pre(_ctx("plan_update", {})) is None
        assert g.check_pre(_ctx("plan_status", {})) is None

    def test_planframing_injects_problem_class_selfcheck(self):
        """Plan-framing injection must pose the P1 self-check questions — name the
        known problem CLASS and its STANDARD METHOD — not just remind to plan.
        These have external referents (a class name, an established technique):
        being unable to answer is the signal the agent has not understood yet."""
        g = PlanGuard()
        result = g.check_pre(_ctx("plan_create", {}))
        assert result is not None and result.action == "inject"
        m = result.message.lower()
        assert "class" in m
        assert "standard method" in m or "standard technique" in m
        # must flag the brute-force tell as the sign the class is still hidden
        assert "brute" in m or "enumeration" in m

    def test_planframing_injects_tool_instance_doc_reminder(self):
        """Plan-framing injection must remind that a qualifier pinning a concrete
        tool instance (version/revision/named model) requires consulting THAT
        instance's own documentation before writing the call — the mteb-retrieve
        failure mode (generic API applied to a specific instance with its own
        usage rules → silently wrong output). This lives in the qualifier-extraction
        gate on purpose: it is a plan-framing concern, sharing one override channel,
        not a content-matching block guard."""
        g = PlanGuard()
        result = g.check_pre(_ctx("plan_create", {}))
        assert result is not None and result.action == "inject"
        m = result.message.lower()
        # names the instance-vs-category distinction and the consult-docs action
        assert "instance" in m
        assert "documentation" in m or "readme" in m or "model card" in m
        # flags the silent-wrong-output trap of the generic API shortcut
        assert "generic" in m
        # near/far: consulting the doc is not enough — must APPLY what it says.
        # Guards the deeper mteb-retrieve failure (agent read the doc, saw the
        # requirement, then skipped it as "not explicitly asked"). Assert the
        # reminder covers the apply-half, not just the consult-half.
        assert "near half" in m and "far half" in m
        assert "apply" in m
        # the reminder must be generic in FORM — not tied to one failure shape
        # (e.g. only "add a prefix"). It should frame the requirement as taking
        # ANY form, so the phrasing must not hardcode a single mechanism.
        assert "any form" in m

    def test_reminds_at_threshold(self):
        g = PlanGuard(task_plan=None)
        for i in range(14):
            ctx = _ctx("read_file", {"path": f"/tmp/f{i}.py"})
            result = g.check_pre(ctx)
            assert result is None
        # 15th call triggers
        ctx = _ctx("shell", {"command": "ls"})
        result = g.check_pre(ctx)
        assert result is not None
        assert result.action == "inject"

    def test_reminds_periodically(self):
        g = PlanGuard(task_plan=None)
        for i in range(30):
            ctx = _ctx("shell", {"command": f"cmd{i}"})
            g.check_pre(ctx)
        # 30th call should also trigger (second reminder)
        assert g._calls_without_plan == 30
        ctx = _ctx("shell", {"command": "extra"})
        result = g.check_pre(ctx)
        # 31st call, not multiple of 15 -> no inject
        assert result is None

    def test_resets_on_plan_create(self):
        g = PlanGuard(task_plan=None)
        g._calls_without_plan = 20
        ctx = _ctx("plan_create", {})
        g.check_post(ctx)
        assert g._calls_without_plan == 0

    def test_does_not_remind_when_plan_exists(self):
        from unittest.mock import MagicMock
        task_plan = MagicMock()
        task_plan.get_active.return_value = {"title": "test", "steps": []}
        g = PlanGuard(task_plan=task_plan)
        for i in range(30):
            ctx = _ctx("read_file", {"path": f"/tmp/f{i}.py"})
            result = g.check_pre(ctx)
            assert result is None

    def test_reset_turn(self):
        g = PlanGuard(task_plan=None)
        g._calls_without_plan = 20
        g.reset_turn()
        assert g._calls_without_plan == 0


class TestPlanGuardSingleShot:
    """Single-shot (unsupervised) mode enforces a plan via block."""

    def test_observation_budget_only_injects(self):
        # Within the observation budget, no block — at most periodic inject.
        g = PlanGuard(task_plan=None, single_shot=True)
        for i in range(PlanGuard.SINGLE_SHOT_BLOCK_THRESHOLD):
            ctx = _ctx("shell", {"command": f"explore{i}"})
            result = g.check_pre(ctx)
            assert result is None or result.action == "inject"

    def test_blocks_after_observation_budget(self):
        g = PlanGuard(task_plan=None, single_shot=True)
        # Consume the observation budget.
        for i in range(PlanGuard.SINGLE_SHOT_BLOCK_THRESHOLD):
            g.check_pre(_ctx("shell", {"command": f"explore{i}"}))
        # The next non-plan call must be blocked.
        result = g.check_pre(_ctx("shell", {"command": "keep_going"}))
        assert result is not None
        assert result.action == "block"
        assert result.category == "plan_required"

    def test_observation_budget_is_generous_for_exploration(self):
        # Regression: the single-shot budget must stay generous (>=20) so an
        # unsupervised run can probe the environment before being forced to plan.
        # Blocking too early forces a plan written before understanding.
        assert PlanGuard.SINGLE_SHOT_BLOCK_THRESHOLD >= 20
        g = PlanGuard(task_plan=None, single_shot=True)
        # Exactly at the budget: still no block (at most a periodic inject).
        for i in range(PlanGuard.SINGLE_SHOT_BLOCK_THRESHOLD):
            r = g.check_pre(_ctx("shell", {"command": f"probe{i}"}))
            assert r is None or r.action == "inject"
        # One past the budget: now it blocks.
        r = g.check_pre(_ctx("shell", {"command": "one_more"}))
        assert r is not None and r.action == "block"

    def test_plan_tools_never_blocked(self):
        g = PlanGuard(task_plan=None, single_shot=True)
        for i in range(PlanGuard.SINGLE_SHOT_BLOCK_THRESHOLD + 5):
            g.check_pre(_ctx("shell", {"command": f"c{i}"}))
        # plan_create itself must always pass through
        assert g.check_pre(_ctx("plan_create", {})) is None

    def test_no_block_when_plan_exists(self):
        from unittest.mock import MagicMock
        task_plan = MagicMock()
        task_plan.get_active.return_value = {"title": "t", "steps": []}
        g = PlanGuard(task_plan=task_plan, single_shot=True)
        for i in range(PlanGuard.SINGLE_SHOT_BLOCK_THRESHOLD + 5):
            assert g.check_pre(_ctx("shell", {"command": f"c{i}"})) is None

    def test_interactive_mode_never_blocks(self):
        # Default (interactive) mode: only inject, never block.
        g = PlanGuard(task_plan=None)
        actions = set()
        for i in range(40):
            r = g.check_pre(_ctx("shell", {"command": f"c{i}"}))
            if r is not None:
                actions.add(r.action)
        assert "block" not in actions

    def test_set_single_shot_runtime_toggle(self):
        g = PlanGuard(task_plan=None)
        g.set_single_shot(True)
        for i in range(PlanGuard.SINGLE_SHOT_BLOCK_THRESHOLD):
            g.check_pre(_ctx("shell", {"command": f"c{i}"}))
        result = g.check_pre(_ctx("shell", {"command": "next"}))
        assert result is not None and result.action == "block"


class TestPlanGuardCompletionGate:
    """Single-shot: [TASK_COMPLETE] is blocked when no plan was ever created."""

    def test_completion_blocked_without_plan_single_shot(self):
        g = PlanGuard(task_plan=None, single_shot=True)
        # A few exploration calls, then straight to completion — no plan_create.
        for i in range(5):
            g.check_pre(_ctx("shell", {"command": f"c{i}"}))
        result = g.check_pre(_ctx("", assistant_text="Done. [TASK_COMPLETE]"))
        assert result is not None
        assert result.action == "block"
        assert result.category == "plan_required"
        assert result.reason == "single_shot_completion_without_plan"

    def test_completion_allowed_after_plan_created(self):
        g = PlanGuard(task_plan=None, single_shot=True)
        # plan_create was called at some point this run.
        g.check_post(_ctx("plan_create", {}))
        result = g.check_pre(_ctx("", assistant_text="All steps done. [TASK_COMPLETE]"))
        assert result is None

    def test_completion_not_blocked_interactive(self):
        # Interactive mode: completion gate never fires, plan or not.
        g = PlanGuard(task_plan=None, single_shot=False)
        result = g.check_pre(_ctx("", assistant_text="[TASK_COMPLETE]"))
        assert result is None

    def test_completion_gate_blocks_every_bare_reemit(self):
        # A bare re-emit of [TASK_COMPLETE] with no plan and no override reason
        # must stay blocked EVERY time — never a blanket fire-once release that
        # lets an unplanned, unverified completion slip through on the 2nd try.
        g = PlanGuard(task_plan=None, single_shot=True)
        first = g.check_pre(_ctx("", assistant_text="[TASK_COMPLETE]"))
        assert first is not None and first.action == "block"
        second = g.check_pre(_ctx("", assistant_text="[TASK_COMPLETE]"))
        assert second is not None and second.action == "block"
        third = g.check_pre(_ctx("", assistant_text="still done [TASK_COMPLETE]"))
        assert third is not None and third.action == "block"

    def test_completion_gate_ignores_need_user_input(self):
        # [NEED_USER_INPUT] is not a completion signal; the gate must not fire.
        g = PlanGuard(task_plan=None, single_shot=True)
        result = g.check_pre(_ctx("", assistant_text="Need more info. [NEED_USER_INPUT]"))
        assert result is None

    def test_completion_gate_no_signal_no_block(self):
        # tool_name=="" but no completion signal in text → no block.
        g = PlanGuard(task_plan=None, single_shot=True)
        result = g.check_pre(_ctx("", assistant_text="still working"))
        assert result is None

    def test_completion_gate_non_overridable(self):
        # The completion gate is now NON-OVERRIDABLE: an override_reason does
        # NOT release it. Only plan_create sets _plan_ever_created and unlocks.
        reg = GuardRegistry()
        g = PlanGuard(task_plan=None, single_shot=True)
        reg.register(g)
        # No override → block returned.
        blocked = reg.check_pre(_ctx("", assistant_text="[TASK_COMPLETE]"))
        assert blocked is not None and blocked.action == "block"
        assert blocked.overridable is False
        # Same guard + an override reason → STILL blocked (non-overridable).
        still_blocked = reg.check_pre(_ctx(
            "", assistant_text="[TASK_COMPLETE]",
            override_reason="genuinely trivial single lookup, nothing to verify",
        ))
        assert still_blocked is not None and still_blocked.action == "block"
        # The ONLY exit: call plan_create, which sets _plan_ever_created.
        from flagscale_agent.react.guard import GuardContext
        ctx = GuardContext(tool_name="plan_create", tool_args={}, tool_result=None)
        g.check_post(ctx)  # sets _plan_ever_created=True
        # Now [TASK_COMPLETE] passes.
        allowed = reg.check_pre(_ctx("", assistant_text="[TASK_COMPLETE]"))
        assert allowed is None


class TestKernelTextOverrideExtraction:
    """kernel._extract_text_override parses the completion-path override channel.

    A [TASK_COMPLETE] signal carries no tool_args, so the completion-gate block
    is overridden by an inline `_override_reason: <reason>` in the assistant text.
    """

    f = staticmethod(AgentKernel._extract_text_override)

    def test_bare_completion_no_override(self):
        assert self.f("The task is done.\n[TASK_COMPLETE]") == ""

    def test_inline_colon_form(self):
        text = "Trivial lookup.\n_override_reason: single fact lookup, no steps\n[TASK_COMPLETE]"
        assert self.f(text) == "single fact lookup, no steps"

    def test_equals_quoted_form(self):
        assert self.f('_override_reason = "checked file, only a typo"') == "checked file, only a typo"

    def test_empty_text(self):
        assert self.f("") == ""

    def test_bare_keyword_no_reason(self):
        # Keyword with no substance → empty; accept_override then keeps it blocked.
        assert self.f("_override_reason:") == ""


# ── GuardRegistry ─────────────────────────────────────────────────────────


class TestGuardRegistry:
    def test_register_and_priority_order(self):
        reg = GuardRegistry()
        g1 = ShellSafetyGuard()  # priority 10
        g2 = ContextPressureGuard()  # priority 60
        reg.register(g2)
        reg.register(g1)
        assert reg.guards[0].priority <= reg.guards[1].priority

    def test_check_pre_first_verdict_wins(self):
        reg = GuardRegistry()
        g = ShellSafetyGuard()
        reg.register(g)
        # is_fatal=False, is_dangerous=True → blocks
        provider = MockProvider(responses=['{"decision": false}', '{"decision": true}'])
        judge = Judge(provider)
        ctx = _ctx("shell", {"command": "rm -rf /"},
                   classify_fn=judge.classify)
        verdict = reg.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"

    def test_reset_turn(self):
        reg = GuardRegistry()
        g = ContextPressureGuard()
        reg.register(g)
        reg.reset_turn()
        # Should not raise — guards can be reset without error


# ── GuardContext ──────────────────────────────────────────────────────────



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





# ── TrainingMonitorGuard ──────────────────────────────────────────────────

class TestTrainingMonitorGuard:

    def test_no_launch_no_block(self):
        g = TrainingMonitorGuard()
        ctx = _ctx("shell", {"command": "ls"})
        assert g.check_pre(ctx) is None

    def test_launch_detected_blocks_non_monitor(self):
        g = TrainingMonitorGuard()
        launch_ctx = _ctx("shell", {"command": "python3 run.py --config-path=conf --config-name=config action=run"})
        g.check_post(launch_ctx)
        next_ctx = _ctx("shell", {"command": "ls"})
        verdict = g.check_pre(next_ctx)
        assert verdict is not None
        assert verdict.action == "block"

    def test_launch_then_monitor_clears(self):
        g = TrainingMonitorGuard()
        launch_ctx = _ctx("shell", {"command": "python3 run.py --config-path=conf --config-name=config action=run"})
        g.check_post(launch_ctx)
        monitor_ctx = _ctx("flagscale_train_monitor", {"output_dir": "/tmp"})
        verdict = g.check_pre(monitor_ctx)
        assert verdict is None
        next_ctx = _ctx("shell", {"command": "ls"})
        assert g.check_pre(next_ctx) is None
