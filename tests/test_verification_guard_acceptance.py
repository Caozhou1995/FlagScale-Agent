"""Unit tests for VerificationGuard acceptance+verification integration."""

import tempfile
import shutil
from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.plan import TaskPlan


def test_step_with_acceptance_requires_verification():
    """Step with acceptance criteria requires verification list."""
    d = tempfile.mkdtemp()
    try:
        plan = TaskPlan(d)
        plan.create("Test", [{"title": "Step A", "acceptance": ["A1", "A2"]}])
        
        guard = VerificationGuard(plan=plan)
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 1}
        )
        
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "acceptance criteria" in verdict.message.lower()
        assert "A1" in verdict.message
        assert "A2" in verdict.message
    finally:
        shutil.rmtree(d)


def test_step_with_acceptance_passes_with_verification():
    """Step with acceptance passes when verification provided."""
    d = tempfile.mkdtemp()
    try:
        plan = TaskPlan(d)
        plan.create("Test", [{"title": "Step A", "acceptance": ["A1", "A2"]}])
        
        guard = VerificationGuard(plan=plan)
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "step_done",
                "step_id": 1,
                "verification": ["Proof A1", "Proof A2"]
            }
        )
        
        verdict = guard.check_pre(ctx)
        assert verdict is None  # Passes
    finally:
        shutil.rmtree(d)


def test_step_without_acceptance_requires_override():
    """Step without acceptance still requires override_reason."""
    d = tempfile.mkdtemp()
    try:
        plan = TaskPlan(d)
        plan.create("Test", ["Simple step"])
        
        guard = VerificationGuard(plan=plan)
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 1}
        )
        
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "_override_reason" in verdict.message
    finally:
        shutil.rmtree(d)


def test_step_without_acceptance_passes_with_override():
    """Step without acceptance passes with override_reason."""
    d = tempfile.mkdtemp()
    try:
        plan = TaskPlan(d)
        plan.create("Test", ["Simple step"])
        
        guard = VerificationGuard(plan=plan)
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={
                "action": "step_done",
                "step_id": 1,
                "_override_reason": "checked, all good"
            },
            override_reason="checked, all good"
        )
        
        verdict = guard.check_pre(ctx)
        assert verdict is None  # Passes
    finally:
        shutil.rmtree(d)


def test_fallback_when_plan_unavailable():
    """When plan is None, fall back to requiring override_reason."""
    guard = VerificationGuard(plan=None)
    ctx = GuardContext(
        tool_name="plan_update",
        tool_args={"action": "step_done", "step_id": 1}
    )
    
    verdict = guard.check_pre(ctx)
    assert verdict is not None
    assert verdict.action == "block"
    assert "_override_reason" in verdict.message


def test_backward_compatibility():
    """Existing tests still work (no plan, override_reason)."""
    guard = VerificationGuard()
    ctx = GuardContext(
        tool_name="plan_update",
        tool_args={
            "action": "step_done",
            "step_id": 1,
            "_override_reason": "verified"
        },
        override_reason="verified"
    )
    
    verdict = guard.check_pre(ctx)
    assert verdict is None  # Passes


def test_other_actions_unaffected():
    """step_doing and add_steps are not blocked."""
    d = tempfile.mkdtemp()
    try:
        plan = TaskPlan(d)
        plan.create("Test", [{"title": "Step A", "acceptance": ["A1"]}])
        
        guard = VerificationGuard(plan=plan)
        
        # step_doing: no block
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_doing", "step_id": 1}
        )
        assert guard.check_pre(ctx) is None
        
        # add_steps: no block
        ctx = GuardContext(
            tool_name="plan_update",
            tool_args={"action": "add_steps", "new_steps": ["New"]}
        )
        assert guard.check_pre(ctx) is None
    finally:
        shutil.rmtree(d)
