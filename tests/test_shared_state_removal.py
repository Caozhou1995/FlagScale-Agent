"""Tests confirming SharedState is fully removed from Guard system."""
import pytest


def test_guard_registry_no_shared_state():
    """GuardRegistry initializes without shared_state attribute."""
    from flagscale_agent.react.guard import GuardRegistry
    reg = GuardRegistry.__new__(GuardRegistry)
    assert not hasattr(reg, 'shared_state')


def test_loop_detect_guard_no_shared_state():
    """LoopDetectGuard has no shared_state or TaskMode remnants."""
    from flagscale_agent.react.guard.loop_detect import LoopDetectGuard
    guard = LoopDetectGuard()
    assert not hasattr(guard, '_shared_state')
    assert not hasattr(guard, 'set_shared_state')
    assert not hasattr(guard, '_task_mode_multiplier')


def test_progress_guard_no_shared_state():
    """ProgressGuard has no set_shared_state method."""
    from flagscale_agent.react.guard.progress import ProgressGuard
    guard = ProgressGuard()
    assert not hasattr(guard, 'set_shared_state')


def test_plan_guard_no_shared_state():
    """PlanGuard has no set_shared_state method."""
    from flagscale_agent.react.guard.plan import PlanGuard
    guard = PlanGuard()
    assert not hasattr(guard, 'set_shared_state')
