"""Tests confirming SharedState removal doesn't break Guard initialization."""
import pytest


def test_guard_registry_no_shared_state():
    """GuardRegistry initializes without shared_state attribute."""
    from flagscale_agent.react.guard import GuardRegistry
    reg = GuardRegistry.__new__(GuardRegistry)
    assert not hasattr(reg, 'shared_state')


def test_loop_detect_guard_no_shared_state():
    """LoopDetectGuard works without shared_state."""
    from flagscale_agent.react.guard.loop_detect import LoopDetectGuard
    guard = LoopDetectGuard()
    # set_shared_state should be no-op
    guard.set_shared_state(None)
    assert guard._task_mode_multiplier == 1.0


def test_progress_guard_no_shared_state():
    """ProgressGuard works without shared_state."""
    from flagscale_agent.react.guard.progress import ProgressGuard
    guard = ProgressGuard()
    guard.set_shared_state(None)
    assert guard._tolerance_multiplier == 1.0


def test_plan_guard_no_shared_state():
    """PlanGuard works without shared_state."""
    from flagscale_agent.react.guard.plan import PlanGuard
    guard = PlanGuard()
    guard.set_shared_state(None)
    assert guard._threshold_multiplier == 1.0
