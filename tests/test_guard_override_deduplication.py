"""
Test guard override deduplication and display fixes.

Tests for the 3 issues fixed:
1. Override显示但实际还是block - Override shown but tool still blocked
2. 重复显示override - Same override message appears multiple times
3. Override不显示 - Override succeeds but no visible feedback
"""
import pytest
from flagscale_agent.react.guard import GuardRegistry, GuardContext, Guard, GuardVerdict


class MockGuard(Guard):
    """Test guard that accepts all overrides."""
    
    def __init__(self, name="test_guard"):
        self.name = name
        self.priority = 50
        self.check_pre_calls = 0
        self.block_count = 0
    
    def check_pre(self, ctx: GuardContext):
        self.check_pre_calls += 1
        if ctx.tool_name == "shell":
            self.block_count += 1
            return GuardVerdict(action="block", message="Test block message", category="test_category", reason="test")
        return None
    
    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        # Accept if reason is valid (non-empty and meaningful)
        return len(reason) > 10


class TestOverrideDeduplication:
    """Test that override display is deduplicated within a turn."""
    
    def test_same_guard_override_shown_once_per_turn(self, capsys):
        """Issue 2 fix: Same guard override should display only once per turn."""
        reg = GuardRegistry()
        guard = MockGuard("dedup_test")
        reg.register(guard)
        
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="Valid override reason that is long enough",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        # First call - override should be displayed
        verdict1 = reg.check_pre(ctx)
        assert verdict1 is None, "Override should pass (returns None)"
        
        # Second call with same override - should NOT display again (deduplicated)
        verdict2 = reg.check_pre(ctx)
        assert verdict2 is None, "Override should pass again"
        
        # Check that guard was called twice
        assert guard.check_pre_calls == 2
        
        # Check output - "Guard override" should appear only once
        captured = capsys.readouterr()
        override_count = captured.out.count("Guard override")
        assert override_count == 1, f"Expected 1 override display, got {override_count}"
    
    def test_override_deduplication_resets_per_turn(self, capsys):
        """Override deduplication should reset each turn."""
        reg = GuardRegistry()
        guard = MockGuard("reset_test")
        reg.register(guard)
        
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="Valid override reason",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        # First turn - override displayed
        verdict1 = reg.check_pre(ctx)
        assert verdict1 is None
        
        # Reset turn
        reg.reset_turn()
        
        # Second turn - override should display again
        verdict2 = reg.check_pre(ctx)
        assert verdict2 is None
        
        # Check output - "Guard override" should appear twice (once per turn)
        captured = capsys.readouterr()
        override_count = captured.out.count("Guard override")
        assert override_count == 2, f"Expected 2 override displays, got {override_count}"
    
    def test_different_guards_both_display_override(self, capsys):
        """Different guards should each display their override (no cross-guard deduplication)."""
        reg = GuardRegistry()
        guard1 = MockGuard("guard1")
        guard2 = MockGuard("guard2")
        reg.register(guard1)
        reg.register(guard2)
        
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="Valid override reason",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        # Both guards should block and accept override
        verdict = reg.check_pre(ctx)
        assert verdict is None, "Both overrides should pass"
        
        # Check output - should see two override displays (one per guard)
        captured = capsys.readouterr()
        assert "guard1" in captured.out, "guard1 override should be displayed"
        assert "guard2" in captured.out, "guard2 override should be displayed"


class TestOverrideDisplayLogic:
    """Test that override display matches actual behavior."""
    
    def test_override_accepted_passes_tool(self):
        """Issue 1 fix: When override is accepted, tool should execute (returns None)."""
        reg = GuardRegistry()
        guard = MockGuard()
        reg.register(guard)
        
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="Valid override reason that is long enough",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        verdict = reg.check_pre(ctx)
        assert verdict is None, "Override accepted should return None (tool executes)"
        assert guard.block_count == 1, "Guard should have attempted to block"
    
    def test_override_rejected_blocks_tool(self):
        """When override is rejected, tool should be blocked."""
        reg = GuardRegistry()
        guard = MockGuard()
        reg.register(guard)
        
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="short",  # Too short, will be rejected
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        verdict = reg.check_pre(ctx)
        assert verdict is not None, "Override rejected should return verdict (blocked)"
        assert verdict.action == "block"
    
    def test_no_override_reason_blocks_tool(self):
        """Without override reason, blocked tool should stay blocked."""
        reg = GuardRegistry()
        guard = MockGuard()
        reg.register(guard)
        
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="",  # No override
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        verdict = reg.check_pre(ctx)
        assert verdict is not None, "No override should keep block"
        assert verdict.action == "block"


class TestNoDuplicateGuardChecks:
    """Test that guards are checked only once (kernel.py does it, not tool_executor.py)."""
    
    def test_guard_registry_has_override_tracking(self):
        """GuardRegistry should have _overridden_this_turn attribute."""
        reg = GuardRegistry()
        assert hasattr(reg, "_overridden_this_turn")
        assert isinstance(reg._overridden_this_turn, set)
        assert len(reg._overridden_this_turn) == 0
    
    def test_override_tracking_persists_within_turn(self):
        """Override tracking should persist within a turn."""
        reg = GuardRegistry()
        guard = MockGuard()
        reg.register(guard)
        
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="Valid override",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        # First call
        reg.check_pre(ctx)
        assert guard.name in reg._overridden_this_turn
        
        # Second call - guard name should still be in set
        reg.check_pre(ctx)
        assert guard.name in reg._overridden_this_turn
    
    def test_override_tracking_clears_on_reset(self):
        """Override tracking should clear when turn resets."""
        reg = GuardRegistry()
        guard = MockGuard()
        reg.register(guard)
        
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            override_reason="Valid override",
            turn_count=1,
            recent_tool_history=[],
            context_pressure=0.0,
            classify_fn=lambda x, y, default=False: default
        )
        
        # Call and populate tracking set
        reg.check_pre(ctx)
        assert len(reg._overridden_this_turn) > 0
        
        # Reset turn
        reg.reset_turn()
        assert len(reg._overridden_this_turn) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
