"""Tests for PostEvictRecoveryGuard and KnowledgeFirstGuard."""

import pytest
from unittest.mock import MagicMock
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.post_evict_recovery import PostEvictRecoveryGuard, EVICT_THRESHOLD
from flagscale_agent.react.guard.knowledge_first import KnowledgeFirstGuard, REMINDER_INTERVAL


def _make_ctx(tool_name=None, tool_result=None, assistant_text=None):
    ctx = MagicMock(spec=GuardContext)
    ctx.tool_name = tool_name
    ctx.tool_result = tool_result
    ctx.assistant_text = assistant_text
    ctx.context_pressure = 0.5
    ctx.evictable_indexes = []
    return ctx


# ── PostEvictRecoveryGuard ──

class TestPostEvictRecoveryGuard:
    def test_no_reminder_without_eviction(self):
        guard = PostEvictRecoveryGuard()
        ctx = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx) is None

    def test_no_reminder_below_threshold(self):
        guard = PostEvictRecoveryGuard()
        # Evict 5 messages (below threshold of 10)
        ctx = _make_ctx(tool_name="evict", tool_result="Evicted 5 message(s), freed ~2000 tokens.")
        guard.check_post(ctx)
        
        # Next tool call should not trigger reminder
        ctx2 = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx2) is None

    def test_reminder_after_heavy_eviction(self):
        guard = PostEvictRecoveryGuard()
        # Evict 15 messages (above threshold)
        ctx = _make_ctx(tool_name="evict", tool_result="Evicted 15 message(s), freed ~5000 tokens.")
        guard.check_post(ctx)
        
        # Next non-recovery tool should trigger reminder
        ctx2 = _make_ctx(tool_name="shell")
        verdict = guard.check_pre(ctx2)
        assert verdict is not None
        assert "evicted" in verdict.message.lower()
        assert "plan_status" in verdict.message
        assert "conversation_full.json" in verdict.message

    def test_no_reminder_for_recovery_tools(self):
        guard = PostEvictRecoveryGuard()
        ctx = _make_ctx(tool_name="evict", tool_result="Evicted 20 message(s), freed ~8000 tokens.")
        guard.check_post(ctx)
        
        # Recovery tools should not trigger reminder
        for tool in ["plan_status", "memory_read", "memory_list", "recall"]:
            ctx2 = _make_ctx(tool_name=tool)
            assert guard.check_pre(ctx2) is None

    def test_recovery_clears_state(self):
        guard = PostEvictRecoveryGuard()
        ctx = _make_ctx(tool_name="evict", tool_result="Evicted 12 message(s), freed ~4000 tokens.")
        guard.check_post(ctx)
        
        # Do recovery
        ctx2 = _make_ctx(tool_name="plan_status", tool_result="Plan: ...")
        guard.check_post(ctx2)
        
        # Next tool should NOT trigger reminder (state cleared)
        ctx3 = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx3) is None

    def test_cumulative_eviction(self):
        guard = PostEvictRecoveryGuard()
        # Two small evictions that sum above threshold
        ctx1 = _make_ctx(tool_name="evict", tool_result="Evicted 6 message(s), freed ~2000 tokens.")
        guard.check_post(ctx1)
        ctx2 = _make_ctx(tool_name="evict", tool_result="Evicted 6 message(s), freed ~2000 tokens.")
        guard.check_post(ctx2)
        
        # Total = 12, above threshold
        ctx3 = _make_ctx(tool_name="shell")
        verdict = guard.check_pre(ctx3)
        assert verdict is not None

    def test_only_reminds_once(self):
        guard = PostEvictRecoveryGuard()
        ctx = _make_ctx(tool_name="evict", tool_result="Evicted 15 message(s), freed ~5000 tokens.")
        guard.check_post(ctx)
        
        # First non-recovery tool: reminder
        ctx2 = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx2) is not None
        
        # Second non-recovery tool: no reminder (already reminded)
        ctx3 = _make_ctx(tool_name="read_file")
        assert guard.check_pre(ctx3) is None


# ── KnowledgeFirstGuard ──

class TestKnowledgeFirstGuard:
    def test_no_reminder_initially(self):
        guard = KnowledgeFirstGuard()
        ctx = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx) is None

    def test_reminder_after_interval(self):
        guard = KnowledgeFirstGuard()
        # Make REMINDER_INTERVAL calls without knowledge loading
        for i in range(REMINDER_INTERVAL - 1):
            ctx = _make_ctx(tool_name="shell")
            guard.check_pre(ctx)
        
        # The Nth call should trigger
        ctx = _make_ctx(tool_name="read_file")
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert "knowledge" in verdict.message.lower()

    def test_load_knowledge_resets(self):
        guard = KnowledgeFirstGuard()
        # Accumulate some calls
        for i in range(REMINDER_INTERVAL - 2):
            ctx = _make_ctx(tool_name="shell")
            guard.check_pre(ctx)
        
        # Load knowledge resets counter
        ctx = _make_ctx(tool_name="load_knowledge")
        guard.check_pre(ctx)
        
        # Now need another full interval before reminder
        for i in range(REMINDER_INTERVAL - 1):
            ctx = _make_ctx(tool_name="shell")
            assert guard.check_pre(ctx) is None

    def test_load_skill_resets(self):
        guard = KnowledgeFirstGuard()
        for i in range(REMINDER_INTERVAL - 2):
            ctx = _make_ctx(tool_name="shell")
            guard.check_pre(ctx)
        
        ctx = _make_ctx(tool_name="load_skill")
        guard.check_pre(ctx)
        
        # Counter reset
        ctx = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx) is None

    def test_meta_tools_dont_count(self):
        guard = KnowledgeFirstGuard()
        # Only meta tools — should never trigger
        for i in range(20):
            ctx = _make_ctx(tool_name="memory_read")
            guard.check_pre(ctx)
        
        ctx = _make_ctx(tool_name="plan_status")
        assert guard.check_pre(ctx) is None

    def test_repeating_reminders(self):
        guard = KnowledgeFirstGuard()
        reminders = 0
        for i in range(REMINDER_INTERVAL * 3):
            ctx = _make_ctx(tool_name="shell")
            verdict = guard.check_pre(ctx)
            if verdict is not None:
                reminders += 1
        
        # Should have reminded 3 times
        assert reminders == 3

    def test_never_blocks(self):
        guard = KnowledgeFirstGuard()
        # Even after many reminders, should only inject
        for i in range(100):
            ctx = _make_ctx(tool_name="shell")
            verdict = guard.check_pre(ctx)
            if verdict is not None:
                assert verdict.action == "inject"
