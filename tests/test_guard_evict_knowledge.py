"""Tests for PostEvictRecoveryGuard and KnowledgeSkillGuard."""

import pytest
from unittest.mock import MagicMock
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.post_evict_recovery import PostEvictRecoveryGuard, EVICT_THRESHOLD
from flagscale_agent.react.guard.knowledge_skill import KnowledgeSkillGuard


def _make_ctx(tool_name=None, tool_result=None, assistant_text=None, tool_args=None):
    ctx = MagicMock(spec=GuardContext)
    ctx.tool_name = tool_name
    ctx.tool_result = tool_result
    ctx.tool_args = tool_args or {}
    ctx.assistant_text = assistant_text
    ctx.context_pressure = 0.5
    ctx.evictable_indexes = []
    return ctx


def _advance(guard, ctx):
    """Simulate one real tool cycle: check_pre decision, then (if the call was
    not blocked) check_post persists the counter. KnowledgeSkillGuard moved the
    count increment into check_post so a blocked/not-yet-run call never inflates
    the counters — tests that accumulate calls must therefore drive BOTH phases.
    Returns the check_pre verdict so callers can assert on it.
    """
    verdict = guard.check_pre(ctx)
    # Mirror the kernel: a blocked call carries the marker in its result and does
    # NOT advance the count. Here the ctx has no such marker, so a real executed
    # call advances via check_post.
    guard.check_post(ctx)
    return verdict


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

    def test_empty_tool_name_safe(self):
        guard = PostEvictRecoveryGuard()
        # Trigger recovery state
        ctx = _make_ctx(tool_name="evict", tool_result="Evicted 20 message(s), freed ~8000 tokens.")
        guard.check_post(ctx)
        
        # Pre-LLM check with empty tool_name should not trigger
        ctx2 = _make_ctx(tool_name="")
        assert guard.check_pre(ctx2) is None
        
        # None tool_name should also be safe
        ctx3 = _make_ctx(tool_name=None)
        assert guard.check_pre(ctx3) is None


# ── KnowledgeSkillGuard ──

class TestKnowledgeSkillGuard:
    def test_no_reminder_initially(self):
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx) is None

    def test_inject_after_threshold(self):
        guard = KnowledgeSkillGuard()
        # Make INJECT_THRESHOLD calls without knowledge loading
        for i in range(guard.INJECT_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            _advance(guard, ctx)
        
        # The Nth call should trigger inject
        ctx = _make_ctx(tool_name="read_file")
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "inject"
        assert "knowledge" in verdict.message.lower()

    def test_block_after_threshold(self):
        guard = KnowledgeSkillGuard()
        # Make BLOCK_THRESHOLD calls
        for i in range(guard.BLOCK_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            _advance(guard, ctx)
        
        # The BLOCK_THRESHOLD-th call should block
        ctx = _make_ctx(tool_name="shell")
        verdict = guard.check_pre(ctx)
        assert verdict is not None
        assert verdict.action == "block"
        assert "override" in verdict.message.lower()

    def test_load_knowledge_resets(self):
        guard = KnowledgeSkillGuard()
        # Accumulate some calls
        for i in range(guard.INJECT_THRESHOLD - 2):
            ctx = _make_ctx(tool_name="shell")
            guard.check_pre(ctx)
        
        # Load knowledge resets counter
        ctx = _make_ctx(tool_name="load_knowledge")
        guard.check_pre(ctx)
        
        # Now need another full threshold before reminder
        for i in range(guard.INJECT_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            assert guard.check_pre(ctx) is None

    def test_load_skill_resets(self):
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            ctx = _make_ctx(tool_name="shell")
            guard.check_pre(ctx)
        
        ctx = _make_ctx(tool_name="load_skill")
        guard.check_pre(ctx)
        
        # Counter reset
        ctx = _make_ctx(tool_name="shell")
        assert guard.check_pre(ctx) is None

    def test_meta_tools_dont_count(self):
        guard = KnowledgeSkillGuard()
        # Only meta tools — should never trigger
        for i in range(50):
            ctx = _make_ctx(tool_name="memory_read")
            guard.check_pre(ctx)
        
        ctx = _make_ctx(tool_name="plan_status")
        assert guard.check_pre(ctx) is None

    def test_accept_override_with_reason(self):
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(tool_name="shell")
        # Should accept override with sufficient reason
        assert guard.accept_override("This task is simple file editing, no domain knowledge needed", ctx)
        # Counter should be reset after override
        assert guard._calls_since_knowledge == 0

    def test_reject_override_without_reason(self):
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(tool_name="shell")
        assert not guard.accept_override("", ctx)
        assert not guard.accept_override("ok", ctx)

    def test_reset_turn_does_nothing(self):
        guard = KnowledgeSkillGuard()
        # Accumulate calls
        for i in range(5):
            ctx = _make_ctx(tool_name="shell")
            _advance(guard, ctx)
        assert guard._calls_since_knowledge == 5
        
        # reset_turn should NOT reset the counter
        guard.reset_turn()
        assert guard._calls_since_knowledge == 5

    def test_web_fetch_resets(self):
        """web_fetch fills an external knowledge gap → should reset counter."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            ctx = _make_ctx(tool_name="shell")
            guard.check_pre(ctx)

        # web_fetch resets counter
        ctx = _make_ctx(tool_name="web_fetch")
        assert guard.check_pre(ctx) is None
        assert guard._calls_since_knowledge == 0

        # Fresh threshold needed before next reminder
        for i in range(guard.INJECT_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            assert guard.check_pre(ctx) is None

    def test_inject_mentions_web_fetch_and_external(self):
        """Inject text must point at BOTH internal and external channels."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            _advance(guard, ctx)
        ctx = _make_ctx(tool_name="shell")
        verdict = guard.check_pre(ctx)
        assert verdict is not None and verdict.action == "inject"
        msg = verdict.message.lower()
        assert "web_fetch" in msg
        assert "external" in msg
        # no-alternative claim is flagged as a knowledge gap
        assert "no other method" in msg or "knowledge gap" in msg

    def test_block_mentions_web_fetch_and_no_alternative(self):
        """Block text must cover web_fetch and the no-alternative-claim trap."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.BLOCK_THRESHOLD - 1):
            ctx = _make_ctx(tool_name="shell")
            _advance(guard, ctx)
        ctx = _make_ctx(tool_name="shell")
        verdict = guard.check_pre(ctx)
        assert verdict is not None and verdict.action == "block"
        msg = verdict.message.lower()
        assert "web_fetch" in msg
        assert "no better" in msg or "no other method" in msg


# ── KnowledgeSkillGuard single-shot early advisory ──

class TestKnowledgeSkillGuardSingleShot:
    def test_default_no_early_advisory(self):
        """Non-single-shot (default) fires no early advisory on first call."""
        guard = KnowledgeSkillGuard()
        verdict = guard.check_pre(_make_ctx(tool_name="shell"))
        # First real call in normal mode: below inject threshold, no verdict.
        assert verdict is None

    def test_set_single_shot_blocks_persistently(self):
        """Single-shot blocks after 3 non-meta calls until a knowledge call clears it."""
        guard = KnowledgeSkillGuard()
        guard.set_single_shot(True)
        v1 = _advance(guard, _make_ctx(tool_name="shell"))
        assert v1 is None  # call 1
        v2 = _advance(guard, _make_ctx(tool_name="shell"))
        assert v2 is None  # call 2
        v3 = guard.check_pre(_make_ctx(tool_name="shell"))
        assert v3 is not None and v3.action == "block"
        assert v3.overridable is False
        assert "research" in v3.message.lower()
        assert "problem class" in v3.message.lower()
        # Fourth real call: still blocked (persistent until knowledge call).
        # (v3 was blocked so check_post would NOT advance; drive via check_pre.)
        v4 = guard.check_pre(_make_ctx(tool_name="shell"))
        assert v4 is not None and v4.action == "block"
        assert v4.overridable is False

    def test_ctor_single_shot_flag(self):
        """single_shot=True via constructor also arms the early gate (blocks after 3 calls)."""
        guard = KnowledgeSkillGuard(single_shot=True)
        v1 = _advance(guard, _make_ctx(tool_name="edit_file"))
        assert v1 is None  # call 1
        v2 = _advance(guard, _make_ctx(tool_name="edit_file"))
        assert v2 is None  # call 2
        v3 = guard.check_pre(_make_ctx(tool_name="edit_file"))
        assert v3 is not None and v3.action == "block"  # call 3

    def test_meta_tool_does_not_fire_early(self):
        """Meta tools (plan/memory) don't consume the early gate."""
        guard = KnowledgeSkillGuard(single_shot=True)
        assert _advance(guard, _make_ctx(tool_name="plan_create")) is None
        assert _advance(guard, _make_ctx(tool_name="memory_write")) is None
        # First 2 real calls pass
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 1
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 2
        # Third real call gets the gate.
        v = guard.check_pre(_make_ctx(tool_name="shell"))
        assert v is not None and v.action == "block"

    def test_knowledge_tool_satisfies_early(self):
        """If agent loads knowledge first, no early gate later."""
        guard = KnowledgeSkillGuard(single_shot=True)
        assert guard.check_pre(_make_ctx(tool_name="web_fetch")) is None
        # Now real calls: gate already satisfied, normal counting resumes.
        # No block until INJECT_THRESHOLD (15) or BLOCK_THRESHOLD (40).
        for i in range(10):
            v = guard.check_pre(_make_ctx(tool_name="shell"))
            assert v is None

    def test_early_gate_non_overridable(self):
        """The early block is NON-overridable — only a real knowledge call clears it."""
        guard = KnowledgeSkillGuard(single_shot=True)
        # First 2 real calls pass; the 3rd trips the non-overridable early gate.
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 1
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 2
        v = guard.check_pre(_make_ctx(tool_name="shell"))  # call 3
        assert v is not None and v.action == "block"
        assert v.overridable is False

    def test_early_gate_cleared_only_by_knowledge_tool(self):
        """After a knowledge call the gate is gone; real calls pass."""
        guard = KnowledgeSkillGuard(single_shot=True)
        # First 2 calls: pass
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 1
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 2
        # Third call would block, but we do a knowledge call instead
        assert _advance(guard, _make_ctx(tool_name="load_knowledge")) is None
        # Now real calls pass (gate cleared, normal counting).
        for i in range(10):
            assert _advance(guard, _make_ctx(tool_name="shell")) is None


# ── KnowledgeSkillGuard network resilience (check_post) ──

class TestKnowledgeSkillGuardNetworkResilience:
    def test_no_inject_when_web_fetch_succeeds(self):
        """No network resilience reminder when web_fetch returns normal content."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="web_fetch",
            tool_result="Here is the documentation page content..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is None

    def test_inject_on_web_fetch_network_error(self):
        """Inject network resilience reminder when web_fetch fails with network error."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_NETWORK_ERROR] Could not retrieve https://example.com: ConnectionError"
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"
        assert verdict.category == "network_resilience"
        msg_lower = verdict.message.lower()
        # Must mention the key troubleshooting steps from Environment Resilience
        assert "proxy" in msg_lower or "http_proxy" in msg_lower
        assert "env -u" in msg_lower or "unset" in msg_lower
        assert "case" in msg_lower  # case sensitivity
        assert "mirror" in msg_lower or "alternative" in msg_lower
        assert "local" in msg_lower or "offline" in msg_lower or "cache" in msg_lower

    def test_inject_case_insensitive_marker(self):
        """Network error marker match is case-insensitive."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="web_fetch",
            tool_result="[web_fetch_network_error] timeout"
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"

    def test_no_inject_for_other_web_fetch_errors(self):
        """Non-network errors (blocked, size exceeded) don't trigger resilience reminder."""
        guard = KnowledgeSkillGuard()
        # SSRF blocked
        ctx1 = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_BLOCKED] Blocked host: localhost"
        )
        assert guard.check_post(ctx1) is None
        
        # Size exceeded
        ctx2 = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_SIZE_EXCEEDED] Response exceeded 5 MB limit..."
        )
        assert guard.check_post(ctx2) is None
        
        # Low content
        ctx3 = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_LOW_CONTENT] returned only 12 chars"
        )
        assert guard.check_post(ctx3) is None

    def test_no_inject_for_non_web_fetch_tools(self):
        """Network resilience only fires for web_fetch, not other tools."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="shell",
            tool_result="curl: (7) Failed to connect"
        )
        assert guard.check_post(ctx) is None

    def test_no_inject_when_tool_result_none(self):
        """Guard handles missing tool_result gracefully."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(tool_name="web_fetch", tool_result=None)
        assert guard.check_post(ctx) is None

    def test_network_resilience_mentions_env_resilience_section(self):
        """Message explicitly references the Environment Resilience section of system prompt."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_NETWORK_ERROR] ProxyError"
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        msg_lower = verdict.message.lower()
        assert "environment resilience" in msg_lower or "system prompt" in msg_lower
