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

    def test_network_probe_passes_through_gate(self):
        """A network probe command (curl -sI) passes through the early gate."""
        guard = KnowledgeSkillGuard(single_shot=True)
        # Exhaust the 2 free calls
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 1
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 2
        # Network probe should pass even on call 3 (would normally block)
        probe_ctx = _make_ctx(tool_name="shell", tool_args={"command": "curl -sI --connect-timeout 3 --max-time 5 https://github.com"})
        v = guard.check_pre(probe_ctx)
        assert v is None  # passes through

    def test_gate_requires_both_probe_and_knowledge(self):
        """Gate stays active if only one of probe/knowledge is done."""
        guard = KnowledgeSkillGuard(single_shot=True)
        # Exhaust 2 free calls first
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 1
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 2
        # Network probe passes through (call 3 would block non-probe)
        probe_ctx = _make_ctx(tool_name="shell", tool_args={"command": "curl -sI --connect-timeout 3 --max-time 5 https://github.com"})
        assert _advance(guard, probe_ctx) is None
        assert guard._network_probed is True
        # Now a regular shell call should still block (knowledge not done yet)
        v = guard.check_pre(_make_ctx(tool_name="shell"))
        assert v is not None and v.action == "block"
        assert "STEP 2" in v.message  # hints at research, not probe
        # Do knowledge call
        assert _advance(guard, _make_ctx(tool_name="load_knowledge")) is None
        # Now both done — regular calls pass
        assert _advance(guard, _make_ctx(tool_name="shell")) is None

    def test_early_gate_cleared_only_by_knowledge_tool(self):
        """After a knowledge call AND a network probe the gate is gone; real calls pass."""
        guard = KnowledgeSkillGuard(single_shot=True)
        # First 2 calls: pass
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 1
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 2
        # Third call would block, but we do a network probe instead (passes through)
        probe_ctx = _make_ctx(tool_name="shell", tool_args={"command": "curl -sI --connect-timeout 3 --max-time 5 https://github.com"})
        assert _advance(guard, probe_ctx) is None  # network probe passes
        assert guard._network_probed is True
        # Then a knowledge call
        assert _advance(guard, _make_ctx(tool_name="load_knowledge")) is None
        # Now real calls pass (gate cleared, both requirements met).
        for i in range(10):
            assert _advance(guard, _make_ctx(tool_name="shell")) is None

    @pytest.mark.parametrize("probe_cmd", [
        "curl -s --connect-timeout 3 --max-time 5 -o /dev/null https://github.com",
        "echo | openssl s_client -connect github.com:443 -servername github.com 2>&1 | head -5",
        "env -u HTTP_PROXY -u HTTPS_PROXY curl -sI --connect-timeout 3 --max-time 5 https://github.com",
        "nslookup github.com",
        "dig github.com",
        "time curl -s --connect-timeout 3 --max-time 10 -o /dev/null https://pypi.org/simple/pip/",
        # degradation-chain tokens: minimal container may lack curl
        "command -v curl wget python3 2>/dev/null",
        "wget -T 5 -t 1 --spider -S https://github.com",
        "python3 -c \"import urllib.request,socket; socket.setdefaulttimeout(5); print(urllib.request.urlopen('https://pypi.org').status)\"",
        "getent hosts github.com",
    ])
    def test_expanded_network_probe_tokens_pass_through(self, probe_cmd):
        """New probe tokens (curl -s --connect-timeout, openssl s_client,
        env -u HTTP_PROXY, DNS tools, time curl) plus degradation-chain tokens
        (command -v, wget -T, python urllib, getent) pass through the early gate."""
        guard = KnowledgeSkillGuard(single_shot=True)
        # Exhaust the 2 free calls
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 1
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 2
        # Each expanded probe command should pass through
        probe_ctx = _make_ctx(tool_name="shell", tool_args={"command": probe_cmd})
        v = guard.check_pre(probe_ctx)
        assert v is None  # passes through as a network probe

    def test_step1_message_is_goal_oriented_and_adaptive(self):
        """The STEP 1 block message must teach the GOAL + an adaptive degradation
        chain (check tool exists, fall back curl→wget→python), NOT a fixed script
        of hardcoded task-irrelevant registries. This is the anti-'copy the list'
        redesign: the compcert run showed the LLM copied ~20 curl lines verbatim
        into a container with no curl and burned 133s."""
        guard = KnowledgeSkillGuard(single_shot=True)
        # Exhaust 2 free calls to trigger the gate
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 1
        assert _advance(guard, _make_ctx(tool_name="shell")) is None  # call 2
        # Call 3 triggers the block with STEP 1 message
        v = guard.check_pre(_make_ctx(tool_name="shell"))
        assert v is not None and v.action == "block"
        msg = v.message
        # 1. States the GOAL, not just commands
        assert "GOAL" in msg
        # 2. Requires task-relevant host selection (not a generic sweep)
        assert "task-relevant" in msg
        assert "sweep" in msg  # explicitly warns against sweeping a generic list
        # 3. Tool-existence check before use (minimal container may lack curl)
        assert "command -v" in msg
        assert "no curl" in msg
        # 4. Degradation chain: curl → wget → python urllib
        assert "curl" in msg
        assert "wget" in msg
        assert "urllib" in msg
        # 5. Still enforces bounded time + memory write
        assert "connect-timeout" in msg or "max-time" in msg
        assert "memory" in msg
        # 6. Failure = one data point, degrade don't repeat
        assert "proxy" in msg
        # 7. Must NOT hardcode task-irrelevant registries any more
        assert "registry.npmjs.org" not in msg
        assert "crates.io" not in msg
        assert "proxy.golang.org" not in msg
        # 8. must NOT leak any task-specific source
        assert "compcert.org" not in msg
        assert "inria.hal.science" not in msg


# ── KnowledgeSkillGuard information-gain inject (check_post) ──

class TestKnowledgeSkillGuardInfoGainInject:
    """check_post now returns inject (not block) for knowledge tools — aligns with
    system prompt's Information Gain section. Inject is advisory, not blocking."""

    def test_inject_when_web_fetch_succeeds(self):
        """Successful web_fetch gets info-gain inject (was None when gate disabled)."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="web_fetch",
            tool_result="Here is the documentation page content..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"
        assert verdict.category == "knowledge_skill"

    def test_inject_on_web_fetch_network_error(self):
        """Network error gets info-gain inject with fallback guidance."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_NETWORK_ERROR] Could not retrieve https://example.com: ConnectionError"
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"
        assert verdict.category == "knowledge_skill"

    def test_inject_case_insensitive_marker(self):
        """Network error marker triggers inject regardless of case."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="web_fetch",
            tool_result="[web_fetch_network_error] timeout"
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"

    def test_no_inject_for_non_network_web_fetch_errors(self):
        """Non-network errors (blocked, size exceeded) still get info-gain inject
        (they are knowledge tools returning a result)."""
        guard = KnowledgeSkillGuard()
        ctx1 = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_BLOCKED] Blocked host: localhost"
        )
        v1 = guard.check_post(ctx1)
        assert v1 is not None
        assert v1.action == "inject"

        ctx2 = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_SIZE_EXCEEDED] Response exceeded 5 MB limit..."
        )
        v2 = guard.check_post(ctx2)
        assert v2 is not None
        assert v2.action == "inject"

        ctx3 = _make_ctx(
            tool_name="web_fetch",
            tool_result="[WEB_FETCH_LOW_CONTENT] returned only 12 chars"
        )
        v3 = guard.check_post(ctx3)
        assert v3 is not None
        assert v3.action == "inject"

    def test_no_inject_for_non_knowledge_tools(self):
        """Info-gain inject only fires for knowledge tools, not shell/etc."""
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

    def test_inject_for_load_knowledge(self):
        """load_knowledge also gets info-gain inject."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="load_knowledge",
            tool_result="NCCL topology detection algorithm..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"

    def test_inject_for_load_skill(self):
        """load_skill also gets info-gain inject."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="load_skill",
            tool_result="Skill content: train-run workflow..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"

    def test_success_inject_presents_three_state_loop(self):
        """Successful fetch inject presents the acquired/missing/self-fillable
        three-state loop so the agent knows its next move."""
        guard = KnowledgeSkillGuard()
        ctx = _make_ctx(
            tool_name="web_fetch",
            tool_result="Here is the documentation page content..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is not None
        assert verdict.action == "inject"
        msg = verdict.message.upper()
        # All three states must be named
        assert "ACQUIRED" in msg
        assert "MISSING" in msg
        assert "SELF-FILLABLE" in msg
        assert verdict.reason == "knowledge_info_gain_three_state"



# ── KnowledgeSkillGuard network shell commands reset counter ──

class TestKnowledgeSkillGuardNetworkShellReset:
    """check_post detects substantive network shell ops (git clone, pip install, wget, curl download)
    and treats them as external info-gain — resets _calls_since_knowledge=0, _early_fired=True.
    Pure probes (command -v, curl -sI) do NOT reset the counter."""

    def test_git_clone_resets_counter(self):
        """git clone is a substantive network op → resets counter."""
        guard = KnowledgeSkillGuard()
        # Advance counter to near inject threshold
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))
        assert guard._calls_since_knowledge == guard.INJECT_THRESHOLD - 2

        # Execute git clone
        ctx = _make_ctx(
            tool_name="shell",
            tool_args={"command": "git clone https://github.com/user/repo.git /tmp/repo"},
            tool_result="Cloning into '/tmp/repo'..."
        )
        verdict = guard.check_post(ctx)
        # check_post returns None (no message), but counter is reset
        assert verdict is None
        assert guard._calls_since_knowledge == 0
        assert guard._early_fired is True

    def test_pip_install_resets_counter(self):
        """pip install is a substantive network op → resets counter."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))
        assert guard._calls_since_knowledge == guard.INJECT_THRESHOLD - 2

        ctx = _make_ctx(
            tool_name="shell",
            tool_args={"command": "pip install numpy"},
            tool_result="Collecting numpy..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is None
        assert guard._calls_since_knowledge == 0
        assert guard._early_fired is True

    def test_apt_install_resets_counter(self):
        """apt-get/apt install is a substantive network op → resets counter."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))

        ctx = _make_ctx(
            tool_name="shell",
            tool_args={"command": "sudo apt-get install -y build-essential"},
            tool_result="Reading package lists..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is None
        assert guard._calls_since_knowledge == 0

    def test_wget_download_resets_counter(self):
        """wget downloading a file is a substantive network op → resets counter."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))

        ctx = _make_ctx(
            tool_name="shell",
            tool_args={"command": "wget https://example.com/file.tar.gz"},
            tool_result="Saving to: 'file.tar.gz'"
        )
        verdict = guard.check_post(ctx)
        assert verdict is None
        assert guard._calls_since_knowledge == 0

    def test_curl_download_resets_counter(self):
        """curl downloading a file (not probe) is a substantive network op → resets counter."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))

        ctx = _make_ctx(
            tool_name="shell",
            tool_args={"command": "curl -L -o /tmp/file.tar.gz https://example.com/file.tar.gz"},
            tool_result="  % Total    % Received..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is None
        assert guard._calls_since_knowledge == 0

    def test_curl_probe_does_not_reset_counter(self):
        """curl -sI (probe) does NOT reset counter — only sets _network_probed."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))
        counter_before = guard._calls_since_knowledge

        ctx = _make_ctx(
            tool_name="shell",
            tool_args={"command": "curl -sI --connect-timeout 3 https://github.com"},
            tool_result="HTTP/2 200"
        )
        verdict = guard.check_post(ctx)
        assert verdict is None
        # Counter incremented by 1 (probe does NOT reset, falls through to increment)
        assert guard._calls_since_knowledge == counter_before + 1
        # Network probe flag set
        assert guard._network_probed is True
        assert guard._network_probed is True

    def test_command_v_probe_does_not_reset_counter(self):
        """command -v (probe) does NOT reset counter."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))
        counter_before = guard._calls_since_knowledge

        ctx = _make_ctx(
            tool_name="shell",
            tool_args={"command": "command -v curl wget"},
            tool_result="/usr/bin/curl\n/usr/bin/wget"
        )
        verdict = guard.check_post(ctx)
        assert verdict is None
        # Counter incremented by 1
        assert guard._calls_since_knowledge == counter_before + 1
        assert guard._network_probed is True

    def test_curl_silent_probe_does_not_reset(self):
        """curl -s --connect-timeout (probe) does NOT reset counter."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))
        counter_before = guard._calls_since_knowledge

        ctx = _make_ctx(
            tool_name="shell",
            tool_args={"command": "curl -s --connect-timeout 3 --max-time 5 -o /dev/null https://pypi.org"},
            tool_result=""
        )
        verdict = guard.check_post(ctx)
        assert verdict is None
        # Counter incremented by 1
        assert guard._calls_since_knowledge == counter_before + 1
        assert guard._network_probed is True

    def test_multiple_network_ops_each_reset(self):
        """Multiple substantive network ops each reset the counter independently."""
        guard = KnowledgeSkillGuard()
        # First git clone
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))
        ctx1 = _make_ctx(
            tool_name="shell",
            tool_args={"command": "git clone https://github.com/A/B.git"},
            tool_result="Cloning..."
        )
        guard.check_post(ctx1)
        assert guard._calls_since_knowledge == 0

        # Build up counter again
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))
        assert guard._calls_since_knowledge == guard.INJECT_THRESHOLD - 2

        # Second pip install also resets
        ctx2 = _make_ctx(
            tool_name="shell",
            tool_args={"command": "pip install requests"},
            tool_result="Collecting requests..."
        )
        guard.check_post(ctx2)
        assert guard._calls_since_knowledge == 0

    def test_regular_shell_command_does_not_reset(self):
        """Regular shell commands (ls, grep, etc.) do NOT reset counter."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))
        counter_before = guard._calls_since_knowledge

        ctx = _make_ctx(
            tool_name="shell",
            tool_args={"command": "ls -la /tmp"},
            tool_result="total 12\ndrwxr-xr-x..."
        )
        verdict = guard.check_post(ctx)
        assert verdict is None
        # Counter incremented by 1 (regular commands don't reset)
        assert guard._calls_since_knowledge == counter_before + 1


    def test_echo_git_clone_does_not_reset(self):
        """Commands that contain network tokens but don't execute them (e.g. echo, grep) should NOT reset.
        This is a regression test for false positives in substring matching."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))
        counter_before = guard._calls_since_knowledge

        # echo containing "git clone" should NOT reset
        ctx = _make_ctx(
            tool_name="shell",
            tool_args={"command": "echo 'git clone https://github.com/user/repo.git'"},
            tool_result="git clone https://github.com/user/repo.git"
        )
        verdict = guard.check_post(ctx)
        assert verdict is None
        # Counter should increment by 1, NOT reset to 0
        assert guard._calls_since_knowledge == counter_before + 1

    def test_grep_pip_install_does_not_reset(self):
        """grep containing 'pip install' should NOT reset counter."""
        guard = KnowledgeSkillGuard()
        for i in range(guard.INJECT_THRESHOLD - 2):
            _advance(guard, _make_ctx(tool_name="shell"))
        counter_before = guard._calls_since_knowledge

        ctx = _make_ctx(
            tool_name="shell",
            tool_args={"command": "grep 'pip install' logfile.txt"},
            tool_result="2024-01-01: pip install numpy"
        )
        verdict = guard.check_post(ctx)
        assert verdict is None
        # Counter should increment by 1, NOT reset to 0
        assert guard._calls_since_knowledge == counter_before + 1
