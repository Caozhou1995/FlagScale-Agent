"""Tests for KnowledgeSkillGuard's network-recovery gate.

After web_fetch hits a network error, the guard must BLOCK substantive
non-network work until the agent makes REQUIRED_RECOVERY_ATTEMPTS distinct
genuine recovery attempts. The exit is the ACTION (real network attempts), not
an argument that the network is unreachable — this targets the "network is
restricted, I'll use prior knowledge" reflex that caps knowledge-gap tasks at
the model's own ceiling.
"""

from unittest.mock import MagicMock
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.knowledge_skill import KnowledgeSkillGuard


def _ctx(tool_name=None, tool_args=None, tool_result=None):
    ctx = MagicMock(spec=GuardContext)
    ctx.tool_name = tool_name
    ctx.tool_args = tool_args or {}
    ctx.tool_result = tool_result
    ctx.assistant_text = None
    ctx.context_pressure = 0.5
    ctx.evictable_indexes = []
    return ctx


def _arm(guard):
    """Drive a web_fetch network failure through check_post to arm the gate."""
    post = guard.check_post(_ctx(
        tool_name="web_fetch",
        tool_result="[WEB_FETCH_NETWORK_ERROR] Could not retrieve https://x: ConnectionError",
    ))
    assert post is not None and post.action == "inject"
    assert guard._network_error_seen is True


class TestNetworkRecoveryGate:
    def test_post_arms_gate(self):
        guard = KnowledgeSkillGuard()
        assert guard._network_error_seen is False
        _arm(guard)

    def test_blocks_non_network_work_after_failure(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        # Agent tries to move on: write a file (prior-knowledge work).
        v = guard.check_pre(_ctx(tool_name="write_file", tool_args={"path": "sol.py"}))
        assert v is not None
        assert v.action == "block"
        assert v.category == "network_resilience"

    def test_shell_non_network_command_blocked(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        # A shell command with no network token is NOT a recovery attempt.
        v = guard.check_pre(_ctx(tool_name="shell", tool_args={"command": "ls -la"}))
        assert v is not None and v.action == "block"

    def test_curl_counts_as_attempt_and_passes(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        v = guard.check_pre(_ctx(
            tool_name="shell",
            tool_args={"command": "env -u HTTP_PROXY curl -sSL https://x"},
        ))
        assert v is None  # genuine attempt passes through
        assert len(guard._recovery_signatures) == 1

    def test_distinct_attempts_release_gate(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        n = guard.REQUIRED_RECOVERY_ATTEMPTS
        # Make N distinct network attempts (each a different URL/technique). The
        # gate must stay armed until the quota is met, then release.
        for i in range(n):
            assert guard.check_pre(_ctx(
                tool_name="shell",
                tool_args={"command": f"env -u HTTP_PROXY curl https://host{i}"},
            )) is None
            if i < n - 1:
                assert guard._network_error_seen is True, f"released early at attempt {i+1}/{n}"
        # Gate should now be released.
        assert guard._network_error_seen is False
        # Subsequent non-network work is allowed.
        assert guard.check_pre(_ctx(tool_name="write_file", tool_args={"path": "s.py"})) is None

    def test_duplicate_attempt_does_not_advance_quota(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        cmd = {"command": "curl https://same"}
        assert guard.check_pre(_ctx(tool_name="shell", tool_args=cmd)) is None
        # Same command again — passes (no deadlock) but quota stays at 1.
        assert guard.check_pre(_ctx(tool_name="shell", tool_args=cmd)) is None
        assert len(guard._recovery_signatures) == 1
        assert guard._network_error_seen is True  # still armed

    def test_web_fetch_retry_counts_as_attempt(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        n = guard.REQUIRED_RECOVERY_ATTEMPTS
        # Retry web_fetch with distinct URLs — each counts via KNOWLEDGE_TOOLS branch.
        assert guard.check_pre(_ctx(tool_name="web_fetch", tool_args={"url": "https://mirror0"})) is None
        assert len(guard._recovery_signatures) == 1
        # Remaining distinct fetches to meet the quota; the last releases the gate.
        for i in range(1, n):
            assert guard.check_pre(_ctx(tool_name="web_fetch", tool_args={"url": f"https://mirror{i}"})) is None
        assert guard._network_error_seen is False

    def test_meta_tools_pass_and_dont_count(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        # plan/memory must not deadlock and must not count as recovery.
        assert guard.check_pre(_ctx(tool_name="plan_update", tool_args={})) is None
        assert guard.check_pre(_ctx(tool_name="memory_write", tool_args={})) is None
        assert len(guard._recovery_signatures) == 0
        assert guard._network_error_seen is True  # still armed

    def test_override_releases_gate_with_evidence(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        ok = guard.accept_override(
            "proxy returns 500 for ossci-datasets.s3.amazonaws.com and direct is refused",
            _ctx(tool_name="write_file"),
        )
        assert ok is True
        assert guard._network_error_seen is False

    def test_override_rejected_when_trivial(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        assert guard.accept_override("x", _ctx(tool_name="write_file")) is False
        assert guard._network_error_seen is True  # still armed

    def test_no_gate_when_web_fetch_succeeds(self):
        guard = KnowledgeSkillGuard()
        assert guard.check_post(_ctx(tool_name="web_fetch", tool_result="doc content")) is None
        assert guard._network_error_seen is False
        # Non-network work is unaffected.
        assert guard.check_pre(_ctx(tool_name="write_file", tool_args={"path": "s.py"})) is None


class TestLocalKnowledgeDoesNotEscapeGate:
    """Regression: a failed web_fetch must NOT be escapable by falling back to
    LOCAL knowledge (load_knowledge/load_skill). Those tools read internal docs
    and do not touch the network, so while the gate is armed they must be BLOCKED
    just like any other non-network work. Only a genuine network attempt
    (web_fetch retry or a network shell cmd) may pass. This is the exact escape
    hatch observed in the wild: agent got the NetworkResilience advisory, then
    called load_knowledge and walked away without a single retry.
    """

    def test_load_knowledge_blocked_while_gate_armed(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        v = guard.check_pre(_ctx(tool_name="load_knowledge",
                                 tool_args={"name": "know-flash-attn"}))
        assert v is not None
        assert v.action == "block"
        assert v.category == "network_resilience"
        assert guard._network_error_seen is True  # not cleared by a local read

    def test_load_skill_blocked_while_gate_armed(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        v = guard.check_pre(_ctx(tool_name="load_skill",
                                 tool_args={"name": "ops-discipline"}))
        assert v is not None and v.action == "block"
        assert v.category == "network_resilience"

    def test_web_fetch_retry_still_passes_while_armed(self):
        """The genuine recovery action (web_fetch) must remain allowed."""
        guard = KnowledgeSkillGuard()
        _arm(guard)
        v = guard.check_pre(_ctx(tool_name="web_fetch",
                                 tool_args={"url": "https://mirror.example/elf"}))
        assert v is None  # allowed as a recovery attempt

    def test_two_web_fetch_retries_release_then_local_knowledge_ok(self):
        """After the quota of distinct web_fetch retries, the gate releases and a
        local knowledge fallback is once again permitted."""
        guard = KnowledgeSkillGuard()
        _arm(guard)
        # N DISTINCT web_fetch retries meet REQUIRED_RECOVERY_ATTEMPTS.
        n = guard.REQUIRED_RECOVERY_ATTEMPTS
        for i in range(n):
            assert guard.check_pre(_ctx(tool_name="web_fetch",
                                        tool_args={"url": f"https://h{i}.example/x"})) is None
        assert guard._network_error_seen is False  # released after the quota
        # Now a local knowledge read is fine (gate no longer armed).
        assert guard.check_pre(_ctx(tool_name="load_knowledge",
                                    tool_args={"name": "know-nccl-core"})) is None

    def test_meta_tools_still_pass_while_gate_armed(self):
        """Meta tools (plan/memory) must never be blocked by the network gate."""
        guard = KnowledgeSkillGuard()
        _arm(guard)
        assert guard.check_pre(_ctx(tool_name="plan_create")) is None
        assert guard.check_pre(_ctx(tool_name="memory_write")) is None
        assert guard._network_error_seen is True  # meta tools don't release it


class TestOverrideCrosstalk:
    """Regression: the registry calls EVERY blocking guard's accept_override with
    the SAME ctx.override_reason. A reason written for a DIFFERENT guard's block
    (e.g. BackupGuard 'backup already made') must NOT discharge the network gate —
    only a reason carrying concrete network-futility evidence may."""

    def test_backup_reason_does_not_release_network_gate(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        # A BackupGuard-flavored reason: substantive (>5 chars) but zero network
        # evidence. Under the old len>5 rule this wrongly released the gate.
        released = guard.accept_override(
            "Backup already made, this is a read-only inspection command",
            _ctx(tool_name="shell"),
        )
        assert released is False
        assert guard._network_error_seen is True  # still armed

    def test_plan_reason_does_not_release_network_gate(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        released = guard.accept_override(
            "Creating the plan before producing any deliverable output",
            _ctx(tool_name="plan_create"),
        )
        assert released is False
        assert guard._network_error_seen is True

    def test_network_evidence_reason_still_releases(self):
        guard = KnowledgeSkillGuard()
        _arm(guard)
        released = guard.accept_override(
            "the proxy refused the connection to this host and DNS fails to resolve",
            _ctx(tool_name="shell"),
        )
        assert released is True
        assert guard._network_error_seen is False

    def test_callcount_override_unaffected_when_gate_not_armed(self):
        """With no network gate armed, a plain substantive reason still overrides
        the call-count block (network-evidence requirement applies ONLY to the
        network gate)."""
        guard = KnowledgeSkillGuard()
        # gate NOT armed
        released = guard.accept_override(
            "This is a simple local file edit, no domain knowledge needed",
            _ctx(tool_name="shell"),
        )
        assert released is True
        assert guard._calls_since_knowledge == 0
