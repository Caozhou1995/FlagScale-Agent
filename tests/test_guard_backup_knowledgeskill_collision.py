"""Test BackupGuard + KnowledgeSkillGuard collision avoidance.

BackupGuard blocks the 1st shell (requires backup or override).
KnowledgeSkillGuard in single-shot mode blocks after 3 non-meta calls (requires research).

With threshold=3, the agent has room to:
  1. Hit BackupGuard block on 1st shell
  2. Satisfy it (override or actually backup with 2nd shell)
  3. Do one more action
  4. Then hit KnowledgeSkill research gate

This prevents the impossible situation where both guards block simultaneously.
"""

import pytest
from unittest.mock import MagicMock
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.backup import BackupGuard
from flagscale_agent.react.guard.knowledge_skill import KnowledgeSkillGuard


def _make_ctx(tool_name=None, tool_args=None, tool_result=None):
    ctx = MagicMock(spec=GuardContext)
    ctx.tool_name = tool_name
    ctx.tool_args = tool_args or {}
    ctx.tool_result = tool_result
    ctx.assistant_text = ""
    return ctx


def _adv(guard, ctx):
    """One real tool cycle: check_pre decision + check_post count persistence.
    KnowledgeSkillGuard advances its counters in check_post (a blocked/not-run
    call must not inflate them), so accumulation tests must drive both phases."""
    verdict = guard.check_pre(ctx)
    guard.check_post(ctx)
    return verdict


class TestBackupKnowledgeSkillCollisionAvoidance:
    def test_no_collision_threshold_3(self):
        """With SINGLE_SHOT_EARLY_THRESHOLD=3, BackupGuard and KnowledgeSkill don't collide."""
        backup = BackupGuard()
        knowledge = KnowledgeSkillGuard(single_shot=True)
        
        # Call 1: shell → BackupGuard blocks
        v1_backup = backup.check_pre(_make_ctx(tool_name="shell", tool_args={"command": "ls"}))
        v1_knowledge = _adv(knowledge, _make_ctx(tool_name="shell", tool_args={"command": "ls"}))
        assert v1_backup is not None and v1_backup.action == "block"
        assert v1_knowledge is None  # KnowledgeSkill: call 1, below threshold 3
        
        # The LLM releases the backup gate with a valid override (in production the
        # GuardRegistry calls accept_override for this). The one-shot flag flips
        # only on release, which is what closes the batched-first-turn leak.
        assert backup.accept_override("input regenerable, no backup needed", _make_ctx(tool_name="shell", tool_args={})) is True

        # Call 2: shell (e.g., cp file.db file.db.bak) → BackupGuard now satisfied
        v2_backup = backup.check_pre(_make_ctx(tool_name="shell", tool_args={"command": "cp file.db file.db.bak"}))
        v2_knowledge = _adv(knowledge, _make_ctx(tool_name="shell", tool_args={"command": "cp file.db file.db.bak"}))
        assert v2_backup is None  # BackupGuard released via override
        assert v2_knowledge is None  # KnowledgeSkill: call 2, still below threshold
        
        # Call 3: another action → NOW KnowledgeSkill blocks (threshold=3 reached)
        v3_knowledge = knowledge.check_pre(_make_ctx(tool_name="read_file", tool_args={"path": "file.db"}))
        assert v3_knowledge is not None
        assert v3_knowledge.action == "block"
        assert v3_knowledge.overridable is False
        assert "research" in v3_knowledge.message.lower()
        assert "3 tool calls" in v3_knowledge.message

    def test_knowledge_call_clears_early_gate(self):
        """If agent loads knowledge early, both gates are satisfied."""
        backup = BackupGuard()
        knowledge = KnowledgeSkillGuard(single_shot=True)
        
        # Call 1: web_fetch (satisfies KnowledgeSkill immediately)
        v1_knowledge = knowledge.check_pre(_make_ctx(tool_name="web_fetch", tool_args={"url": "https://example.com"}))
        assert v1_knowledge is None
        
        # Call 2: shell → only BackupGuard blocks
        v2_backup = backup.check_pre(_make_ctx(tool_name="shell", tool_args={"command": "ls"}))
        v2_knowledge = knowledge.check_pre(_make_ctx(tool_name="shell", tool_args={"command": "ls"}))
        assert v2_backup is not None and v2_backup.action == "block"
        assert v2_knowledge is None  # KnowledgeSkill gate already cleared
        
        # Call 3+: no blocks
        for i in range(10):
            assert knowledge.check_pre(_make_ctx(tool_name="shell")) is None

    def test_meta_tools_dont_advance_threshold(self):
        """Meta tools (plan/memory) don't count toward the threshold."""
        knowledge = KnowledgeSkillGuard(single_shot=True)
        
        # Many meta tools: should never trigger
        for i in range(20):
            assert _adv(knowledge, _make_ctx(tool_name="plan_status")) is None
            assert _adv(knowledge, _make_ctx(tool_name="memory_read")) is None
        
        # First real call: still triggers threshold counting
        v1 = _adv(knowledge, _make_ctx(tool_name="shell"))
        assert v1 is None  # call 1
        v2 = _adv(knowledge, _make_ctx(tool_name="shell"))
        assert v2 is None  # call 2
        v3 = knowledge.check_pre(_make_ctx(tool_name="shell"))
        assert v3 is not None and v3.action == "block"  # call 3, threshold reached

    def test_override_backup_then_research(self):
        """Agent can override BackupGuard, then must satisfy KnowledgeSkill."""
        backup = BackupGuard()
        knowledge = KnowledgeSkillGuard(single_shot=True)
        
        # Call 1: shell → BackupGuard blocks
        v1 = backup.check_pre(_make_ctx(tool_name="shell", tool_args={"command": "ls"}))
        assert v1 is not None
        
        # Agent overrides (no irreplaceable inputs)
        backup.accept_override("No irreplaceable inputs in this task", _make_ctx())
        
        # Calls 1-2: pass through
        _adv(knowledge, _make_ctx(tool_name="shell"))  # 1
        _adv(knowledge, _make_ctx(tool_name="read_file"))  # 2
        
        # Call 3: KnowledgeSkill blocks
        v3 = knowledge.check_pre(_make_ctx(tool_name="shell"))
        assert v3 is not None
        assert v3.action == "block"
        assert v3.overridable is False
