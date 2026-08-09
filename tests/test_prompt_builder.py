"""Tests for PromptBuilder._build_dashboard and _build_memory_keys_summary."""

import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ──────────────────────────────────────────────────────────────

def make_builder():
    """Create a PromptBuilder with a mock SkillManager."""
    from flagscale_agent.react.prompt_builder import PromptBuilder
    skill_mgr = MagicMock()
    skill_mgr.list_skills.return_value = []
    return PromptBuilder(skill_mgr)


# ── _build_dashboard ─────────────────────────────────────────────────────

class TestBuildDashboard:
    def test_turn_always_present(self):
        b = make_builder()
        b._turn_count = 7
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Turn: 7" in result

    def test_no_plan_no_task_step(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Task:" not in result
        assert "Step:" not in result

    def test_plan_title_extracted(self):
        b = make_builder()
        b._turn_count = 1
        plan_ctx = '<active-plan title="My Task" status="active">'
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard(plan_ctx, session_dir="")
        assert "Task: My Task" in result

    def test_plan_step_doing(self):
        b = make_builder()
        b._turn_count = 1
        plan_ctx = (
            '<active-plan title="T">\n'
            '[✅] Step 1: done\n'
            '[🔄] Step 2: in progress\n'
            '[⬜] Step 3: pending\n'
        )
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard(plan_ctx, session_dir="")
        assert "Step: 2/3" in result

    def test_plan_step_pending_when_no_doing(self):
        b = make_builder()
        b._turn_count = 1
        plan_ctx = (
            '<active-plan title="T">\n'
            '[✅] Step 1: done\n'
            '[⬜] Step 2: next\n'
            '[⬜] Step 3: later\n'
        )
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard(plan_ctx, session_dir="")
        assert "Step: 2/3" in result

    def test_session_dir_injected(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="/tmp/sess123")
        assert "Session: /tmp/sess123" in result
        assert "conversation.json: /tmp/sess123/conversation.json" in result
        assert "conversation_full.json: /tmp/sess123/conversation_full.json" in result

    def test_no_session_dir_omits_paths(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "conversation.json" not in result
        assert "Session:" not in result

    def test_memory_keys_present(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value="fact/a, pitfall/b"):
            result = b._build_dashboard("", session_dir="")
        assert "Memory keys: fact/a, pitfall/b" in result

    def test_empty_memory_keys_omitted(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Memory keys" not in result

    def test_full_dashboard_all_parts(self):
        """All parts present when plan + session + memory all provided."""
        b = make_builder()
        b._turn_count = 5
        plan_ctx = '<active-plan title="Deploy" >\n[🔄] Step 1:\n[⬜] Step 2:\n'
        with patch.object(b, "_build_memory_keys_summary", return_value="fact/x"):
            result = b._build_dashboard(plan_ctx, session_dir="/home/user/.flagscale/sessions/abc")
        assert "Task: Deploy" in result
        assert "Step: 1/2" in result
        assert "Turn: 5" in result
        assert "Session: /home/user/.flagscale/sessions/abc" in result
        assert "Memory keys: fact/x" in result

# ── _build_memory_keys_summary ───────────────────────────────────────────

class TestBuildMemoryKeysSummary:
    def test_returns_keys_only(self, tmp_path):
        """Keys listed, values not included."""
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))
        mem.put("fact/cluster/port", "fact", "值: 22")
        mem.put("pitfall/nccl/hang", "pitfall", "现象: hang")

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()

        assert "fact/cluster/port" in result
        assert "pitfall/nccl/hang" in result
        assert "值: 22" not in result
        assert "现象: hang" not in result

    def test_empty_memory_returns_empty_string(self, tmp_path):
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()
        assert result == ""

    def test_exception_returns_empty_string(self):
        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.Memory", side_effect=Exception("boom")):
            result = b._build_memory_keys_summary()
        assert result == ""

    def test_multiple_keys_comma_separated(self, tmp_path):
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))
        mem.put("fact/a/b", "fact", "x")
        mem.put("fact/c/d", "fact", "y")
        mem.put("insight/agent/loop", "insight", "z")

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()

        keys = [k.strip() for k in result.split(",")]
        assert "fact/a/b" in keys
        assert "fact/c/d" in keys
        assert "insight/agent/loop" in keys


# ── refresh() session_dir passthrough ───────────────────────────────────

class TestRefreshSessionDir:
    def test_session_dir_appears_in_system_prompt(self, tmp_path):
        """refresh() passes session_dir all the way into the system prompt."""
        from flagscale_agent.react.prompt_builder import PromptBuilder

        skill_mgr = MagicMock()
        skill_mgr.list_skills.return_value = []
        builder = PromptBuilder(skill_mgr)

        history = MagicMock()
        captured = {}
        history.set_system_prompt.side_effect = lambda p: captured.__setitem__("prompt", p)

        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory") as mock_mem_cls:
            mock_mem_cls.return_value.list_entries.return_value = []
            builder.refresh(
                history=history,
                active_skill_content={},
                shared_storage_paths=[],
                session_dir="/fake/session/xyz",
            )

        prompt = captured.get("prompt", "")
        assert "/fake/session/xyz" in prompt
        assert "conversation_full.json" in prompt
        assert "conversation.json" in prompt
