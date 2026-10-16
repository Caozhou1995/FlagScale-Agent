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
        assert "Turn: 7R" in result

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

    def test_memory_domains_present(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value="fact/a(2) pitfall/b(1) (3 keys total; memory_list(keyword=...) to search)"):
            result = b._build_dashboard("", session_dir="")
        assert "Memory domains: fact/a(2) pitfall/b(1)" in result

    def test_memory_keys_omitted_when_empty(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Memory domains" not in result

    def test_full_dashboard_all_parts(self):
        """All parts present when plan + session + memory all provided."""
        b = make_builder()
        b._turn_count = 5
        plan_ctx = '<active-plan title="Deploy" >\n[🔄] Step 1:\n[⬜] Step 2:\n'
        with patch.object(b, "_build_memory_keys_summary", return_value="fact/x"):
            result = b._build_dashboard(plan_ctx, session_dir="/home/user/.flagscale/sessions/abc")
        assert "Task: Deploy" in result
        assert "Step: 1/2" in result
        assert "Turn: 5R" in result
        assert "Session: /home/user/.flagscale/sessions/abc" in result
        assert "Memory domains: fact/x" in result

# ── Runtime gauges (Ctx / Time / BG) ────────────────────────────────────

class TestDashboardRuntimeGauges:
    """Dashboard gauges: Ctx pressure/evictable/evicted, Time budget, BG jobs."""

    def test_ctx_from_history(self):
        b = make_builder()
        b._turn_count = 1
        hist = MagicMock()
        hist.get_context_pressure.return_value = 0.83
        hist.get_evictable_indexes.return_value = [1, 2, 3, 4, 5]
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="", history=hist)
        assert "Ctx: 83% evictable=5" in result

    def test_ctx_from_runtime_stats_snapshot(self):
        """When no history passed, gauge falls back to runtime_stats snapshot."""
        b = make_builder()
        b._turn_count = 1
        b.runtime_stats = {"ctx_pressure": 0.5, "evictable": 12, "evict_count": 30}
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Ctx: 50% evictable=12 evicted=30" in result

    def test_ctx_absent_when_no_data(self):
        """No history, no snapshot → no Ctx gauge (never a fabricated 0%)."""
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Ctx:" not in result

    def test_time_budget_rendered(self):
        b = make_builder()
        b._turn_count = 1
        b.runtime_stats = {"budget": {"elapsed": 600, "budget": 3600,
                                      "remaining": 3000, "pct": 16.7}}
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Time: 17% used, 50m left" in result

    def test_time_absent_when_no_budget(self):
        """No external deadline → _task_budget_stats()=None → no Time gauge."""
        b = make_builder()
        b._turn_count = 1
        b.runtime_stats = {"budget": None}
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Time:" not in result

    def test_time_elapsed_shown_when_no_budget(self):
        """No budget but a live turn stopwatch → pure elapsed, no % / no remaining."""
        b = make_builder()
        b._turn_count = 1
        b.runtime_stats = {"budget": None, "turn_elapsed": 125.0}
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Time: 2m05s elapsed" in result
        assert "%" not in result.split("Time:")[1].split("\n")[0]
        assert "left" not in result

    def test_time_elapsed_hour_format(self):
        b = make_builder()
        b._turn_count = 1
        b.runtime_stats = {"budget": None, "turn_elapsed": 3725.0}
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Time: 1h02m elapsed" in result

    def test_time_elapsed_absent_when_key_missing(self):
        """No budget AND no stopwatch snapshot → no Time gauge at all."""
        b = make_builder()
        b._turn_count = 1
        b.runtime_stats = {}
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Time:" not in result

    def test_time_budget_takes_precedence_over_elapsed(self):
        """When a real budget exists, only the pct/remaining line renders."""
        b = make_builder()
        b._turn_count = 1
        b.runtime_stats = {"budget": {"elapsed": 600, "budget": 3600,
                                      "remaining": 3000, "pct": 16.7},
                           "turn_elapsed": 600.0}
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Time: 17% used, 50m left" in result
        assert "elapsed" not in result.split("Time:")[1].split("\n")[0]

    def test_bg_jobs_listed(self):
        b = make_builder()
        b._turn_count = 1
        job = MagicMock()
        job.job_id = "job3"
        job.status_str.return_value = "running"
        job.start = 0.0  # elapsed = time.time() - 0 → large but formatted
        from flagscale_agent.react.tools import shell as shell_mod
        saved = shell_mod._JOB_REGISTRY
        shell_mod._JOB_REGISTRY = MagicMock()
        shell_mod._JOB_REGISTRY.all.return_value = [job]
        try:
            with patch.object(b, "_build_memory_keys_summary", return_value=""):
                result = b._build_dashboard("", session_dir="")
        finally:
            shell_mod._JOB_REGISTRY = saved
        assert "BG: job3:running(" in result

    def test_bg_absent_when_no_jobs(self):
        b = make_builder()
        b._turn_count = 1
        from flagscale_agent.react.tools import shell as shell_mod
        saved = shell_mod._JOB_REGISTRY
        shell_mod._JOB_REGISTRY = MagicMock()
        shell_mod._JOB_REGISTRY.all.return_value = []
        try:
            with patch.object(b, "_build_memory_keys_summary", return_value=""):
                result = b._build_dashboard("", session_dir="")
        finally:
            shell_mod._JOB_REGISTRY = saved
        assert "BG:" not in result

# ── _build_memory_keys_summary ───────────────────────────────────────────

class TestBuildMemoryKeysSummary:
    def test_domain_counts_no_values(self, tmp_path):
        """Second-level groups with counts; values never leak into the summary."""
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))
        mem.put("fact/cluster/port", "fact", "value: 22")
        mem.put("fact/cluster/ip", "fact", "value: 10.0.0.1")
        mem.put("pitfall/nccl/hang", "pitfall", "symptom: hang")

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()

        assert "fact/cluster(2)" in result
        assert "pitfall/nccl(1)" in result
        # full keys and values must NOT appear
        assert "fact/cluster/port" not in result
        assert "value: 22" not in result
        assert "symptom: hang" not in result
        # total count suffix
        assert "(3 keys total" in result

    def test_malformed_key_falls_back_to_type(self, tmp_path):
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))
        mem.put("fact/cluster/ok", "fact", "x")
        mem.put("weird", "fact", "y")  # no second segment

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()

        assert "fact/cluster(1)" in result
        assert "weird(1)" in result

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

    def test_multiple_domains_sorted_with_counts(self, tmp_path):
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))
        mem.put("fact/agent/a", "fact", "x")
        mem.put("fact/agent/b", "fact", "y")
        mem.put("fact/agent/c", "fact", "z")
        mem.put("pitfall/baseline/p1", "pitfall", "p")
        mem.put("pitfall/baseline/p2", "pitfall", "q")
        mem.put("insight/agent/i1", "insight", "i")

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()

        # sorted alphabetically, counts correct, no truncation
        assert result.startswith("fact/agent(3) insight/agent(1) pitfall/baseline(2)")
        assert "(6 keys total" in result
        # no per-key truncation: every token before " (" is a full "type/domain(n)" pair
        for tok in result.split(" (")[0].split():
            assert tok in ("fact/agent(3)", "insight/agent(1)", "pitfall/baseline(2)")


# ── SYSTEM_PROMPT_STATIC content (regression guards) ─────────────────────

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


# ── Analyze-don't-just-use principle (20260918) ──────────────────────────
