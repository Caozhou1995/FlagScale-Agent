"""Tests for the background-shell and time-budget guidance added to the
system prompt (prompt.py SYSTEM_PROMPT_STATIC)."""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


class TestBackgroundToolGuidance:
    def test_tool_guide_mentions_background_launch(self):
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "background=true" in low
        assert "shell_jobs" in low

    def test_tool_guide_lists_poll_and_wait(self):
        # The agent must know how to check a backgrounded job later.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert 'action="poll"' in low
        assert 'action="wait"' in low

    def test_tool_guide_covers_long_running_classes(self):
        low = SYSTEM_PROMPT_STATIC.lower()
        # Names the kinds of commands that should be backgrounded.
        assert "training" in low
        assert "download" in low or "build" in low

    def test_mentions_auto_detach_by_monitor(self):
        # The prompt should tell the agent the monitor may auto-detach a
        # healthy-but-long command and hand back a job handle.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "detach" in low

    def test_warns_against_wait_spin_loop(self):
        # Task 2 (prompt-only): backgrounding then repeatedly waiting with no
        # other work is synchronous blocking in disguise.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "synchronous blocking" in low
        # Must frame the win as doing OTHER work while the job runs.
        assert "other useful work" in low or "other work" in low

    def test_says_background_jobs_stay_monitored(self):
        # Task 1 surfaced in the prompt: backgrounded/auto-detached jobs stay
        # health-monitored and can be terminated if they later hang.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "health-monitored" in low or "health monitored" in low


class TestTimeBudgetGuidance:
    def test_has_time_budget_section(self):
        assert "Time Budget" in SYSTEM_PROMPT_STATIC

    def test_budget_section_says_plan_whole_task(self):
        low = SYSTEM_PROMPT_STATIC.lower()
        # Front-load expensive steps / allocate the shared clock up front.
        assert "front-load" in low or "front load" in low
        assert "budget" in low

    def test_budget_section_warns_against_blocking(self):
        # Counter-example: sit blocked on a long run doing nothing.
        low = SYSTEM_PROMPT_STATIC.lower()
        assert "block" in low
        assert "background" in low

    def test_budget_order_preserved(self):
        # BUDGET ORDER guidance must still be present.
        assert "BUDGET ORDER" in SYSTEM_PROMPT_STATIC
