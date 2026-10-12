# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for display verbosity control (show_thinking / show_guard_advisory).

Covers the display-policy setter + per-surface gating, the AgentConfig env
override, and the CLI flag precedence. The autouse fixture restores the module
policy to its defaults after every test so global state never leaks.
"""
import os

import pytest
from typer.testing import CliRunner

import flagscale_agent.react.agent as agent_mod
from flagscale_agent.react import display
from flagscale_agent.react.config import AgentConfig
from flagscale_agent.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_display_policy():
    display.set_policy(show_thinking=False, show_guard_advisory=False)
    yield
    display.set_policy(show_thinking=False, show_guard_advisory=False)


class TestDisplayGating:
    def test_guard_inject_hidden_when_off(self, capsys):
        display.set_policy(show_guard_advisory=False)
        display.guard_inject("[TestGuard] advisory text")
        assert capsys.readouterr().out == ""

    def test_guard_inject_visible_when_on(self, capsys):
        display.set_policy(show_guard_advisory=True)
        display.guard_inject("[TestGuard] advisory text")
        out = capsys.readouterr().out
        assert "🛡" in out and "advisory text" in out

    def test_guard_block_always_visible_even_when_advisory_off(self, capsys):
        # Block/escalate explain why a tool did not run — never hideable.
        display.set_policy(show_thinking=False, show_guard_advisory=False)
        display.guard_block("[Blocked] reason")
        display.guard_escalate("[Escalated] reason")
        out = capsys.readouterr().out
        assert "🚫" in out and "⚠️" in out

    def test_thinking_off_still_starts_spinner(self, capsys):
        # The ⏳ spinner is the "thinking mode" busy indicator and is shown
        # UNCONDITIONALLY; only the 💭 CONTENT is policy-gated.
        display.set_policy(show_thinking=False)
        display.thinking()
        assert display._thinking_anim is not None
        display.thinking_done()
        assert display._thinking_anim is None

    def test_thinking_off_hides_content_but_keeps_spinner(self, capsys):
        # policy OFF: content stays hidden (thinking_enabled False) yet the
        # spinner still starts — the two are independent.
        display.set_policy(show_thinking=False)
        assert display.thinking_enabled() is False
        display.thinking()
        assert display._thinking_anim is not None
        display.thinking_done()

    def test_thinking_on_starts_spinner(self, capsys):
        display.set_policy(show_thinking=True)
        display.thinking()
        # Under pytest stdout is not a tty, so the anim thread does not run,
        # but the state object is created (the gate is what we assert).
        assert display._thinking_anim is not None
        display.thinking_done()
        assert display._thinking_anim is None

    def test_policy_accessors(self):
        display.set_policy(show_thinking=False, show_guard_advisory=False)
        assert display.thinking_enabled() is False
        assert display.guard_advisory_enabled() is False
        display.set_policy(show_thinking=True)
        assert display.thinking_enabled() is True
        # Unspecified arg leaves the other flag untouched.
        assert display.guard_advisory_enabled() is False


class TestConfigEnvOverride:
    def test_defaults_hide(self, monkeypatch):
        # Default is OFF (cleaner terminal) — thinking + advisory hidden unless
        # explicitly enabled via env/YAML/CLI.
        monkeypatch.delenv("FLAGSCALE_AGENT_SHOW_THINKING", raising=False)
        monkeypatch.delenv("FLAGSCALE_AGENT_SHOW_GUARD_ADVISORY", raising=False)
        c = AgentConfig(provider="anthropic", model="m1")
        assert c.show_thinking is False
        assert c.show_guard_advisory is False

    def test_env_can_enable_from_default_off(self, monkeypatch):
        # The OFF default must be overridable OFF->ON via each explicit channel.
        monkeypatch.setenv("FLAGSCALE_AGENT_SHOW_THINKING", "1")
        monkeypatch.setenv("FLAGSCALE_AGENT_SHOW_GUARD_ADVISORY", "1")
        c = AgentConfig(provider="anthropic", model="m1")
        assert c.show_thinking is True
        assert c.show_guard_advisory is True

    @pytest.mark.parametrize("val,expected", [
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("1", True), ("true", True), ("yes", True), ("anything", True),
    ])
    def test_env_parsing(self, monkeypatch, val, expected):
        monkeypatch.setenv("FLAGSCALE_AGENT_SHOW_THINKING", val)
        c = AgentConfig(provider="anthropic", model="m1")
        assert c.show_thinking is expected

    def test_env_beats_yaml(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FLAGSCALE_AGENT_SHOW_GUARD_ADVISORY", "true")
        p = tmp_path / "c.yaml"
        p.write_text("provider: anthropic\nmodel: m1\nshow_guard_advisory: false\n")
        c = AgentConfig.from_yaml(str(p))
        assert c.show_guard_advisory is True  # env wins over YAML false

    def test_env_unset_yaml_stands(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FLAGSCALE_AGENT_SHOW_THINKING", raising=False)
        p = tmp_path / "c.yaml"
        p.write_text("provider: anthropic\nmodel: m1\nshow_thinking: false\n")
        c = AgentConfig.from_yaml(str(p))
        assert c.show_thinking is False

    def test_reload_reapplies_env(self, monkeypatch, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("provider: anthropic\nmodel: m1\nshow_thinking: true\n")
        monkeypatch.setenv("FLAGSCALE_AGENT_SHOW_THINKING", "0")
        c = AgentConfig.from_yaml(str(p))
        assert c.show_thinking is False
        c.show_thinking = True  # simulate a post-init flip
        assert c.reload() is True
        assert c.show_thinking is False  # env still wins after reload


class _CaptureAgent:
    last_cfg = None

    def __init__(self, cfg):
        _CaptureAgent.last_cfg = cfg

    def run(self, single_shot_query=None):
        return None


def _patch_worker_agent(monkeypatch):
    monkeypatch.setattr(agent_mod, "WorkerAgent", _CaptureAgent)


def test_cli_no_show_thinking(monkeypatch):
    _patch_worker_agent(monkeypatch)
    # Seed an OPPOSITE (True) baseline so the assertion discriminates: the flag
    # must actively override env=1 -> False, not merely agree with the new default.
    monkeypatch.setenv("FLAGSCALE_AGENT_SHOW_THINKING", "1")
    result = runner.invoke(app, ["--no-show-thinking", "--model", "test-model-x"])
    assert result.exit_code == 0
    assert _CaptureAgent.last_cfg.show_thinking is False


def test_cli_flag_beats_env(monkeypatch):
    _patch_worker_agent(monkeypatch)
    monkeypatch.setenv("FLAGSCALE_AGENT_SHOW_THINKING", "0")
    result = runner.invoke(app, ["--show-thinking", "--model", "test-model-x"])
    assert result.exit_code == 0
    assert _CaptureAgent.last_cfg.show_thinking is True


def test_cli_no_flag_leaves_env(monkeypatch):
    _patch_worker_agent(monkeypatch)
    # Opposite-of-default env value (True) with NO flag: proves the env channel
    # still flows when CLI is silent (post-flip the default is False, so
    # asserting True here discriminates).
    monkeypatch.setenv("FLAGSCALE_AGENT_SHOW_GUARD_ADVISORY", "1")
    result = runner.invoke(app, ["--model", "test-model-x"])
    assert result.exit_code == 0
    assert _CaptureAgent.last_cfg.show_guard_advisory is True


class TestBehaviorNeutrality:
    """Hiding advisory display must not change what the model receives."""

    def _make_kernel(self, shared_display):
        from unittest.mock import MagicMock
        import types
        from flagscale_agent.react.kernel import AgentKernel, KernelDeps
        from flagscale_agent.react.guard import GuardRegistry, Guard, GuardVerdict

        class _History:
            def __init__(self):
                self.messages = [{"role": "user", "content": "go"}]

            def append(self, m):
                self.messages.append(m)

            def get_messages(self):
                return self.messages

            def get_context_pressure(self):
                return 0.1

            def get_evictable_indexes(self):
                return []

            def report_actual_tokens(self, n):
                pass

        class _StubInjectGuard(Guard):
            name = "stub_inject"
            priority = 90

            def check_pre(self, ctx):
                if not ctx.tool_name and "[TASK_COMPLETE]" in (ctx.assistant_text or ""):
                    return GuardVerdict.inject(
                        "[StubAdvisory] hidden-but-delivered",
                        reason="stub", category="stub")
                return None

        history = _History()
        registry = GuardRegistry()
        registry.register(_StubInjectGuard())
        config = types.SimpleNamespace(
            max_iterations=10, max_continuations=200, mode="auto", _turn_count=0)
        provider = MagicMock()
        provider.format_assistant_message.side_effect = (
            lambda r: {"role": "assistant", "content": r.get("content", "")})
        deps = KernelDeps(
            provider=provider, history=history, tool_registry=MagicMock(),
            judge=MagicMock(), guard_registry=registry, config=config,
            display=shared_display, get_schemas_fn=lambda: [],
            inject_message_fn=lambda m: history.append({"role": "user", "content": m}),
            append_tool_results_fn=lambda r: None,
            format_tool_result_fn=lambda tid, r: {},
            execute_tools_fn=lambda t: [],
            is_context_limit_error_fn=lambda e: False,
            call_llm_fn=lambda m, s: ({"content": "[TASK_COMPLETE]", "tool_calls": []},
                                      {"input_tokens": 1, "output_tokens": 1}),
        )
        return AgentKernel(deps), history

    def test_advisory_reaches_model_even_when_display_hidden(self, monkeypatch, capsys):
        import flagscale_agent.react.kernel as kernel_mod
        from unittest.mock import MagicMock

        shared = MagicMock()
        monkeypatch.setattr(kernel_mod, "display", shared)
        display.set_policy(show_guard_advisory=False)
        kernel, history = self._make_kernel(shared)
        kernel.run_turn()
        # FAR END: the advisory text is in conversation history (model sees it)...
        blob = " ".join(str(m.get("content", "")) for m in history.get_messages())
        assert "hidden-but-delivered" in blob
        # ...while the terminal echo was suppressed (capsys captured nothing).
        assert "🛡" not in capsys.readouterr().out



class TestReloadResync:
    """F1 regression: a hot config reload must re-push the display policy.

    The /reload config path (commands.py) calls config.reload() then
    agent._sync_display_policy(); without that second call a config-only
    edit leaves the old display policy active until a full restart.
    """

    def test_sync_display_policy_pushes_config(self, monkeypatch):
        cfg = AgentConfig()
        cfg.show_thinking = False
        cfg.show_guard_advisory = False
        # Bind the unbound method to a lightweight stand-in with .config
        class _A:
            pass
        a = _A()
        a.config = cfg
        agent_mod.WorkerAgent._sync_display_policy(a)
        assert display.thinking_enabled() is False
        assert display.guard_advisory_enabled() is False

    def test_reload_handler_resyncs_display(self, monkeypatch, tmp_path):
        """The /reload config handler re-syncs display after config.reload()."""
        import flagscale_agent.react.commands as commands_mod

        p = tmp_path / "c.yaml"
        p.write_text("provider: anthropic\nmodel: m1\nshow_thinking: true\n")
        cfg = AgentConfig.from_yaml(str(p))

        synced = {"n": 0}

        class _Agent:
            def __init__(self):
                self.config = cfg
                self.skill_manager = type("S", (), {"invalidate_cache": lambda s: None})()
                self._session_id = "sess"

            def _sync_display_policy(self):
                synced["n"] += 1
                display.set_policy(show_thinking=self.config.show_thinking)

            def _refresh_system_prompt(self):
                pass

        # Prepare the flipped YAML, then invoke the handler.
        p.write_text("provider: anthropic\nmodel: m1\nshow_thinking: false\n")
        handler = commands_mod.CommandHandler(_Agent())
        handler._handle_reload("/reload config")
        assert synced["n"] == 1
        assert display.thinking_enabled() is False
