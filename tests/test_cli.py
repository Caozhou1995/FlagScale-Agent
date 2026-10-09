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

"""Tests for the CLI entry point (flagscale_agent/cli.py).

The CLI wiring is exercised through CliRunner with WorkerAgent patched out,
capturing the AgentConfig the CLI would hand to the agent.
"""

from typer.testing import CliRunner

import flagscale_agent.react.agent as agent_mod
from flagscale_agent.cli import app

runner = CliRunner()


class _CaptureAgent:
    """Stands in for WorkerAgent; records the config it receives."""

    last_cfg = None

    def __init__(self, cfg):
        _CaptureAgent.last_cfg = cfg

    def run(self, single_shot_query=None):
        return None


def _patch_worker_agent(monkeypatch):
    monkeypatch.setattr(agent_mod, "WorkerAgent", _CaptureAgent)


def test_cli_max_context_tokens_override(monkeypatch):
    """--max-context-tokens explicitly sets cfg.max_context_tokens."""
    _patch_worker_agent(monkeypatch)
    result = runner.invoke(app, ["--max-context-tokens", "150000", "--model", "test-model-x"])
    assert result.exit_code == 0
    assert _CaptureAgent.last_cfg is not None
    assert _CaptureAgent.last_cfg.max_context_tokens == 150000


def test_cli_max_context_tokens_default_autodetect(monkeypatch):
    """Without the flag, 0 falls through to model-based auto-detection.

    Uses gpt-4o (window 128000) rather than a 200000-window model so the
    assertion cannot be satisfied by a hardcoded DEFAULT_CONTEXT_TOKENS
    fallback — it genuinely exercises the model->window lookup.
    """
    _patch_worker_agent(monkeypatch)
    result = runner.invoke(app, ["--model", "gpt-4o"])
    assert result.exit_code == 0
    assert _CaptureAgent.last_cfg is not None
    assert _CaptureAgent.last_cfg.max_context_tokens == 128000
