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

"""Tests for guard and kernel modules."""

import sys
import importlib
import pytest
from unittest.mock import MagicMock

# Import new modules directly to avoid agent.py's heavy dependencies
import importlib.util, pathlib

def _load(rel):
    base = pathlib.Path(__file__).parent.parent / "flagscale_agent" / "react"
    spec = importlib.util.spec_from_file_location(rel, base / (rel.replace(".", "/") + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[rel] = mod  # register before exec so dataclasses can find __module__
    spec.loader.exec_module(mod)
    return mod


# guard/__init__.py
_guard_spec = importlib.util.spec_from_file_location(
    "guard", pathlib.Path(__file__).parent.parent / "flagscale_agent" / "react" / "guard" / "__init__.py"
)
_guard = importlib.util.module_from_spec(_guard_spec)
sys.modules["guard"] = _guard
_guard_spec.loader.exec_module(_guard)
Guard = _guard.Guard
GuardContext = _guard.GuardContext
GuardVerdict = _guard.GuardVerdict
GuardRegistry = _guard.GuardRegistry


# ── Guard tests ───────────────────────────────────────────────────────────────

class ConcreteGuard(Guard):
    name = "test_guard"
    priority = 10

    def __init__(self, verdict=None):
        self._verdict = verdict

    def check_pre(self, ctx):
        return self._verdict

    def check_post(self, ctx):
        return self._verdict


class TestGuardContext:
    def test_default_context(self):
        ctx = GuardContext()
        assert ctx.tool_name == ""

    def test_context_with_values(self):
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
        )
        assert ctx.tool_name == "shell"


class TestGuardVerdict:
    def test_block_factory(self):
        v = GuardVerdict.block("stop!", reason="dangerous", category="test")
        assert v.action == "block"
        assert v.message == "stop!"
        assert v.reason == "dangerous"

    def test_inject_factory(self):
        v = GuardVerdict.inject("reminder", reason="test", category="test")
        assert v.action == "inject"

    def test_escalate_factory(self):
        v = GuardVerdict.escalate("review needed", reason="test", category="test")
        assert v.action == "escalate"

class TestGuardRegistry:
    def test_register_and_priority_order(self):
        reg = GuardRegistry()
        g1 = ConcreteGuard()
        g1.priority = 20
        g2 = ConcreteGuard()
        g2.priority = 5
        reg.register(g1)
        reg.register(g2)
        assert reg.guards[0].priority == 5
        assert reg.guards[1].priority == 20

    def test_check_pre_block_wins_with_inject_prepended(self):
        """Block verdict takes precedence; inject messages are NOT prepended (clean signal)."""
        reg = GuardRegistry()
        g1 = ConcreteGuard(verdict=GuardVerdict.block("blocked by g1", reason="test", category="test"))
        g1.priority = 10
        g2 = ConcreteGuard(verdict=GuardVerdict.inject("injected by g2", reason="test", category="test"))
        g2.priority = 20
        reg.register(g1)
        reg.register(g2)
        ctx = GuardContext()
        verdict = reg.check_pre(ctx)
        assert verdict.action == "block"
        # v4: Inject messages from other guards are dropped when a block fires
        # to avoid confusing multi-signal noise. Block message is self-sufficient.
        assert "injected by g2" not in verdict.message
        assert "blocked by g1" in verdict.message

    def test_check_pre_block_only_no_inject(self):
        """Block verdict alone — no inject merging."""
        reg = GuardRegistry()
        g1 = ConcreteGuard(verdict=GuardVerdict.block("blocked", reason="test", category="test"))
        g1.priority = 10
        g2 = ConcreteGuard(verdict=None)
        g2.priority = 20
        reg.register(g1)
        reg.register(g2)
        ctx = GuardContext()
        verdict = reg.check_pre(ctx)
        assert verdict.action == "block"
        # Block verdicts get override hint
        assert "blocked" in verdict.message

    def test_check_pre_returns_none_when_all_allow(self):
        reg = GuardRegistry()
        g = ConcreteGuard(verdict=None)
        reg.register(g)
        ctx = GuardContext()
        assert reg.check_pre(ctx) is None

    def test_reset_turn_called_on_all_guards(self):
        reg = GuardRegistry()
        g1 = MagicMock(spec=Guard)
        g1.priority = 10
        g2 = MagicMock(spec=Guard)
        g2.priority = 20
        reg._guards = [g1, g2]
        reg.reset_turn()
        g1.reset_turn.assert_called_once()
        g2.reset_turn.assert_called_once()


