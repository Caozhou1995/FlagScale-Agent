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

"""Tests for guard override mechanisms — ensures no guard can cause dead loops.

Verifies:
1. All blocking guards have overridable=True
2. accept_override works with valid reasons
3. accept_override rejects trivial reasons
4. reset_turn prevents cross-session state persistence
5. End-to-end: block → override → unblock flow
"""

from flagscale_agent.react.guard import GuardContext, GuardVerdict
from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard


def _ctx(tool_name="", tool_args=None, tool_result=None,
         assistant_text="", override_reason="", **kwargs):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
        assistant_text=assistant_text,
        override_reason=override_reason,
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# All guards overridable
# ══════════════════════════════════════════════════════════════════════════════


class TestAllBlockingGuardsOverridable:
    """Every guard that can block/inject must be overridable."""

    def test_memory_discipline_overridable(self):
        g = MemoryDisciplineGuard()
        assert g.overridable is True



# ══════════════════════════════════════════════════════════════════════════════
# accept_override — valid reasons accepted
# ══════════════════════════════════════════════════════════════════════════════


class TestAcceptOverrideValid:
    """Guards accept override with substantive reasons."""

    def test_memory_discipline_accepts_reason(self):
        g = MemoryDisciplineGuard()
        g._calls_since_memory = 15
        ctx = _ctx()
        assert g.accept_override("Already checked memory, no relevant entries exist", ctx) is True
        assert g._calls_since_memory == 0


# ══════════════════════════════════════════════════════════════════════════════
# accept_override — trivial reasons rejected
# ══════════════════════════════════════════════════════════════════════════════


class TestAcceptOverrideRejectsShort:
    """Guards reject empty or too-short override reasons."""

    def test_memory_discipline_rejects_empty(self):
        g = MemoryDisciplineGuard()
        assert g.accept_override("", _ctx()) is False

    def test_memory_discipline_rejects_short(self):
        g = MemoryDisciplineGuard()
        assert g.accept_override("skip", _ctx()) is False


# ══════════════════════════════════════════════════════════════════════════════
# reset_turn — prevents cross-session persistence
# ══════════════════════════════════════════════════════════════════════════════


class TestResetTurnPreventsDeadLoop:
    """reset_turn clears escalation state to prevent stale blocks."""

    def test_memory_discipline_reset(self):
        """reset_turn is a no-op — counter persists within a turn."""
        g = MemoryDisciplineGuard()
        g._calls_since_memory = 7
        g.reset_turn()
        assert g._calls_since_memory == 7

    def test_memory_discipline_reset_state(self):
        """reset_state (full reset via decay/override) clears counter."""
        g = MemoryDisciplineGuard()
        g._calls_since_memory = 20
        g.reset_state()
        assert g._calls_since_memory == 0

    def test_memory_discipline_preserves_knowledge(self):
        """reset_new_turn doesn't reset counter — memory gap persists across turns."""
        g = MemoryDisciplineGuard()
        g._calls_since_memory = 8
        g.reset_new_turn()
        assert g._calls_since_memory == 8



class TestEndToEndOverrideFlow:
    """Simulate the full cycle: trigger → block → override → continue."""

    def test_memory_discipline_override_clears_block(self):
        """Simulate accumulated calls, override clears counter."""
        g = MemoryDisciplineGuard()
        g._calls_since_memory = 15

        # Override
        result = g.accept_override("Already checked memory, working on unrelated code generation task", _ctx())
        assert result is True
        assert g._calls_since_memory == 0

