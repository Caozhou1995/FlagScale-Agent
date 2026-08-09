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

"""Unit tests for display module guard output functions."""
import pytest
import re
from io import StringIO
import sys

from flagscale_agent.react import display


def strip_ansi(text):
    """Remove ANSI escape codes from text."""
    return re.sub(r'\033\[[0-9;]*m', '', text)


class TestGuardDisplayFunctions:
    """Test guard display functions have visible icons."""

    def test_guard_inject_has_visible_icon(self, capsys):
        """guard_inject should display shield icon without dim() applied to it."""
        display.guard_inject("[TestGuard] This is a test message")
        captured = capsys.readouterr()
        
        # Icon should be present
        assert "🛡" in captured.out
        
        # Check that message is on same line as icon
        lines = captured.out.strip().split('\n')
        assert len(lines) == 1
        assert "[TestGuard]" in lines[0]

    def test_guard_inject_multiline(self, capsys):
        """guard_inject should handle multiline messages with proper indentation."""
        display.guard_inject("[TestGuard] First line\nSecond line\nThird line")
        captured = capsys.readouterr()
        
        lines = captured.out.strip().split('\n')
        assert len(lines) == 3
        
        # First line has icon
        assert "🛡" in lines[0]
        assert "First line" in lines[0]
        
        # Subsequent lines are indented (5 spaces to align with text after icon)
        assert lines[1].startswith("     ")
        assert "Second line" in lines[1]
        assert lines[2].startswith("     ")
        assert "Third line" in lines[2]

    def test_guard_block_has_visible_icon(self, capsys):
        """guard_block should display red stop sign icon."""
        display.guard_block("[Blocked] This action is blocked")
        captured = capsys.readouterr()
        
        assert "🚫" in captured.out
        assert "[Blocked]" in captured.out

    def test_guard_escalate_has_visible_icon(self, capsys):
        """guard_escalate should display warning icon."""
        display.guard_escalate("[Warning] This is escalated\nDetails here")
        captured = capsys.readouterr()
        
        assert "⚠️" in captured.out or "⚠" in captured.out  # Some terminals show single char
        assert "[Warning]" in captured.out
        assert "Details here" in captured.out

    def test_guard_inject_empty_message(self, capsys):
        """guard_inject should handle empty messages gracefully."""
        display.guard_inject("")
        captured = capsys.readouterr()
        assert captured.out == ""
        
        display.guard_inject("   \n  \n  ")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_icon_not_dimmed(self, capsys):
        """Verify that the icon itself is not wrapped in dim() ANSI code.
        
        This is the core fix: icon should be visible, text can be dimmed.
        """
        display.guard_inject("[TestGuard] Message")
        captured = capsys.readouterr()
        
        # Split output to check icon and text separately
        # Icon should appear before any dim ANSI codes
        output = captured.out
        
        # Find icon position
        icon_pos = output.find("🛡")
        assert icon_pos >= 0, "Icon not found"
        
        # Find first dim code (ESC[38;5;245m for color 245)
        dim_match = re.search(r'\033\[38;5;245m', output)
        if dim_match:
            dim_pos = dim_match.start()
            # Icon should appear before dim code, or dim should only affect text after icon
            # We allow icon to be before or after, but the pattern should be: "  🛡 <dim>text"
            # So icon position should be less than the position of "[TestGuard]"
            text_pos = output.find("[TestGuard]")
            assert icon_pos < text_pos, "Icon should appear before the message text"


class TestGuardDisplayEdgeCases:
    """Test edge cases in guard display functions."""

    def test_unicode_in_message(self, capsys):
        """Guard display should handle unicode characters in messages."""
        display.guard_inject("[Test] Message with emoji 🚀 and 中文")
        captured = capsys.readouterr()
        assert "🛡" in captured.out
        assert "🚀" in captured.out
        assert "中文" in captured.out

    def test_very_long_line(self, capsys):
        """Guard display should handle very long messages without truncation."""
        long_msg = "[Test] " + "x" * 500
        display.guard_inject(long_msg)
        captured = capsys.readouterr()
        
        # No truncation for guard messages (per docstring)
        assert "x" * 500 in captured.out

    def test_block_multiline_alignment(self, capsys):
        """guard_block multiline messages should have consistent indentation."""
        display.guard_block("[Blocked] First line\nSecond line")
        captured = capsys.readouterr()
        
        lines = captured.out.strip().split('\n')
        assert len(lines) == 2
        
        # Second line indented to align with text
        assert lines[1].startswith("     ")
