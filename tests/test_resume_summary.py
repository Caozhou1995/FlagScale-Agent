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

"""Tests for session resume summary generation fix.

Bug: _generate_missing_summaries and CommandHandler._generate_resume_summary
used `stream, _ = provider.chat_stream(...)` which fails because chat_stream
returns an Iterator (generator), not a tuple. Fixed to use provider.chat().
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestResumeSummaryGeneration:
    """Verify that session summary generation uses simple text format (no LLM)."""

    def test_agent_generate_missing_summaries(self, tmp_path):
        """_generate_missing_summaries should generate simple text summary without LLM."""
        from flagscale_agent.react.agent import WorkerAgent

        # Create a fake conversation.json
        conv_data = {
            "messages": [
                {"role": "user", "content": "帮我分析代码"},
                {"role": "assistant", "content": [{"type": "text", "text": "好的"}]},
                {"role": "user", "content": "继续完善文档"},
            ]
        }
        session_dir = str(tmp_path / "session1")
        os.makedirs(session_dir)
        conv_path = os.path.join(session_dir, "conversation.json")
        with open(conv_path, "w", encoding="utf-8") as f:
            json.dump(conv_data, f, ensure_ascii=False)

        # Create a minimal agent mock
        agent = MagicMock(spec=WorkerAgent)
        agent._sessions_root = str(tmp_path)

        # Bind the real method
        import types
        agent._generate_missing_summaries = types.MethodType(
            WorkerAgent._generate_missing_summaries, agent
        )

        sessions = [{"session_dir": session_dir, "session_summary": ""}]
        agent._generate_missing_summaries(sessions)

        # Verify simple text summary was generated (no LLM call)
        # Format: [1] <first message>\n[2] <second message>
        assert sessions[0]["session_summary"]
        summary = sessions[0]["session_summary"]
        assert "[1]" in summary
        assert "[2]" in summary
        assert "帮我分析代码" in summary
        assert "继续完善文档" in summary

    def test_agent_handles_list_content_response(self, tmp_path):
        """Handle messages with list-based content (tool_result format)."""
        from flagscale_agent.react.agent import WorkerAgent

        conv_data = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "训练模型"}]},
            ]
        }
        session_dir = str(tmp_path / "session2")
        os.makedirs(session_dir)
        conv_path = os.path.join(session_dir, "conversation.json")
        with open(conv_path, "w", encoding="utf-8") as f:
            json.dump(conv_data, f, ensure_ascii=False)

        agent = MagicMock(spec=WorkerAgent)

        import types
        agent._generate_missing_summaries = types.MethodType(
            WorkerAgent._generate_missing_summaries, agent
        )

        sessions = [{"session_dir": session_dir, "session_summary": ""}]
        agent._generate_missing_summaries(sessions)

        assert "训练模型" in sessions[0]["session_summary"]

    def test_agent_handles_error_gracefully(self, tmp_path):
        """If JSON parsing fails, session is skipped without crashing."""
        from flagscale_agent.react.agent import WorkerAgent

        session_dir = str(tmp_path / "session3")
        os.makedirs(session_dir)
        conv_path = os.path.join(session_dir, "conversation.json")
        # Write invalid JSON
        with open(conv_path, "w", encoding="utf-8") as f:
            f.write("invalid json{")

        agent = MagicMock(spec=WorkerAgent)

        import types
        agent._generate_missing_summaries = types.MethodType(
            WorkerAgent._generate_missing_summaries, agent
        )

        sessions = [{"session_dir": session_dir, "session_summary": ""}]
        # Should not raise
        agent._generate_missing_summaries(sessions)
        # Summary remains empty (not updated due to JSON error)
        assert sessions[0]["session_summary"] == ""


class TestCommandHandlerResumeSummary:
    """Verify CommandHandler._generate_resume_summary uses provider.chat()."""

    def test_generates_summary_via_chat(self, tmp_path):
        """_generate_resume_summary should use provider.chat, not chat_stream."""
        from flagscale_agent.react.commands import CommandHandler

        conv_data = {
            "messages": [
                {"role": "user", "content": "分析pipeline并行"},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                {"role": "user", "content": "继续"},
            ]
        }
        session_dir = str(tmp_path / "cmd_session")
        os.makedirs(session_dir)
        conv_path = os.path.join(session_dir, "conversation.json")
        with open(conv_path, "w", encoding="utf-8") as f:
            json.dump(conv_data, f, ensure_ascii=False)

        # Create handler with mock agent
        handler = CommandHandler.__new__(CommandHandler)
        handler.agent = MagicMock()
        handler.agent.provider = MagicMock()
        handler.agent.provider.chat.return_value = {
            "content": "分析并行策略\n已完成pipeline分析\n下一步分析TP",
            "tool_calls": None,
        }

        session_info = {"session_dir": session_dir}
        summary = handler._generate_resume_summary(session_info)

        # Verify chat (not chat_stream) was called
        handler.agent.provider.chat.assert_called_once()
        handler.agent.provider.chat_stream.assert_not_called()
        assert "分析" in summary

    def test_handles_error_returns_fallback(self, tmp_path):
        """On error, returns error message instead of crashing."""
        from flagscale_agent.react.commands import CommandHandler

        handler = CommandHandler.__new__(CommandHandler)
        handler.agent = MagicMock()
        handler.agent.provider = MagicMock()
        handler.agent.provider.chat.side_effect = Exception("timeout")

        session_info = {"session_dir": str(tmp_path / "nonexist")}
        summary = handler._generate_resume_summary(session_info)
        assert "失败" in summary or "无法" in summary
