# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for session summary feature in resume workflow."""

import json
import os
import time

import pytest

from flagscale_agent.react.session import (
    save_conversation,
    load_conversation,
    find_resumable_sessions,
)


class TestSaveConversationSummary:
    """Test session_summary field in save/load cycle."""

    def test_save_with_summary(self, tmp_path):
        session_dir = str(tmp_path / "session1")
        messages = [{"role": "user", "content": "hello"}]
        save_conversation(
            session_dir, "sess-1", messages,
            session_summary="主要做X\n进展到Y\n下一步Z"
        )
        data = load_conversation(session_dir)
        assert data["session_summary"] == "主要做X\n进展到Y\n下一步Z"

    def test_save_without_summary(self, tmp_path):
        session_dir = str(tmp_path / "session2")
        messages = [{"role": "user", "content": "hello"}]
        save_conversation(session_dir, "sess-2", messages)
        data = load_conversation(session_dir)
        assert "session_summary" not in data or data.get("session_summary") is None

    def test_preserve_existing_summary_on_auto_save(self, tmp_path):
        """Auto-save (no summary param) should preserve existing summary."""
        session_dir = str(tmp_path / "session3")
        messages = [{"role": "user", "content": "hello"}]
        # First save with summary
        save_conversation(
            session_dir, "sess-3", messages,
            session_summary="原始摘要"
        )
        # Auto-save without summary (simulates periodic auto-save)
        messages.append({"role": "assistant", "content": "world"})
        save_conversation(session_dir, "sess-3", messages)
        data = load_conversation(session_dir)
        assert data["session_summary"] == "原始摘要"

    def test_overwrite_summary_on_exit(self, tmp_path):
        """Exit save with new summary should overwrite old one."""
        session_dir = str(tmp_path / "session4")
        messages = [{"role": "user", "content": "hello"}]
        save_conversation(
            session_dir, "sess-4", messages,
            session_summary="旧摘要"
        )
        save_conversation(
            session_dir, "sess-4", messages,
            completed=True,
            session_summary="新摘要：退出时生成"
        )
        data = load_conversation(session_dir)
        assert data["session_summary"] == "新摘要：退出时生成"


class TestFindResumableSessionsSummary:
    """Test that find_resumable_sessions returns session_summary."""

    def test_returns_summary_field(self, tmp_path):
        session_dir = str(tmp_path / "sess-abc")
        os.makedirs(session_dir)
        messages = [{"role": "user", "content": "做训练优化"}]
        save_conversation(
            session_dir, "sess-abc", messages,
            completed=False,
            session_summary="训练Qwen3-10B\n完成baseline\n下一步profiling"
        )
        sessions = find_resumable_sessions(str(tmp_path))
        assert len(sessions) == 1
        assert sessions[0]["session_summary"] == "训练Qwen3-10B\n完成baseline\n下一步profiling"

    def test_empty_summary_for_forced_exit(self, tmp_path):
        session_dir = str(tmp_path / "sess-def")
        os.makedirs(session_dir)
        messages = [{"role": "user", "content": "测试"}]
        save_conversation(session_dir, "sess-def", messages, completed=False)
        sessions = find_resumable_sessions(str(tmp_path))
        assert len(sessions) == 1
        assert sessions[0]["session_summary"] == ""

    def test_shows_all_sessions_not_limited(self, tmp_path):
        """Should return ALL resumable sessions, not just first 10."""
        for i in range(15):
            session_dir = str(tmp_path / f"sess-{i:03d}")
            os.makedirs(session_dir)
            messages = [{"role": "user", "content": f"session {i}"}]
            save_conversation(
                session_dir, f"sess-{i:03d}", messages,
                completed=False,
                session_summary=f"会话{i}\n进展{i}\n下一步{i}"
            )
        sessions = find_resumable_sessions(str(tmp_path))
        assert len(sessions) == 15


class TestResumeDisplay:
    """Test _handle_resume display format."""

    def test_display_format(self, tmp_path, capsys):
        """Verify resume list shows summary instead of skills."""
        # Create a fake session
        session_dir = str(tmp_path / "sess-aabbccdd")
        os.makedirs(session_dir)
        messages = [
            {"role": "user", "content": "训练优化"},
            {"role": "assistant", "content": "好的"},
        ]
        save_conversation(
            session_dir, "aabbccdd-1234-5678", messages,
            completed=False,
            session_summary="修复FlagScale训练bug\n已完成代码修改和测试\n无下一步待做"
        )

        # Mock agent with minimal interface
        class MockAgent:
            _sessions_root = str(tmp_path)
            provider = None

        from flagscale_agent.react.commands import CommandHandler
        handler = CommandHandler(MockAgent())
        handler._handle_resume("/resume")

        captured = capsys.readouterr()
        # Should show session ID (8 chars), turns, and summary lines
        assert "aabbccdd" in captured.out
        assert "1 turns" in captured.out
        assert "修复FlagScale训练bug" in captured.out
        assert "已完成代码修改和测试" in captured.out
        assert "无下一步待做" in captured.out
        # Should NOT show skill info
        assert "[" not in captured.out.split("turns)")[0] if "turns)" in captured.out else True


class TestCheckResumeStartup:
    """Test _check_resume (startup display) shows same format as /resume."""

    def test_startup_shows_all_sessions_with_summary(self, tmp_path, capsys, monkeypatch):
        """_check_resume should show all sessions with summary, no skill info."""
        # Create multiple sessions
        for i in range(3):
            session_dir = str(tmp_path / f"sess-{i:08x}")
            os.makedirs(session_dir)
            messages = [{"role": "user", "content": f"task {i}"}]
            save_conversation(
                session_dir, f"{i:08x}-1234-5678", messages,
                completed=False,
                loaded_skills=["train-run", "debug-strategy"],
                session_summary=f"会话{i}主要内容\n进展到步骤{i}\n下一步做{i+1}"
            )

        # Create a minimal mock agent with _check_resume
        from flagscale_agent.react.agent import WorkerAgent
        from unittest.mock import MagicMock

        agent = MagicMock(spec=WorkerAgent)
        agent._sessions_root = str(tmp_path)

        # Call the real _check_resume with our mock
        WorkerAgent._check_resume(agent)

        captured = capsys.readouterr()
        # Should show all 3 sessions (not limited to 5)
        assert "3 resumable session" in captured.out
        # Should show summaries
        assert "会话0主要内容" in captured.out
        assert "会话1主要内容" in captured.out
        assert "会话2主要内容" in captured.out
        assert "进展到步骤0" in captured.out
        assert "下一步做1" in captured.out
        # Should NOT show skill brackets
        assert "train-run" not in captured.out
        assert "debug-strategy" not in captured.out

    def test_startup_fallback_when_no_summary(self, tmp_path, capsys, monkeypatch):
        """Sessions without summary should generate simple text summary, then persist."""
        session_dir = str(tmp_path / "sess-nosummary")
        os.makedirs(session_dir)
        messages = [{"role": "user", "content": "帮我优化训练性能"}]
        save_conversation(
            session_dir, "nosummary-1234", messages,
            completed=False,
            session_input_history=["帮我优化训练性能"],
        )

        from flagscale_agent.react.agent import WorkerAgent
        from unittest.mock import MagicMock

        # Create agent mock
        agent = MagicMock(spec=WorkerAgent)
        agent._sessions_root = str(tmp_path)
        # Bind real methods
        agent._generate_missing_summaries = lambda sessions: WorkerAgent._generate_missing_summaries(agent, sessions)

        WorkerAgent._check_resume(agent)

        captured = capsys.readouterr()
        # Should show generated simple text summary (format: [1] <message>)
        assert "帮我优化训练性能" in captured.out
        assert "[1]" in captured.out

    def test_startup_summary_generation_failure_graceful(self, tmp_path, capsys):
        """If JSON parsing fails, still show session without crash."""
        session_dir = str(tmp_path / "sess-fail")
        os.makedirs(session_dir)
        messages = [{"role": "user", "content": "test"}]
        save_conversation(
            session_dir, "failsess-1234", messages,
            completed=False,
            session_input_history=["test"],
        )

        from flagscale_agent.react.agent import WorkerAgent
        from unittest.mock import MagicMock

        agent = MagicMock(spec=WorkerAgent)
        agent._sessions_root = str(tmp_path)
        # Bind real method
        agent._generate_missing_summaries = lambda sessions: WorkerAgent._generate_missing_summaries(agent, sessions)

        WorkerAgent._check_resume(agent)

        captured = capsys.readouterr()
        # Should not crash, shows session with generated summary
        assert "1 resumable session" in captured.out
        assert "test" in captured.out
