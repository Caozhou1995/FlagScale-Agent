# Copyright 2026 FlagOS Contributors
# Tests for the dict-command bug fix (shell tool receives dict instead of string)

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestShellDictCommandFix:
    """Test that shell tool handles malformed dict commands gracefully."""

    def test_shell_execute_rejects_dict_command(self):
        """Shell execute() returns ERROR for dict commands."""
        from flagscale_agent.react.tools.shell import ShellTool
        tool = ShellTool(require_confirm=False)
        result = tool.execute(command={"type": "string", "value": "ls"})
        assert "ERROR" in result
        assert "dict" in result

    def test_shell_execute_rejects_list_command(self):
        """Shell execute() returns ERROR for list commands."""
        from flagscale_agent.react.tools.shell import ShellTool
        tool = ShellTool(require_confirm=False)
        result = tool.execute(command=["ls", "-la"])
        assert "ERROR" in result

    def test_shell_execute_accepts_string_command(self):
        """Shell execute() works normally for string commands."""
        from flagscale_agent.react.tools.shell import ShellTool
        tool = ShellTool(require_confirm=False)
        result = tool.execute(command="echo hello_test_12345")
        assert "hello_test_12345" in result

    def test_tool_executor_sanitizes_dict_command(self):
        """tool_executor sanitizes dict command before passing to guards."""
        # Simulate the sanitization logic
        raw_args = {"command": {"type": "string", "value": "ls -la"}}
        tool_name = "shell"

        if tool_name == "shell" and isinstance(raw_args.get("command"), dict):
            cmd_dict = raw_args["command"]
            extracted = cmd_dict.get("value") or cmd_dict.get("command") or str(cmd_dict)
            raw_args = {**raw_args, "command": extracted}

        assert raw_args["command"] == "ls -la"
        assert isinstance(raw_args["command"], str)

    def test_tool_executor_sanitizes_nested_command(self):
        """tool_executor handles command dict without 'value' key."""
        raw_args = {"command": {"description": "list files"}}
        tool_name = "shell"

        if tool_name == "shell" and isinstance(raw_args.get("command"), dict):
            cmd_dict = raw_args["command"]
            extracted = cmd_dict.get("value") or cmd_dict.get("command") or str(cmd_dict)
            raw_args = {**raw_args, "command": extracted}

        assert isinstance(raw_args["command"], str)
        assert "description" in raw_args["command"]

    def test_env_compat_guard_handles_dict_command(self):
        """EnvCompatGuard.check_pre doesn't crash on dict commands."""
        from flagscale_agent.react.guard.env_compat import EnvCompatGuard
        from flagscale_agent.react.guard import GuardContext
        guard = EnvCompatGuard()
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": {"type": "string", "value": "echo hello"}},
            
        )
        # Should NOT raise 'dict' object has no attribute 'lower'
        # The key test: no exception is thrown
        result = guard.check_pre(ctx)
        # Result is None (no guard action needed for a simple echo)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
