"""Tests for refactored tools/base.py and guard/utils.py READ_ONLY_TOOLS."""

from flagscale_agent.react.tools.base import Tool
from flagscale_agent.react.guard.utils import READ_ONLY_TOOLS


class TestToolBase:
    """Tool base class after ToolEffect removal."""

    def test_tool_is_abstract(self):
        """Tool requires name, description, parameters, execute."""
        import pytest
        with pytest.raises(TypeError):
            Tool()

    def test_tool_subclass_minimal(self):
        class MyTool(Tool):
            name = "my_tool"
            description = "A test tool"
            parameters = {"type": "object", "properties": {}}
            async def execute(self, **kwargs):
                return "ok"
        t = MyTool()
        assert t.name == "my_tool"
        assert t.max_result_size == 50000  # default

    def test_tool_no_effects_attribute(self):
        """ToolEffect is gone — no .effects attribute."""
        class MyTool(Tool):
            name = "t"
            description = "d"
            parameters = {}
            async def execute(self, **kwargs):
                return ""
        t = MyTool()
        assert not hasattr(t, 'effects')

    def test_max_result_size_override(self):
        class BigTool(Tool):
            name = "big"
            description = "d"
            parameters = {}
            max_result_size = 200000
            async def execute(self, **kwargs):
                return ""
        assert BigTool().max_result_size == 200000


class TestReadOnlyTools:
    """READ_ONLY_TOOLS constant in guard/utils.py."""

    def test_is_frozenset(self):
        assert isinstance(READ_ONLY_TOOLS, frozenset)

    def test_contains_expected(self):
        expected = {"read_file", "memory_read", "memory_list", "plan_status",
                    "load_skill", "load_knowledge", "web_fetch"}
        assert expected.issubset(READ_ONLY_TOOLS)

    def test_write_tools_excluded(self):
        write_tools = {"write_file", "edit_file", "shell", "memory_write",
                       "plan_create", "plan_update"}
        assert READ_ONLY_TOOLS.isdisjoint(write_tools)
