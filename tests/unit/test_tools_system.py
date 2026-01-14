"""Unit tests for system tool."""

import pytest

from agent.tools.system import SystemTool
from agent.runtime.schema import OperationNotSupportedError
from agent.tools.base import ToolResult


class TestSystemTool:
    """Test SystemTool."""

    @pytest.fixture
    def tool(self):
        """Create system tool."""
        return SystemTool({
            "enabled": True,
            "allowed_operations": ["get_status", "get_info"],
        })

    def test_tool_name(self, tool):
        """Test tool name."""
        assert tool.name == "system"

    def test_tool_description(self, tool):
        """Test tool description."""
        assert "system" in tool.description.lower()

    def test_get_status(self, tool):
        """Test get_status operation."""
        import asyncio
        result = asyncio.run(tool.execute("get_status", {}))

        assert result.success is True
        assert isinstance(result.output, dict)
        assert "platform" in result.output
        assert "python_version" in result.output
        assert "status" in result.output

    def test_get_info(self, tool):
        """Test get_info operation."""
        import asyncio
        result = asyncio.run(tool.execute("get_info", {}))

        assert result.success is True
        assert isinstance(result.output, dict)
        assert "platform" in result.output
        assert "python_version" in result.output

    def test_reject_execution_operations(self, tool):
        """Test that execution operations are rejected."""
        import asyncio
        
        with pytest.raises(OperationNotSupportedError):
            asyncio.run(tool.execute("execute_command", {"command": "ls"}))

        with pytest.raises(OperationNotSupportedError):
            asyncio.run(tool.execute("run_script", {"script": "test.sh"}))

    def test_reject_file_operations(self, tool):
        """Test that file operations are rejected."""
        import asyncio
        
        with pytest.raises(OperationNotSupportedError):
            asyncio.run(tool.execute("read_file", {"path": "/tmp/test.txt"}))

        with pytest.raises(OperationNotSupportedError):
            asyncio.run(tool.execute("write_file", {"path": "/tmp/test.txt", "content": "test"}))

    def test_operation_not_allowed(self, tool):
        """Test operation not in allowed list."""
        import asyncio
        with pytest.raises(OperationNotSupportedError):
            asyncio.run(tool.execute("invalid_operation", {}))

    def test_tool_disabled(self):
        """Test disabled tool."""
        tool = SystemTool({"enabled": False, "allowed_operations": []})
        import asyncio
        result = asyncio.run(tool.execute("get_status", {}))
        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_returns_structured_output(self, tool):
        """Test that tool returns structured dict output."""
        import asyncio
        result = asyncio.run(tool.execute("get_status", {}))
        
        assert result.success is True
        assert isinstance(result.output, dict)
        # Should not be a string
        assert not isinstance(result.output, str)
