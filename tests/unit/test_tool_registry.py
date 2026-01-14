"""Unit tests for tool registry."""

import pytest

from agent.tools.base import Tool, ToolRegistry, ToolResult
from agent.runtime.schema import ToolNotFoundError


class MockTool(Tool):
    """Mock tool for testing."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "Mock tool for testing"

    async def execute(self, operation: str, arguments: dict) -> ToolResult:
        return ToolResult(success=True, output={"result": "mock"})


class TestToolRegistry:
    """Test ToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = MockTool({"enabled": True})
        
        registry.register(tool)
        
        assert registry.get("mock_tool") == tool
        assert "mock_tool" in registry.list()

    def test_get_nonexistent_tool(self):
        """Test getting a non-existent tool."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_validate_tool_exists(self):
        """Test validating an existing tool."""
        registry = ToolRegistry()
        tool = MockTool({"enabled": True})
        registry.register(tool)
        
        validated = registry.validate_tool("mock_tool")
        assert validated == tool

    def test_validate_tool_not_found(self):
        """Test validating a non-existent tool raises error."""
        registry = ToolRegistry()
        
        with pytest.raises(ToolNotFoundError):
            registry.validate_tool("nonexistent")

    def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()
        tool1 = MockTool({"enabled": True})
        tool2 = MockTool({"enabled": True})
        tool2._name = "mock_tool_2"
        
        registry.register(tool1)
        registry.register(tool2)
        
        tools = registry.list()
        assert "mock_tool" in tools
        assert "mock_tool_2" in tools

    def test_list_enabled_tools(self):
        """Test listing only enabled tools."""
        registry = ToolRegistry()
        enabled_tool = MockTool({"enabled": True})
        disabled_tool = MockTool({"enabled": False})
        disabled_tool._name = "disabled_tool"
        
        registry.register(enabled_tool)
        registry.register(disabled_tool)
        
        enabled = registry.list_enabled()
        assert "mock_tool" in enabled
        assert "disabled_tool" not in enabled

    def test_validate_operation(self):
        """Test validating an operation."""
        registry = ToolRegistry()
        tool = MockTool({"enabled": True})
        registry.register(tool)
        
        # Should not raise
        registry.validate_operation("mock_tool", "some_operation")

    def test_validate_operation_tool_not_found(self):
        """Test validating operation for non-existent tool."""
        registry = ToolRegistry()
        
        with pytest.raises(ToolNotFoundError):
            registry.validate_operation("nonexistent", "operation")
