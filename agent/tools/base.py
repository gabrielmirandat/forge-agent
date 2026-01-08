"""Base tool interface and registry.

Tools are the execution layer of the system. They can only be invoked by the
Executor component. The LLM and Planner have no direct access to tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result from a tool execution."""

    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class Tool(ABC):
    """Base class for all tools.

    Tools are the execution layer. They can only be invoked by the Executor.
    The LLM and Planner have no direct access to tools - they only propose
    tool calls that the Executor validates and executes.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize tool with configuration.

        Args:
            config: Tool-specific configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return tool description."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute tool operation.

        This method is called by the Executor component. Tools are never
        invoked directly by the LLM or Planner.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result
        """
        pass

    def validate(self, **kwargs: Any) -> bool:
        """Validate tool parameters before execution.

        Args:
            **kwargs: Tool parameters to validate

        Returns:
            True if valid, False otherwise
        """
        return True


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        """Initialize empty registry."""
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def list_enabled(self) -> List[str]:
        """List enabled tool names.

        Returns:
            List of enabled tool names
        """
        return [name for name, tool in self._tools.items() if tool.enabled]

