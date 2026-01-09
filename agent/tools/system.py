"""System information and status tool."""

import platform
import sys
from typing import Any

from agent.tools.base import Tool, ToolResult


class SystemTool(Tool):
    """Tool for system information and status."""

    @property
    def name(self) -> str:
        """Return tool name."""
        return "system"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "System information and status"

    def __init__(self, config: dict[str, Any]):
        """Initialize system tool.

        Args:
            config: Tool configuration with allowed_operations
        """
        super().__init__(config)
        self.allowed_operations = set(config.get("allowed_operations", ["status", "info"]))

    async def execute(self, operation: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute system operation.

        Args:
            operation: Operation type (get_status, get_info)
            arguments: Operation arguments dict (typically empty for system operations)

        Returns:
            Tool execution result
        """
        if not self.enabled:
            return ToolResult(success=False, output=None, error="System tool is disabled")

        if operation not in self.allowed_operations:
            from agent.runtime.schema import OperationNotSupportedError
            raise OperationNotSupportedError(self.name, operation)

        try:
            # Map operation names from schema to internal methods
            if operation == "get_status":
                return await self._status()
            elif operation == "get_info":
                return await self._info()
            else:
                from agent.runtime.schema import OperationNotSupportedError
                raise OperationNotSupportedError(self.name, operation)
        except OperationNotSupportedError:
            raise
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    async def _status(self) -> ToolResult:
        """Get system status."""
        # TODO: Implement system status
        info = {
            "platform": platform.system(),
            "python_version": sys.version,
            "status": "operational",
        }
        return ToolResult(success=True, output=info)

    async def _info(self) -> ToolResult:
        """Get system information."""
        # TODO: Implement system info
        info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.machine(),
        }
        return ToolResult(success=True, output=info)

