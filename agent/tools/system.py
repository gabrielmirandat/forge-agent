"""System information and status tool.

CONTRACT:
- Purpose: Side-effect-free system introspection
- Allowed: Query platform info, Python version, system status
- Forbidden: Execution semantics, file operations, state mutation
- Security: All operations MUST be read-only and side-effect-free

See docs/tools/system.md for full contract documentation.
"""

import platform
import sys
from typing import Any

from agent.tools.base import Tool, ToolResult


class SystemTool(Tool):
    """Tool for system information and status.
    
    CONTRACT ENFORCEMENT:
    - System tool MUST be side-effect-free (no execution, no mutation)
    - All operations MUST be read-only introspection
    - No silent fallback to other tools
    
    See docs/tools/system.md for full contract.
    """

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
        self.allowed_operations = set(config.get("allowed_operations", ["get_status", "get_info"]))
        
        # INTERNAL ASSERTION: System tool must be side-effect-free
        # All operations must be read-only introspection, no execution, no mutation
        # This is enforced by rejecting execution/file operations in execute()
        
        # INTERNAL ASSERTION: System tool operations must be introspection-only
        # Operations like "execute", "run", "command" are explicitly rejected
        # This prevents misuse of system tool for execution semantics

    async def execute(self, operation: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute system operation.

        CONTRACT ENFORCEMENT:
        - System tool MUST reject any operation that implies execution
        - All operations MUST be side-effect-free and read-only
        - Operations MUST return structured data, not raw text

        Args:
            operation: Operation type (get_status, get_info)
            arguments: Operation arguments dict (typically empty for system operations)

        Returns:
            Tool execution result

        Raises:
            OperationNotSupportedError: If operation implies execution or is invalid
        """
        if not self.enabled:
            return ToolResult(success=False, output=None, error="System tool is disabled")

        # CONTRACT ASSERTION: System tool MUST reject execution attempts
        if "execute" in operation.lower() or "command" in operation.lower() or "run" in operation.lower():
            from agent.runtime.schema import OperationNotSupportedError
            raise OperationNotSupportedError(
                self.name, operation
            )  # Error message: "System tool does not support execution operations. Use shell tool for commands."

        # CONTRACT ASSERTION: System tool MUST reject file operations
        if "file" in operation.lower() or "read" in operation.lower() or "write" in operation.lower():
            from agent.runtime.schema import OperationNotSupportedError
            raise OperationNotSupportedError(
                self.name, operation
            )  # Error message: "System tool does not support file operations. Use filesystem tool for file I/O."

        if operation not in self.allowed_operations:
            from agent.runtime.schema import OperationNotSupportedError
            raise OperationNotSupportedError(
                self.name, operation
            )  # Error message includes allowed operations in base message

        try:
            # Map operation names from schema to internal methods
            if operation == "get_status":
                result = await self._status()
            elif operation == "get_info":
                result = await self._info()
            else:
                from agent.runtime.schema import OperationNotSupportedError
                raise OperationNotSupportedError(self.name, operation)

            # CONTRACT ASSERTION: System tool MUST return structured data
            if result.output and not isinstance(result.output, dict):
                # This should never happen, but enforce the contract
                return ToolResult(
                    success=False,
                    output=None,
                    error="System tool must return structured output (dict), not raw text"
                )

            return result
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

