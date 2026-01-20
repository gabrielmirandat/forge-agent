"""Runtime exceptions.

Minimal exception definitions for the runtime system.
"""


class OperationNotSupportedError(Exception):
    """Raised when a tool operation is not supported."""

    def __init__(self, tool_name: str, operation: str):
        """Initialize error.
        
        Args:
            tool_name: Name of the tool
            operation: Operation that is not supported
        """
        self.tool_name = tool_name
        self.operation = operation
        super().__init__(f"Operation '{operation}' is not supported by tool '{tool_name}'")


class ToolNotFoundError(Exception):
    """Raised when a tool is not found in the registry."""

    def __init__(self, tool_name: str):
        """Initialize error.
        
        Args:
            tool_name: Name of the tool that was not found
        """
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found in registry")
