"""Runtime exceptions."""


class OperationNotSupportedError(Exception):
    """Raised when a tool operation is not supported."""

    def __init__(self, tool_name: str, operation: str):
        self.tool_name = tool_name
        self.operation = operation
        super().__init__(f"Operation '{operation}' is not supported by tool '{tool_name}'")


class ToolNotFoundError(Exception):
    """Raised when a tool is not found in the registry."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found in registry")


class MCPConnectionError(Exception):
    """Raised when a connection to an MCP server fails."""

    def __init__(self, server_name: str, reason: str = ""):
        self.server_name = server_name
        msg = f"Failed to connect to MCP server '{server_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class MCPToolExecutionError(Exception):
    """Raised when an MCP tool call returns an error."""

    def __init__(self, tool_name: str, reason: str = ""):
        self.tool_name = tool_name
        msg = f"MCP tool '{tool_name}' execution failed"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class ModelTimeoutError(Exception):
    """Raised when the LLM exceeds its configured timeout."""

    def __init__(self, model: str, timeout_seconds: float):
        self.model = model
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Model '{model}' timed out after {timeout_seconds}s")


class AgentMaxIterationsError(Exception):
    """Raised when the agent reaches max_iterations without producing a final answer."""

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        super().__init__(f"Agent stopped after reaching max_iterations={max_iterations}")
