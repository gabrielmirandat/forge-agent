"""Shell command execution tool."""

import subprocess
from typing import Any

from agent.tools.base import Tool, ToolResult


class ShellTool(Tool):
    """Tool for executing shell commands."""

    @property
    def name(self) -> str:
        """Return tool name."""
        return "shell"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "Execute shell commands safely"

    def __init__(self, config: dict[str, Any]):
        """Initialize shell tool.

        Args:
            config: Tool configuration with allowed_commands and restricted_commands
        """
        super().__init__(config)
        self.allowed_commands = set(config.get("allowed_commands", []))
        self.restricted_commands = set(config.get("restricted_commands", []))

    def _check_command(self, command: str) -> bool:
        """Check if command is allowed.

        Args:
            command: Command to check

        Returns:
            True if allowed, False otherwise
        """
        cmd_base = command.split()[0] if command else ""

        if cmd_base in self.restricted_commands:
            return False

        if self.allowed_commands and cmd_base not in self.allowed_commands:
            return False

        return True

    async def execute(self, operation: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute shell command.

        Args:
            operation: Operation type (execute_command)
            arguments: Operation arguments dict with 'command' and optional 'cwd', 'env', 'timeout'

        Returns:
            Tool execution result
        """
        if not self.enabled:
            return ToolResult(success=False, output=None, error="Shell tool is disabled")

        # Extract command from arguments
        if operation != "execute_command":
            from agent.runtime.schema import OperationNotSupportedError
            raise OperationNotSupportedError(self.name, operation)

        command = arguments.get("command")
        if not command:
            return ToolResult(
                success=False, output=None, error="Missing required argument: 'command'"
            )

        if not self._check_command(command):
            return ToolResult(
                success=False, output=None, error=f"Command not allowed: {command}"
            )

        try:
            # TODO: Implement safe command execution
            # - Use subprocess with proper timeout
            # - Capture stdout/stderr
            # - Handle errors gracefully
            raise NotImplementedError("Shell execution not yet implemented")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

