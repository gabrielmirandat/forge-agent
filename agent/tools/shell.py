"""Shell command execution tool.

CONTRACT:
- Purpose: Command execution with explicit execution semantics
- Allowed: Execute commands, run programs, capture output
- Forbidden: Information-only operations (use system/filesystem tools)
- Security: Command validation is mandatory, no unrestricted execution

See docs/tools/shell.md for full contract documentation.
"""

import subprocess
from typing import Any

from agent.tools.base import Tool, ToolResult


class ShellTool(Tool):
    """Tool for executing shell commands.
    
    CONTRACT ENFORCEMENT:
    - Shell tool MUST only be used when execution semantics are required
    - Command validation is mandatory and non-negotiable
    - No silent fallback to other tools for read-only operations
    
    See docs/tools/shell.md for full contract.
    """

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
        
        # INTERNAL ASSERTION: Shell tool must have command validation configured
        # This ensures security boundaries are explicit, not implicit
        if not self.allowed_commands and not self.restricted_commands:
            import warnings
            warnings.warn(
                "Shell tool has no command restrictions configured. "
                "This is a security risk. Configure allowed_commands or restricted_commands.",
                UserWarning
            )
        
        # INTERNAL ASSERTION: Shell tool must declare execution intent
        # This is enforced by only supporting execute_command operation
        # The tool cannot be used for read-only operations (use filesystem/system tools)

    def _check_command(self, command: str) -> bool:
        """Check if command is allowed.

        Args:
            command: Command to check (may contain shell operators like &&, |, etc.)

        Returns:
            True if allowed, False otherwise
        """
        if not command:
            return False
        
        # For commands with shell operators (&&, |, ;, etc.), check each command separately
        # Split by common operators and check each part
        import re
        # Split by &&, ||, |, ; but preserve the structure
        parts = re.split(r'\s*(?:&&|\|\||\||;)\s*', command)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Extract the base command (first word)
            cmd_base = part.split()[0] if part else ""
            
            # FIRST: Check if base command is explicitly restricted (highest priority)
            if cmd_base in self.restricted_commands:
                return False
        
        # SECOND: If allowed_commands is configured and non-empty, check against it
        # If allowed_commands is empty or not configured, allow all commands (except restricted)
        if self.allowed_commands:
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                cmd_base = part.split()[0] if part else ""
                if cmd_base not in self.allowed_commands:
                    return False
        
        # If we get here, command is allowed:
        # - Not in restricted_commands
        # - Either allowed_commands is empty/not configured, OR command is in allowed_commands
        return True

    async def execute(self, operation: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute shell command.

        CONTRACT ENFORCEMENT:
        - Shell tool MUST only support execute_command operation
        - Command validation is mandatory and non-negotiable
        - No silent fallback to other tools

        Args:
            operation: Operation type (execute_command)
            arguments: Operation arguments dict with 'command' and optional 'cwd', 'env', 'timeout'

        Returns:
            Tool execution result

        Raises:
            OperationNotSupportedError: If operation is not execute_command
        """
        if not self.enabled:
            return ToolResult(success=False, output=None, error="Shell tool is disabled")

        # CONTRACT ASSERTION: Shell tool MUST only support execute_command
        if operation != "execute_command":
            from agent.runtime.schema import OperationNotSupportedError
            raise OperationNotSupportedError(
                self.name, operation
            )  # Error message: "Shell tool only supports execute_command operation"

        command = arguments.get("command")
        if not command:
            return ToolResult(
                success=False, output=None, error="Missing required argument: 'command'"
            )

        # CONTRACT ENFORCEMENT: Command validation is mandatory
        if not self._check_command(command):
            return ToolResult(
                success=False, output=None, error=f"Command not allowed: {command}"
            )

        # CONTRACT ASSERTION: Shell tool MUST only be used for execution semantics
        # Note: We cannot detect misuse here (e.g., using shell for file reads),
        # but the Planner should prefer filesystem/system tools for read-only operations.
        # This is documented in the contract and enforced at planning time.

        try:
            import asyncio
            
            if not command or not command.strip():
                return ToolResult(
                    success=False, output=None, error="Empty command"
                )
            
            # Get timeout from arguments (default: 30 seconds)
            timeout = arguments.get("timeout", 30)
            
            # session_id is mandatory - always execute in tmux
            session_id = arguments.get("_session_id")
            if not session_id:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Missing required argument: '_session_id'. Session ID is mandatory.",
                )
            
            # Execute command in tmux session (tmux is mandatory)
            from agent.tools.tmux import get_tmux_manager
            from agent.storage.sqlite import SQLiteStorage
            
            tmux_manager = get_tmux_manager()
            storage = SQLiteStorage("forge_agent.db")
            
            # Get tmux session (mandatory)
            tmux_session = await storage.get_tmux_session(session_id)
            if not tmux_session:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Tmux session not found for session {session_id}",
                )
            
            # Execute command in tmux session
            return_code, stdout_text, stderr_text = await tmux_manager.execute_command(
                tmux_session,
                command,
                timeout=timeout,
            )
            
            success = return_code == 0
            output = {
                "stdout": stdout_text,
                "stderr": stderr_text,
                "return_code": return_code,
                "command": command,
            }
            
            if not success:
                error_msg = stderr_text or f"Command failed with return code {return_code}"
                return ToolResult(success=False, output=output, error=error_msg)
            
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

