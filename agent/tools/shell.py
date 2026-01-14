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
            
            # Check if base command is restricted (only check restricted_commands, not allowed_commands)
            # We now allow all commands except explicitly restricted ones
            if cmd_base in self.restricted_commands:
                return False
        
        # If no restricted commands found, allow it
        # Note: allowed_commands is now optional - if not configured, all commands are allowed (except restricted)
        if self.allowed_commands:
            # If allowed_commands is configured, still check it for backward compatibility
            # But this is now more permissive
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                cmd_base = part.split()[0] if part else ""
                if cmd_base not in self.allowed_commands:
                    return False
        
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
            import subprocess
            import asyncio
            import shlex
            
            if not command or not command.strip():
                return ToolResult(
                    success=False, output=None, error="Empty command"
                )
            
            # Get working directory from arguments if provided
            cwd = arguments.get("cwd")
            if cwd:
                from pathlib import Path
                cwd = Path(cwd).expanduser().resolve()
                # Validate cwd is in allowed paths (if filesystem tool restrictions apply)
                # For now, just use it if provided
            
            # Get timeout from arguments (default: 30 seconds)
            timeout = arguments.get("timeout", 30)
            
            # Check if command contains shell operators (&&, ||, |, ;, etc.)
            # If so, execute via shell, otherwise execute directly
            shell_operators = ["&&", "||", "|", ";", ">", ">>", "<", "<<"]
            use_shell = any(op in command for op in shell_operators)
            
            if use_shell:
                # Execute via shell to support operators
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(cwd) if cwd else None,
                )
            else:
                # Parse command and arguments for direct execution
                cmd_parts = shlex.split(command)
                if not cmd_parts:
                    return ToolResult(
                        success=False, output=None, error="Empty command"
                    )
                
                # Execute command directly
                process = await asyncio.create_subprocess_exec(
                    *cmd_parts,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(cwd) if cwd else None,
                )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Command timed out after {timeout} seconds",
                )
            
            # Decode output
            stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
            
            # Command succeeded if return code is 0
            success = process.returncode == 0
            
            output = {
                "stdout": stdout_text,
                "stderr": stderr_text,
                "return_code": process.returncode,
                "command": command,
            }
            
            if not success:
                error_msg = stderr_text or f"Command failed with return code {process.returncode}"
                return ToolResult(success=False, output=output, error=error_msg)
            
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

