"""System information tool.

CONTRACT:
- Purpose: System introspection and information retrieval
- Allowed: Read-only system information (OS, runtime, config, environment)
- Forbidden: Any operations that modify system state
- Security: No side effects, read-only operations only

This tool provides safe, read-only access to system information.
"""

import os
import platform
import sys
from typing import Any

from agent.tools.base import Tool, ToolResult


class SystemTool(Tool):
    """Tool for retrieving system information.
    
    CONTRACT ENFORCEMENT:
    - System tool MUST only provide read-only information
    - No operations that modify system state
    - Safe for introspection and diagnostics
    """

    @property
    def name(self) -> str:
        """Return tool name."""
        return "system"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "Get system information (OS, runtime, environment)"

    async def execute(self, operation: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute system operation.
        
        CONTRACT ENFORCEMENT:
        - System tool MUST only support read-only operations
        - No operations that modify system state
        
        Args:
            operation: Operation type (get_os_info, get_runtime_info, get_agent_config)
            arguments: Operation arguments
            
        Returns:
            Tool execution result with system information
        """
        if not self.enabled:
            return ToolResult(
                success=False,
                output=None,
                error="System tool is disabled",
            )

        try:
            if operation == "get_os_info":
                return await self._get_os_info()
            elif operation == "get_runtime_info":
                return await self._get_runtime_info()
            elif operation == "get_agent_config":
                return await self._get_agent_config()
            else:
                from agent.runtime.schema import OperationNotSupportedError
                raise OperationNotSupportedError(self.name, operation)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"System operation failed: {str(e)}",
            )

    async def _get_os_info(self) -> ToolResult:
        """Get operating system information.
        
        Returns:
            ToolResult with OS information
        """
        info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        }
        
        return ToolResult(
            success=True,
            output=info,
        )

    async def _get_runtime_info(self) -> ToolResult:
        """Get Python runtime information.
        
        Returns:
            ToolResult with runtime information
        """
        info = {
            "python_version": sys.version,
            "python_version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
                "releaselevel": sys.version_info.releaselevel,
                "serial": sys.version_info.serial,
            },
            "executable": sys.executable,
            "platform": sys.platform,
        }
        
        return ToolResult(
            success=True,
            output=info,
        )

    async def _get_agent_config(self) -> ToolResult:
        """Get agent configuration summary.
        
        Returns:
            ToolResult with agent configuration summary
        """
        # Get configuration from self.config (passed during initialization)
        config_summary = {
            "enabled": self.enabled,
            "workspace": {
                "allowed_paths": self.config.get("allowed_paths", []),
                "restricted_paths": self.config.get("restricted_paths", []),
            },
        }
        
        return ToolResult(
            success=True,
            output=config_summary,
        )
