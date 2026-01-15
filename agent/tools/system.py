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

    def __init__(self, config: dict[str, Any]):
        """Initialize system tool.
        
        Args:
            config: Tool configuration (may include full agent config via _agent_config key)
        """
        super().__init__(config)
        # Store reference to full agent config if available
        self._agent_config = config.get("_agent_config")

    @property
    def name(self) -> str:
        """Return tool name."""
        return "system"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "Get system information (OS, runtime, agent status, workspace) - read-only, no sensitive data"

    async def execute(self, operation: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute system operation.
        
        CONTRACT ENFORCEMENT:
        - System tool MUST only support read-only operations
        - No operations that modify system state
        - session_id is mandatory (even though this tool doesn't execute commands in tmux)
        
        Args:
            operation: Operation type (get_os_info, get_runtime_info, get_agent_status, get_workspace_info)
            arguments: Operation arguments (must include _session_id)
            
        Returns:
            Tool execution result with system information
        """
        if not self.enabled:
            return ToolResult(
                success=False,
                output=None,
                error="System tool is disabled",
            )
        
        # session_id is mandatory (even though this tool doesn't execute commands in tmux)
        session_id = arguments.get("_session_id")
        if not session_id:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required argument: '_session_id'. Session ID is mandatory.",
            )

        try:
            if operation == "get_os_info":
                return await self._get_os_info()
            elif operation == "get_runtime_info":
                return await self._get_runtime_info()
            elif operation == "get_agent_status":
                return await self._get_agent_status()
            elif operation == "get_workspace_info":
                return await self._get_workspace_info()
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

    async def _get_agent_status(self) -> ToolResult:
        """Get agent status and basic configuration (safe, no sensitive data).
        
        Returns:
            ToolResult with agent status information
        """
        from pathlib import Path
        
        # Get agent name and version from agent config if available
        agent_config = self._agent_config
        if agent_config:
            agent_name = getattr(agent_config, "name", "forge-agent")
            agent_version = getattr(agent_config, "version", "unknown")
            
            # Get tools status from agent config
            tools_status = {}
            if hasattr(agent_config, "tools"):
                tools_config = agent_config.tools.model_dump() if hasattr(agent_config.tools, "model_dump") else {}
                for tool_name in ["filesystem", "git", "github", "shell", "system"]:
                    tool_config = tools_config.get(tool_name, {})
                    tools_status[tool_name] = {
                        "enabled": tool_config.get("enabled", True)
                    }
            
            # Get LLM provider info (safe info only, no tokens/keys)
            llm_info = {}
            if hasattr(agent_config, "llm"):
                llm_config = agent_config.llm
                llm_info = {
                    "provider": getattr(llm_config, "provider", "unknown"),
                    "model": getattr(llm_config, "model", "unknown"),
                    "base_url": getattr(llm_config, "base_url", "unknown"),
                    # DO NOT expose: temperature, max_tokens, timeout (internal config)
                }
            
            # Get runtime info
            runtime_info = {}
            if hasattr(agent_config, "runtime"):
                runtime_config = agent_config.runtime
                runtime_info = {
                    "safety_checks": getattr(runtime_config, "safety_checks", True),
                }
        else:
            # Fallback if agent config not available
            agent_name = "forge-agent"
            agent_version = "unknown"
            tools_status = {}
            llm_info = {}
            runtime_info = {}
        
        # Get workspace info (sanitized) from tool config
        allowed_paths = self.config.get("allowed_paths", [])
        # Sanitize paths - only show relative or ~ paths, not full absolute paths
        sanitized_paths = []
        for path in allowed_paths:
            try:
                p = Path(path).expanduser()
                # If path is in home directory, show as ~/relative
                if str(p).startswith(str(Path.home())):
                    sanitized_paths.append(f"~/{p.relative_to(Path.home())}")
                else:
                    # For other paths, just show the path as-is (already safe)
                    sanitized_paths.append(path)
            except Exception:
                sanitized_paths.append(path)
        
        status = {
            "agent": {
                "name": agent_name,
                "version": agent_version,
            },
            "workspace": {
                "allowed_paths": sanitized_paths,
                "base_path": self.config.get("base_path", "unknown"),
            },
            "tools": tools_status,
            "llm": llm_info,
            "runtime": runtime_info,
        }
        
        return ToolResult(
            success=True,
            output=status,
        )

    async def _get_workspace_info(self) -> ToolResult:
        """Get workspace information (safe, sanitized paths only).
        
        Returns:
            ToolResult with workspace information
        """
        from pathlib import Path
        
        allowed_paths = self.config.get("allowed_paths", [])
        restricted_paths = self.config.get("restricted_paths", [])
        
        # Sanitize paths - show relative paths when possible
        sanitized_allowed = []
        for path in allowed_paths:
            try:
                p = Path(path).expanduser()
                if str(p).startswith(str(Path.home())):
                    sanitized_allowed.append(f"~/{p.relative_to(Path.home())}")
                else:
                    sanitized_allowed.append(str(p))
            except Exception:
                sanitized_allowed.append(path)
        
        # For restricted paths, just show count and categories (not full paths)
        restricted_categories = []
        for path in restricted_paths:
            if path == "/":
                restricted_categories.append("root (/)")
            elif path.startswith("/"):
                # Extract top-level directory name
                parts = path.strip("/").split("/")
                if parts[0]:
                    restricted_categories.append(f"/{parts[0]}/...")
        
        workspace_info = {
            "base_path": self.config.get("base_path", "unknown"),
            "allowed_paths": sanitized_allowed,
            "restricted_areas": list(set(restricted_categories)),  # Unique categories
            "persistent": self.config.get("persistent", True),
        }
        
        return ToolResult(
            success=True,
            output=workspace_info,
        )
