"""Base tool interface and registry.

Tools are the execution layer of the system. They can only be invoked by the
Executor component. The LLM and Planner have no direct access to tools.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result from a tool execution."""

    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


@dataclass
class ToolContext:
    """Context passed to tools during execution."""

    session_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize extra dict if None."""
        if self.extra is None:
            self.extra = {}


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
    async def execute(self, operation: str, arguments: Dict[str, Any], ctx: Optional[ToolContext] = None) -> ToolResult:
        """Execute tool operation.

        This method is called by the Executor component. Tools are never
        invoked directly by the LLM or Planner.

        Args:
            operation: Operation name to execute
            arguments: Operation arguments as a dictionary
            ctx: Optional tool context with session_id and extra data

        Returns:
            Tool execution result
        """
        pass

    async def rollback(
        self, operation: str, arguments: Dict[str, Any], execution_output: Any
    ) -> ToolResult:
        """Rollback a previously executed operation.

        This method is OPTIONAL. If not implemented, rollback for this tool
        will be skipped and recorded in the execution result.

        Rollback MUST NOT call LLMs or perform reasoning. It is best-effort only.

        Args:
            operation: Operation name that was executed
            arguments: Original operation arguments
            execution_output: Output from the original execution

        Returns:
            Tool execution result indicating rollback success/failure

        Raises:
            NotImplementedError: If tool does not support rollback
        """
        raise NotImplementedError(f"Tool '{self.name}' does not support rollback")

    def validate(self, **kwargs: Any) -> bool:
        """Validate tool parameters before execution.

        Args:
            **kwargs: Tool parameters to validate

        Returns:
            True if valid, False otherwise
        """
        return True


class ToolRegistry:
    """Registry for managing available tools.

    Validates tool existence and operation support before execution.
    Rejects unknown tools immediately.
    """

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

    def validate_tool(self, name: str) -> Tool:
        """Validate that a tool exists and return it.

        Args:
            name: Tool name

        Returns:
            Tool instance

        Raises:
            ToolNotFoundError: If tool is not registered
        """
        tool = self._tools.get(name)
        if tool is None:
            from agent.runtime.exceptions import ToolNotFoundError
            raise ToolNotFoundError(name)
        return tool

    def validate_operation(self, tool_name: str, operation: str) -> None:
        """Validate that a tool supports an operation.

        This is a basic check - actual operation support is validated
        by the tool itself during execution.

        Args:
            tool_name: Tool name
            operation: Operation name

        Raises:
            ToolNotFoundError: If tool is not registered
            OperationNotSupportedError: If operation is not supported (if tool provides this info)
        """
        tool = self.validate_tool(tool_name)
        # Tools will validate operations during execute()
        # This method exists for future extension if needed

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
    
    async def get_langchain_tools(self, session_id: Optional[str] = None, config: Optional[Any] = None) -> List[Any]:
        """Get all enabled tools as LangChain tools using langchain-mcp-adapters (official pattern).
        
        All tools come from MCP servers (Docker containers).
        Uses MultiServerMCPClient from langchain-mcp-adapters to load tools directly from MCP servers.
        Queries ALL configured MCP servers dynamically to ensure all tools are available.
        
        Args:
            session_id: Optional session ID for tool context (not used for MCP tools)
            config: AgentConfig to get MCP server configurations and workspace path
            
        Returns:
            List of LangChain StructuredTool instances from MCP servers
        """
        from pathlib import Path
        import logging

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:
            raise ImportError(
                "langchain-mcp-adapters is required for MCP tools. "
                "Install it with: pip install langchain-mcp-adapters"
            ) from exc

        if config is None:
            raise ValueError("AgentConfig 'config' is required to load MCP tools via adapters")

        logger = logging.getLogger(__name__)

        # Build MultiServerMCPClient configuration from config directly
        # This ensures we query ALL configured servers, not just already-connected ones
        mcp_configs: Dict[str, Dict[str, Any]] = {}
        workspace_base = Path(config.workspace.base_path).expanduser().resolve()

        # Get all MCP server configurations from config
        all_mcp_configs = config.mcp or {}

        # Build client config for each enabled MCP server
        for server_name, mcp_config in all_mcp_configs.items():
            # Skip disabled servers
            if mcp_config.get("enabled") is False:
                logger.debug(f"Skipping disabled MCP server: {server_name}")
                continue

            server_type = mcp_config.get("type", "docker")
            
            if server_type == "docker":
                # Handle Docker-type MCP servers
                image = mcp_config.get("image")
                if not image:
                    logger.warning(f"MCP server '{server_name}' has no image configured, skipping")
                    continue

                # Build docker command
                docker_cmd: List[str] = ["docker", "run", "-i", "--rm"]

                # Add volumes
                volumes = mcp_config.get("volumes", [])
                for volume in volumes:
                    if "{{workspace.base_path}}" in volume:
                        volume = volume.replace("{{workspace.base_path}}", str(workspace_base))
                    if volume.startswith("~"):
                        volume = str(Path(volume).expanduser())
                    docker_cmd.extend(["-v", volume])

                # Add environment variables
                env_vars = mcp_config.get("environment", {})
                resolved_env: Dict[str, str] = {}
                for key, value in env_vars.items():
                    if isinstance(value, str) and value.startswith("{{env:") and value.endswith("}}"):
                        env_var_name = value[6:-2]
                        resolved_env[key] = os.getenv(env_var_name, "") or ""
                    else:
                        resolved_env[key] = str(value)

                for key, value in resolved_env.items():
                    if value:
                        docker_cmd.extend(["-e", f"{key}={value}"])

                # Add image and args
                docker_cmd.append(image)
                docker_cmd.extend(mcp_config.get("args", []))

                # Configure for MultiServerMCPClient
                mcp_configs[server_name] = {
                    "command": docker_cmd[0],
                    "args": docker_cmd[1:],
                    "transport": "stdio",
                }
                logger.debug(f"Configured Docker MCP server '{server_name}' for tool loading")
            
            elif server_type == "local":
                # Handle local MCP servers (e.g., filesystem)
                command = mcp_config.get("command")
                if not command:
                    logger.warning(f"MCP server '{server_name}' has no command configured, skipping")
                    continue
                
                # Resolve command (can be string or list)
                if isinstance(command, str):
                    cmd_list = [command]
                else:
                    cmd_list = list(command)
                
                # Resolve workspace path in command arguments
                resolved_cmd = []
                for arg in cmd_list:
                    if isinstance(arg, str):
                        # Replace workspace placeholder
                        if "{{workspace.base_path}}" in arg:
                            arg = arg.replace("{{workspace.base_path}}", str(workspace_base))
                        # Expand user home directory
                        if arg.startswith("~"):
                            arg = str(Path(arg).expanduser())
                    resolved_cmd.append(arg)
                
                # Resolve environment variables
                env_vars = mcp_config.get("environment", {})
                resolved_env: Dict[str, str] = {}
                for key, value in env_vars.items():
                    if isinstance(value, str):
                        # Replace workspace placeholder
                        if "{{workspace.base_path}}" in value:
                            resolved_env[key] = value.replace("{{workspace.base_path}}", str(workspace_base))
                        # Support {{env:VAR_NAME}} syntax
                        elif value.startswith("{{env:") and value.endswith("}}"):
                            env_var_name = value[6:-2]
                            resolved_env[key] = os.getenv(env_var_name, "") or ""
                        else:
                            resolved_env[key] = value
                    else:
                        resolved_env[key] = str(value)
                
                # Configure for MultiServerMCPClient
                mcp_configs[server_name] = {
                    "command": resolved_cmd[0],
                    "args": resolved_cmd[1:] if len(resolved_cmd) > 1 else [],
                    "transport": "stdio",
                    "env": resolved_env if resolved_env else None,
                }
                logger.debug(f"Configured local MCP server '{server_name}' for tool loading")

        if not mcp_configs:
            logger.warning("No enabled MCP servers configured for langchain-mcp-adapters")
            return []

        logger.info(
            f"Querying {len(mcp_configs)} MCP server(s) for tools: {list(mcp_configs.keys())}"
        )

        # Use MultiServerMCPClient (official pattern)
        # This creates connections to all servers and queries tools dynamically
        client = MultiServerMCPClient(mcp_configs)
        tools = await client.get_tools()

        logger.info(
            f"Loaded {len(tools)} LangChain tools from {len(mcp_configs)} MCP server(s)",
            extra={"servers": list(mcp_configs.keys()), "tools_count": len(tools)},
        )

        return tools

