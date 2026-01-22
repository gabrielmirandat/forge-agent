"""Dependency providers for FastAPI application.

All dependencies are singleton-scoped and configured at startup.
"""

import asyncio
from functools import lru_cache
from typing import Optional

from fastapi import Depends

from agent.config.loader import AgentConfig, ConfigLoader
from agent.llm.base import LLMProvider
from agent.llm.ollama import OllamaProvider
from agent.storage import Storage
from agent.storage.json_storage import JSONStorage
from agent.tools.base import ToolRegistry


_config_cache: AgentConfig | None = None


def get_config() -> AgentConfig:
    """Get agent configuration (singleton).

    Returns:
        AgentConfig instance
    """
    global _config_cache
    if _config_cache is None:
        loader = ConfigLoader()
        _config_cache = loader.load()
    return _config_cache


# Global LLM provider cache (can be cleared to force reload)
_llm_provider_cache: Optional[LLMProvider] = None


def get_llm_provider(config: AgentConfig = Depends(get_config)) -> LLMProvider:
    """Get LLM provider (singleton, can be reloaded at runtime).

    Args:
        config: Agent configuration (injected)

    Returns:
        LLMProvider instance (OllamaProvider)
    """
    global _llm_provider_cache
    
    # Check if we need to reload (cache cleared or config changed)
    if _llm_provider_cache is None:
        provider_name = config.llm.provider.lower()
        
        # Convert LLMConfig to dict format expected by providers
        llm_config_dict = {
            "model": config.llm.model,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
            "timeout": config.llm.timeout,
        }
        
        # Only Ollama is supported
        if provider_name != "ollama":
            raise ValueError(
                f"Unsupported LLM provider: {provider_name}. "
                "Only 'ollama' is supported. "
                "Configure provider: ollama in your config file."
            )
        
        llm_config_dict["base_url"] = config.llm.base_url
        _llm_provider_cache = OllamaProvider(llm_config_dict)
    
    return _llm_provider_cache


# Global tool registry singleton
_tool_registry: Optional[ToolRegistry] = None


async def initialize_tool_registry(config: AgentConfig) -> ToolRegistry:
    """Initialize tool registry (called at startup).

    Args:
        config: Agent configuration

    Returns:
        ToolRegistry with all tools registered
    """
    global _tool_registry
    
    if _tool_registry is not None:
        return _tool_registry
    
    registry = ToolRegistry()

    # All tools come from MCP servers (Docker containers)
    # Local tools have been removed - we use only MCP-standard tools
    # MCP tools are registered as classes and converted to LangChain tools via adapter
    await _register_mcp_tools(registry, config)

    _tool_registry = registry
    return registry


def get_tool_registry() -> ToolRegistry:
    """Get tool registry (singleton, must be initialized first).

    Returns:
        ToolRegistry with all tools registered

    Raises:
        RuntimeError: If registry not initialized
    """
    global _tool_registry
    if _tool_registry is None:
        raise RuntimeError("Tool registry not initialized. Call initialize_tool_registry() first.")
    return _tool_registry


async def _register_mcp_tools(registry: ToolRegistry, config: AgentConfig):
    """Register MCP server tools.

    Args:
        registry: Tool registry
        config: Agent configuration
    """
    from agent.runtime.mcp_client import get_mcp_manager
    from agent.tools.mcp_tool import MCPTool
    from agent.observability import get_logger

    logger = get_logger("mcp", "dependencies")
    mcp_manager = get_mcp_manager()
    mcp_configs = config.mcp or {}

    for mcp_name, mcp_config in mcp_configs.items():
        # Skip if disabled
        if mcp_config.get("enabled") is False:
            logger.info(f"MCP server {mcp_name} is disabled, skipping")
            continue
        
        try:
            # Resolve workspace path for all server types
            if mcp_config.get("type") == "docker":
                logger.info(f"MCP server {mcp_name} is Docker-type, connecting via MCPClient")
                mcp_config_resolved = mcp_config.copy()
                # Pass workspace_path to MCPClient for Docker volume resolution
                mcp_config_resolved["workspace_path"] = config.workspace.base_path
                
                # Resolve workspace path in volumes
                if "volumes" in mcp_config_resolved:
                    from pathlib import Path
                    workspace_base = str(Path(config.workspace.base_path).expanduser().resolve())
                    resolved_volumes = []
                    for volume in mcp_config_resolved["volumes"]:
                        if "{{workspace.base_path}}" in volume:
                            volume = volume.replace("{{workspace.base_path}}", workspace_base)
                        if volume.startswith("~"):
                            volume = str(Path(volume).expanduser())
                        resolved_volumes.append(volume)
                    mcp_config_resolved["volumes"] = resolved_volumes
            else:
                # Resolve workspace path placeholder in command if present
                mcp_config_resolved = mcp_config.copy()
                # Pass workspace_path to MCPClient for local servers
                mcp_config_resolved["workspace_path"] = config.workspace.base_path if hasattr(config, "workspace") else "~/repos"
                
                if "command" in mcp_config_resolved:
                    workspace_base = config.workspace.base_path if hasattr(config, "workspace") else "~/repos"
                    from pathlib import Path
                    workspace_base = str(Path(workspace_base).expanduser())
                    
                    resolved_command = []
                    for arg in mcp_config_resolved["command"]:
                        if isinstance(arg, str):
                            # Replace workspace placeholder
                            arg = arg.replace("{{workspace.base_path}}", workspace_base)
                            # Expand user home directory
                            if arg.startswith("~"):
                                arg = str(Path(arg).expanduser())
                        resolved_command.append(arg)
                    mcp_config_resolved["command"] = resolved_command
            
            # Add MCP server with timeout to prevent blocking
            success = await asyncio.wait_for(
                mcp_manager.add_server(mcp_name, mcp_config_resolved),
                timeout=10.0  # 10 second timeout per MCP
            )
            if not success:
                logger.warning(f"Failed to register MCP server: {mcp_name}")
                continue
            
            # Get tools from MCP server
            tools = mcp_manager.get_all_tools().get(mcp_name, [])
            
            # Register each tool as a Forge Agent tool
            for mcp_tool in tools:
                tool_wrapper = MCPTool(
                    mcp_name=mcp_name,
                    mcp_tool=mcp_tool,
                    config={"enabled": True},
                )
                registry.register(tool_wrapper)
                
            logger.info(f"MCP server '{mcp_name}' registered with {len(tools)} tools")
            
            # Log if desktop-commander MCP is enabled (provides file operations)
            if mcp_name == "desktop_commander":
                logger.info(
                    "MCP desktop-commander server enabled. Provides file operations and terminal commands."
                )
        except asyncio.TimeoutError:
            logger.warning(f"MCP server {mcp_name} connection timed out. Continuing without it.")
            continue
        except Exception as e:
            logger.warning(f"Failed to register MCP server {mcp_name}: {e}. Continuing without it.")
            continue


# Removed get_planner and get_executor - no longer needed with direct tool calling

_storage_cache: Storage | None = None


def get_storage() -> Storage:
    """Get storage instance (singleton, default: JSON).

    Returns:
        Storage instance
    """
    global _storage_cache
    if _storage_cache is None:
        # Use JSON storage (like OpenCode)
        _storage_cache = JSONStorage("~/.forge-agent/sessions")
    return _storage_cache

