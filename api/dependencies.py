"""Dependency providers for FastAPI application.

All dependencies are singleton-scoped and configured at startup.
"""

from functools import lru_cache

from fastapi import Depends

from agent.config.loader import AgentConfig, ConfigLoader
from agent.llm.ollama import OllamaProvider
from agent.runtime.executor import Executor
from agent.runtime.planner import Planner
from agent.storage import SQLiteStorage, Storage
from agent.tools.base import ToolRegistry
from agent.tools.filesystem import FilesystemTool
from agent.tools.git import GitTool
from agent.tools.github import GitHubTool
from agent.tools.shell import ShellTool
from agent.tools.system import SystemTool


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


def get_llm_provider(config: AgentConfig = Depends(get_config)) -> OllamaProvider:
    """Get LLM provider (singleton).

    Args:
        config: Agent configuration (injected)

    Returns:
        OllamaProvider instance
    """
    # Convert LLMConfig to dict format expected by OllamaProvider
    llm_config_dict = {
        "base_url": config.llm.base_url,
        "model": config.llm.model,
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens,
        "timeout": config.llm.timeout,
    }
    return OllamaProvider(llm_config_dict)


def get_tool_registry(config: AgentConfig = Depends(get_config)) -> ToolRegistry:
    """Get tool registry (singleton, configured at startup).

    Args:
        config: Agent configuration (injected)

    Returns:
        ToolRegistry with all tools registered
    """
    registry = ToolRegistry()

    # Get tool configurations (with security settings)
    tools_config = config.tools.model_dump()
    workspace_config = config.workspace.model_dump()

    # Register all tools with their security configurations
    # Merge workspace config with tool-specific config
    filesystem_config = {**workspace_config, **tools_config.get("filesystem", {})}
    git_config = {**workspace_config, **tools_config.get("git", {})}
    github_config = {**workspace_config, **tools_config.get("github", {})}
    shell_config = {**workspace_config, **tools_config.get("shell", {})}
    system_config = {**workspace_config, **tools_config.get("system", {})}

    registry.register(FilesystemTool(filesystem_config))
    registry.register(GitTool(git_config))
    registry.register(GitHubTool(github_config))
    registry.register(ShellTool(shell_config))
    registry.register(SystemTool(system_config))

    return registry


def get_planner(
    config: AgentConfig = Depends(get_config),
    llm_provider: OllamaProvider = Depends(get_llm_provider),
) -> Planner:
    """Get planner instance.

    Args:
        config: Agent configuration (injected)
        llm_provider: LLM provider (injected)

    Returns:
        Planner instance
    """
    return Planner(config, llm_provider)


def get_executor(
    config: AgentConfig = Depends(get_config),
    tool_registry: ToolRegistry = Depends(get_tool_registry),
) -> Executor:
    """Get executor instance.

    Args:
        config: Agent configuration (injected)
        tool_registry: Tool registry (injected)

    Returns:
        Executor instance
    """
    return Executor(config, tool_registry)


_storage_cache: Storage | None = None


def get_storage() -> Storage:
    """Get storage instance (singleton, default: SQLite).

    Returns:
        Storage instance
    """
    global _storage_cache
    if _storage_cache is None:
        # Default to SQLite storage
        # In future, this could be config-driven
        _storage_cache = SQLiteStorage("forge_agent.db")
    return _storage_cache

