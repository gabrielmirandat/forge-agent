"""Dependency providers for FastAPI application.

All dependencies are singleton-scoped and configured at startup.
"""

from functools import lru_cache

from fastapi import Depends

from agent.config.loader import AgentConfig, ConfigLoader
from agent.llm.airllm import AirLLMProvider
from agent.llm.base import LLMProvider
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


def get_llm_provider(config: AgentConfig = Depends(get_config)) -> LLMProvider:
    """Get LLM provider (singleton).

    Args:
        config: Agent configuration (injected)

    Returns:
        LLMProvider instance (OllamaProvider, AirLLMProvider, or LocalAIProvider)
    """
    provider_name = config.llm.provider.lower()
    
    # Convert LLMConfig to dict format expected by providers
    llm_config_dict = {
        "model": config.llm.model,
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens,
        "timeout": config.llm.timeout,
    }
    
    # Add provider-specific config
    if provider_name == "ollama":
        llm_config_dict["base_url"] = config.llm.base_url
        return OllamaProvider(llm_config_dict)
    elif provider_name == "airllm":
        # AirLLM-specific config (from config.llm dict if present)
        llm_config_raw = config.llm.model_dump()
        if "compression" in llm_config_raw:
            llm_config_dict["compression"] = llm_config_raw["compression"]
        if "hf_token" in llm_config_raw:
            llm_config_dict["hf_token"] = llm_config_raw["hf_token"]
        if "profiling_mode" in llm_config_raw:
            llm_config_dict["profiling_mode"] = llm_config_raw["profiling_mode"]
        if "layer_shards_saving_path" in llm_config_raw:
            llm_config_dict["layer_shards_saving_path"] = llm_config_raw["layer_shards_saving_path"]
        if "delete_original" in llm_config_raw:
            llm_config_dict["delete_original"] = llm_config_raw["delete_original"]
        return AirLLMProvider(llm_config_dict)
    elif provider_name == "localai":
        llm_config_dict["base_url"] = config.llm.base_url
        from agent.llm.localai import LocalAIProvider
        return LocalAIProvider(llm_config_dict)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}. Supported: ollama, airllm, localai")


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
    # Git tool needs access to filesystem allowed_paths for path validation
    git_config = {
        **workspace_config,
        **tools_config.get("git", {}),
        "allowed_paths": filesystem_config.get("allowed_paths", []),
        "restricted_paths": filesystem_config.get("restricted_paths", []),
    }
    github_config = {**workspace_config, **tools_config.get("github", {})}
    shell_config = {**workspace_config, **tools_config.get("shell", {})}
    # System tool needs access to full agent config for status info
    system_config = {**workspace_config, **tools_config.get("system", {}), "_agent_config": config}

    registry.register(FilesystemTool(filesystem_config))
    registry.register(GitTool(git_config))
    registry.register(GitHubTool(github_config))
    registry.register(ShellTool(shell_config))
    registry.register(SystemTool(system_config))

    return registry


def get_planner(
    config: AgentConfig = Depends(get_config),
    llm_provider: LLMProvider = Depends(get_llm_provider),
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

