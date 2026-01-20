"""Configuration loader for agent settings."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class WorkspaceConfig(BaseModel):
    """Workspace configuration."""

    base_path: str = Field(default="~/repos", description="Base path for repositories")
    persistent: bool = Field(default=True, description="Use persistent workspace")
    cleanup_on_exit: bool = Field(default=False, description="Cleanup on exit")


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = Field(default="ollama", description="LLM provider name (only 'ollama' is supported)")
    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    model: str = Field(default="gpt-3.5-turbo", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Max tokens")
    timeout: int = Field(default=300, gt=0, description="Request timeout in seconds")
    
    
    class Config:
        """Pydantic config to allow extra fields."""
        extra = "allow"  # Allow extra fields from YAML for provider-specific config


class HumanInTheLoopConfig(BaseModel):
    """Human-in-the-Loop configuration."""

    enabled: bool = Field(default=False, description="Enable HITL approval workflow")


class RuntimeConfig(BaseModel):
    """Runtime execution configuration."""

    max_iterations: int = Field(default=50, gt=0, description="Max planning iterations")
    timeout_seconds: int = Field(default=3600, gt=0, description="Total timeout")
    enable_parallel_tools: bool = Field(default=False, description="Enable parallel tool execution")
    safety_checks: bool = Field(default=True, description="Enable safety checks")
    auto_execution: bool = Field(default=True, description="Enable automatic tool execution (OpenCode-style)")
    continuous_loop: bool = Field(default=True, description="Enable continuous loop for auto-correction (OpenCode-style)")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format")
    output_path: str = Field(default="./workspace/logs", description="Log output path")


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) server configuration."""

    type: str = Field(..., description="MCP server type: 'local', 'remote', or 'docker'")
    enabled: bool = Field(default=True, description="Enable this MCP server")
    timeout: int = Field(default=5000, description="Connection timeout in milliseconds")
    
    # Local server config
    command: list[str] | None = Field(default=None, description="Command to run local MCP server")
    environment: Dict[str, str] | None = Field(default=None, description="Environment variables for local server")
    
    # Remote server config
    url: str | None = Field(default=None, description="URL for remote MCP server")
    headers: Dict[str, str] | None = Field(default=None, description="HTTP headers for remote server")
    oauth: bool | Dict[str, Any] = Field(default=True, description="OAuth configuration (True/False or config dict)")
    
    # Docker server config
    image: str | None = Field(default=None, description="Docker image name for docker type")
    volumes: list[str] | None = Field(default=None, description="Volume mounts for docker type")
    args: list[str] | None = Field(default=None, description="Arguments for docker container")
    environment: Dict[str, str] | None = Field(default=None, description="Environment variables for Docker container (e.g., GITHUB_TOKEN)")


class AgentConfig(BaseModel):
    """Main agent configuration."""

    name: str = Field(default="forge-agent", description="Agent name")
    version: str = Field(default="0.1.0", description="Agent version")
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging settings")
    human_in_the_loop: HumanInTheLoopConfig = Field(
        default_factory=HumanInTheLoopConfig, description="Human-in-the-Loop configuration"
    )
    mcp: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="MCP server configurations"
    )
    system_prompt_template: str | None = Field(
        default=None,
        description="System prompt template with placeholders: {workspace_base}, {tools_count}, {all_mcp_tools}"
    )


class ConfigLoader:
    """Loads and validates agent configuration from YAML files."""

    def __init__(self, config_path: str | None = None):
        """Initialize config loader.

        Args:
            config_path: Path to config file. If None, uses CONFIG_PATH env var or default.
        """
        if config_path is None:
            # Try to find a default config file
            default_path = os.getenv("CONFIG_PATH")
            if default_path:
                config_path = default_path
            else:
                # Try agent.ollama.yaml first, then agent.yaml
                config_dir = Path("config")
                if (config_dir / "agent.ollama.yaml").exists():
                    config_path = "config/agent.ollama.yaml"
                else:
                    config_path = "config/agent.yaml"

        self.config_path = Path(config_path).expanduser().resolve()

    def load(self) -> AgentConfig:
        """Load configuration from file.

        Returns:
            Validated agent configuration.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config is invalid.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        if not raw_config or "agent" not in raw_config:
            raise ValueError("Invalid config: missing 'agent' section")

        agent_config = raw_config["agent"]
        
        # Ensure mcp is always a dict (not None) if present but empty
        if "mcp" in agent_config and agent_config["mcp"] is None:
            agent_config["mcp"] = {}
        
        return AgentConfig(**agent_config)

