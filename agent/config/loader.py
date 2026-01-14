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

    provider: str = Field(default="localai", description="LLM provider name (ollama, airllm, localai)")
    base_url: str = Field(default="http://localhost:8080", description="LLM API base URL (for ollama/localai)")
    model: str = Field(default="gpt-3.5-turbo", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Max tokens")
    timeout: int = Field(default=300, gt=0, description="Request timeout in seconds")
    
    # AirLLM-specific optional fields
    compression: str | None = Field(default=None, description="AirLLM compression: '4bit', '8bit', or None")
    hf_token: str | None = Field(default=None, description="HuggingFace token for gated models")
    profiling_mode: bool = Field(default=False, description="AirLLM profiling mode")
    layer_shards_saving_path: str | None = Field(default=None, description="AirLLM layer shards path")
    delete_original: bool = Field(default=False, description="AirLLM: delete original model after splitting")
    
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


class ToolsConfig(BaseModel):
    """Tool security and configuration settings."""

    filesystem: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "allowed_paths": ["~/repos"],
            "restricted_paths": ["/", "/home", "/etc", "/usr", "/var", "/sys", "/proc"],
        },
        description="Filesystem tool configuration with security paths",
    )
    git: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "default_branch_prefix": "agent/",
            "auto_commit": False,
        },
        description="Git tool configuration",
    )
    github: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "base_url": "https://api.github.com",
            "auto_create_pr": False,
        },
        description="GitHub tool configuration",
    )
    shell: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "allowed_commands": ["git", "npm", "python", "python3", "docker", "docker-compose"],
            "restricted_commands": ["rm", "sudo", "chmod", "chown", "mkfs", "fdisk", "dd", "mount", "umount"],
        },
        description="Shell tool configuration with security command restrictions",
    )
    system: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
        },
        description="System tool configuration for read-only system introspection",
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format")
    output_path: str = Field(default="./workspace/logs", description="Log output path")


class AgentConfig(BaseModel):
    """Main agent configuration."""

    name: str = Field(default="forge-agent", description="Agent name")
    version: str = Field(default="0.1.0", description="Agent version")
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tools: ToolsConfig = Field(
        default_factory=ToolsConfig, description="Tool security and configuration (CRITICAL for security)"
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging settings")
    human_in_the_loop: HumanInTheLoopConfig = Field(
        default_factory=HumanInTheLoopConfig, description="Human-in-the-Loop configuration"
    )


class ConfigLoader:
    """Loads and validates agent configuration from YAML files."""

    def __init__(self, config_path: str | None = None):
        """Initialize config loader.

        Args:
            config_path: Path to config file. If None, uses CONFIG_PATH env var or default.
        """
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "config/agent.yaml")

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

        return AgentConfig(**raw_config["agent"])

