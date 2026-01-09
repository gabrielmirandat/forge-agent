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

    provider: str = Field(default="localai", description="LLM provider name")
    base_url: str = Field(default="http://localhost:8080", description="LLM API base URL")
    model: str = Field(default="gpt-3.5-turbo", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Max tokens")
    timeout: int = Field(default=300, gt=0, description="Request timeout in seconds")


class HumanInTheLoopConfig(BaseModel):
    """Human-in-the-Loop configuration."""

    enabled: bool = Field(default=False, description="Enable HITL approval workflow")


class RuntimeConfig(BaseModel):
    """Runtime execution configuration."""

    max_iterations: int = Field(default=50, gt=0, description="Max planning iterations")
    timeout_seconds: int = Field(default=3600, gt=0, description="Total timeout")
    enable_parallel_tools: bool = Field(default=False, description="Enable parallel tool execution")
    safety_checks: bool = Field(default=True, description="Enable safety checks")


class AgentConfig(BaseModel):
    """Main agent configuration."""

    name: str = Field(default="forge-agent", description="Agent name")
    version: str = Field(default="0.1.0", description="Agent version")
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
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

