"""Unit tests for configuration loader."""

import pytest
import tempfile
import yaml
from pathlib import Path

from agent.config.loader import (
    AgentConfig,
    ConfigLoader,
    LLMConfig,
    ToolsConfig,
    WorkspaceConfig,
    RuntimeConfig,
    LoggingConfig,
    HumanInTheLoopConfig,
)


class TestConfigLoader:
    """Test configuration loading."""

    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "agent": {
                    "name": "test-agent",
                    "version": "1.0.0",
                    "workspace": {"base_path": "~/test-repos"},
                    "llm": {
                        "provider": "ollama",
                        "base_url": "http://localhost:11434",
                        "model": "test-model",
                    },
                    "tools": {
                        "filesystem": {"allowed_paths": ["~/test"]},
                    },
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            loader = ConfigLoader(config_path)
            config = loader.load()

            assert config.name == "test-agent"
            assert config.version == "1.0.0"
            assert config.workspace.base_path == "~/test-repos"
            assert config.llm.provider == "ollama"
            assert config.llm.model == "test-model"
        finally:
            Path(config_path).unlink()

    def test_load_missing_file(self):
        """Test loading a non-existent configuration file."""
        loader = ConfigLoader("/nonexistent/path/config.yaml")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_invalid_config_missing_agent_section(self):
        """Test loading config without 'agent' section."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"not_agent": {}}, f)
            config_path = f.name

        try:
            loader = ConfigLoader(config_path)
            with pytest.raises(ValueError, match="missing 'agent' section"):
                loader.load()
        finally:
            Path(config_path).unlink()

    def test_load_empty_config(self):
        """Test loading empty config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            config_path = f.name

        try:
            loader = ConfigLoader(config_path)
            with pytest.raises(ValueError, match="missing 'agent' section"):
                loader.load()
        finally:
            Path(config_path).unlink()

    def test_default_config_values(self):
        """Test that default values are used when not specified."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {"agent": {}}
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            loader = ConfigLoader(config_path)
            config = loader.load()

            assert config.name == "forge-agent"
            assert config.version == "0.1.0"
            assert config.workspace.base_path == "~/repos"
            assert config.llm.provider == "localai"
            assert config.llm.model == "gpt-3.5-turbo"
        finally:
            Path(config_path).unlink()

    def test_config_path_expansion(self):
        """Test that config path with ~ is expanded."""
        import os
        home = os.path.expanduser("~")
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=home, delete=False) as f:
            config_data = {"agent": {"name": "test"}}
            yaml.dump(config_data, f)
            config_path = f"~/{Path(f.name).name}"

        try:
            loader = ConfigLoader(config_path)
            assert str(loader.config_path).startswith(home)
        finally:
            Path(f.name).unlink()


class TestAgentConfig:
    """Test AgentConfig model."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = AgentConfig()
        assert config.name == "forge-agent"
        assert config.version == "0.1.0"
        assert isinstance(config.workspace, WorkspaceConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.tools, ToolsConfig)
        assert isinstance(config.runtime, RuntimeConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.human_in_the_loop, HumanInTheLoopConfig)

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = AgentConfig(
            name="custom-agent",
            version="2.0.0",
            workspace=WorkspaceConfig(base_path="/custom/path"),
        )
        assert config.name == "custom-agent"
        assert config.version == "2.0.0"
        assert config.workspace.base_path == "/custom/path"


class TestLLMConfig:
    """Test LLMConfig model."""

    def test_default_llm_config(self):
        """Test default LLM config values."""
        config = LLMConfig()
        assert config.provider == "localai"
        assert config.base_url == "http://localhost:8080"
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 300

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperature
        config = LLMConfig(temperature=0.5)
        assert config.temperature == 0.5

        # Invalid: too low
        with pytest.raises(Exception):  # pydantic validation
            LLMConfig(temperature=-1.0)

        # Invalid: too high
        with pytest.raises(Exception):  # pydantic validation
            LLMConfig(temperature=3.0)


class TestToolsConfig:
    """Test ToolsConfig model."""

    def test_default_tools_config(self):
        """Test default tools config."""
        config = ToolsConfig()
        assert config.filesystem["enabled"] is True
        assert "allowed_paths" in config.filesystem
        assert config.shell["enabled"] is True
        assert "allowed_commands" in config.shell
        assert config.system["enabled"] is True

    def test_custom_tools_config(self):
        """Test custom tools config."""
        config = ToolsConfig(
            filesystem={"enabled": False, "allowed_paths": ["/custom"]}
        )
        assert config.filesystem["enabled"] is False
        assert config.filesystem["allowed_paths"] == ["/custom"]
