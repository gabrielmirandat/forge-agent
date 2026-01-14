"""Unit tests for filesystem tool."""

import pytest
import tempfile
from pathlib import Path

from agent.tools.filesystem import FilesystemTool
from agent.runtime.schema import OperationNotSupportedError
from agent.tools.base import ToolResult


class TestFilesystemTool:
    """Test FilesystemTool."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def tool(self, temp_dir):
        """Create filesystem tool with temp directory as allowed path."""
        return FilesystemTool({
            "enabled": True,
            "allowed_paths": [str(temp_dir)],
            "restricted_paths": [],
        })

    def test_tool_name(self, tool):
        """Test tool name."""
        assert tool.name == "filesystem"

    def test_tool_description(self, tool):
        """Test tool description."""
        assert "file" in tool.description.lower()

    def test_read_file_allowed_path(self, tool, temp_dir):
        """Test reading file from allowed path."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        import asyncio
        result = asyncio.run(tool.execute("read_file", {"path": str(test_file)}))

        assert result.success is True
        assert result.output == "test content"

    def test_read_file_restricted_path(self, tool):
        """Test reading file from restricted path."""
        import asyncio
        result = asyncio.run(tool.execute("read_file", {"path": "/etc/passwd"}))

        assert result.success is False
        assert "not allowed" in result.error.lower() or "restricted" in result.error.lower()

    def test_write_file_allowed_path(self, tool, temp_dir):
        """Test writing file to allowed path."""
        test_file = temp_dir / "write_test.txt"

        import asyncio
        result = asyncio.run(tool.execute("write_file", {
            "path": str(test_file),
            "content": "test content"
        }))

        assert result.success is True
        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_list_directory_allowed_path(self, tool, temp_dir):
        """Test listing directory in allowed path."""
        (temp_dir / "file1.txt").touch()
        (temp_dir / "file2.txt").touch()
        (temp_dir / "subdir").mkdir()

        import asyncio
        result = asyncio.run(tool.execute("list_directory", {"path": str(temp_dir)}))

        assert result.success is True
        assert "entries" in result.output
        entries = result.output["entries"]
        assert len(entries) >= 2  # At least file1 and file2

    def test_operation_not_supported(self, tool):
        """Test unsupported operation."""
        import asyncio
        with pytest.raises(OperationNotSupportedError):
            asyncio.run(tool.execute("invalid_operation", {}))

    def test_tool_disabled(self):
        """Test disabled tool."""
        tool = FilesystemTool({"enabled": False, "allowed_paths": []})
        import asyncio
        result = asyncio.run(tool.execute("read_file", {"path": "/tmp/test.txt"}))
        assert result.success is False
        assert "disabled" in result.error.lower()
