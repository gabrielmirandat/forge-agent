"""Filesystem operations tool."""

import os
from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolResult


class FilesystemTool(Tool):
    """Tool for filesystem operations."""

    @property
    def name(self) -> str:
        """Return tool name."""
        return "filesystem"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "Read, write, and manage files and directories"

    def __init__(self, config: dict[str, Any]):
        """Initialize filesystem tool.

        Args:
            config: Tool configuration with allowed_paths and restricted_paths
        """
        super().__init__(config)
        self.allowed_paths = [Path(p).expanduser().resolve() for p in config.get("allowed_paths", [])]
        self.restricted_paths = [Path(p).expanduser().resolve() for p in config.get("restricted_paths", [])]

    def _check_path(self, path: Path) -> bool:
        """Check if path is allowed.

        Args:
            path: Path to check

        Returns:
            True if allowed, False otherwise
        """
        resolved = path.resolve()

        # Check restricted paths first
        for restricted in self.restricted_paths:
            if resolved.is_relative_to(restricted):
                return False

        # Check allowed paths
        if not self.allowed_paths:
            return True

        for allowed in self.allowed_paths:
            if resolved.is_relative_to(allowed):
                return True

        return False

    async def execute(self, operation: str, path: str, **kwargs: Any) -> ToolResult:
        """Execute filesystem operation.

        Args:
            operation: Operation type (read, write, list, create, delete)
            path: File or directory path
            **kwargs: Additional operation-specific parameters

        Returns:
            Tool execution result
        """
        if not self.enabled:
            return ToolResult(success=False, output=None, error="Filesystem tool is disabled")

        file_path = Path(path).expanduser().resolve()

        if not self._check_path(file_path):
            return ToolResult(
                success=False, output=None, error=f"Path not allowed: {path}"
            )

        try:
            if operation == "read":
                return await self._read(file_path)
            elif operation == "write":
                content = kwargs.get("content", "")
                return await self._write(file_path, content)
            elif operation == "list":
                return await self._list(file_path)
            elif operation == "create":
                return await self._create(file_path, kwargs.get("is_dir", False))
            elif operation == "delete":
                return await self._delete(file_path)
            else:
                return ToolResult(success=False, output=None, error=f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    async def _read(self, path: Path) -> ToolResult:
        """Read file contents."""
        # TODO: Implement file reading
        raise NotImplementedError("File reading not yet implemented")

    async def _write(self, path: Path, content: str) -> ToolResult:
        """Write file contents."""
        # TODO: Implement file writing
        raise NotImplementedError("File writing not yet implemented")

    async def _list(self, path: Path) -> ToolResult:
        """List directory contents."""
        # TODO: Implement directory listing
        raise NotImplementedError("Directory listing not yet implemented")

    async def _create(self, path: Path, is_dir: bool) -> ToolResult:
        """Create file or directory."""
        # TODO: Implement file/directory creation
        raise NotImplementedError("File/directory creation not yet implemented")

    async def _delete(self, path: Path) -> ToolResult:
        """Delete file or directory."""
        # TODO: Implement file/directory deletion
        raise NotImplementedError("File/directory deletion not yet implemented")

