"""Filesystem operations tool.

CONTRACT:
- Purpose: Structured file and directory operations with explicit path validation
- Allowed: Read/write files, list directories, create/delete files
- Forbidden: Execution semantics, system introspection, network I/O
- Security: Path validation is mandatory, no silent failures, no symlink escapes

See docs/tools/filesystem.md for full contract documentation.
"""

import os
from pathlib import Path
from typing import Any

from agent.runtime.schema import OperationNotSupportedError
from agent.tools.base import Tool, ToolResult


class FilesystemTool(Tool):
    """Tool for filesystem operations.
    
    CONTRACT ENFORCEMENT:
    - Path validation is mandatory for all operations
    - No execution semantics (use shell tool for commands)
    - No system introspection (use system tool for system info)
    - All mutations are explicit and traceable
    
    See docs/tools/filesystem.md for full contract.
    """

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
        
        # INTERNAL ASSERTION: Filesystem tool must have path validation configured
        # This ensures security boundaries are explicit, not implicit
        if not self.allowed_paths and not self.restricted_paths:
            import warnings
            warnings.warn(
                "Filesystem tool has no path restrictions configured. "
                "This is a security risk. Configure allowed_paths or restricted_paths.",
                UserWarning
            )

    def _check_path(self, path: Path) -> bool:
        """Check if path is allowed.

        Args:
            path: Path to check

        Returns:
            True if allowed, False otherwise
        """
        resolved = path.resolve()

        # Check allowed paths first (allowed_paths take priority over restricted_paths)
        if self.allowed_paths:
            for allowed in self.allowed_paths:
                # Path is allowed if it's exactly the allowed path or a subdirectory
                if resolved == allowed or resolved.is_relative_to(allowed):
                    # Double-check it's not in a restricted path (unless the allowed path itself is in restricted)
                    # This handles cases where allowed_paths are subdirectories of restricted_paths
                    is_restricted = False
                    for restricted in self.restricted_paths:
                        # Only block if the path is in restricted AND not in any allowed path
                        if resolved.is_relative_to(restricted) and not any(
                            allowed_path.is_relative_to(restricted) for allowed_path in self.allowed_paths
                        ):
                            is_restricted = True
                            break
                    if not is_restricted:
                        return True

        # If no allowed_paths specified, check restricted paths
        if not self.allowed_paths:
            for restricted in self.restricted_paths:
                if resolved.is_relative_to(restricted):
                    return False
            return True

        return False

    async def execute(self, operation: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute filesystem operation.

        CONTRACT ENFORCEMENT:
        - Filesystem tool MUST NOT support execution operations
        - Path validation is mandatory and non-negotiable
        - No silent fallback to other tools

        Args:
            operation: Operation type (read_file, write_file, list_directory, create_file, delete_file)
            arguments: Operation arguments dict with 'path' and operation-specific keys

        Returns:
            Tool execution result

        Raises:
            OperationNotSupportedError: If operation implies execution or is invalid
        """
        if not self.enabled:
            return ToolResult(success=False, output=None, error="Filesystem tool is disabled")

        # CONTRACT ASSERTION: Filesystem tool MUST NOT support execution operations
        if "execute" in operation.lower() or "command" in operation.lower() or "run" in operation.lower():
            raise OperationNotSupportedError(
                self.name, operation
            )  # Error message: "Filesystem tool does not support execution operations. Use shell tool for commands."

        # CONTRACT ASSERTION: Filesystem tool MUST NOT support system introspection
        if "system" in operation.lower() or "info" in operation.lower() or "status" in operation.lower():
            raise OperationNotSupportedError(
                self.name, operation
            )  # Error message: "Filesystem tool does not support system introspection. Use system tool for system info."

        # Extract path from arguments
        path = arguments.get("path")
        if not path:
            return ToolResult(
                success=False, output=None, error="Missing required argument: 'path'"
            )

        file_path = Path(path).expanduser().resolve()

        # CONTRACT ENFORCEMENT: Path validation is mandatory
        if not self._check_path(file_path):
            # Show both original and resolved path for debugging
            allowed_str = ", ".join([str(p) for p in self.allowed_paths])
            return ToolResult(
                success=False,
                output=None,
                error=f"Path not allowed: {path} (resolved: {file_path}). Allowed paths: {allowed_str}"
            )

        try:
            # Map operation names from schema to internal methods
            if operation == "read_file":
                return await self._read(file_path)
            elif operation == "write_file":
                content = arguments.get("content", "")
                return await self._write(file_path, content)
            elif operation == "list_directory":
                return await self._list(file_path)
            elif operation == "create_file":
                return await self._create(file_path, arguments.get("is_dir", False))
            elif operation == "delete_file":
                return await self._delete(file_path)
            else:
                raise OperationNotSupportedError(self.name, operation)
        except OperationNotSupportedError:
            raise
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
        try:
            if not path.exists():
                return ToolResult(
                    success=False, output=None, error=f"Path does not exist: {path}"
                )
            if not path.is_dir():
                return ToolResult(
                    success=False, output=None, error=f"Path is not a directory: {path}"
                )
            
            entries = []
            for entry in path.iterdir():
                entry_info = {
                    "name": entry.name,
                    "path": str(entry),
                    "is_directory": entry.is_dir(),
                    "is_file": entry.is_file(),
                }
                if entry.is_file():
                    entry_info["size"] = entry.stat().st_size
                entries.append(entry_info)
            
            # Sort: directories first, then files, both alphabetically
            entries.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))
            
            return ToolResult(success=True, output={"entries": entries, "path": str(path)})
        except PermissionError as e:
            return ToolResult(success=False, output=None, error=f"Permission denied: {e}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Failed to list directory: {e}")

    async def _create(self, path: Path, is_dir: bool) -> ToolResult:
        """Create file or directory."""
        # TODO: Implement file/directory creation
        raise NotImplementedError("File/directory creation not yet implemented")

    async def _delete(self, path: Path) -> ToolResult:
        """Delete file or directory."""
        # TODO: Implement file/directory deletion
        raise NotImplementedError("File/directory deletion not yet implemented")

