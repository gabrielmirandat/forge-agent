"""Filesystem operations tool with security boundaries.

CONTRACT:
- PRIMARY PURPOSE: Security boundaries and path validation for file operations
- Simple CRUD operations: read, write, list, create, delete files and directories
- Security: Path validation is MANDATORY - enforces allowed_paths and restricted_paths
- For elaborate operations: Use shell commands (head, tail, grep, sed, awk, etc.) for text processing

The filesystem tool enforces security boundaries through path validation. It provides
simple file CRUD operations. For elaborate text processing, filtering, or complex
operations, use shell commands instead.
"""

import os
from pathlib import Path
from typing import Any

from agent.runtime.schema import OperationNotSupportedError
from agent.tools.base import Tool, ToolResult


class FilesystemTool(Tool):
    """Tool for filesystem operations with security boundaries.
    
    PRIMARY PURPOSE: Security boundaries and path validation.
    
    This tool enforces security through mandatory path validation against
    allowed_paths and restricted_paths. It provides simple file operations:
    - read_file: Read file contents
    - write_file: Write content to a file
    - list_directory: List files and directories
    - create_file: Create a new file
    - delete_file: Delete a file
    
    For elaborate operations (text processing, filtering, extraction):
    - Use shell commands: head, tail, grep, sed, awk, cut, sort, etc.
    - Example: To show first 10 lines, use shell.execute_command with "head -n 10"
    
    CONTRACT ENFORCEMENT:
    - SECURITY: Path validation is MANDATORY for all operations
    - Validates paths against allowed_paths and restricted_paths
    - No operation can bypass path validation
    - Simple operations: use filesystem tool
    - Elaborate operations: use shell tool
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
                # create_file can create both files and directories
                # If content is provided, it's a file with content; otherwise check is_dir
                content = arguments.get("content")
                if content is not None:
                    # Create file with content (use write_file logic)
                    return await self._write(file_path, content)
                else:
                    # Create empty file or directory based on is_dir
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
        try:
            if not path.exists():
                return ToolResult(
                    success=False, output=None, error=f"Path does not exist: {path}"
                )
            if not path.is_file():
                return ToolResult(
                    success=False, output=None, error=f"Path is not a file: {path}"
                )
            
            # Read file content
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try binary mode for non-text files
                with open(path, "rb") as f:
                    content = f.read()
                    # For binary files, return a message instead of raw bytes
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"File appears to be binary (not text): {path}",
                    )
            except PermissionError as e:
                return ToolResult(
                    success=False, output=None, error=f"Permission denied: {e}"
                )
            
            # Get file size
            file_size = path.stat().st_size
            
            return ToolResult(
                success=True,
                output={
                    "path": str(path),
                    "content": content,
                    "size": file_size,
                    "lines": len(content.splitlines()) if content else 0,
                },
            )
        except PermissionError as e:
            return ToolResult(success=False, output=None, error=f"Permission denied: {e}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Failed to read file: {e}")

    async def _write(self, path: Path, content: str) -> ToolResult:
        """Write file contents.
        
        Args:
            path: Path to the file
            content: Content to write
            
        Returns:
            ToolResult with write status
        """
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Get file size after writing
            file_size = path.stat().st_size
            
            return ToolResult(
                success=True,
                output={
                    "path": str(path),
                    "size": file_size,
                    "lines": len(content.splitlines()) if content else 0,
                },
            )
        except PermissionError as e:
            return ToolResult(success=False, output=None, error=f"Permission denied: {e}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Failed to write file: {e}")

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
        """Create file or directory.
        
        Args:
            path: Path to create
            is_dir: If True, create directory; if False, create empty file
            
        Returns:
            ToolResult with creation status
        """
        try:
            # Check if path already exists
            if path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Path already exists: {path}",
                )
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if is_dir:
                # Create directory
                path.mkdir(parents=True, exist_ok=True)
                return ToolResult(
                    success=True,
                    output={"path": str(path), "type": "directory"},
                )
            else:
                # Create empty file
                path.touch()
                return ToolResult(
                    success=True,
                    output={"path": str(path), "type": "file", "size": 0},
                )
        except PermissionError as e:
            return ToolResult(success=False, output=None, error=f"Permission denied: {e}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Failed to create: {e}")

    async def _delete(self, path: Path) -> ToolResult:
        """Delete file or directory."""
        # TODO: Implement file/directory deletion
        raise NotImplementedError("File/directory deletion not yet implemented")

