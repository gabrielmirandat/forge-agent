"""Native shell / filesystem helpers that complement MCP tools.

These tools handle operations that the MCP filesystem server does not support
(e.g. recursive directory deletion).  They are registered as native LangChain
StructuredTools alongside the MCP tools.
"""

import shutil
from pathlib import Path
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolContext, ToolResult


# ── config-level tool class (for ToolRegistry) ────────────────────────────────

class ShellTool(Tool):
    """Provides lightweight filesystem helpers: delete_directory."""

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return "Lightweight filesystem helpers (delete directory, etc.)"

    async def execute(
        self,
        operation: str,
        arguments: dict,
        ctx: Optional[ToolContext] = None,
    ) -> ToolResult:
        if operation == "delete_directory":
            path = arguments.get("path", "")
            try:
                p = Path(path).resolve()
                if not p.exists():
                    return ToolResult(success=True, output=f"Path {path} does not exist (nothing to delete)")
                if not p.is_dir():
                    return ToolResult(success=False, output="", error=f"{path} is not a directory")
                shutil.rmtree(p)
                return ToolResult(success=True, output=f"Deleted {path}")
            except Exception as exc:
                return ToolResult(success=False, output="", error=str(exc))
        return ToolResult(success=False, output="", error=f"Unknown operation: {operation}")


# ── LangChain tool factory ────────────────────────────────────────────────────

class _DeleteDirInput(BaseModel):
    path: str = Field(..., description="Absolute host path to the directory to delete recursively (e.g. /home/user/repos/my-project)")


def _delete_directory_sync(path: str) -> str:
    import subprocess
    p = Path(path).resolve()
    if not p.exists():
        return f"Path {path} does not exist (nothing to delete)"
    if not p.is_dir():
        return f"Error: {path} is not a directory"
    try:
        shutil.rmtree(p)
        return f"Deleted {path} successfully"
    except PermissionError:
        # Files may be root-owned (written by a Docker MCP container).
        # Run a privileged Docker container to perform the deletion.
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{p.parent}:/target",
                "alpine",
                "rm", "-rf", f"/target/{p.name}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return f"Deleted {path} successfully"
        return f"Error deleting {path}: {result.stderr or result.stdout}"


async def _delete_directory_async(path: str) -> str:
    import asyncio
    return await asyncio.get_running_loop().run_in_executor(None, _delete_directory_sync, path)


def create_delete_directory_tool(workspace_base: str = "~/repos") -> StructuredTool:
    """Create a LangChain StructuredTool for recursive directory deletion.

    The tool accepts a HOST-SIDE absolute path (e.g. /home/user/repos/foo).
    Since the filesystem MCP server does not expose a delete/rm capability,
    this native tool fills the gap.
    """
    workspace_resolved = str(Path(workspace_base).expanduser().resolve())

    return StructuredTool.from_function(
        func=_delete_directory_sync,
        coroutine=_delete_directory_async,
        name="delete_directory",
        description=(
            f"Recursively delete a directory and all its contents. "
            f"Use HOST-SIDE absolute paths. "
            f"The user's ~/repos directory is at {workspace_resolved} on the host. "
            f"Example: to delete ~/repos/my-project pass path='{workspace_resolved}/my-project'."
        ),
        args_schema=_DeleteDirInput,
    )
