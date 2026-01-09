"""Git operations tool."""

from pathlib import Path
from typing import Any

from agent.tools.base import Tool, ToolResult


class GitTool(Tool):
    """Tool for Git operations."""

    @property
    def name(self) -> str:
        """Return tool name."""
        return "git"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "Git repository operations (branch, commit, push)"

    def __init__(self, config: dict[str, Any]):
        """Initialize Git tool.

        Args:
            config: Tool configuration with branch_prefix, auto_commit, etc.
        """
        super().__init__(config)
        self.branch_prefix = config.get("default_branch_prefix", "agent/")
        self.auto_commit = config.get("auto_commit", False)
        self.commit_template = config.get("commit_message_template", "Agent: {description}")

    async def execute(self, operation: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute Git operation.

        Args:
            operation: Operation type (create_branch, commit, push, status, diff)
            arguments: Operation arguments dict with 'repo_path' and operation-specific keys

        Returns:
            Tool execution result
        """
        if not self.enabled:
            return ToolResult(success=False, output=None, error="Git tool is disabled")

        # Extract repo_path from arguments
        repo_path = arguments.get("repo_path", ".")
        repo = Path(repo_path).expanduser().resolve()

        if not repo.exists():
            return ToolResult(success=False, output=None, error=f"Repository not found: {repo_path}")

        try:
            # Map operation names from schema to internal methods
            if operation == "create_branch":
                branch_name = arguments.get("branch_name", "")
                return await self._create_branch(repo, branch_name)
            elif operation == "commit":
                message = arguments.get("message", "")
                files = arguments.get("files", [])
                return await self._commit(repo, message, files)
            elif operation == "push":
                branch = arguments.get("branch", "")
                remote = arguments.get("remote", "origin")
                return await self._push(repo, branch, remote)
            elif operation == "status":
                return await self._status(repo)
            elif operation == "diff":
                return await self._diff(repo)
            else:
                from agent.runtime.schema import OperationNotSupportedError
                raise OperationNotSupportedError(self.name, operation)
        except OperationNotSupportedError:
            raise
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    async def _create_branch(self, repo: Path, branch_name: str) -> ToolResult:
        """Create a new branch."""
        # TODO: Implement branch creation
        raise NotImplementedError("Branch creation not yet implemented")

    async def _commit(self, repo: Path, message: str, files: list[str]) -> ToolResult:
        """Create a commit."""
        # TODO: Implement commit creation
        raise NotImplementedError("Commit creation not yet implemented")

    async def _push(self, repo: Path, branch: str, remote: str) -> ToolResult:
        """Push branch to remote."""
        # TODO: Implement push
        raise NotImplementedError("Push not yet implemented")

    async def _status(self, repo: Path) -> ToolResult:
        """Get repository status."""
        # TODO: Implement status
        raise NotImplementedError("Status not yet implemented")

    async def _diff(self, repo: Path) -> ToolResult:
        """Get repository diff."""
        # TODO: Implement diff
        raise NotImplementedError("Diff not yet implemented")

