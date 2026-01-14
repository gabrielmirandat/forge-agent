"""Git operations tool."""

import asyncio
import subprocess
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
        # Path validation: Use same allowed_paths/restricted_paths as filesystem tool
        self.allowed_paths = [Path(p).expanduser().resolve() for p in config.get("allowed_paths", [])]
        self.restricted_paths = [Path(p).expanduser().resolve() for p in config.get("restricted_paths", [])]

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
        
        # Validate path for all operations (except init and clone which validate internally)
        if operation not in ("init", "clone") and not self._check_path(repo):
            return ToolResult(
                success=False,
                output=None,
                error=f"Path is not allowed: {repo_path}. Repository must be within allowed paths.",
            )

        if operation not in ("init", "clone") and not repo.exists():
            return ToolResult(success=False, output=None, error=f"Repository not found: {repo_path}")

        try:
            # Map operation names from schema to internal methods
            if operation == "init":
                return await self._init(repo)
            elif operation == "clone":
                repo_url = arguments.get("repo_url", "")
                target_path = Path(arguments.get("target_path", repo_path))
                return await self._clone(repo_url, target_path)
            elif operation == "add":
                files = arguments.get("files", [])
                return await self._add(repo, files)
            elif operation == "create_branch":
                branch_name = arguments.get("branch_name", "")
                return await self._create_branch(repo, branch_name)
            elif operation == "checkout":
                branch_name = arguments.get("branch_name", "")
                return await self._checkout(repo, branch_name)
            elif operation == "commit":
                message = arguments.get("message", "")
                files = arguments.get("files", [])
                return await self._commit(repo, message, files)
            elif operation == "push":
                branch = arguments.get("branch", "")
                remote = arguments.get("remote", "origin")
                return await self._push(repo, branch, remote)
            elif operation == "pull":
                remote = arguments.get("remote", "origin")
                branch = arguments.get("branch", "")
                return await self._pull(repo, remote, branch)
            elif operation == "fetch":
                remote = arguments.get("remote", "origin")
                return await self._fetch(repo, remote)
            elif operation == "merge":
                branch = arguments.get("branch", "")
                return await self._merge(repo, branch)
            elif operation == "status":
                return await self._status(repo)
            elif operation == "diff":
                return await self._diff(repo)
            elif operation == "log":
                limit = arguments.get("limit", 10)
                return await self._log(repo, limit)
            elif operation == "tag":
                tag_name = arguments.get("tag_name", "")
                message = arguments.get("message", "")
                return await self._tag(repo, tag_name, message)
            elif operation == "add_remote":
                remote_name = arguments.get("remote_name", "origin")
                remote_url = arguments.get("remote_url", "")
                return await self._add_remote(repo, remote_name, remote_url)
            else:
                from agent.runtime.schema import OperationNotSupportedError
                raise OperationNotSupportedError(self.name, operation)
        except OperationNotSupportedError:
            raise
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    def _check_path(self, path: Path) -> bool:
        """Check if path is allowed (same logic as filesystem tool).
        
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

    async def _init(self, repo: Path) -> ToolResult:
        """Initialize a new Git repository.
        
        Args:
            repo: Path where to initialize the repository (directory will be created if it doesn't exist)
            
        Returns:
            ToolResult with initialization status
        """
        # Validate path before creating directory
        if not self._check_path(repo):
            return ToolResult(
                success=False,
                output=None,
                error=f"Path is not allowed: {repo}. Repository must be created within allowed paths.",
            )
        
        # Create directory if it doesn't exist
        if not repo.exists():
            try:
                repo.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Failed to create directory: {e}",
                )
        elif not repo.is_dir():
            return ToolResult(success=False, output=None, error=f"Path exists but is not a directory: {repo}")
        
        # Check if already a git repository
        git_dir = repo / ".git"
        if git_dir.exists():
            return ToolResult(
                success=False,
                output=None,
                error=f"Directory is already a Git repository: {repo}",
            )
        
        # Initialize repository (default branch is "master" in older Git versions)
        return_code, stdout, stderr = await self._run_git_command(repo, "init")
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output={"path": str(repo), "error": stderr},
                error=f"Failed to initialize repository: {stderr}",
            )
        
        # Get the default branch name (usually "master")
        return_code_branch, branch, _ = await self._run_git_command(repo, "rev-parse", "--abbrev-ref", "HEAD")
        branch = branch.strip() if branch else "master"
        
        return ToolResult(
            success=True,
            output={"path": str(repo), "branch": branch, "message": stdout.strip() or f"Initialized empty Git repository in {repo} with branch '{branch}'"},
        )

    async def _run_git_command(self, repo: Path, *args: str) -> tuple[int, str, str]:
        """Run a git command and return (return_code, stdout, stderr).
        
        Args:
            repo: Repository path
            *args: Git command arguments (e.g., "status", "--porcelain")
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(repo),
            )
            stdout, stderr = await process.communicate()
            return (
                process.returncode,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except Exception as e:
            return (1, "", str(e))

    async def _add(self, repo: Path, files: list[str]) -> ToolResult:
        """Add files to staging area.
        
        Args:
            repo: Repository path
            files: List of files to add (empty list means add all changes)
            
        Returns:
            ToolResult with add status
        """
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # Add files if specified
        if files:
            for file in files:
                return_code, stdout, stderr = await self._run_git_command(repo, "add", file)
                if return_code != 0:
                    return ToolResult(
                        success=False,
                        output={"files": files, "error": stderr},
                        error=f"Failed to add file '{file}': {stderr}",
                    )
        else:
            # Add all changes
            return_code, stdout, stderr = await self._run_git_command(repo, "add", "-A")
            if return_code != 0:
                return ToolResult(
                    success=False,
                    output={"error": stderr},
                    error=f"Failed to stage files: {stderr}",
                )
        
        return ToolResult(
            success=True,
            output={"files": files if files else "all changes", "message": stdout.strip() or f"Added {len(files) if files else 'all changes'} to staging area"},
        )

    async def _create_branch(self, repo: Path, branch_name: str) -> ToolResult:
        """Create a new branch.
        
        Args:
            repo: Repository path
            branch_name: Name of the branch to create
            
        Returns:
            ToolResult with branch creation status
        """
        if not branch_name:
            return ToolResult(success=False, output=None, error="Branch name is required")
        
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # Create branch
        return_code, stdout, stderr = await self._run_git_command(repo, "checkout", "-b", branch_name)
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output={"branch": branch_name, "error": stderr},
                error=f"Failed to create branch: {stderr}",
            )
        
        return ToolResult(
            success=True,
            output={"branch": branch_name, "message": stdout.strip() or f"Branch '{branch_name}' created"},
        )

    async def _commit(self, repo: Path, message: str, files: list[str] = None) -> ToolResult:
        """Create a commit.
        
        Args:
            repo: Repository path
            message: Commit message
            files: Optional list of files to commit (if provided, will stage them first; if None or empty, commits all staged files)
            
        Returns:
            ToolResult with commit status
        """
        if not message:
            return ToolResult(success=False, output=None, error="Commit message is required")
        
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # If files are specified, stage them first (optional - LLM can use git.add separately)
        if files:
            for file in files:
                return_code, _, stderr = await self._run_git_command(repo, "add", file)
                if return_code != 0:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Failed to add file '{file}': {stderr}",
                    )
        
        # Create commit (commits all staged files)
        return_code, stdout, stderr = await self._run_git_command(
            repo, "commit", "-m", message
        )
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output={"message": message, "error": stderr},
                error=f"Failed to create commit: {stderr}",
            )
        
        # Get commit hash
        return_code, commit_hash, _ = await self._run_git_command(repo, "rev-parse", "HEAD")
        commit_hash = commit_hash.strip() if commit_hash else "unknown"
        
        return ToolResult(
            success=True,
            output={
                "commit": commit_hash,
                "message": message,
                "files": files if files else "all staged files",
            },
        )

    async def _push(self, repo: Path, branch: str, remote: str) -> ToolResult:
        """Push branch to remote.
        
        Args:
            repo: Repository path
            branch: Branch name to push
            remote: Remote name (default: "origin")
            
        Returns:
            ToolResult with push status
        """
        if not branch:
            return ToolResult(success=False, output=None, error="Branch name is required")
        
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # Get remote URL to check if it needs authentication
        return_code, remote_url, _ = await self._run_git_command(repo, "remote", "get-url", remote)
        if return_code != 0:
            return ToolResult(
                success=False,
                output=None,
                error=f"Remote '{remote}' not found. Use git.add_remote to add it first.",
            )
        
        # If remote URL is HTTPS and contains github.com, inject token for authentication
        import os
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token and "github.com" in remote_url and remote_url.startswith("https://"):
            # Inject token into URL: https://github.com/user/repo.git -> https://TOKEN@github.com/user/repo.git
            if f":{github_token}@" not in remote_url and "@github.com" not in remote_url:
                # Extract the path part (user/repo.git)
                url_parts = remote_url.replace("https://", "").split("/", 1)
                if len(url_parts) == 2:
                    authenticated_url = f"https://{github_token}@github.com/{url_parts[1]}"
                    # Temporarily update remote URL for this push
                    await self._run_git_command(repo, "remote", "set-url", remote, authenticated_url)
                    try:
                        # Push branch
                        return_code, stdout, stderr = await self._run_git_command(repo, "push", "-u", remote, branch)
                    finally:
                        # Restore original URL (without token)
                        await self._run_git_command(repo, "remote", "set-url", remote, remote_url)
                else:
                    return_code, stdout, stderr = await self._run_git_command(repo, "push", "-u", remote, branch)
            else:
                # Already has authentication, just push
                return_code, stdout, stderr = await self._run_git_command(repo, "push", "-u", remote, branch)
        else:
            # Not a GitHub HTTPS URL or no token, push normally
            return_code, stdout, stderr = await self._run_git_command(repo, "push", "-u", remote, branch)
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output={"branch": branch, "remote": remote, "error": stderr},
                error=f"Failed to push branch: {stderr}",
            )
        
        return ToolResult(
            success=True,
            output={"branch": branch, "remote": remote, "message": stdout.strip() or f"Pushed {branch} to {remote}"},
        )

    async def _status(self, repo: Path) -> ToolResult:
        """Get repository status.
        
        Args:
            repo: Repository path
            
        Returns:
            ToolResult with git status
        """
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # Get status
        return_code, stdout, stderr = await self._run_git_command(repo, "status", "--porcelain")
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to get status: {stderr}",
            )
        
        # Get current branch
        return_code_branch, branch, _ = await self._run_git_command(repo, "rev-parse", "--abbrev-ref", "HEAD")
        branch = branch.strip() if branch else "unknown"
        
        return ToolResult(
            success=True,
            output={
                "branch": branch,
                "status": stdout.strip() or "working tree clean",
                "has_changes": bool(stdout.strip()),
            },
        )

    async def _diff(self, repo: Path) -> ToolResult:
        """Get repository diff.
        
        Args:
            repo: Repository path
            
        Returns:
            ToolResult with git diff
        """
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # Get diff
        return_code, stdout, stderr = await self._run_git_command(repo, "diff")
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to get diff: {stderr}",
            )
        
        return ToolResult(
            success=True,
            output={"diff": stdout.strip() or "No changes"},
        )

    async def _clone(self, repo_url: str, target_path: Path) -> ToolResult:
        """Clone a repository.
        
        Args:
            repo_url: URL of the repository to clone
            target_path: Path where to clone the repository (directory will be created)
            
        Returns:
            ToolResult with clone status
        """
        if not repo_url:
            return ToolResult(success=False, output=None, error="Repository URL is required")
        
        # Validate target path
        if not self._check_path(target_path):
            return ToolResult(
                success=False,
                output=None,
                error=f"Path is not allowed: {target_path}. Repository must be cloned within allowed paths.",
            )
        
        # Check if target directory already exists
        if target_path.exists():
            return ToolResult(
                success=False,
                output=None,
                error=f"Directory already exists: {target_path}. Cannot clone into existing directory.",
            )
        
        # Clone repository - git clone creates the directory, so we run from parent
        parent_dir = target_path.parent
        repo_name = target_path.name
        
        # Ensure parent directory exists
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Failed to create parent directory: {e}",
                )
        
        # Clone repository (git clone creates the target directory)
        return_code, stdout, stderr = await self._run_git_command(
            parent_dir,
            "clone",
            repo_url,
            repo_name,
        )
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output={"url": repo_url, "path": str(target_path), "error": stderr},
                error=f"Failed to clone repository: {stderr}",
            )
        
        return ToolResult(
            success=True,
            output={"url": repo_url, "path": str(target_path), "message": stdout.strip() or f"Cloned repository to {target_path}"},
        )

    async def _checkout(self, repo: Path, branch_name: str) -> ToolResult:
        """Checkout a branch.
        
        Args:
            repo: Repository path
            branch_name: Name of the branch to checkout
            
        Returns:
            ToolResult with checkout status
        """
        if not branch_name:
            return ToolResult(success=False, output=None, error="Branch name is required")
        
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # Checkout branch
        return_code, stdout, stderr = await self._run_git_command(repo, "checkout", branch_name)
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output={"branch": branch_name, "error": stderr},
                error=f"Failed to checkout branch: {stderr}",
            )
        
        return ToolResult(
            success=True,
            output={"branch": branch_name, "message": stdout.strip() or f"Checked out branch '{branch_name}'"},
        )

    async def _pull(self, repo: Path, remote: str, branch: str) -> ToolResult:
        """Pull changes from remote.
        
        Args:
            repo: Repository path
            remote: Remote name (default: "origin")
            branch: Branch name to pull (default: current branch)
            
        Returns:
            ToolResult with pull status
        """
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # Pull changes
        if branch:
            return_code, stdout, stderr = await self._run_git_command(repo, "pull", remote, branch)
        else:
            return_code, stdout, stderr = await self._run_git_command(repo, "pull", remote)
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output={"remote": remote, "branch": branch, "error": stderr},
                error=f"Failed to pull: {stderr}",
            )
        
        return ToolResult(
            success=True,
            output={"remote": remote, "branch": branch, "message": stdout.strip() or f"Pulled from {remote}"},
        )

    async def _fetch(self, repo: Path, remote: str) -> ToolResult:
        """Fetch changes from remote without merging.
        
        Args:
            repo: Repository path
            remote: Remote name (default: "origin")
            
        Returns:
            ToolResult with fetch status
        """
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # Fetch changes
        return_code, stdout, stderr = await self._run_git_command(repo, "fetch", remote)
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output={"remote": remote, "error": stderr},
                error=f"Failed to fetch: {stderr}",
            )
        
        return ToolResult(
            success=True,
            output={"remote": remote, "message": stdout.strip() or f"Fetched from {remote}"},
        )

    async def _merge(self, repo: Path, branch: str) -> ToolResult:
        """Merge a branch into current branch.
        
        Args:
            repo: Repository path
            branch: Branch name to merge
            
        Returns:
            ToolResult with merge status
        """
        if not branch:
            return ToolResult(success=False, output=None, error="Branch name is required")
        
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # Merge branch
        return_code, stdout, stderr = await self._run_git_command(repo, "merge", branch)
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output={"branch": branch, "error": stderr},
                error=f"Failed to merge branch: {stderr}",
            )
        
        return ToolResult(
            success=True,
            output={"branch": branch, "message": stdout.strip() or f"Merged branch '{branch}'"},
        )

    async def _log(self, repo: Path, limit: int = 10) -> ToolResult:
        """Get commit history.
        
        Args:
            repo: Repository path
            limit: Maximum number of commits to return (default: 10)
            
        Returns:
            ToolResult with commit log
        """
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # Get log
        return_code, stdout, stderr = await self._run_git_command(
            repo, "log", f"--max-count={limit}", "--oneline", "--decorate"
        )
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to get log: {stderr}",
            )
        
        return ToolResult(
            success=True,
            output={"log": stdout.strip() or "No commits", "limit": limit},
        )

    async def _tag(self, repo: Path, tag_name: str, message: str = "") -> ToolResult:
        """Create a tag.
        
        Args:
            repo: Repository path
            tag_name: Name of the tag
            message: Optional tag message (creates annotated tag if provided)
            
        Returns:
            ToolResult with tag creation status
        """
        if not tag_name:
            return ToolResult(success=False, output=None, error="Tag name is required")
        
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # Create tag
        if message:
            return_code, stdout, stderr = await self._run_git_command(repo, "tag", "-a", tag_name, "-m", message)
        else:
            return_code, stdout, stderr = await self._run_git_command(repo, "tag", tag_name)
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output={"tag": tag_name, "error": stderr},
                error=f"Failed to create tag: {stderr}",
            )
        
        return ToolResult(
            success=True,
            output={"tag": tag_name, "message": message or f"Tag '{tag_name}' created", "output": stdout.strip() or f"Tag '{tag_name}' created"},
        )

    async def _add_remote(self, repo: Path, remote_name: str, remote_url: str) -> ToolResult:
        """Add a remote repository.
        
        Args:
            repo: Repository path
            remote_name: Name of the remote (default: "origin")
            remote_url: URL of the remote repository
            
        Returns:
            ToolResult with remote addition status
        """
        if not remote_url:
            return ToolResult(success=False, output=None, error="Remote URL is required")
        
        # Check if repo is a git repository
        return_code, _, stderr = await self._run_git_command(repo, "rev-parse", "--git-dir")
        if return_code != 0:
            return ToolResult(success=False, output=None, error=f"Not a git repository: {stderr}")
        
        # Check if remote already exists
        return_code, _, _ = await self._run_git_command(repo, "remote", "get-url", remote_name)
        if return_code == 0:
            # Remote exists, update it
            return_code, stdout, stderr = await self._run_git_command(repo, "remote", "set-url", remote_name, remote_url)
        else:
            # Remote doesn't exist, add it
            return_code, stdout, stderr = await self._run_git_command(repo, "remote", "add", remote_name, remote_url)
        
        if return_code != 0:
            return ToolResult(
                success=False,
                output={"remote": remote_name, "url": remote_url, "error": stderr},
                error=f"Failed to add/update remote: {stderr}",
            )
        
        return ToolResult(
            success=True,
            output={"remote": remote_name, "url": remote_url, "message": f"Remote '{remote_name}' added/updated"},
        )
