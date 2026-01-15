"""GitHub API operations tool."""

import os
from typing import Any

from agent.tools.base import Tool, ToolResult


class GitHubTool(Tool):
    """Tool for GitHub API operations."""

    @property
    def name(self) -> str:
        """Return tool name."""
        return "github"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "GitHub API operations (PRs, issues, comments)"

    def __init__(self, config: dict[str, Any]):
        """Initialize GitHub tool.

        Args:
            config: Tool configuration with base_url, templates, etc.
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.github.com")
        self.token = os.getenv("GITHUB_TOKEN")
        self.auto_create_pr = config.get("auto_create_pr", False)
        self.pr_title_template = config.get("pr_title_template", "Agent: {title}")
        self.pr_body_template = config.get("pr_body_template", "Automated changes by forge-agent")

    async def execute(self, operation: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute GitHub operation.

        Args:
            operation: Operation type (create_pr, list_prs, comment_pr)
            arguments: Operation arguments dict with operation-specific keys

        Returns:
            Tool execution result
        """
        if not self.enabled:
            return ToolResult(success=False, output=None, error="GitHub tool is disabled")

        # For CLI operations, token is optional (gh CLI uses its own auth)
        # For API operations, token is required
        if operation != "create_repository_with_cli" and not self.token:
            return ToolResult(success=False, output=None, error="GITHUB_TOKEN not set")

        try:
            # Map operation names from schema to internal methods
            if operation == "create_repository":
                name = arguments.get("name", "")
                description = arguments.get("description", "")
                private = arguments.get("private", False)
                return await self._create_repository(name, description, private)
            elif operation == "create_repository_with_cli":
                name = arguments.get("name", "")
                description = arguments.get("description", "")
                private = arguments.get("private", False)
                repo_path = arguments.get("repo_path", ".")
                remote = arguments.get("remote", "origin")
                push = arguments.get("push", True)
                
                # session_id is mandatory - get current directory from tmux session when repo_path is "."
                session_id = arguments.get("_session_id")
                if not session_id:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="Missing required argument: '_session_id'. Session ID is mandatory.",
                    )
                
                if repo_path == ".":
                    from agent.tools.tmux import get_tmux_manager
                    from agent.storage.sqlite import SQLiteStorage
                    
                    tmux_manager = get_tmux_manager()
                    storage = SQLiteStorage("forge_agent.db")
                    
                    # Get tmux session (mandatory)
                    tmux_session = await storage.get_tmux_session(session_id)
                    if not tmux_session:
                        return ToolResult(
                            success=False,
                            output=None,
                            error=f"Tmux session not found for session {session_id}",
                        )
                    
                    # Get current working directory from tmux session
                    cwd = await tmux_manager.get_working_directory(tmux_session)
                    if not cwd:
                        return ToolResult(
                            success=False,
                            output=None,
                            error=f"Failed to get working directory from tmux session {tmux_session}",
                        )
                    
                    repo_path = cwd
                
                return await self._create_repository_with_cli(name, description, private, repo_path, remote, push)
            elif operation == "create_pr":
                repo = arguments.get("repo", "")
                title = arguments.get("title", "")
                body = arguments.get("body", "")
                head = arguments.get("head", "")
                base = arguments.get("base", "main")
                return await self._create_pr(repo, title, body, head, base)
            elif operation == "list_prs":
                repo = arguments.get("repo", "")
                state = arguments.get("state", "open")
                return await self._list_prs(repo, state)
            elif operation == "comment_pr":
                repo = arguments.get("repo", "")
                pr_number = arguments.get("pr_number", 0)
                body = arguments.get("body", "")
                return await self._comment(repo, pr_number, body)
            else:
                from agent.runtime.schema import OperationNotSupportedError
                raise OperationNotSupportedError(self.name, operation)
        except OperationNotSupportedError:
            raise
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    async def _create_repository_with_cli(
        self, name: str, description: str, private: bool, repo_path: str, remote: str, push: bool
    ) -> ToolResult:
        """Create a new GitHub repository using GitHub CLI (gh).
        
        This method uses `gh repo create` which can create the remote repository
        and optionally push the local repository in one command.
        
        Args:
            name: Repository name
            description: Repository description
            private: Whether the repository should be private
            repo_path: Path to the local repository (default: current directory)
            remote: Remote name (default: "origin")
            push: Whether to push local commits (default: True)
            
        Returns:
            ToolResult with repository creation status and URL
        """
        if not name:
            return ToolResult(success=False, output=None, error="Repository name is required")
        
        import asyncio
        from pathlib import Path
        
        repo_dir = Path(repo_path).expanduser().resolve()
        
        # Build gh repo create command
        cmd_parts = ["gh", "repo", "create", name]
        
        if description:
            cmd_parts.extend(["--description", description])
        
        if private:
            cmd_parts.append("--private")
        else:
            cmd_parts.append("--public")
        
        # Add source directory and remote options
        cmd_parts.extend(["--source", str(repo_dir)])
        cmd_parts.extend(["--remote", remote])
        
        if push:
            cmd_parts.append("--push")
        else:
            cmd_parts.append("--no-push")
        
        try:
            # Execute gh repo create command
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(repo_dir),
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60.0)
            
            stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
            
            if process.returncode == 0:
                # Extract repository URL from output
                # gh repo create outputs: "https://github.com/username/repo-name.git"
                repo_url = stdout_text.strip() or f"https://github.com/{name}"
                
                return ToolResult(
                    success=True,
                    output={
                        "name": name,
                        "url": repo_url,
                        "clone_url": repo_url,
                        "remote": remote,
                        "pushed": push,
                        "message": f"Repository '{name}' created successfully using GitHub CLI",
                    },
                )
            else:
                error_msg = stderr_text or f"gh repo create failed with return code {process.returncode}"
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Failed to create repository with GitHub CLI: {error_msg}",
                )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output=None,
                error="GitHub CLI command timed out after 60 seconds",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error creating repository with GitHub CLI: {str(e)}",
            )

    async def _create_repository(self, name: str, description: str, private: bool) -> ToolResult:
        """Create a new GitHub repository.
        
        Args:
            name: Repository name
            description: Repository description
            private: Whether the repository should be private
            
        Returns:
            ToolResult with repository creation status and URL
        """
        if not name:
            return ToolResult(success=False, output=None, error="Repository name is required")
        
        import httpx
        
        url = f"{self.base_url}/user/repos"
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        data = {
            "name": name,
            "description": description or "",
            "private": private,
            "auto_init": False,  # Don't initialize with README, we'll do it ourselves
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers, timeout=30.0)
                
                if response.status_code == 201:
                    repo_data = response.json()
                    repo_url = repo_data.get("html_url", "")
                    clone_url = repo_data.get("clone_url", "")
                    ssh_url = repo_data.get("ssh_url", "")
                    
                    return ToolResult(
                        success=True,
                        output={
                            "name": name,
                            "url": repo_url,
                            "clone_url": clone_url,
                            "ssh_url": ssh_url,
                            "private": private,
                            "message": f"Repository '{name}' created successfully",
                        },
                    )
                else:
                    error_text = response.text
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Failed to create repository: {response.status_code} - {error_text}",
                    )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error creating repository: {str(e)}",
            )

    async def _create_pr(self, repo: str, title: str, body: str, head: str, base: str) -> ToolResult:
        """Create a pull request."""
        # TODO: Implement PR creation using GitHub API
        raise NotImplementedError("PR creation not yet implemented")

    async def _list_prs(self, repo: str, state: str) -> ToolResult:
        """List pull requests."""
        # TODO: Implement PR listing
        raise NotImplementedError("PR listing not yet implemented")

    async def _comment(self, repo: str, pr_number: int, body: str) -> ToolResult:
        """Add comment to PR."""
        # TODO: Implement PR commenting
        raise NotImplementedError("PR commenting not yet implemented")

    async def _get_pr(self, repo: str, pr_number: int) -> ToolResult:
        """Get PR details."""
        # TODO: Implement PR retrieval
        raise NotImplementedError("PR retrieval not yet implemented")

