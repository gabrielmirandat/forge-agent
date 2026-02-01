"""MCP tool call interceptor for path normalization and parameter correction.

This interceptor automatically corrects tool parameters before execution,
allowing the LLM to use natural paths (e.g., "forge-agent") while the
interceptor converts them to the correct Docker paths (e.g., "/workspace/forge-agent").
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from langchain_mcp_adapters.interceptors import ToolCallInterceptor, MCPToolCallRequest, MCPToolCallResult
    MCP_INTERCEPTOR_AVAILABLE = True
except ImportError:
    MCP_INTERCEPTOR_AVAILABLE = False
    logging.warning("langchain-mcp-adapters interceptors not available")


class MCPPathInterceptor(ToolCallInterceptor):
    """Interceptor that normalizes paths and corrects tool parameters.
    
    This interceptor:
    - Converts relative repository names to absolute Docker paths
    - Normalizes filesystem paths to use /projects
    - Normalizes git paths to use /workspace
    - Handles workspace base path resolution
    """
    
    def __init__(self, workspace_base: Optional[Path] = None):
        """Initialize path interceptor.
        
        Args:
            workspace_base: Base workspace path (defaults to ~/repos)
        """
        if not MCP_INTERCEPTOR_AVAILABLE:
            raise ImportError(
                "langchain-mcp-adapters interceptors not available. "
                "Install with: pip install langchain-mcp-adapters"
            )
        
        self.workspace_base = workspace_base or Path.home() / "repos"
        self.workspace_base = self.workspace_base.expanduser().resolve()
        self.logger = logging.getLogger(__name__)
        
        # Known repository names that should map to workspace paths
        # This can be extended to read from config or discover automatically
        self.known_repos = ["forge-agent"]  # Can be expanded
    
    def _normalize_filesystem_path(self, path: str) -> str:
        """Normalize filesystem tool paths to use /projects.
        
        Args:
            path: Original path from LLM
            
        Returns:
            Normalized path using /projects base
        """
        if not path:
            return path
        
        # If already absolute and starts with /projects, return as-is
        if path.startswith("/projects"):
            return path
        
        # Expand ~ and $HOME first
        expanded_path = path
        if expanded_path.startswith("~") or expanded_path.startswith("$HOME"):
            expanded_path = str(Path(expanded_path).expanduser())
        
        # If it's a known repository name, convert to /projects/repo_name
        if path in self.known_repos or path.endswith(tuple(f"/{repo}" for repo in self.known_repos)):
            repo_name = path.split("/")[-1] if "/" in path else path
            return f"/projects/{repo_name}"
        
        # If it's a relative path, assume it's relative to workspace
        if not path.startswith("/"):
            # Check if it's a repository name
            if path in self.known_repos:
                return f"/projects/{path}"
            # Otherwise, treat as relative to /projects
            return f"/projects/{path}"
        
        # If it's an absolute path starting with workspace_base, convert to /projects
        workspace_str = str(self.workspace_base)
        if expanded_path.startswith(workspace_str):
            relative = expanded_path[len(workspace_str):].lstrip("/")
            return f"/projects/{relative}"
        
        # If it starts with ~/repos or similar, convert
        if path.startswith("~/repos") or path.startswith("$HOME/repos"):
            # Extract the part after repos/
            if "repos/" in path:
                relative = path.split("repos/", 1)[-1]
            else:
                # Just the repo name
                relative = path.split("/")[-1]
            return f"/projects/{relative}"
        
        # Default: assume it's already correct or return as-is
        return path
    
    def _normalize_git_path(self, repo_path: str) -> str:
        """Normalize git tool paths to use /workspace.
        
        Args:
            repo_path: Original repo_path from LLM
            
        Returns:
            Normalized path using /workspace base
        """
        if not repo_path:
            return repo_path
        
        # If already absolute and starts with /workspace, return as-is
        if repo_path.startswith("/workspace"):
            return repo_path
        
        # Expand ~ and $HOME first
        expanded_path = repo_path
        if expanded_path.startswith("~") or expanded_path.startswith("$HOME"):
            expanded_path = str(Path(expanded_path).expanduser())
        
        # If it's just a repository name (e.g., "forge-agent"), convert to /workspace/repo_name
        if repo_path in self.known_repos or (not "/" in repo_path and not repo_path.startswith("/")):
            return f"/workspace/{repo_path}"
        
        # If it starts with /app (incorrect), convert to /workspace
        if repo_path.startswith("/app/"):
            relative = repo_path[len("/app/"):]
            return f"/workspace/{relative}"
        
        # If it starts with ~/repos or similar, extract relative part
        if repo_path.startswith("~/repos") or repo_path.startswith("$HOME/repos"):
            # Extract the part after repos/
            if "repos/" in repo_path:
                relative = repo_path.split("repos/", 1)[-1]
            else:
                # Just the repo name
                relative = repo_path.split("/")[-1]
            return f"/workspace/{relative}"
        
        # If it's an absolute path starting with workspace_base (after expansion), convert to /workspace
        workspace_str = str(self.workspace_base)
        if expanded_path.startswith(workspace_str):
            relative = expanded_path[len(workspace_str):].lstrip("/")
            return f"/workspace/{relative}"
        
        # If it's a relative path, assume it's relative to workspace
        if not repo_path.startswith("/"):
            return f"/workspace/{repo_path}"
        
        # Default: assume it's already correct or return as-is
        return repo_path
    
    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Any,  # Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]
    ) -> MCPToolCallResult:
        """Intercept tool call and normalize paths.
        
        Args:
            request: Tool call request with name and args
            handler: Original handler to call after normalization
            
        Returns:
            Tool call result
        """
        # Get server name and tool name from request
        server_name = request.server_name
        tool_name = request.name
        args = request.args
        
        # Create a copy of args to modify
        normalized_args = args.copy() if isinstance(args, dict) else {}
        args_changed = False
        
        # Normalize paths based on server type
        if server_name == "filesystem":
            # Filesystem tools use "path" parameter (and potentially other path-related params)
            path_params = ["path", "file_path", "directory", "source", "destination", "target"]
            for param_name in path_params:
                if param_name in normalized_args:
                    original_path = normalized_args[param_name]
                    normalized_path = self._normalize_filesystem_path(str(original_path))
                    if normalized_path != original_path:
                        self.logger.debug(
                            f"Normalized filesystem {param_name}: {original_path} → {normalized_path}",
                            extra={"tool": tool_name, "param": param_name, "original": original_path, "normalized": normalized_path}
                        )
                        normalized_args[param_name] = normalized_path
                        args_changed = True
        
        elif server_name == "git":
            # Git tools use "repo_path" parameter
            if "repo_path" in normalized_args:
                original_path = normalized_args["repo_path"]
                normalized_path = self._normalize_git_path(str(original_path))
                if normalized_path != original_path:
                    self.logger.info(
                        f"Normalized git repo_path: {original_path} → {normalized_path}",
                        extra={"tool": tool_name, "original": original_path, "normalized": normalized_path}
                    )
                    normalized_args["repo_path"] = normalized_path
                    args_changed = True
        
        # Create new request with normalized args if changed
        # MCPToolCallRequest is a dataclass, so we create a new instance
        if args_changed:
            modified_request = MCPToolCallRequest(
                name=request.name,
                args=normalized_args,
                server_name=request.server_name,
                headers=request.headers,
                runtime=request.runtime
            )
            # Call handler with modified request
            return await handler(modified_request)
        else:
            # No changes needed, call handler with original request
            return await handler(request)
