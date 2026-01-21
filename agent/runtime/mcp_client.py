"""MCP (Model Context Protocol) client integration.

Similar to OpenCode's MCP integration for connecting to external MCP servers.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from agent.observability import get_logger

try:
    from mcp import ClientSession, types
    from mcp.client.auth import OAuthClientProvider, TokenStorage
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.client.sse import sse_client
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    get_logger("mcp", "mcp").warning("MCP SDK not available. Install with: pip install mcp")


class InMemoryTokenStorage(TokenStorage):
    """In-memory token storage for MCP OAuth."""

    def __init__(self):
        """Initialize token storage."""
        self.tokens: Optional[OAuthToken] = None
        self.client_info: Optional[OAuthClientInformationFull] = None

    async def get_tokens(self) -> Optional[OAuthToken]:
        """Get stored tokens."""
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store tokens."""
        self.tokens = tokens

    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        """Get stored client information."""
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store client information."""
        self.client_info = client_info


class MCPClient:
    """MCP client wrapper for connecting to MCP servers.
    
    This client implements the Model Context Protocol (MCP) specification
    to connect to MCP servers via stdio, SSE, or Streamable HTTP transports.
    
    According to MCP Python SDK documentation:
    - Supports stdio transport for local servers
    - Supports SSE and Streamable HTTP for remote servers
    - Handles OAuth authentication when required
    - Manages tool discovery and execution
    
    Reference: _helpers/python-sdk/README.md
    """

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        token_storage: Optional[TokenStorage] = None,
    ) -> None:
        """Initialize MCP client.

        Args:
            name: MCP server name for identification and logging.
            config: MCP server configuration dict containing:
                - type: Transport type ("local", "remote", "docker")
                - command: Command to run (for stdio/local)
                - url: Server URL (for remote/HTTP)
                - Other transport-specific settings
            token_storage: Optional token storage for OAuth authentication.
                If None, uses InMemoryTokenStorage.
        
        Raises:
            ImportError: If MCP SDK is not available.
        """
        if not MCP_AVAILABLE:
            raise ImportError("MCP SDK not available. Install with: pip install mcp")

        self.name = name
        self.config = config
        self.token_storage = token_storage or InMemoryTokenStorage()
        self.logger = get_logger(f"mcp.{name}", "mcp")
        self._session: Optional[ClientSession] = None
        self._transport = None
        self._tools: List[types.Tool] = []
        self._status: str = "disconnected"

    async def connect(self) -> bool:
        """Connect to MCP server using the configured transport.
        
        This method establishes a connection to the MCP server based on the
        transport type specified in the configuration. It handles initialization
        and tool discovery automatically.

        Returns:
            True if connected successfully and tools are available, False otherwise.
        
        Raises:
            ValueError: If the MCP type is unknown or unsupported.
            ConnectionError: If the connection fails.
        """
        try:
            mcp_type = self.config.get("type", "local")
            
            if mcp_type == "local":
                await self._connect_local()
            elif mcp_type == "remote":
                await self._connect_remote()
            elif mcp_type == "docker":
                await self._connect_docker()
            else:
                raise ValueError(f"Unknown MCP type: {mcp_type}")

            if self._session:
                # Initialize session
                await self._session.initialize()
                
                # List available tools
                tools_result = await self._session.list_tools()
                self._tools = tools_result.tools
                
                self._status = "connected"
                self.logger.info(f"MCP server '{self.name}' connected with {len(self._tools)} tools")
                return True
            return False
        except Exception as e:
            self._status = "failed"
            self.logger.error(f"Failed to connect to MCP server '{self.name}': {e}", exc_info=True)
            return False

    async def _connect_local(self):
        """Connect to local MCP server via stdio."""
        from mcp import StdioServerParameters

        command = self.config.get("command", [])
        if not command:
            raise ValueError("Local MCP server requires 'command' configuration")

        env = self.config.get("environment", {})
        
        server_params = StdioServerParameters(
            command=command[0],
            args=command[1:] if len(command) > 1 else [],
            env=env,
        )

        self._transport = stdio_client(server_params)
        read, write = await self._transport.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
    
    async def _connect_docker(self):
        """Connect to Docker-based MCP server via stdio (docker run -i).

        For MCP servers, each connection creates a new server instance.
        We use docker run -i --rm to run the server for this connection.
        """
        from mcp import StdioServerParameters
        from pathlib import Path

        image = self.config.get("image")
        if not image:
            raise ValueError(f"Docker MCP server '{self.name}' missing 'image' configuration")

        volumes = self.config.get("volumes", [])
        args = self.config.get("args", [])
        env_vars = self.config.get("environment", {})
        workspace_path = self.config.get("workspace_path", os.getenv("WORKSPACE_BASE_PATH", "~/repos"))

        # Build docker run command
        docker_cmd = ["docker", "run", "-i", "--rm"]  # Interactive, remove on exit

        # Add environment variables from config
        # Also check system environment for tokens (e.g., GITHUB_TOKEN, GITHUB_PERSONAL_ACCESS_TOKEN)
        resolved_env = {}
        if env_vars:
            for key, value in env_vars.items():
                # Support {{env:VAR_NAME}} syntax to read from system environment
                if isinstance(value, str) and value.startswith("{{env:") and value.endswith("}}"):
                    env_var_name = value[6:-2]  # Extract VAR_NAME from {{env:VAR_NAME}}
                    resolved_env[key] = os.getenv(env_var_name, "")
                else:
                    resolved_env[key] = value
        
        # Add common token environment variables if not already set
        # Check for tokens in system environment and add to resolved_env
        token_mappings = {
            "GITHUB_TOKEN": "GITHUB_TOKEN",
            "GITHUB_PERSONAL_ACCESS_TOKEN": "GITHUB_PERSONAL_ACCESS_TOKEN",
        }
        for env_key, system_key in token_mappings.items():
            if env_key not in resolved_env:
                system_value = os.getenv(system_key)
                if system_value:
                    resolved_env[env_key] = system_value

        # Add volumes
        for volume in volumes:
            if "{{workspace.base_path}}" in volume:
                # Resolve workspace path
                workspace_base = str(Path(workspace_path).expanduser())
                volume = volume.replace("{{workspace.base_path}}", workspace_base)
            if volume.startswith("~"):
                volume = str(Path(volume).expanduser())
            docker_cmd.extend(["-v", volume])

        # Add environment variables to docker command
        for key, value in resolved_env.items():
            if value:  # Only add if value is not empty
                docker_cmd.extend(["-e", f"{key}={value}"])

        # Add image and args
        docker_cmd.append(image)
        docker_cmd.extend(args)

        # Use docker run -i for stdio connection
        # Each MCP connection gets its own container instance
        server_params = StdioServerParameters(
            command=docker_cmd[0],  # "docker"
            args=docker_cmd[1:],    # ["run", "-i", "--rm", "-e", "KEY=value", ...]
            env=resolved_env,  # Also pass to stdio client
        )

        self._transport = stdio_client(server_params)
        read, write = await self._transport.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()

    async def _connect_remote(self):
        """Connect to remote MCP server via HTTP/SSE."""
        url = self.config.get("url")
        if not url:
            raise ValueError("Remote MCP server requires 'url' configuration")

        headers = self.config.get("headers", {})
        oauth_enabled = self.config.get("oauth", True)
        
        # Try streamable HTTP first, then SSE
        try:
            if oauth_enabled:
                from mcp.shared.auth import OAuthClientMetadata
                client_metadata = OAuthClientMetadata(
                    redirect_uris=["http://localhost:8000/oauth/callback"],
                    client_name="forge-agent",
                    client_uri="https://github.com/forge-agent",
                    grant_types=["authorization_code", "refresh_token"],
                    response_types=["code"],
                    token_endpoint_auth_method="none",
                )
                oauth_provider = OAuthClientProvider(
                    server_url=url,
                    storage=self.token_storage,
                    client_metadata=client_metadata,
                    redirect_handler=self._handle_oauth_redirect,
                    callback_handler=self._handle_oauth_callback,
                )
                # Use httpx with OAuth
                import httpx
                http_client = httpx.AsyncClient(auth=oauth_provider, follow_redirects=True)
                self._transport = streamable_http_client(url, http_client=http_client)
            else:
                self._transport = streamable_http_client(url, headers=headers)
            
            read, write, _ = await self._transport.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
        except Exception as e:
            self.logger.warning(f"Streamable HTTP failed, trying SSE: {e}")
            # Fallback to SSE
            try:
                if oauth_enabled:
                    from mcp.shared.auth import OAuthClientMetadata
                    client_metadata = OAuthClientMetadata(
                        redirect_uris=["http://localhost:8000/oauth/callback"],
                        client_name="forge-agent",
                        client_uri="https://github.com/forge-agent",
                        grant_types=["authorization_code", "refresh_token"],
                        response_types=["code"],
                        token_endpoint_auth_method="none",
                    )
                    oauth_provider = OAuthClientProvider(
                        server_url=url,
                        storage=self.token_storage,
                        client_metadata=client_metadata,
                        redirect_handler=self._handle_oauth_redirect,
                        callback_handler=self._handle_oauth_callback,
                    )
                    import httpx
                    http_client = httpx.AsyncClient(auth=oauth_provider, follow_redirects=True)
                    self._transport = sse_client(url, http_client=http_client)
                else:
                    self._transport = sse_client(url, headers=headers)
                
                read, write, _ = await self._transport.__aenter__()
                self._session = ClientSession(read, write)
                await self._session.__aenter__()
            except Exception as e2:
                raise Exception(f"Both HTTP and SSE transports failed: {e}, {e2}") from e2

    async def _handle_oauth_redirect(self, url: str) -> None:
        """Handle OAuth redirect.

        Args:
            url: OAuth authorization URL
        """
        self.logger.info(f"OAuth redirect required for {self.name}: {url}")
        # In a real implementation, this would open a browser
        # For now, just log
        print(f"Please visit: {url}")

    async def _handle_oauth_callback(self) -> tuple[str, Optional[str]]:
        """Handle OAuth callback.

        Returns:
            Tuple of (code, state)
        """
        # In a real implementation, this would handle the callback
        code = input("Enter OAuth code: ")
        return code, None

    async def disconnect(self):
        """Disconnect from MCP server.
        
        Tolerates cancellation errors during shutdown - these are expected
        when the application is shutting down and tasks are being cancelled.
        """
        import asyncio
        
        errors = []
        
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except asyncio.CancelledError:
                # Expected during shutdown - just log and continue
                self.logger.debug(f"Session disconnect cancelled for {self.name} (expected during shutdown)")
            except Exception as e:
                # Only log non-cancellation errors
                error_msg = f"Error disconnecting session for {self.name}: {e}"
                self.logger.debug(error_msg)  # Use debug level to avoid noise during shutdown
                errors.append(error_msg)
        
        if self._transport:
            try:
                await self._transport.__aexit__(None, None, None)
            except asyncio.CancelledError:
                # Expected during shutdown - just log and continue
                self.logger.debug(f"Transport disconnect cancelled for {self.name} (expected during shutdown)")
            except Exception as e:
                # Only log non-cancellation errors
                error_msg = f"Error disconnecting transport for {self.name}: {e}"
                self.logger.debug(error_msg)  # Use debug level to avoid noise during shutdown
                errors.append(error_msg)
        
        self._status = "disconnected"
        self._session = None
        self._transport = None
        
        # Don't raise on cancellation errors - they're expected during shutdown
        non_cancel_errors = [e for e in errors if "cancel" not in e.lower()]
        if non_cancel_errors:
            self.logger.warning(f"Non-cancellation errors disconnecting {self.name}: {'; '.join(non_cancel_errors)}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
        """Call an MCP tool.

        Args:
            tool_name: Tool name (with or without server prefix)
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self._session:
            raise RuntimeError(f"MCP client '{self.name}' is not connected")

        # Remove server prefix if present (e.g., "github_list_repos" -> "list_repos")
        if tool_name.startswith(f"{self.name}_"):
            tool_name = tool_name[len(f"{self.name}_"):]

        result = await self._session.call_tool(tool_name, arguments)
        return result

    def get_tools(self) -> List[types.Tool]:
        """Get available tools from MCP server.

        Returns:
            List of available tools
        """
        return self._tools

    def get_status(self) -> str:
        """Get connection status.

        Returns:
            Status string
        """
        return self._status
    
    def get_session(self) -> Optional[ClientSession]:
        """Get the MCP client session (for use with langchain-mcp-adapters).

        Returns:
            ClientSession if connected, None otherwise
        """
        return self._session if self._status == "connected" else None


class MCPManager:
    """Manages multiple MCP clients."""

    def __init__(self):
        """Initialize MCP manager."""
        self._clients: Dict[str, MCPClient] = {}
        self.logger = get_logger("mcp.manager", "mcp")

    async def add_server(self, name: str, config: Dict[str, Any]) -> bool:
        """Add and connect to an MCP server.

        Args:
            name: Server name
            config: Server configuration

        Returns:
            True if added successfully, False otherwise
        """
        if not MCP_AVAILABLE:
            self.logger.warning("MCP SDK not available, skipping server")
            return False

        if config.get("enabled", True) is False:
            self.logger.info(f"MCP server '{name}' is disabled")
            return False

        try:
            client = MCPClient(name, config)
            success = await client.connect()
            
            if success:
                self._clients[name] = client
                self.logger.info(f"MCP server '{name}' added successfully")
                return True
            else:
                self.logger.error(f"Failed to connect to MCP server '{name}'")
                return False
        except Exception as e:
            self.logger.error(f"Error adding MCP server '{name}': {e}", exc_info=True)
            return False

    async def remove_server(self, name: str):
        """Remove an MCP server.

        Args:
            name: Server name
        """
        if name in self._clients:
            client = self._clients[name]
            await client.disconnect()
            del self._clients[name]
            self.logger.info(f"MCP server '{name}' removed")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
        """Call a tool from any connected MCP server.

        Args:
            tool_name: Tool name (may include server prefix, e.g., "github_list_repos")
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        # Try to find server by prefix
        for server_name, client in self._clients.items():
            if tool_name.startswith(f"{server_name}_"):
                return await client.call_tool(tool_name, arguments)
        
        # If no prefix, try all servers
        for server_name, client in self._clients.items():
            tools = client.get_tools()
            tool_names = [t.name for t in tools]
            if tool_name in tool_names:
                return await client.call_tool(tool_name, arguments)
        
        raise ValueError(f"Tool '{tool_name}' not found in any MCP server")

    def get_all_tools(self) -> Dict[str, List[types.Tool]]:
        """Get all tools from all connected servers.

        Returns:
            Dictionary mapping server name to list of tools
        """
        return {name: client.get_tools() for name, client in self._clients.items()}

    def get_server_status(self, name: str) -> Optional[str]:
        """Get status of a specific server.

        Args:
            name: Server name

        Returns:
            Status string or None if server not found
        """
        if name in self._clients:
            return self._clients[name].get_status()
        return None

    async def disconnect_all(self):
        """Disconnect all MCP servers.
        
        Tolerates cancellation errors during shutdown.
        """
        import asyncio
        
        disconnected_count = 0
        for name, client in list(self._clients.items()):
            try:
                await client.disconnect()
                disconnected_count += 1
            except asyncio.CancelledError:
                # Expected during shutdown
                self.logger.debug(f"MCP server '{name}' disconnect cancelled (expected during shutdown)")
            except Exception as e:
                # Log but don't fail - shutdown should be graceful
                self.logger.debug(f"Error disconnecting MCP server '{name}': {e}")
        
        self._clients.clear()
        
        if disconnected_count > 0:
            self.logger.info(f"Disconnected {disconnected_count} MCP server(s)")


# Global MCP manager
_mcp_manager: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    """Get global MCP manager.

    Returns:
        MCPManager instance
    """
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager()
    return _mcp_manager
