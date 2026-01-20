"""MCP server for exposing Forge Agent tools via Model Context Protocol.

This allows all Forge Agent tools to be used via MCP, making them compatible
with LangChain's MCP integration and any other MCP clients.
"""

import json
from typing import Any, Dict, List, Optional

from agent.observability import get_logger
from agent.tools.base import ToolRegistry, ToolContext, ToolResult

try:
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    get_logger("mcp.server", "mcp").warning("MCP SDK not available. Install with: pip install mcp")


class ForgeAgentMCPServer:
    """MCP server that exposes Forge Agent tools via MCP protocol."""
    
    def __init__(self, tool_registry: ToolRegistry, session_id: Optional[str] = None):
        """Initialize MCP server.
        
        Args:
            tool_registry: Tool registry with all Forge Agent tools
            session_id: Optional session ID for tool context
        """
        if not MCP_AVAILABLE:
            raise ImportError("MCP SDK not available. Install with: pip install mcp")
        
        self.tool_registry = tool_registry
        self.session_id = session_id
        self.logger = get_logger("mcp.server", "mcp")
        self.server = Server("forge-agent-tools")
        
        # Register MCP handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register MCP protocol handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available tools."""
            tools = []
            
            for tool_name in self.tool_registry.list_enabled():
                tool = self.tool_registry.get(tool_name)
                if not tool:
                    continue
                
                # Convert Forge Agent tool to MCP Tool
                mcp_tool = Tool(
                    name=tool_name,
                    description=tool.description,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "description": f"Operation to perform on {tool_name} tool (e.g., 'read', 'write', 'edit', 'search', 'execute')"
                            },
                            "arguments": {
                                "type": "object",
                                "description": "Arguments for the operation as a JSON object",
                                "additionalProperties": True
                            }
                        },
                        "required": ["operation", "arguments"]
                    }
                )
                tools.append(mcp_tool)
            
            return tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Call a tool by name.
            
            Args:
                name: Tool name
                arguments: Tool arguments (must include 'operation' and 'arguments')
                
            Returns:
                List of text content with tool result
            """
            try:
                # Get tool from registry
                tool = self.tool_registry.get(name)
                if not tool:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "success": False,
                                "error": f"Tool '{name}' not found"
                            })
                        )
                    ]
                
                # Extract operation and arguments
                operation = arguments.get("operation")
                tool_arguments = arguments.get("arguments", {})
                
                # Create tool context
                ctx = ToolContext(session_id=self.session_id)
                
                # Execute tool
                result = await tool.execute(
                    operation=operation,
                    arguments=tool_arguments,
                    ctx=ctx
                )
                
                # Format result as JSON
                result_dict = {
                    "success": result.success,
                    "output": result.output,
                }
                if result.error:
                    result_dict["error"] = result.error
                if result.metadata:
                    result_dict["metadata"] = result.metadata
                
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result_dict)
                    )
                ]
                
            except Exception as e:
                self.logger.error(f"Error calling tool {name}: {e}", exc_info=True)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "error": str(e)
                        })
                    )
                ]
    
    async def run(self):
        """Run MCP server (stdio)."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def create_mcp_server(tool_registry: ToolRegistry, session_id: Optional[str] = None) -> ForgeAgentMCPServer:
    """Create MCP server for Forge Agent tools.
    
    Args:
        tool_registry: Tool registry with all tools
        session_id: Optional session ID
        
    Returns:
        MCP server instance
    """
    return ForgeAgentMCPServer(tool_registry, session_id)
