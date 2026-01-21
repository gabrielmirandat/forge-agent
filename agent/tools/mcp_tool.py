"""MCP tool wrapper - wraps MCP server tools as Forge Agent tools.

This allows MCP tools to be used seamlessly alongside native tools.
"""

from typing import Any, Dict, Optional

from agent.tools.base import Tool, ToolContext, ToolResult


class MCPTool(Tool):
    """Tool wrapper for MCP server tools.
    
    This class wraps MCP (Model Context Protocol) server tools to make them
    compatible with the Forge Agent tool system. It handles tool execution
    through the MCP client and converts MCP responses to ToolResult format.
    
    According to MCP Python SDK documentation, tools can return:
    - TextContent: Plain text results
    - Structured content: JSON-serializable data
    - Error responses: Indicated by isError flag
    
    Reference: _helpers/python-sdk/README.md
    """

    def __init__(
        self,
        mcp_name: str,
        mcp_tool: Any,  # types.Tool from MCP SDK
        config: Dict[str, Any],
    ) -> None:
        """Initialize MCP tool wrapper.

        Args:
            mcp_name: Name of the MCP server providing this tool.
            mcp_tool: MCP tool definition from the MCP SDK (types.Tool).
            config: Tool configuration dict containing enabled flag and other settings.
        """
        super().__init__(config)
        self.mcp_name = mcp_name
        self.mcp_tool = mcp_tool
        self._tool_name = f"{mcp_name}_{mcp_tool.name}"

    @property
    def name(self) -> str:
        """Return tool name (with server prefix)."""
        return self._tool_name

    @property
    def description(self) -> str:
        """Return tool description."""
        return self.mcp_tool.description or f"MCP tool from {self.mcp_name}"

    async def execute(
        self,
        operation: str,
        arguments: Dict[str, Any],
        ctx: ToolContext | None = None
    ) -> ToolResult:
        """Execute MCP tool via the MCP client.
        
        This method calls the MCP server tool and converts the response
        to a ToolResult. It handles both text and structured content,
        as well as error responses.

        Args:
            operation: Operation name (should match tool name without server prefix).
            arguments: Operation arguments as a dictionary.
            ctx: Optional tool context (not currently used for MCP tools).

        Returns:
            ToolResult with success status, output data, and optional error message.
            The output can be text, structured data, or a combination of both.
        """
        if not self.enabled:
            return ToolResult(success=False, output=None, error="MCP tool is disabled")

        from agent.runtime.mcp_client import get_mcp_manager

        manager = get_mcp_manager()

        try:
            # Call MCP tool
            result = await manager.call_tool(self._tool_name, arguments)

            if result.isError:
                error_text = ""
                for content in result.content:
                    if hasattr(content, "text"):
                        error_text += content.text
                return ToolResult(
                    success=False,
                    output=None,
                    error=error_text or "MCP tool execution failed",
                )

            # Extract output from result
            output_text = ""
            output_data = []
            
            for content in result.content:
                if hasattr(content, "text"):
                    output_text += content.text
                elif hasattr(content, "data"):
                    # Handle structured content
                    output_data.append(content.data)
                else:
                    # Handle other content types
                    output_data.append(str(content))

            output = output_text if output_text else output_data if output_data else None

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "mcp_server": self.mcp_name,
                    "tool_name": self.mcp_tool.name,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"MCP tool execution error: {e}",
            )
