"""LangChain adapter for Forge Agent tools.

Converts Forge Agent Tool classes to LangChain tools for use with LangChain agents.
"""

import json
from typing import Any, Dict, Optional

from langchain_core.tools import BaseTool, tool
from langchain_core.callbacks import CallbackManagerForToolRun

from agent.tools.base import Tool, ToolContext, ToolResult


class LangChainToolAdapter(BaseTool):
    """Adapter that wraps a Forge Agent Tool as a LangChain tool.
    
    This allows Forge Agent tools to be used with LangChain's agent system
    while maintaining compatibility with the existing tool interface.
    """
    
    name: str
    description: str
    forge_tool: Tool
    session_id: Optional[str] = None
    
    def __init__(
        self,
        forge_tool: Tool,
        session_id: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize LangChain tool adapter.
        
        Args:
            forge_tool: Forge Agent Tool instance
            session_id: Optional session ID for tool context
            **kwargs: Additional arguments for BaseTool
        """
        super().__init__(
            name=forge_tool.name,
            description=forge_tool.description,
            forge_tool=forge_tool,
            session_id=session_id,
            **kwargs
        )
    
    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Synchronous tool execution (wraps async).
        
        Args:
            *args: Positional arguments (not used)
            **kwargs: Keyword arguments - for MCP tools, these are the schema parameters directly
                      For non-MCP tools, expects 'operation' and 'arguments'
            
        Returns:
            Tool result as JSON string
        """
        import asyncio
        
        # Run async method in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to use a different approach
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return asyncio.run(self._arun(*args, **kwargs))
                except ImportError:
                    # Fallback: create a new event loop in a thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._arun(*args, **kwargs))
                        return future.result()
            else:
                return loop.run_until_complete(self._arun(*args, **kwargs))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self._arun(*args, **kwargs))
    
    async def _arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Asynchronous tool execution.
        
        Args:
            *args: Positional arguments (not used)
            **kwargs: Keyword arguments
                      - For MCP tools: these are the schema parameters directly (e.g., path, content, mode)
                      - For non-MCP tools: expects 'operation' and 'arguments'
            
        Returns:
            Tool result as JSON string
        """
        # Check if this is an MCP tool (has mcp_name attribute)
        is_mcp_tool = hasattr(self.forge_tool, 'mcp_name')
        
        if is_mcp_tool:
            # MCP tools: LangChain passes schema parameters directly (e.g., path, content, mode)
            # The MCP tool expects these parameters directly, not wrapped in 'arguments'
            # For MCP tools, we call the tool with the actual MCP tool name (not the prefixed name)
            mcp_tool_name = self.forge_tool.mcp_tool.name  # e.g., "write_file"
            mcp_arguments = kwargs.copy()  # e.g., {"path": "...", "content": "...", "mode": "rewrite"}
            
            # Call MCP tool directly via manager
            from agent.runtime.mcp_client import get_mcp_manager
            manager = get_mcp_manager()
            
            try:
                # Use the full tool name (with server prefix) for the manager
                result_mcp = await manager.call_tool(self.forge_tool.name, mcp_arguments)
                
                if result_mcp.isError:
                    error_text = ""
                    for content in result_mcp.content:
                        if hasattr(content, "text"):
                            error_text += content.text
                    return json.dumps({
                        "success": False,
                        "error": error_text or "MCP tool execution failed"
                    })
                
                # Extract output from result
                output_text = ""
                output_data = []
                
                for content in result_mcp.content:
                    if hasattr(content, "text"):
                        output_text += content.text
                    elif hasattr(content, "data"):
                        output_data.append(content.data)
                    else:
                        output_data.append(str(content))
                
                output = output_text if output_text else output_data if output_data else None
                
                return json.dumps({
                    "success": True,
                    "output": output,
                    "metadata": {
                        "mcp_server": self.forge_tool.mcp_name,
                        "tool_name": mcp_tool_name,
                    }
                })
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "error": f"MCP tool execution error: {e}"
                })
        else:
            # Non-MCP tools: extract operation and arguments
            operation = kwargs.get("operation", "")
            arguments = kwargs.get("arguments", {})
            
            if not operation:
                return json.dumps({"success": False, "error": "Missing 'operation' argument"})
            
            if not isinstance(arguments, dict):
                return json.dumps({"success": False, "error": "Arguments must be a dictionary"})
            
            # Create tool context
            ctx = ToolContext(session_id=self.session_id)
            
            # Execute tool
            result = await self.forge_tool.execute(
                operation=operation,
                arguments=arguments,
                ctx=ctx
            )
        
        # Format result as JSON string (LangChain expects string return)
        result_dict = {
            "success": result.success,
            "output": result.output,
        }
        if result.error:
            result_dict["error"] = result.error
        if result.metadata:
            result_dict["metadata"] = result.metadata
        
        return json.dumps(result_dict)
    
    # Removed args_schema property to avoid recursion issues
    # For MCP tools, we need to use StructuredTool.from_function with proper schema
    # This will be handled in get_langchain_tools by creating tools directly from MCP schemas


def forge_tool_to_langchain(
    forge_tool: Tool,
    session_id: Optional[str] = None,
) -> BaseTool:
    """Convert a Forge Agent Tool to a LangChain tool.
    
    Args:
        forge_tool: Forge Agent Tool instance
        session_id: Optional session ID for tool context
        
    Returns:
        LangChain BaseTool instance
    """
    return LangChainToolAdapter(
        forge_tool=forge_tool,
        session_id=session_id
    )


def tools_to_langchain(
    tools: list[Tool],
    session_id: Optional[str] = None,
) -> list[BaseTool]:
    """Convert a list of Forge Agent Tools to LangChain tools.
    
    Args:
        tools: List of Forge Agent Tool instances
        session_id: Optional session ID for tool context
        
    Returns:
        List of LangChain BaseTool instances
    """
    return [
        forge_tool_to_langchain(tool, session_id=session_id)
        for tool in tools
        if tool.enabled
    ]
