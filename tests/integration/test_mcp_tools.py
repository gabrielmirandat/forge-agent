#!/usr/bin/env python3
"""Test MCP tools registration."""

import asyncio
import sys
# Project root is already in path when running from tests/

from pathlib import Path
from agent.config.loader import ConfigLoader
from agent.tools.base import ToolRegistry
from api.dependencies import _register_mcp_tools


async def test_mcp_tools():
    """Test MCP tools registration."""
    print("🔍 Testing MCP tools registration...")
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "agent.ollama.yaml"
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    
    # Create tool registry
    registry = ToolRegistry()
    
    # Register MCP tools
    print("   Registering MCP tools...")
    await _register_mcp_tools(registry, config)
    
    # List registered tools
    tools = registry.list_enabled()
    print(f"   ✅ Registered {len(tools)} tools")
    
    # Show tools with 'write' or 'file' in name
    relevant_tools = [t for t in tools if 'write' in t.lower() or 'file' in t.lower()]
    print(f"   File-related tools: {relevant_tools}")
    
    # Show all tools
    print(f"\n   All tools:")
    for tool_name in tools[:10]:  # First 10
        tool = registry.get(tool_name)
        if tool:
            print(f"      - {tool_name}: {tool.description[:60]}...")
    
    return len(tools) > 0


if __name__ == "__main__":
    success = asyncio.run(test_mcp_tools())
    exit(0 if success else 1)
