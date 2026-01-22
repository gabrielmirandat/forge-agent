#!/usr/bin/env python3
"""Test 3: Direct MCP tool call (write_file)."""

import asyncio
import sys
import tempfile
from pathlib import Path
# Project root is already in path when running from tests/

from agent.config.loader import ConfigLoader
from agent.runtime.mcp_client import get_mcp_manager
from api.dependencies import _register_mcp_tools
from agent.tools.base import ToolRegistry


async def test_tool_direct():
    """Test direct MCP tool call."""
    print("🔍 Test 3: Direct MCP tool call (write_file)...")
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "agent.ollama.yaml"
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    
    # Register tools
    registry = ToolRegistry()
    await _register_mcp_tools(registry, config)
    
    # Get MCP manager
    manager = get_mcp_manager()
    
    # Create temp workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "hello-world.py"
        print(f"   📁 Workspace: {tmpdir}")
        print(f"   📄 Target file: {test_file}")
        
        # Call tool directly
        print("   🔧 Calling filesystem_write_file...")
        try:
            result = await manager.call_tool(
                "filesystem_write_file",
                {
                    "path": str(test_file),
                    "content": "print('Hello, World!')",
                    "mode": "rewrite"  # Valid values: "rewrite" or "append"
                }
            )
            
            print(f"   📊 Tool result: {result}")
            
            # Check if file was created
            if test_file.exists():
                content = test_file.read_text()
                print(f"   ✅ File created!")
                print(f"   📄 Content: {content}")
                return True
            else:
                print(f"   ❌ File was not created")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = asyncio.run(test_tool_direct())
    exit(0 if success else 1)
