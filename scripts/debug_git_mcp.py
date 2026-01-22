#!/usr/bin/env python3
"""Debug script for Git MCP server connection.

This script helps diagnose issues with the git MCP server by:
1. Testing direct connection to the git MCP server
2. Listing available tools
3. Testing a simple tool call (git_status)
4. Capturing all JSON messages exchanged
5. Identifying path-related errors like /app/f
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("❌ MCP SDK not available. Install with: pip install mcp")
    sys.exit(1)


async def debug_git_mcp():
    """Debug the git MCP server connection."""
    print("🔍 Debugging Git MCP Server Connection\n")
    print("=" * 60)
    
    # Configuration
    workspace_path = Path.home() / "repos"
    repo_path = "/workspace/forge-agent"
    
    # Build docker command
    docker_cmd = [
        "docker",
        "run",
        "-i",
        "--rm",
        "-v",
        f"{workspace_path}:/workspace",
        "mcp/git:latest",
    ]
    
    print(f"📋 Configuration:")
    print(f"   Workspace: {workspace_path}")
    print(f"   Repo path: {repo_path}")
    print(f"   Docker command: {' '.join(docker_cmd[:3])} ... {docker_cmd[-1]}")
    print()
    
    # Create server parameters
    server_params = StdioServerParameters(
        command=docker_cmd[0],
        args=docker_cmd[1:],
    )
    
    print("🔌 Connecting to git MCP server...")
    
    try:
        # Connect via stdio
        async with stdio_client(server_params) as (read, write):
            session = ClientSession(read, write)
            
            print("✅ Connected! Initializing session...")
            
            # Initialize
            init_result = await session.initialize()
            print(f"✅ Initialized: {init_result.protocol_version}")
            print()
            
            # List tools
            print("📋 Listing available tools...")
            tools_result = await session.list_tools()
            print(f"✅ Found {len(tools_result.tools)} tools:")
            for tool in tools_result.tools:
                print(f"   - {tool.name}: {tool.description[:60]}...")
            print()
            
            # Find git_status tool
            status_tool = None
            for tool in tools_result.tools:
                if "status" in tool.name.lower():
                    status_tool = tool
                    break
            
            if not status_tool:
                print("❌ git_status tool not found!")
                return
            
            print(f"🧪 Testing tool: {status_tool.name}")
            print(f"   Description: {status_tool.description}")
            print()
            
            # Test git_status with different paths
            test_paths = [
                "/workspace/forge-agent",
                "/workspace",
                "/app/forge-agent",  # This might cause the /app/f error
                "/app/f",  # This is the problematic path
            ]
            
            for test_path in test_paths:
                print(f"🔍 Testing with repo_path: {test_path}")
                try:
                    result = await session.call_tool(
                        status_tool.name,
                        {"repo_path": test_path}
                    )
                    
                    print(f"   ✅ Success!")
                    if result.content:
                        content = result.content[0]
                        if hasattr(content, "text"):
                            text = content.text[:200]
                            print(f"   Response: {text}...")
                        else:
                            print(f"   Response: {content}")
                    print()
                    
                except Exception as e:
                    print(f"   ❌ Error: {type(e).__name__}: {e}")
                    print(f"   Error details: {str(e)[:200]}")
                    print()
            
            print("✅ Debug session completed!")
            
    except Exception as e:
        print(f"❌ Connection failed: {type(e).__name__}: {e}")
        print(f"   Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(debug_git_mcp())
    sys.exit(0 if success else 1)
