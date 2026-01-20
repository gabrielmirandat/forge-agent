#!/usr/bin/env python3
"""Test 2: LangChain Executor with simple prompt."""

import asyncio
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, '/home/gabriel-miranda/repos/forge-agent')

from agent.config.loader import ConfigLoader
from agent.runtime.langchain_executor import LangChainExecutor
from agent.tools.base import ToolRegistry
from api.dependencies import _register_mcp_tools


async def test_executor_simple():
    """Test executor with simple prompt."""
    print("🔍 Test 2: LangChain Executor with 'Hey'...")
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "agent.ollama.yaml"
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    
    # Create tool registry
    registry = ToolRegistry()
    await _register_mcp_tools(registry, config)
    print(f"   ✅ Registered {len(registry.list_enabled())} tools")
    
    # Create executor
    executor = LangChainExecutor(
        config=config,
        tool_registry=registry,
        session_id="test_hey",
        max_iterations=5,
    )
    print("   ✅ Executor created")
    
    # Test simple prompt
    print("   📤 Sending: 'Hey'")
    result = await executor.run(
        user_message="Hey",
        conversation_history=None,
    )
    
    print(f"   📊 Result: success={result.get('success')}")
    print(f"   📝 Response: {result.get('response', 'No response')[:200]}")
    
    if result.get('error'):
        print(f"   ❌ Error: {result.get('error')}")
    
    return result.get('success') is True


if __name__ == "__main__":
    success = asyncio.run(test_executor_simple())
    exit(0 if success else 1)
