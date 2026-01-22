#!/usr/bin/env python3
"""Test 5: Executor with English prompt."""

import asyncio
import sys
import tempfile
from pathlib import Path
# Project root is already in path when running from tests/

from agent.config.loader import ConfigLoader
from agent.runtime.langchain_executor import LangChainExecutor
from agent.tools.base import ToolRegistry
from api.dependencies import _register_mcp_tools


async def test_executor_english():
    """Test executor with English prompt."""
    print("🔍 Test 5: Executor with English prompt...")
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "agent.ollama.yaml"
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    
    # Create temp workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        config.workspace.base_path = tmpdir
        print(f"   📁 Workspace: {tmpdir}")
        
        # Create tool registry
        registry = ToolRegistry()
        await _register_mcp_tools(registry, config)
        
        # Create executor
        executor = LangChainExecutor(
            config=config,
            tool_registry=registry,
            session_id="test_english",
            max_iterations=10,
        )
        
        # Test with English prompt
        prompt = "Create a hello world Python file in the current directory"
        print(f"   📤 Prompt: {prompt}")
        
        result = await executor.run(
            user_message=prompt,
            conversation_history=None,
        )
        
        print(f"   📊 Result: success={result.get('success')}")
        print(f"   📝 Response: {result.get('response', 'No response')[:300]}")
        
        if result.get('error'):
            print(f"   ❌ Error: {result.get('error')}")
        
        # Check if file was created
        test_file = Path(tmpdir) / "hello-world.py"
        if not test_file.exists():
            # Try other possible names
            all_files = list(Path(tmpdir).iterdir())
            print(f"   📁 Files in workspace: {[f.name for f in all_files]}")
            for f in all_files:
                if f.suffix == '.py':
                    print(f"   ✅ Found Python file: {f.name}")
                    print(f"   📄 Content: {f.read_text()[:200]}")
                    return True
        
        if test_file.exists():
            content = test_file.read_text()
            print(f"   ✅ File created: {test_file}")
            print(f"   📄 Content: {content}")
            return True
        else:
            return False


if __name__ == "__main__":
    success = asyncio.run(test_executor_english())
    exit(0 if success else 1)
