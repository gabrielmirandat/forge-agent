#!/usr/bin/env python3
"""Test 4: Executor with tool calling prompt."""

import asyncio
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, '/home/gabriel-miranda/repos/forge-agent')

from agent.config.loader import ConfigLoader
from agent.runtime.langchain_executor import LangChainExecutor
from agent.tools.base import ToolRegistry
from api.dependencies import _register_mcp_tools


async def test_executor_with_tool():
    """Test executor with tool calling."""
    print("🔍 Test 4: Executor with tool calling...")
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "agent.ollama.yaml"
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
            session_id="test_tool",
            max_iterations=10,
        )
        
        # List available tools
        tools = executor.langchain_tools
        write_tools = [t.name for t in tools if 'write' in t.name.lower() or 'file' in t.name.lower()]
        print(f"   🔧 Available write/file tools: {write_tools[:5]}")
        
        # Test with generic prompt (in Portuguese as requested)
        prompt = "Crie um hello world em python no diretorio atual"
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
        if test_file.exists():
            content = test_file.read_text()
            print(f"   ✅ File created: {test_file}")
            print(f"   📄 Content: {content}")
            return True
        else:
            all_files = list(Path(tmpdir).iterdir())
            print(f"   ❌ File not found. Files in workspace: {[f.name for f in all_files]}")
            return False


if __name__ == "__main__":
    success = asyncio.run(test_executor_with_tool())
    exit(0 if success else 1)
