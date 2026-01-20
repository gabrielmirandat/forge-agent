"""Integration test for LangChain executor with Ollama and MCP tools.

Tests the complete flow:
1. LangChain executor initialization
2. Ollama LLM connection
3. MCP tools availability
4. Agent execution with tool calling
5. File creation via desktop_commander tool
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from agent.config.loader import AgentConfig, ConfigLoader
from agent.runtime.langchain_executor import LangChainExecutor
from agent.tools.base import ToolRegistry


@pytest.fixture
def test_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_config(test_workspace):
    """Create test configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "agent.ollama.yaml"
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    
    # Override workspace path for testing
    config.workspace.base_path = test_workspace
    
    return config


@pytest.fixture
def tool_registry(test_config):
    """Initialize tool registry with MCP tools."""
    import asyncio
    import nest_asyncio
    from api.dependencies import _register_mcp_tools
    
    # Apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()
    
    registry = ToolRegistry()
    
    # Register MCP tools (run async code in sync fixture)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, use nest_asyncio
            asyncio.run(_register_mcp_tools(registry, test_config))
        else:
            loop.run_until_complete(_register_mcp_tools(registry, test_config))
    except RuntimeError:
        # No event loop, create one
        asyncio.run(_register_mcp_tools(registry, test_config))
    
    return registry


@pytest.mark.asyncio
async def test_langchain_create_file(test_config, tool_registry, test_workspace):
    """Test LangChain executor creating a file via tool calling.
    
    This test verifies:
    1. LangChain executor can be initialized
    2. Ollama LLM can be connected
    3. MCP tools are available
    4. Agent can execute tool calls
    5. File is created successfully
    """
    # Change to test workspace
    original_cwd = os.getcwd()
    try:
        os.chdir(test_workspace)
        
        # Initialize LangChain executor
        executor = LangChainExecutor(
            config=test_config,
            tool_registry=tool_registry,
            session_id="test_session",
            max_iterations=10,
        )
        
        # Verify executor is initialized (agent graph will be created lazily on first run)
        assert executor is not None
        
        # Execute prompt to create hello-world.py
        # Use the exact prompt requested by user (in Portuguese)
        prompt = "Crie um hello world em python no diretorio atual"
        
        print(f"\n🔍 Executing prompt: {prompt}")
        print(f"📁 Working directory: {test_workspace}")
        print(f"🔧 Available tools: {[tool.name for tool in executor.langchain_tools if 'write' in tool.name.lower() or 'file' in tool.name.lower()]}")
        
        result = await executor.run(
            user_message=prompt,
            conversation_history=None,
        )
        
        # Verify execution succeeded
        assert result is not None, "Result is None"
        print(f"\n📊 Execution result: {result}")
        
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            print(f"❌ Execution failed: {error_msg}")
            print(f"📝 Response: {result.get('response', 'No response')}")
        
        assert result.get("success") is True, f"Execution failed: {result.get('error', 'Unknown error')}. Response: {result.get('response', 'No response')}"
        
        # Verify file was created
        file_path = Path(test_workspace) / "hello-world.py"
        if not file_path.exists():
            # List all files in workspace for debugging
            all_files = list(Path(test_workspace).iterdir())
            print(f"\n📁 Files in workspace: {[f.name for f in all_files]}")
            print(f"📝 Agent response: {result.get('response', 'No response')}")
        
        assert file_path.exists(), f"File {file_path} was not created. Files in workspace: {[f.name for f in Path(test_workspace).iterdir()]}"
        
        # Verify file content
        content = file_path.read_text()
        assert "Hello, World!" in content or "hello" in content.lower(), f"Unexpected file content: {content}"
        
        print(f"✅ File created successfully: {file_path}")
        print(f"📄 File content:\n{content}")
        
    finally:
        os.chdir(original_cwd)


@pytest.mark.asyncio
async def test_langchain_simple_prompt(test_config, tool_registry):
    """Test LangChain executor with a simple prompt (no tools).
    
    This test verifies basic LLM communication without tool calling.
    """
    # Initialize LangChain executor
    executor = LangChainExecutor(
        config=test_config,
        tool_registry=tool_registry,
        session_id="test_session_simple",
        max_iterations=5,
    )
    
    # Simple prompt that doesn't require tools
    prompt = "What is 2 + 2? Answer with just the number."
    
    print(f"\n🔍 Executing simple prompt: {prompt}")
    
    result = await executor.run(
        user_message=prompt,
        conversation_history=None,
    )
    
    # Verify execution succeeded
    assert result is not None
    assert result.get("success") is True, f"Execution failed: {result.get('error', 'Unknown error')}"
    
    response = result.get("response", "")
    assert response is not None
    assert len(response) > 0
    
    print(f"✅ LLM response: {response}")
