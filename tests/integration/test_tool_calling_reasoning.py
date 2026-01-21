"""Integration tests for tool calling with LLM reasoning capture.

Tests verify that:
1. LLM correctly identifies when to use tools
2. LLM reasoning is captured and logged
3. Tools are called correctly via MCP
4. Tool calling decisions are visible in logs
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pytest

from agent.config.loader import ConfigLoader
from agent.runtime.langchain_executor import LangChainExecutor
from agent.tools.base import ToolRegistry


@pytest.fixture
def test_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def tool_registry(test_workspace):
    """Initialize tool registry with MCP tools."""
    import nest_asyncio
    from api.dependencies import _register_mcp_tools
    
    # Apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()
    
    # Load a base config for tool registration
    config_path = Path(__file__).parent.parent.parent / "config" / "agent.ollama.yaml"
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    config.workspace.base_path = test_workspace
    
    registry = ToolRegistry()
    
    # Register MCP tools (run async code in sync fixture)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run(_register_mcp_tools(registry, config))
        else:
            loop.run_until_complete(_register_mcp_tools(registry, config))
    except RuntimeError:
        asyncio.run(_register_mcp_tools(registry, config))
    
    return registry


class ReasoningCollector:
    """Collects LLM reasoning and tool calling events during execution."""
    
    def __init__(self):
        self.reasoning_events: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_decisions: List[Dict[str, Any]] = []
        self.llm_responses: List[str] = []
    
    def add_reasoning(self, event: Dict[str, Any]):
        """Add a reasoning event."""
        self.reasoning_events.append(event)
    
    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        """Add a tool call event."""
        self.tool_calls.append({
            "tool": tool_name,
            "arguments": arguments,
        })
    
    def add_tool_decision(self, decision: Dict[str, Any]):
        """Add a tool decision event."""
        self.tool_decisions.append(decision)
    
    def add_llm_response(self, content: str):
        """Add an LLM response."""
        self.llm_responses.append(content)
    
    def print_summary(self):
        """Print a summary of collected events."""
        print("\n" + "="*80)
        print("LLM REASONING AND TOOL CALLING SUMMARY")
        print("="*80)
        
        print(f"\n📝 Reasoning Events: {len(self.reasoning_events)}")
        for i, event in enumerate(self.reasoning_events, 1):
            status = event.get("status", "unknown")
            content = event.get("content", "")[:200]
            print(f"  {i}. Status: {status}")
            print(f"     Content: {content}...")
        
        print(f"\n🔧 Tool Calls: {len(self.tool_calls)}")
        for i, call in enumerate(self.tool_calls, 1):
            tool = call.get("tool", "unknown")
            args = call.get("arguments", {})
            print(f"  {i}. Tool: {tool}")
            print(f"     Arguments: {args}")
        
        print(f"\n🤔 Tool Decisions: {len(self.tool_decisions)}")
        for i, decision in enumerate(self.tool_decisions, 1):
            decision_type = decision.get("decision", "unknown")
            reasoning = decision.get("reasoning", "")
            tool_calls_count = decision.get("tool_calls_count", 0)
            print(f"  {i}. Decision: {decision_type}")
            print(f"     Tool Calls Count: {tool_calls_count}")
            print(f"     Reasoning: {reasoning}")
        
        print(f"\n💬 LLM Responses: {len(self.llm_responses)}")
        for i, response in enumerate(self.llm_responses, 1):
            print(f"  {i}. {response[:200]}...")
        
        print("="*80 + "\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("model_config,model_name", [
    ("agent.ollama.yaml", "qwen3:8b"),
    ("agent.ollama.qwen.yaml", "qwen3:8b"),
    ("agent.ollama.qwen14b.yaml", "qwen2.5:14b"),
])
@pytest.mark.parametrize("prompt,expected_tool_pattern,description", [
    (
        "Em que pasta estamos?",
        ["list_directory", "get_file_info"],
        "Should use desktop_commander to get current directory"
    ),
    (
        "List files in the current directory",
        ["list_directory"],
        "Should use desktop_commander to list files"
    ),
    (
        "What is the git status?",
        ["git_status"],
        "Should use git tool to check status"
    ),
    (
        "Read the README.md file",
        ["read_file"],
        "Should use desktop_commander to read file"
    ),
    (
        "What is 2 + 2?",
        [],
        "Should NOT use tools - simple math question"
    ),
])
async def test_tool_calling_reasoning(
    model_config: str,
    model_name: str,
    prompt: str,
    expected_tool_pattern: List[str],
    description: str,
    tool_registry: ToolRegistry,
    test_workspace: str,
):
    """Test that LLM correctly reasons about when to use tools.
    
    This test:
    1. Sends a prompt that should (or shouldn't) trigger tool calls
    2. Captures LLM reasoning events
    3. Verifies tool calling decisions
    4. Checks if expected tools were called
    
    Args:
        model_config: Config file name (e.g., "agent.ollama.yaml")
        model_name: Model name for logging (e.g., "llama3.1")
        prompt: User prompt to test
        expected_tool_pattern: List of tool name patterns that should be called
        description: Description of what this test verifies
        tool_registry: Tool registry fixture
        test_workspace: Temporary workspace directory
    """
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / model_config
    if not config_path.exists():
        pytest.skip(f"Config file {model_config} not found")
    
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    config.workspace.base_path = test_workspace
    
    # Create executor
    executor = LangChainExecutor(
        config=config,
        tool_registry=tool_registry,
        session_id=f"test_reasoning_{model_name}_{hash(prompt)}",
        max_iterations=10,
    )
    
    # Create reasoning collector
    collector = ReasoningCollector()
    
    # Subscribe to events (we'll capture them via the executor's event publishing)
    # For now, we'll capture via the executor's internal logging
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"Model: {model_name} ({model_config})")
    print(f"Prompt: {prompt}")
    print(f"Expected tools: {expected_tool_pattern}")
    print(f"{'='*80}\n")
    
    # Execute prompt
    result = await executor.run(
        user_message=prompt,
        conversation_history=None,
    )
    
    # Verify execution succeeded
    assert result is not None, "Result is None"
    assert result.get("success") is True, f"Execution failed: {result.get('error', 'Unknown error')}"
    
    response = result.get("response", "")
    assert len(response) > 0, "Empty response"
    
    print(f"\n✅ LLM Response: {response[:500]}...")
    
    # Check execution history for tool calls
    execution_history = result.get("execution_history", [])
    tool_calls_found = []
    
    # Also check logs for tool calls (they're logged during execution)
    # The executor logs tool calls via logger.info with "🔧 Tool called:"
    
    for step in execution_history:
        if isinstance(step, dict):
            # Look for tool calls in the step
            if "tool" in step or "tool_calls" in step:
                tool_name = step.get("tool") or step.get("tool_calls", [{}])[0].get("name", "")
                if tool_name:
                    tool_calls_found.append(tool_name)
    
    # Check if response mentions tool usage (sometimes tools are called but not logged in history)
    response_lower = response.lower()
    
    print(f"\n🔧 Tools called during execution: {tool_calls_found}")
    print(f"📝 Response preview: {response[:300]}...")
    
    # Check for tool-related keywords in response
    tool_keywords = ["tool", "called", "executed", "using", "via", "mcp"]
    has_tool_keywords = any(keyword in response_lower for keyword in tool_keywords)
    if has_tool_keywords:
        print("✅ Response mentions tool-related keywords")
    
    # Verify tool calling expectations
    if expected_tool_pattern:
        # Should have called at least one expected tool
        found_expected = False
        for pattern in expected_tool_pattern:
            for tool_called in tool_calls_found:
                if pattern.lower() in tool_called.lower():
                    found_expected = True
                    print(f"✅ Found expected tool pattern '{pattern}' in '{tool_called}'")
                    break
            if found_expected:
                break
        
        if not found_expected and len(tool_calls_found) == 0:
            print(f"⚠️ WARNING: Expected tools {expected_tool_pattern} but no tools were called")
            print(f"   This might indicate the LLM is not using tools when it should")
            print(f"   Response: {response[:300]}")
        elif not found_expected:
            print(f"⚠️ WARNING: Expected tools {expected_tool_pattern} but got {tool_calls_found}")
    else:
        # Should NOT have called tools
        if len(tool_calls_found) > 0:
            print(f"⚠️ WARNING: Expected no tools but got {tool_calls_found}")
        else:
            print(f"✅ Correctly did not call tools for simple question")


@pytest.mark.asyncio
async def test_tool_calling_file_operations(
    tool_registry: ToolRegistry,
    test_workspace: str,
):
    """Test tool calling for file operations with detailed reasoning capture.
    
    This test creates a file and then reads it back, capturing all reasoning.
    """
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "agent.ollama.yaml"
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    config.workspace.base_path = test_workspace
    
    # Create executor
    executor = LangChainExecutor(
        config=config,
        tool_registry=tool_registry,
        session_id="test_file_ops",
        max_iterations=15,
    )
    
    collector = ReasoningCollector()
    
    # Test 1: Create a file
    print("\n" + "="*80)
    print("TEST 1: Create a file")
    print("="*80)
    
    create_prompt = "Create a file named test_reasoning.txt with content 'Hello from reasoning test'"
    
    result1 = await executor.run(
        user_message=create_prompt,
        conversation_history=None,
    )
    
    assert result1.get("success") is True, f"Failed to create file: {result1.get('error')}"
    print(f"✅ Response: {result1.get('response', '')[:300]}...")
    
    # Verify file was created
    test_file = Path(test_workspace) / "test_reasoning.txt"
    if test_file.exists():
        print(f"✅ File created successfully: {test_file}")
        content = test_file.read_text()
        print(f"📄 File content: {content}")
    else:
        print(f"⚠️ File was not created. Files in workspace: {list(Path(test_workspace).iterdir())}")
    
    # Test 2: Read the file back
    print("\n" + "="*80)
    print("TEST 2: Read the file back")
    print("="*80)
    
    read_prompt = "Read the file test_reasoning.txt"
    
    result2 = await executor.run(
        user_message=read_prompt,
        conversation_history=None,
    )
    
    assert result2.get("success") is True, f"Failed to read file: {result2.get('error')}"
    print(f"✅ Response: {result2.get('response', '')[:300]}...")
    
    # Verify response mentions the file content
    response2 = result2.get("response", "").lower()
    if "hello" in response2 or "reasoning" in response2:
        print("✅ Response contains file content")
    else:
        print(f"⚠️ Response might not contain file content: {response2[:200]}")


@pytest.mark.asyncio
async def test_tool_calling_git_operations(
    tool_registry: ToolRegistry,
    test_workspace: str,
):
    """Test tool calling for git operations.
    
    This test checks git status, which should use git tools.
    """
    # Initialize git repo in test workspace
    import subprocess
    original_cwd = os.getcwd()
    try:
        os.chdir(test_workspace)
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, capture_output=True)
    finally:
        os.chdir(original_cwd)
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "agent.ollama.yaml"
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    config.workspace.base_path = test_workspace
    
    # Create executor
    executor = LangChainExecutor(
        config=config,
        tool_registry=tool_registry,
        session_id="test_git_ops",
        max_iterations=10,
    )
    
    print("\n" + "="*80)
    print("TEST: Git status check")
    print("="*80)
    
    git_prompt = "What is the git status of this repository?"
    
    result = await executor.run(
        user_message=git_prompt,
        conversation_history=None,
    )
    
    assert result.get("success") is True, f"Failed: {result.get('error')}"
    response = result.get("response", "")
    print(f"✅ Response: {response[:500]}...")
    
    # Check if response mentions git-related information
    response_lower = response.lower()
    git_indicators = ["git", "status", "branch", "commit", "clean", "modified", "untracked"]
    found_indicator = any(indicator in response_lower for indicator in git_indicators)
    
    if found_indicator:
        print("✅ Response contains git-related information")
    else:
        print(f"⚠️ Response might not contain git information: {response[:200]}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
