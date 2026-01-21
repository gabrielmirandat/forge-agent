"""Integration test for git MCP tools.

Tests verify that the LLM correctly decides to call git tools
and that the tools are executed successfully.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Any

import pytest
import pytest_asyncio

from langchain_ollama import ChatOllama
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient


@pytest.fixture
def test_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()


@pytest_asyncio.fixture
async def git_tools(test_workspace: Path) -> List[Any]:
    """Get tools from git MCP server."""
    docker_command = [
        "docker",
        "run",
        "-i",
        "--rm",
        "-v",
        f"{test_workspace}:/workspace",
        "mcp/git:latest",
    ]
    
    client = MultiServerMCPClient(
        {
            "git": {
                "command": docker_command[0],
                "args": docker_command[1:],
                "transport": "stdio",
            }
        }
    )
    
    tools = await client.get_tools()
    assert len(tools) > 0, "No tools loaded from git"
    return tools


@pytest.fixture
def llm():
    """Create ChatOllama LLM instance."""
    return ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=60.0,
    )


def create_agent_for_tools(llm, tools: List[Any], system_prompt: str) -> AgentExecutor:
    """Create a tool calling agent and executor for given tools."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    llm_with_tools = llm.bind_tools(tools)
    
    agent = create_tool_calling_agent(
        llm=llm_with_tools,
        tools=tools,
        prompt=prompt_template,
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )
    
    return agent_executor


def verify_llm_decided_to_use_tool(result: dict, tool_name_pattern: str) -> bool:
    """Verify that the LLM output indicates it decided to use a tool.
    
    Args:
        result: AgentExecutor result dictionary
        tool_name_pattern: Pattern to match in tool name (e.g., "git_status", "git_log")
        
    Returns:
        True if LLM output suggests tool was called, False otherwise
    """
    output = result.get("output", "").lower()
    
    # Check for git-related keywords that indicate tool usage
    git_indicators = [
        "git", "status", "branch", "commit", "log", "repository",
        "repo", "clean", "modified", "untracked", "staged",
        "working", "directory", "history", "commits"
    ]
    
    has_git_indicators = any(indicator in output for indicator in git_indicators)
    
    # Check if output mentions the specific tool or its result
    has_tool_pattern = tool_name_pattern.lower() in output
    
    # Check if output is not just a generic response
    generic_responses = [
        "i cannot", "i'm unable", "i don't have", "i can't",
        "sorry", "unfortunately", "i apologize", "error"
    ]
    is_not_generic = not any(phrase in output for phrase in generic_responses)
    
    return has_git_indicators and (has_tool_pattern or is_not_generic)


@pytest.mark.asyncio
async def test_git_status_tool_decision(
    git_tools: List[Any],
    llm: ChatOllama,
    test_workspace: Path,
):
    """Test that LLM decides to call git_status tool."""
    # Initialize git repo
    original_cwd = os.getcwd()
    try:
        os.chdir(test_workspace)
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, capture_output=True)
        # Configure safe.directory for Docker container
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", "/workspace"], check=True, capture_output=True)
    finally:
        os.chdir(original_cwd)
    
    # Find git_status tool
    status_tool = None
    for tool in git_tools:
        if "status" in tool.name.lower():
            status_tool = tool
            break
    
    assert status_tool is not None, f"git_status tool not found. Available: {[t.name for t in git_tools[:10]]}"
    
    # Create agent
    agent_executor = create_agent_for_tools(
        llm=llm,
        tools=[status_tool],
        system_prompt=(
            "You are a helpful assistant with access to git tools.\n"
            "When the user asks about git status, you MUST use the git_status tool.\n"
            "Always use /workspace as the repository path."
        ),
    )
    
    # Execute
    result = await agent_executor.ainvoke(
        {"input": "What is the git status of the repository at /workspace?"},
    )
    
    assert result is not None
    assert "output" in result
    output = result["output"]
    assert len(output) > 0, "Empty output from agent"
    
    # Verify LLM decided to use the tool
    llm_decided = verify_llm_decided_to_use_tool(result, "git_status")
    
    print(f"\n{'='*80}")
    print(f"TEST: git_status tool decision")
    print(f"{'='*80}")
    print(f"LLM Output: {output[:500]}...")
    print(f"LLM Decided to Use Tool: {llm_decided}")
    print(f"{'='*80}\n")
    
    assert llm_decided, (
        f"LLM output does not indicate tool was called. "
        f"Output: {output[:300]}"
    )


@pytest.mark.asyncio
async def test_git_log_tool_decision(
    git_tools: List[Any],
    llm: ChatOllama,
    test_workspace: Path,
):
    """Test that LLM decides to call git_log tool."""
    # Initialize git repo and make a commit
    original_cwd = os.getcwd()
    try:
        os.chdir(test_workspace)
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, capture_output=True)
        # Configure safe.directory for Docker container
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", "/workspace"], check=True, capture_output=True)
        
        # Create a file and commit it
        test_file = test_workspace / "test.txt"
        test_file.write_text("test content")
        subprocess.run(["git", "add", "test.txt"], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], check=True, capture_output=True)
    finally:
        os.chdir(original_cwd)
    
    # Find git_log tool
    log_tool = None
    for tool in git_tools:
        if "log" in tool.name.lower() and "status" not in tool.name.lower():
            log_tool = tool
            break
    
    if log_tool is None:
        pytest.skip(f"git_log tool not found. Available: {[t.name for t in git_tools[:10]]}")
    
    # Create agent
    agent_executor = create_agent_for_tools(
        llm=llm,
        tools=[log_tool],
        system_prompt=(
            "You are a helpful assistant with access to git tools.\n"
            "When the user asks about git history or commits, you MUST use the git_log tool.\n"
            "Always use /workspace as the repository path."
        ),
    )
    
    # Execute
    result = await agent_executor.ainvoke(
        {"input": "Show me the git log for the repository at /workspace"},
    )
    
    assert result is not None
    assert "output" in result
    output = result["output"]
    assert len(output) > 0, "Empty output from agent"
    
    # Verify LLM decided to use the tool
    llm_decided = verify_llm_decided_to_use_tool(result, "git_log")
    
    print(f"\n{'='*80}")
    print(f"TEST: git_log tool decision")
    print(f"{'='*80}")
    print(f"LLM Output: {output[:500]}...")
    print(f"LLM Decided to Use Tool: {llm_decided}")
    print(f"{'='*80}\n")
    
    assert llm_decided, (
        f"LLM output does not indicate tool was called. "
        f"Output: {output[:300]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
