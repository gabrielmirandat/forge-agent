"""Integration test for filesystem MCP tools.

Tests verify that the LLM correctly decides to call filesystem tools
and that the tools are executed successfully.
"""

import tempfile
import os
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
async def filesystem_tools(test_workspace: Path) -> List[Any]:
    """Get tools from filesystem MCP server (Docker)."""
    # Use official Docker MCP server
    docker_command = [
        "docker",
        "run",
        "-i",
        "--rm",
        "-v",
        f"{test_workspace}:/workspace",
        "mcp/filesystem:latest",
        "/workspace",
    ]
    
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "command": docker_command[0],
                "args": docker_command[1:],
                "transport": "stdio",
            }
        }
    )
    
    tools = await client.get_tools()
    assert len(tools) > 0, "No tools loaded from filesystem"
    return tools


@pytest.fixture
def llm():
    """Create ChatOllama LLM instance."""
    return ChatOllama(
        model="hhao/qwen2.5-coder-tools",
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
        tool_name_pattern: Pattern to match in tool name (e.g., "list_directory", "write_file")
        
    Returns:
        True if LLM output suggests tool was called, False otherwise
    """
    output = result.get("output", "").lower()
    
    # Check for tool-related keywords that indicate tool usage
    tool_indicators = [
        "using", "called", "executed", "invoked", "tool",
        "file", "directory", "created", "read", "wrote",
        "listing", "found", "contents", "path"
    ]
    
    has_tool_indicators = any(indicator in output for indicator in tool_indicators)
    
    # Check if output mentions the specific tool or its result
    has_tool_pattern = tool_name_pattern.lower() in output
    
    # Check if output is not just a generic response
    generic_responses = [
        "i cannot", "i'm unable", "i don't have", "i can't",
        "sorry", "unfortunately", "i apologize"
    ]
    is_not_generic = not any(phrase in output for phrase in generic_responses)
    
    return has_tool_indicators and (has_tool_pattern or is_not_generic)


@pytest.mark.asyncio
async def test_list_directory_tool_decision(
    filesystem_tools: List[Any],
    llm: ChatOllama,
    test_workspace: Path,
):
    """Test that LLM decides to call list_directory tool."""
    # Find list_directory tool
    list_dir_tool = None
    for tool in filesystem_tools:
        if "list_directory" in tool.name.lower() or "list_dir" in tool.name.lower():
            list_dir_tool = tool
            break
    
    assert list_dir_tool is not None, f"list_directory tool not found. Available: {[t.name for t in filesystem_tools[:10]]}"
    
    # Create agent
    agent_executor = create_agent_for_tools(
        llm=llm,
        tools=[list_dir_tool],
        system_prompt=(
            "You are a helpful assistant with access to filesystem tools.\n"
            "When the user asks to list files or directories, you MUST use the list_directory tool.\n"
            f"Use {test_workspace} as the base path (workspace_base parameter)."
        ),
    )
    
    # Execute
    result = await agent_executor.ainvoke(
        {"input": f"List the files in the directory {test_workspace}"},
    )
    
    assert result is not None
    assert "output" in result
    output = result["output"]
    assert len(output) > 0, "Empty output from agent"
    
    # Verify LLM decided to use the tool
    llm_decided = verify_llm_decided_to_use_tool(result, "list_directory")
    
    print(f"\n{'='*80}")
    print(f"TEST: list_directory tool decision")
    print(f"{'='*80}")
    print(f"LLM Output: {output[:500]}...")
    print(f"LLM Decided to Use Tool: {llm_decided}")
    print(f"{'='*80}\n")
    
    assert llm_decided, (
        f"LLM output does not indicate tool was called. "
        f"Output: {output[:300]}"
    )


@pytest.mark.asyncio
async def test_write_file_tool_decision(
    filesystem_tools: List[Any],
    llm: ChatOllama,
    test_workspace: Path,
):
    """Test that LLM decides to call write_file tool."""
    # Find write_file tool
    write_tool = None
    for tool in filesystem_tools:
        if "write_file" in tool.name.lower():
            write_tool = tool
            break
    
    assert write_tool is not None, f"write_file tool not found. Available: {[t.name for t in filesystem_tools[:10]]}"
    
    # Create agent
    agent_executor = create_agent_for_tools(
        llm=llm,
        tools=[write_tool],
        system_prompt=(
            "You are a helpful assistant with access to filesystem tools.\n"
            "When the user asks you to create or write a file, you MUST call the write_file tool.\n"
            "Do not just suggest code - you MUST actually call the tool.\n"
            f"Use {test_workspace} as the workspace_base parameter for files."
        ),
    )
    
    # Execute
    test_file = test_workspace / "test_write_decision.txt"
    result = await agent_executor.ainvoke(
        {"input": f"Create a file named {test_file.name} with content 'Hello from tool decision test' in {test_workspace}"},
    )
    
    assert result is not None
    assert "output" in result
    output = result["output"]
    assert len(output) > 0, "Empty output from agent"
    
    # Verify file was created (tool was executed)
    assert test_file.exists(), f"File {test_file} was not created - tool was not executed"
    content = test_file.read_text()
    assert "Hello" in content or "tool decision" in content.lower(), f"Unexpected file content: {content}"
    
    # Verify LLM decided to use the tool
    llm_decided = verify_llm_decided_to_use_tool(result, "write_file")
    
    print(f"\n{'='*80}")
    print(f"TEST: write_file tool decision")
    print(f"{'='*80}")
    print(f"LLM Output: {output[:500]}...")
    print(f"File Created: {test_file.exists()}")
    print(f"File Content: {content[:200]}...")
    print(f"LLM Decided to Use Tool: {llm_decided}")
    print(f"{'='*80}\n")
    
    assert llm_decided, (
        f"LLM output does not indicate tool was called. "
        f"Output: {output[:300]}"
    )


@pytest.mark.asyncio
async def test_read_file_tool_decision(
    filesystem_tools: List[Any],
    llm: ChatOllama,
    test_workspace: Path,
):
    """Test that LLM decides to call read_file tool."""
    # First create a test file
    test_file = test_workspace / "test_read_decision.txt"
    test_file.write_text("This is test content for reading - tool decision test")
    
    # Find read_file tool
    read_tool = None
    for tool in filesystem_tools:
        if "read_file" in tool.name.lower():
            read_tool = tool
            break
    
    assert read_tool is not None, f"read_file tool not found. Available: {[t.name for t in filesystem_tools[:10]]}"
    
    # Create agent
    agent_executor = create_agent_for_tools(
        llm=llm,
        tools=[read_tool],
        system_prompt=(
            "You are a helpful assistant with access to filesystem tools.\n"
            "When the user asks to read a file, you MUST use the read_file tool.\n"
            f"Use {test_workspace} as the workspace_base parameter."
        ),
    )
    
    # Execute
    result = await agent_executor.ainvoke(
        {"input": f"Read the file {test_file.name} from {test_workspace}"},
    )
    
    assert result is not None
    assert "output" in result
    output = result["output"]
    assert len(output) > 0, "Empty output from agent"
    
    # Verify LLM decided to use the tool
    llm_decided = verify_llm_decided_to_use_tool(result, "read_file")
    
    # Check if response mentions file content
    has_file_content = "test content" in output.lower() or "reading" in output.lower() or "tool decision" in output.lower()
    
    print(f"\n{'='*80}")
    print(f"TEST: read_file tool decision")
    print(f"{'='*80}")
    print(f"LLM Output: {output[:500]}...")
    print(f"Has File Content: {has_file_content}")
    print(f"LLM Decided to Use Tool: {llm_decided}")
    print(f"{'='*80}\n")
    
    assert llm_decided, (
        f"LLM output does not indicate tool was called. "
        f"Output: {output[:300]}"
    )
    
    assert has_file_content, (
        f"LLM output does not contain file content. "
        f"Output: {output[:300]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
