"""Integration test to verify LLM lists available tools correctly.

This test asks the LLM "Quais tools você tem disponível?" and verifies
that it correctly lists all available tools, including filesystem.
"""

import tempfile
from pathlib import Path
from typing import List, Any

import pytest
import pytest_asyncio

from langchain_ollama import ChatOllama
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.config.loader import ConfigLoader
from agent.runtime.langchain_executor import LangChainExecutor


@pytest.fixture
def test_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()


@pytest.fixture
def llm():
    """Create ChatOllama LLM instance."""
    return ChatOllama(
        model="hhao/qwen2.5-coder-tools",
        base_url="http://localhost:11434",
        temperature=0.0,  # Deterministic output
        timeout=60.0,
    )


@pytest_asyncio.fixture
async def all_tools(test_workspace: Path) -> List[Any]:
    """Get all tools from configured MCP servers."""
    # Load config to get MCP server configurations
    config_path = Path(__file__).parent.parent.parent / "config" / "agent.ollama.yaml"
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    
    # Build MultiServerMCPClient config from config (same as LangChainExecutor)
    from pathlib import Path as PathLib
    import os
    
    mcp_configs = {}
    workspace_base = PathLib(config.workspace.base_path).expanduser().resolve()
    all_mcp_configs = config.mcp or {}
    
    for server_name, mcp_config in all_mcp_configs.items():
        if mcp_config.get("enabled") is False:
            continue
        
        server_type = mcp_config.get("type", "docker")
        
        if server_type == "docker":
            image = mcp_config.get("image")
            if not image:
                continue
            
            docker_cmd = ["docker", "run", "-i", "--rm"]
            
            volumes = mcp_config.get("volumes", [])
            for volume in volumes:
                if "{{workspace.base_path}}" in volume:
                    volume = volume.replace("{{workspace.base_path}}", str(workspace_base))
                if volume.startswith("~"):
                    volume = str(PathLib(volume).expanduser())
                docker_cmd.extend(["-v", volume])
            
            env_vars = mcp_config.get("environment", {})
            resolved_env = {}
            for key, value in env_vars.items():
                if isinstance(value, str) and value.startswith("{{env:") and value.endswith("}}"):
                    env_var_name = value[6:-2]
                    resolved_env[key] = os.getenv(env_var_name, "") or ""
                else:
                    resolved_env[key] = str(value)
            
            for key, value in resolved_env.items():
                if value:
                    docker_cmd.extend(["-e", f"{key}={value}"])
            
            docker_cmd.append(image)
            docker_cmd.extend(mcp_config.get("args", []))
            
            mcp_configs[server_name] = {
                "command": docker_cmd[0],
                "args": docker_cmd[1:],
                "transport": "stdio",
            }
    
    # Get tools from all servers
    if mcp_configs:
        client = MultiServerMCPClient(mcp_configs)
        tools = await client.get_tools()
        return tools
    
    return []


async def create_agent_for_tools(llm, tools: List[Any], config) -> AgentExecutor:
    """Create a tool calling agent and executor for given tools."""
    # Generate system prompt using LangChainExecutor's method
    system_prompt_str = await LangChainExecutor.format_system_prompt(config, tools)
    
    # Debug: print system prompt to verify it contains filesystem
    print(f"\n{'='*80}")
    print("SYSTEM PROMPT BEING USED:")
    print(f"{'='*80}")
    if "filesystem" in system_prompt_str.lower():
        print("✅ System prompt CONTAINS 'filesystem'")
        # Show filesystem line
        for line in system_prompt_str.split("\n"):
            if "filesystem -" in line.lower():
                print(f"   {line[:150]}...")
                break
    else:
        print("❌ System prompt DOES NOT contain 'filesystem'!")
    print(f"{'='*80}\n")
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_str),
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
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )
    
    return agent_executor


@pytest.mark.asyncio
async def test_llm_lists_available_tools_including_filesystem(
    llm: ChatOllama,
    all_tools: List[Any],
    test_workspace: Path,
):
    """Test that LLM correctly lists all available tools when asked, including filesystem."""
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "agent.ollama.yaml"
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    
    # Create agent
    agent_executor = await create_agent_for_tools(llm, all_tools, config)
    
    # Ask the exact question
    question = "Quais tools você tem disponível?"
    
    print(f"\n{'='*80}")
    print(f"TEST: LLM lists available tools")
    print(f"{'='*80}")
    print(f"Question: {question}")
    print()
    
    result = await agent_executor.ainvoke({"input": question})
    
    output = result.get("output", "")
    
    print(f"LLM Response:")
    print(output)
    print(f"{'='*80}\n")
    
    # Verify filesystem is mentioned
    output_lower = output.lower()
    
    # Check for filesystem mentions
    filesystem_mentioned = (
        "filesystem" in output_lower or
        "file system" in output_lower or
        "sistema de arquivos" in output_lower
    )
    
    # Check for filesystem tools
    filesystem_tools_mentioned = any(
        tool in output_lower
        for tool in ["list_directory", "read_file", "write_file", "create_directory"]
    )
    
    assert filesystem_mentioned or filesystem_tools_mentioned, (
        f"LLM did not mention filesystem in response. "
        f"Response: {output[:500]}"
    )
    
    # Also verify that the response is not just generic
    # It should mention specific tools or servers
    generic_responses = [
        "variedade de ferramentas",
        "diferentes tarefas",
        "principais categorias",
    ]
    
    is_too_generic = all(phrase in output_lower for phrase in generic_responses[:2])
    
    if is_too_generic:
        # If too generic, check if it at least mentions some specific tools
        specific_tools = [
            "filesystem", "git", "playwright", "openapi", "python",
            "list_directory", "read_file", "status", "navigate", "click"
        ]
        has_specific_tools = any(tool in output_lower for tool in specific_tools)
        
        assert has_specific_tools, (
            f"LLM response is too generic and doesn't mention specific tools. "
            f"Response: {output[:500]}"
        )
    
    print("✅ Test passed: LLM correctly mentions filesystem or filesystem tools")
