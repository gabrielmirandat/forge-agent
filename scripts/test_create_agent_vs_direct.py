#!/usr/bin/env python3
"""Compare create_agent vs direct invocation to find the issue."""

import asyncio
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, '/home/gabriel-miranda/repos/forge-agent')

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool


@tool
def list_directory(path: str = ".") -> str:
    """List files and directories in the given path."""
    dir_path = Path(path)
    if not dir_path.exists():
        return f"Error: Path {path} does not exist"
    items = []
    for item in dir_path.iterdir():
        items.append(f"{'DIR' if item.is_dir() else 'FILE'}: {item.name}")
    return "\n".join(items) if items else "Directory is empty"


async def test_direct_invocation():
    """Test direct invocation (this works)."""
    print("\n" + "="*80)
    print("TEST 1: Direct invocation (llm_with_tools.ainvoke)")
    print("="*80)
    
    llm = ChatOllama(
        model="llama3.1",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=60.0,
    )
    
    tools = [list_directory]
    llm_with_tools = llm.bind_tools(tools)
    
    message = HumanMessage(content="Em que pasta estamos? List the current directory.")
    
    response = await llm_with_tools.ainvoke([message])
    
    print(f"Response type: {type(response)}")
    print(f"Response content: {response.content[:200] if response.content else '(empty)'}")
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"✅ SUCCESS: Tool calls found: {len(response.tool_calls)}")
        for tc in response.tool_calls:
            print(f"   Tool: {tc.get('name', 'unknown')}, Args: {tc.get('args', {})}")
        return True
    else:
        print(f"❌ FAILED: No tool calls")
        return False


async def test_create_agent():
    """Test create_agent (this doesn't work)."""
    print("\n" + "="*80)
    print("TEST 2: create_agent")
    print("="*80)
    
    llm = ChatOllama(
        model="llama3.1",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=60.0,
    )
    
    tools = [list_directory]
    llm_with_tools = llm.bind_tools(tools)
    
    agent = create_agent(
        model=llm_with_tools,
        tools=tools,
        system_prompt="You are a helpful assistant. When asked about directories, use the list_directory tool.",
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            input_data = {
                "messages": [HumanMessage(content="Em que pasta estamos? List the current directory.")]
            }
            
            tool_called = False
            async for event in agent.astream_events(
                input_data,
                version="v2",
                include_names=["tool", "llm", "chain"],
            ):
                event_name = event.get("event", "")
                event_data = event.get("data", {})
                
                if event_name == "on_tool_start":
                    tool_name = event_data.get("name", "")
                    print(f"✅ Tool called: {tool_name}")
                    tool_called = True
                    break  # Found tool call, success
            
            if tool_called:
                print(f"✅ SUCCESS: Tool was called via create_agent")
                return True
            else:
                print(f"❌ FAILED: No tool was called via create_agent")
                return False
                
        finally:
            os.chdir(original_cwd)


async def main():
    """Run both tests."""
    print("\n🔍 Comparing direct invocation vs create_agent\n")
    
    result1 = await test_direct_invocation()
    result2 = await test_create_agent()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Direct invocation: {'✅ WORKS' if result1 else '❌ FAILED'}")
    print(f"create_agent:      {'✅ WORKS' if result2 else '❌ FAILED'}")
    print("="*80)
    
    if result1 and not result2:
        print("\n⚠️ ISSUE FOUND: create_agent is not calling tools correctly!")
        print("   Direct invocation works, but create_agent doesn't.")
        print("   This suggests a problem with how create_agent manages tool calling.")


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
