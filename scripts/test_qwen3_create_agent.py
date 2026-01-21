#!/usr/bin/env python3
"""Test qwen3:8b with create_agent to verify it works."""

import asyncio
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, '/home/gabriel-miranda/repos/forge-agent')

from langchain_ollama import ChatOllama
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

@tool
def list_directory(path: str = ".") -> str:
    """List files and directories in the given path."""
    import os
    try:
        items = os.listdir(path)
        return f"Directory contents: {', '.join(items[:10])}"
    except Exception as e:
        return f"Error: {e}"

async def test_qwen3_create_agent():
    """Test if qwen3:8b works with create_agent."""
    print("🔍 Testing qwen3:8b with create_agent...")
    
    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=60.0,
    )
    
    tools = [list_directory]
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"\n✅ LLM created with {len(tools)} tool(s)")
    print(f"   Model: qwen3:8b")
    print(f"   Tool: {tools[0].name}")
    
    # Create agent with state_schema and checkpointer
    checkpointer = MemorySaver()
    agent = create_agent(
        model=llm_with_tools,
        tools=tools,
        system_prompt="You are a helpful assistant. When asked about the current directory, use the list_directory tool.",
        state_schema=AgentState,
        checkpointer=checkpointer,
    )
    
    print("\n✅ Agent created with create_agent")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            input_data = {"messages": [HumanMessage(content="Em que pasta estamos?")]}
            
            print("\n📤 Invoking agent with message: 'Em que pasta estamos?'")
            print("   Waiting for response...")
            
            result = await agent.ainvoke(
                input_data,
                config={"configurable": {"thread_id": "test_qwen3"}},
            )
            
            print("\n📥 Agent result received:")
            print(f"   Type: {type(result)}")
            print(f"   Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            
            if isinstance(result, dict) and "messages" in result:
                messages = result["messages"]
                print(f"   Messages count: {len(messages)}")
                
                # Check for tool calls
                tool_calls_found = False
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_calls_found = True
                        print(f"\n   ✅ SUCCESS: Tool calls found in message!")
                        for i, tool_call in enumerate(msg.tool_calls):
                            print(f"      Tool call {i+1}:")
                            print(f"         Name: {tool_call.get('name', 'N/A')}")
                            print(f"         Args: {tool_call.get('args', {})}")
                
                # Get final response
                for msg in reversed(messages):
                    if hasattr(msg, "content") and msg.content:
                        print(f"\n   Final response: {msg.content[:200]}")
                        break
                
                if tool_calls_found:
                    return True
                else:
                    print("\n   ❌ FAILED: No tool calls found in agent result")
                    return False
            else:
                print("\n   ⚠️ Unexpected result format")
                return False
                
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    success = asyncio.run(test_qwen3_create_agent())
    exit(0 if success else 1)
