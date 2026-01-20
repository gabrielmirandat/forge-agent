#!/usr/bin/env python3
"""Test create_agent with tool calling to see if it executes tools."""

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
def create_file(filename: str, content: str) -> str:
    """Create a file with the given filename and content."""
    file_path = Path(filename)
    file_path.write_text(content)
    return f"File {filename} created successfully with content: {content}"


async def test_create_agent_tool_calling():
    """Test if create_agent executes tools when model calls them."""
    print("🔍 Testing create_agent with tool calling...")
    
    # Create LLM
    llm = ChatOllama(
        model="llama3.1",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=60.0,
    )
    
    # Bind tools to LLM
    tools = [create_file]
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"   ✅ LLM created with {len(tools)} tool(s)")
    
    # Create agent
    agent = create_agent(
        model=llm_with_tools,
        tools=tools,
        system_prompt="You are a helpful assistant. When asked to create a file, use the create_file tool.",
    )
    
    print(f"   ✅ Agent created")
    
    # Test in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        print(f"   📁 Test directory: {tmpdir}")
        print(f"   📄 Target file: {test_file}")
        
        # Change to temp directory
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            # Test with prompt
            print("\n📤 Testing with prompt: 'Create a file named test.txt with content Hello World'")
            input_data = {
                "messages": [HumanMessage(content="Create a file named test.txt with content Hello World")]
            }
            
            # Stream events to see what happens
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
                    tool_input = event_data.get("input", {})
                    print(f"   🔧 Tool called: {tool_name} with input: {tool_input}")
                    tool_called = True
                elif event_name == "on_tool_end":
                    tool_output = event_data.get("output", "")
                    print(f"   ✅ Tool completed: {tool_output[:100]}")
                elif event_name == "on_chain_end":
                    output = event_data.get("output", {})
                    if isinstance(output, dict) and "messages" in output:
                        messages = output["messages"]
                        for msg in messages:
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                print(f"   🔧 Found tool_calls in message: {msg.tool_calls}")
            
            # Check if file was created
            if test_file.exists():
                content = test_file.read_text()
                print(f"\n   ✅ File created successfully!")
                print(f"   📄 Content: {content}")
                return True
            else:
                print(f"\n   ❌ File was not created")
                print(f"   Tool called: {tool_called}")
                return False
                
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    success = asyncio.run(test_create_agent_tool_calling())
    exit(0 if success else 1)
