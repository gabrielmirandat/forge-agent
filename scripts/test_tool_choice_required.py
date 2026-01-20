#!/usr/bin/env python3
"""Test tool calling with tool_choice="required" to force tool usage."""

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


async def test_tool_choice_required():
    """Test if tool_choice='required' forces tool usage."""
    print("🔍 Testing tool_choice='required' with ChatOllama...")
    
    # Create LLM
    llm = ChatOllama(
        model="llama3.1",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=60.0,
    )
    
    # Bind tools to LLM with tool_choice="required"
    tools = [list_directory]
    
    print(f"\n📋 Testing bind_tools with tool_choice='required'...")
    try:
        llm_with_tools = llm.bind_tools(tools, tool_choice="required")
        print(f"   ✅ bind_tools() with tool_choice='required' succeeded")
    except TypeError as e:
        print(f"   ⚠️ bind_tools() doesn't support tool_choice parameter: {e}")
        print(f"   📝 Falling back to default bind_tools()...")
        llm_with_tools = llm.bind_tools(tools)
    
    print(f"   ✅ LLM created with {len(tools)} tool(s)")
    print(f"   Tool: {tools[0].name} - {tools[0].description}")
    
    # Create agent
    agent = create_agent(
        model=llm_with_tools,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant with access to filesystem tools.\n"
            "When asked about directories or files, you MUST use the list_directory tool.\n"
            "Do not say you don't have access - use the tools!"
        ),
    )
    
    print(f"   ✅ Agent created")
    
    # Test in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n📁 Test directory: {tmpdir}")
        
        # Change to temp directory
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Hello World")
            print(f"   📄 Created test file: {test_file}")
            
            # Test with prompt that should trigger tool
            print("\n📤 Testing with prompt: 'Em que pasta estamos?'")
            input_data = {
                "messages": [HumanMessage(content="Em que pasta estamos?")]
            }
            
            # Stream events to see what happens
            tool_called = False
            tool_name_called = None
            response_text = ""
            
            async for event in agent.astream_events(
                input_data,
                version="v2",
                include_names=["tool", "llm", "chain"],
            ):
                event_name = event.get("event", "")
                event_data = event.get("data", {})
                
                if event_name == "on_tool_start":
                    tool_name_called = event_data.get("name", "")
                    tool_input = event_data.get("input", {})
                    print(f"   🔧 Tool called: {tool_name_called} with input: {tool_input}")
                    tool_called = True
                elif event_name == "on_tool_end":
                    tool_output = event_data.get("output", "")
                    print(f"   ✅ Tool completed: {tool_output[:200]}")
                elif event_name == "on_llm_new_token":
                    token = event_data.get("chunk", "")
                    if token:
                        response_text += token
                elif event_name == "on_chain_end":
                    output = event_data.get("output", {})
                    if isinstance(output, dict) and "messages" in output:
                        messages = output["messages"]
                        for msg in messages:
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                print(f"   🔧 Found tool_calls in message: {msg.tool_calls}")
                            if hasattr(msg, "content") and msg.content:
                                response_text = msg.content
            
            print(f"\n📝 Final response: {response_text[:300]}")
            
            if tool_called:
                print(f"\n   ✅ SUCCESS: Tool was called!")
                print(f"   Tool name: {tool_name_called}")
                return True
            else:
                print(f"\n   ❌ FAILED: No tool was called")
                print(f"   Response: {response_text[:200]}")
                return False
                
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    success = asyncio.run(test_tool_choice_required())
    exit(0 if success else 1)
