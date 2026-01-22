#!/usr/bin/env python3
"""Test direct tool calling with LangChain - verify if model can call tools."""

import asyncio
import sys
# Project root is already in path when running from tests/

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool


@tool
def create_file(filename: str, content: str) -> str:
    """Create a file with the given filename and content."""
    with open(filename, 'w') as f:
        f.write(content)
    return f"File {filename} created successfully with content: {content}"


async def test_tool_calling():
    """Test if model can call tools directly."""
    print("🔍 Testing direct tool calling with LangChain...")
    
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
    print(f"   Tool: {tools[0].name} - {tools[0].description}")
    
    # Test 1: Simple message that should trigger tool
    print("\n📤 Test 1: Asking to create a file...")
    message = HumanMessage(content="Create a file named test.txt with content 'Hello World'")
    
    try:
        response = await llm_with_tools.ainvoke([message])
        
        print(f"   Response type: {type(response)}")
        print(f"   Response content: {response.content[:200]}")
        
        # Check for tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"   ✅ Tool calls found: {len(response.tool_calls)}")
            for i, tool_call in enumerate(response.tool_calls):
                print(f"      Tool call {i+1}:")
                print(f"         Name: {tool_call.get('name', 'N/A')}")
                print(f"         Args: {tool_call.get('args', {})}")
            return True
        else:
            print(f"   ❌ No tool calls found in response")
            print(f"   Response attributes: {dir(response)}")
            if hasattr(response, 'response_metadata'):
                print(f"   Response metadata: {response.response_metadata}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_tool_calling())
    exit(0 if success else 1)
