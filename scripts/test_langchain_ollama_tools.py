#!/usr/bin/env python3
"""Test LangChain ChatOllama with tools to see what's being sent to Ollama."""

import asyncio
import sys
sys.path.insert(0, '/home/gabriel-miranda/repos/forge-agent')

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool


@tool
def list_directory(path: str = ".") -> str:
    """List files and directories in the given path."""
    return f"Listing directory: {path}"


async def test_langchain_ollama_tools():
    """Test LangChain ChatOllama with tools to see the actual request."""
    print("🔍 Testing LangChain ChatOllama with tools...")
    
    # Create LLM
    llm = ChatOllama(
        model="llama3.1",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=60.0,
    )
    
    # Bind tools
    tools = [list_directory]
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"\n✅ LLM created with {len(tools)} tool(s)")
    print(f"   Tool: {tools[0].name}")
    print(f"   Tool description: {tools[0].description}")
    
    # Check what bind_tools did
    if hasattr(llm_with_tools, "bound_tools"):
        print(f"\n📋 Bound tools: {llm_with_tools.bound_tools}")
    
    # Test invocation
    print(f"\n📤 Invoking LLM with message: 'Em que pasta estamos?'")
    message = HumanMessage(content="Em que pasta estamos? List the current directory.")
    
    try:
        # Enable debug logging to see what's being sent
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        response = await llm_with_tools.ainvoke([message])
        
        print(f"\n📥 Response received:")
        print(f"   Type: {type(response)}")
        print(f"   Content: {response.content[:200] if response.content else '(empty)'}")
        
        # Check for tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"\n   ✅ SUCCESS: Tool calls found!")
            for i, tool_call in enumerate(response.tool_calls):
                print(f"      Tool call {i+1}:")
                print(f"         Name: {tool_call.get('name', 'unknown')}")
                print(f"         Args: {tool_call.get('args', {})}")
            return True
        else:
            print(f"\n   ❌ FAILED: No tool_calls in response")
            print(f"   Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
            if hasattr(response, 'response_metadata'):
                print(f"   Response metadata: {response.response_metadata}")
            return False
            
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_langchain_ollama_tools())
    exit(0 if success else 1)
