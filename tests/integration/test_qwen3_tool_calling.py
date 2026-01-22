#!/usr/bin/env python3
"""Test hhao/qwen2.5-coder-tools tool calling directly to verify it works."""

import asyncio
import sys
from pathlib import Path

# Project root is already in path when running from tests/

from langchain_ollama import ChatOllama
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

async def test_qwen3_tool_calling():
    """Test if hhao/qwen2.5-coder-tools can call tools."""
    print("🔍 Testing hhao/qwen2.5-coder-tools with tool calling...")
    
    llm = ChatOllama(
        model="hhao/qwen2.5-coder-tools",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=60.0,
    )
    
    tools = [list_directory]
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"\n✅ LLM created with {len(tools)} tool(s)")
    print(f"   Model: hhao/qwen2.5-coder-tools")
    print(f"   Tool: {tools[0].name} - {tools[0].description}")
    
    message = HumanMessage(content="Em que pasta estamos? List the current directory.")
    
    print("\n📤 Invoking LLM with message: 'Em que pasta estamos? List the current directory.'")
    print("   Waiting for response...")
    
    try:
        response = await llm_with_tools.ainvoke([message])
        
        print("\n📥 Response received:")
        print(f"   Type: {type(response)}")
        print(f"   Content: {response.content[:200] if hasattr(response, 'content') and response.content else '(empty)'}")
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"\n   ✅ SUCCESS: Tool calls found: {len(response.tool_calls)}")
            for i, tool_call in enumerate(response.tool_calls):
                print(f"      Tool call {i+1}:")
                print(f"         Name: {tool_call.get('name', 'N/A')}")
                print(f"         Args: {tool_call.get('args', {})}")
            return True
        else:
            print("\n   ❌ FAILED: No tool calls found in response.")
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
    success = asyncio.run(test_qwen3_tool_calling())
    exit(0 if success else 1)
