#!/usr/bin/env python3
"""Test 1: Simple 'Hey' message to LLM via LangChain."""

import asyncio
import sys
sys.path.insert(0, '/home/gabriel-miranda/repos/forge-agent')

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


async def test_hey():
    """Test simple 'Hey' message."""
    print("🔍 Test 1: Sending 'Hey' to LLM...")
    
    try:
        llm = ChatOllama(
            model="llama3.1",
            base_url="http://localhost:11434",
            temperature=0.1,
            timeout=60.0,
        )
        
        print("   📤 Sending: 'Hey'")
        response = await llm.ainvoke([HumanMessage(content="Hey")])
        
        print(f"   ✅ Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_hey())
    exit(0 if success else 1)
