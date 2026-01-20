#!/usr/bin/env python3
"""Test LangChain ChatOllama directly (without executor)."""

import asyncio
import sys
sys.path.insert(0, '/home/gabriel-miranda/repos/forge-agent')

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


async def test_langchain_ollama():
    """Test LangChain ChatOllama connection."""
    print("🔍 Testing LangChain ChatOllama...")
    
    try:
        # Create ChatOllama instance
        llm = ChatOllama(
            model="llama3.1",
            base_url="http://localhost:11434",
            temperature=0.1,
            timeout=60.0,
        )
        print(f"   ✅ ChatOllama created")
        print(f"   Model: llama3.1")
        print(f"   Base URL: http://localhost:11434")
        
        # Test simple message
        print("\n📤 Sending message...")
        message = HumanMessage(content="What is 2 + 2? Answer with just the number.")
        
        print("   Waiting for response...")
        response = await llm.ainvoke([message])
        
        print(f"   ✅ Response received: {response.content}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_langchain_ollama())
    exit(0 if success else 1)
