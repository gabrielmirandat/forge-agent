#!/usr/bin/env python3
"""Test Ollama connection directly (without LangChain)."""

import asyncio
import httpx
import json


async def test_ollama_connection():
    """Test direct connection to Ollama."""
    base_url = "http://localhost:11434"
    
    print("🔍 Testing Ollama connection...")
    print(f"   Base URL: {base_url}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test 1: Check if Ollama is running
            print("\n1️⃣ Testing Ollama API availability...")
            try:
                response = await client.get(f"{base_url}/api/tags")
                print(f"   Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    print(f"   ✅ Ollama is running")
                    print(f"   Available models: {[m.get('name') for m in models]}")
                else:
                    print(f"   ❌ Unexpected status: {response.status_code}")
                    return False
            except httpx.ConnectError:
                print(f"   ❌ Cannot connect to Ollama at {base_url}")
                print(f"   💡 Make sure Ollama Docker container is running:")
                print(f"      docker ps | grep ollama")
                return False
            except Exception as e:
                print(f"   ❌ Error: {e}")
                return False
            
            # Test 2: Test simple chat with mistral
            print("\n2️⃣ Testing simple chat...")
            try:
                chat_data = {
                    "model": "llama3.1",
                    "messages": [
                        {"role": "user", "content": "Say 'Hello' and nothing else."}
                    ],
                    "stream": False
                }
                response = await client.post(
                    f"{base_url}/api/chat",
                    json=chat_data,
                    timeout=60.0
                )
                print(f"   Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    message = data.get("message", {})
                    content = message.get("content", "")
                    print(f"   ✅ Chat response: {content[:100]}")
                    return True
                else:
                    print(f"   ❌ Chat failed: {response.text[:200]}")
                    return False
            except Exception as e:
                print(f"   ❌ Chat error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_ollama_connection())
    exit(0 if success else 1)
