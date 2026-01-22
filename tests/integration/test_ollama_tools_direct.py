#!/usr/bin/env python3
"""Test Ollama tool calling directly without LangChain to see the raw format."""

import asyncio
import json
import httpx

async def test_ollama_tools_direct():
    """Test Ollama API directly with tools to see what format it expects."""
    print("🔍 Testing Ollama API directly with tools...")
    
    # Define a simple tool in OpenAI format
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files and directories in the given path",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to list (default: current directory)"
                        }
                    }
                }
            }
        }
    ]
    
    # Test message
    messages = [
        {
            "role": "user",
            "content": "Em que pasta estamos? List the current directory."
        }
    ]
    
    # Prepare request
    payload = {
        "model": "llama3.1",
        "messages": messages,
        "tools": tools,
        "stream": False,
    }
    
    print(f"\n📤 Sending request to Ollama...")
    print(f"   Model: llama3.1")
    print(f"   Tools: {len(tools)} tool(s)")
    print(f"   Tool format: {json.dumps(tools[0], indent=2)}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                "http://localhost:11434/api/chat",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            print(f"\n📥 Response from Ollama:")
            print(f"   Status: {response.status_code}")
            print(f"   Full response: {json.dumps(result, indent=2)}")
            
            # Check for tool calls
            if "message" in result:
                message = result["message"]
                if "tool_calls" in message and message["tool_calls"]:
                    print(f"\n   ✅ SUCCESS: Tool calls found!")
                    for tool_call in message["tool_calls"]:
                        print(f"      Tool: {tool_call.get('function', {}).get('name', 'unknown')}")
                        print(f"      Args: {tool_call.get('function', {}).get('arguments', {})}")
                    return True
                else:
                    print(f"\n   ❌ FAILED: No tool_calls in response")
                    print(f"   Message content: {message.get('content', '')[:200]}")
                    return False
            else:
                print(f"\n   ⚠️ Unexpected response format")
                return False
                
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = asyncio.run(test_ollama_tools_direct())
    exit(0 if success else 1)
