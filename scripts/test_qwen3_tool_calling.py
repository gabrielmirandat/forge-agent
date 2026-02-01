#!/usr/bin/env python3
"""Test Qwen3 model tool calling - specifically filesystem to list projects."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.config.loader import ConfigLoader
from agent.runtime.langchain_executor import LangChainExecutor
from agent.tools.base import ToolRegistry
from api.dependencies import _register_mcp_tools


async def test_qwen3_filesystem_tool_calling():
    """Test Qwen3 model making tool calls to list projects using filesystem."""
    print("=" * 80)
    print("Testing Qwen3 Tool Calling - Filesystem List Projects")
    print("=" * 80)
    
    # Load Qwen3 config
    print("\n1. Loading Qwen3 configuration...")
    config_path = project_root / "config" / "agent.ollama.qwen.yaml"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    print(f"✅ Config loaded: {config.llm.model}")
    print(f"   Workspace: {config.workspace.base_path}")
    
    # Note: Not forcing tool_choice - using default "auto"
    print("\n   ℹ️  Using default tool_choice='auto' (not forcing tool calling)")
    if hasattr(config.llm, 'tool_choice'):
        original_tool_choice = config.llm.tool_choice
        print(f"   Current tool_choice: {original_tool_choice}")
    else:
        print(f"   No tool_choice set (will default to 'auto')")
    
    # Create tool registry
    print("\n2. Creating tool registry...")
    registry = ToolRegistry()
    await _register_mcp_tools(registry, config)
    enabled_tools = registry.list_enabled()
    print(f"✅ Registered {len(enabled_tools)} tools")
    
    # Show filesystem-related tools
    filesystem_tools = [t for t in enabled_tools if 'filesystem' in t.lower() or 'list' in t.lower() or 'directory' in t.lower()]
    if filesystem_tools:
        print(f"   Filesystem tools: {', '.join(filesystem_tools[:5])}")
    
    # Create executor
    print("\n3. Creating LangChain executor...")
    executor = LangChainExecutor(
        config=config,
        tool_registry=registry,
        session_id="test_qwen3_tool_calling",
        max_iterations=10,
    )
    print("✅ Executor created")
    
    # Test prompt asking to list projects
    print("\n4. Testing tool calling...")
    # Use a more direct prompt that should trigger tool calling
    prompt = "List all directories in /projects"
    print(f"📤 Prompt: {prompt}")
    print("   (Expected: Qwen3 should call filesystem_list_directory tool)")
    
    # Show available filesystem tools
    langchain_tools = await registry.get_langchain_tools(config=config)
    filesystem_tools = [t for t in langchain_tools if 'filesystem' in t.name.lower() and 'list' in t.name.lower()]
    if filesystem_tools:
        print(f"   Available list tools: {[t.name for t in filesystem_tools]}")
    
    try:
        print("\n   ⏳ Executing (this may take a moment)...")
        result = await executor.run(
            user_message=prompt,
            conversation_history=None,
        )
        
        print(f"\n📊 Result:")
        print(f"   Success: {result.get('success')}")
        
        if result.get('success'):
            response = result.get('response', 'No response')
            print(f"\n✅ Response received (full response):")
            print(f"   {response}")
            print(f"\n   Response length: {len(response)} characters")
            
            # Check for actual tool calls
            intermediate_steps = result.get('intermediate_steps', [])
            tool_calls_made = False
            
            if intermediate_steps:
                print(f"\n🔧 Tool calls detected: {len(intermediate_steps)} step(s)")
                for i, step in enumerate(intermediate_steps[:5], 1):
                    if isinstance(step, tuple) and len(step) >= 2:
                        tool_call = step[0]
                        tool_result = step[1]
                        
                        # Try to extract tool name
                        tool_name = "unknown"
                        if hasattr(tool_call, 'name'):
                            tool_name = tool_call.name
                        elif isinstance(tool_call, dict):
                            tool_name = tool_call.get('name', 'unknown')
                        else:
                            tool_name = str(tool_call)[:50]
                        
                        print(f"   Step {i}: {tool_name}")
                        
                        # Show tool result
                        if tool_result:
                            result_str = str(tool_result)[:300]
                            print(f"      Result: {result_str}...")
                            tool_calls_made = True
            else:
                print("\n⚠️  No intermediate steps found - tool may not have been called")
            
            # Check if tool was actually called
            if tool_calls_made:
                print("\n✅ Tool calling worked! Tools were actually invoked.")
            elif 'filesystem' in response.lower() or 'list_directory' in response.lower():
                print("\n⚠️  Response mentions filesystem, but no tool calls detected")
                print("   (Model may have suggested code instead of calling tools)")
            else:
                print("\n❌ Tool calling may not have been triggered")
                print("   (No tool calls detected in intermediate steps)")
        else:
            error = result.get('error', 'Unknown error')
            print(f"\n❌ Error: {error}")
            
            # Show more details if available
            if 'intermediate_steps' in result:
                print(f"\n   Intermediate steps: {len(result['intermediate_steps'])}")
            
        return result.get('success') is True
        
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Qwen3 Tool Calling Test")
    print("=" * 80)
    print("\nThis test will:")
    print("1. Load Qwen3 configuration")
    print("2. Initialize tools (including filesystem MCP)")
    print("3. Ask Qwen3 to list projects using filesystem tool")
    print("4. Verify tool calling works correctly")
    print("\nNote: Using default tool_choice='auto' (not forcing tool calling)")
    print("\n" + "=" * 80 + "\n")
    
    success = asyncio.run(test_qwen3_filesystem_tool_calling())
    
    print("\n" + "=" * 80)
    if success:
        print("✅ TEST PASSED")
    else:
        print("❌ TEST FAILED")
    print("=" * 80)
    
    sys.exit(0 if success else 1)
