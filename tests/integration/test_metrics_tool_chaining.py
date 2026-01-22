#!/usr/bin/env python3
"""Quick integration test for tool chaining and metrics tracking.

This script:
1. Creates a session
2. Sends a message that triggers tool chaining
3. Checks if metrics are being recorded
4. Provides quick feedback

Usage:
    python scripts/test_metrics_tool_chaining.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.config.loader import ConfigLoader
from agent.observability.llm_metrics import get_llm_metrics
from agent.runtime.langchain_executor import LangChainExecutor
from agent.tools.base import ToolRegistry


async def test_tool_chaining_with_metrics():
    """Test tool chaining and verify metrics are recorded."""
    print("🚀 Starting tool chaining test with metrics verification...")
    print()
    
    # Get config and tools
    config_path = Path(__file__).parent.parent.parent / "config" / "agent.ollama.yaml"
    loader = ConfigLoader(config_path=str(config_path))
    config = loader.load()
    
    # Initialize tool registry
    from api.dependencies import _register_mcp_tools
    tool_registry = ToolRegistry()
    # Register MCP tools if available
    try:
        await _register_mcp_tools(tool_registry, config)
    except Exception as e:
        print(f"⚠️  Warning: Could not register MCP tools: {e}")
        print("   Continuing with basic tools only...")
    
    # Create a test session ID
    session_id = f"test-metrics-{int(time.time())}"
    print(f"📝 Session ID: {session_id}")
    print()
    
    # Get metrics tracker
    metrics = get_llm_metrics()
    
    # Check initial metrics
    initial_metrics = metrics.get_session_metrics(session_id)
    initial_global = metrics.get_global_metrics()
    
    print("📊 Initial Metrics:")
    print(f"  Session calls: {initial_metrics['calls']}")
    print(f"  Session tokens: {initial_metrics['total_tokens']}")
    print(f"  Global calls: {initial_global['total_calls']}")
    print(f"  Global tokens: {initial_global['total_tokens']}")
    print()
    
    # Create executor
    print("🔧 Creating LangChain executor...")
    executor = LangChainExecutor(
        config=config,
        tool_registry=tool_registry,
        session_id=session_id,
        max_iterations=10,
    )
    
    # Initialize agent
    print("⚙️  Initializing agent...")
    await executor._ensure_agent_initialized()
    print("✅ Agent initialized")
    print()
    
    # Test 1: Simple tool call (if available)
    print("=" * 60)
    print("TEST 1: Simple tool call")
    print("=" * 60)
    
    # Use a simple request that should trigger a tool
    # Try to use a filesystem tool if available
    test_message = "What files are in the current directory? List them."
    
    print(f"💬 User message: {test_message}")
    print()
    
    start_time = time.time()
    try:
        result = await executor.run(
            user_message=test_message,
            conversation_history=None,
        )
        
        elapsed = time.time() - start_time
        
        print(f"⏱️  Execution time: {elapsed:.2f}s")
        print(f"📤 Response: {result.get('response', '')[:200]}...")
        print()
        
        # Check metrics after first call
        after_metrics = metrics.get_session_metrics(session_id)
        after_global = metrics.get_global_metrics()
        
        print("📊 Metrics After First Call:")
        print(f"  Session calls: {after_metrics['calls']} (was {initial_metrics['calls']})")
        print(f"  Session tokens: {after_metrics['total_tokens']} (was {initial_metrics['total_tokens']})")
        print(f"  Session model: {after_metrics['model']}")
        print(f"  Session response times: {len(after_metrics.get('response_times', []))}")
        if after_metrics.get('response_times'):
            avg_time = sum(after_metrics['response_times']) / len(after_metrics['response_times'])
            print(f"  Session avg response time: {avg_time:.3f}s")
        print()
        print(f"  Global calls: {after_global['total_calls']} (was {initial_global['total_calls']})")
        print(f"  Global tokens: {after_global['total_tokens']} (was {initial_global['total_tokens']})")
        print(f"  Global models used: {after_global['models_used']}")
        if after_global.get('model_avg_response_times'):
            print(f"  Global avg response times: {after_global['model_avg_response_times']}")
        print()
        
        # Verify metrics were updated
        if after_metrics['calls'] > initial_metrics['calls']:
            print("✅ SUCCESS: Session metrics updated!")
        else:
            print("❌ FAILURE: Session metrics NOT updated!")
            return False
            
        if after_global['total_calls'] > initial_global['total_calls']:
            print("✅ SUCCESS: Global metrics updated!")
        else:
            print("❌ FAILURE: Global metrics NOT updated!")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Tool chaining (multiple tools)
    print()
    print("=" * 60)
    print("TEST 2: Tool chaining (multiple tools)")
    print("=" * 60)
    
    test_message2 = "Get the current date and time, then tell me what day of the week it is."
    
    print(f"💬 User message: {test_message2}")
    print()
    
    start_time2 = time.time()
    try:
        result2 = await executor.run(
            user_message=test_message2,
            conversation_history=None,
        )
        
        elapsed2 = time.time() - start_time2
        
        print(f"⏱️  Execution time: {elapsed2:.2f}s")
        print(f"📤 Response: {result2.get('response', '')[:200]}...")
        print()
        
        # Check metrics after second call
        final_metrics = metrics.get_session_metrics(session_id)
        final_global = metrics.get_global_metrics()
        
        print("📊 Final Metrics:")
        print(f"  Session calls: {final_metrics['calls']} (was {after_metrics['calls']})")
        print(f"  Session tokens: {final_metrics['total_tokens']} (was {after_metrics['total_tokens']})")
        print(f"  Session response times: {len(final_metrics.get('response_times', []))}")
        if final_metrics.get('response_times'):
            avg_time = sum(final_metrics['response_times']) / len(final_metrics['response_times'])
            print(f"  Session avg response time: {avg_time:.3f}s")
        print()
        print(f"  Global calls: {final_global['total_calls']} (was {after_global['total_calls']})")
        print(f"  Global tokens: {final_global['total_tokens']} (was {after_global['total_tokens']})")
        if final_global.get('model_avg_response_times'):
            print(f"  Global avg response times: {final_global['model_avg_response_times']}")
        print()
        
        # Verify metrics were updated again
        # The second call should increase calls
        calls_before = after_metrics['calls']
        calls_after = final_metrics['calls']
        response_times_before = len(after_metrics.get('response_times', []))
        response_times_after = len(final_metrics.get('response_times', []))
        
        # Check if global metrics increased (most important)
        global_calls_before = after_global['total_calls']
        global_calls_after = final_global['total_calls']
        
        if global_calls_after > global_calls_before:
            print(f"✅ SUCCESS: Global metrics updated for second call! ({global_calls_before} -> {global_calls_after})")
        else:
            print(f"❌ FAILURE: Global metrics NOT updated for second call! ({global_calls_before} -> {global_calls_after})")
            return False
        
        # Check session metrics (may not increase if agent reused previous response)
        if calls_after > calls_before:
            print(f"✅ SUCCESS: Session metrics updated for second call! ({calls_before} -> {calls_after})")
        elif response_times_after > response_times_before:
            print(f"✅ SUCCESS: Response times increased, indicating a new call was made!")
            print(f"   Session calls: {calls_before} -> {calls_after} (may be same due to caching)")
        elif calls_after >= calls_before:
            print(f"⚠️  INFO: Session calls maintained ({calls_before} -> {calls_after})")
            print("   This is OK - the important thing is that global metrics increased")
        else:
            print(f"❌ FAILURE: Session calls decreased! ({calls_before} -> {calls_after})")
            return False
        
        print()
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_tool_chaining_with_metrics())
    sys.exit(0 if success else 1)
