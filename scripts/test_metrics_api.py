#!/usr/bin/env python3
"""Quick test to verify metrics API endpoint returns correct data.

This script:
1. Makes a real API call to create a session and send a message
2. Checks the observability endpoint for metrics
3. Provides quick feedback

Usage:
    # Start the API server first:
    # uvicorn api.app:app --reload
    
    # Then run this script:
    python scripts/test_metrics_api.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

BASE_URL = "http://localhost:8000/api/v1"


async def test_metrics_api():
    """Test metrics API endpoint."""
    print("🚀 Testing metrics API endpoint...")
    print()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Create a session
        print("📝 Step 1: Creating session...")
        create_response = await client.post(f"{BASE_URL}/sessions")
        if create_response.status_code != 200:
            print(f"❌ Failed to create session: {create_response.status_code}")
            print(create_response.text)
            return False
        
        session_data = create_response.json()
        session_id = session_data["session_id"]
        print(f"✅ Session created: {session_id}")
        print()
        
        # Step 2: Check initial metrics
        print("📊 Step 2: Checking initial metrics...")
        try:
            # Use SSE endpoint to get metrics
            metrics_url = f"{BASE_URL}/observability/metrics"
            
            # We'll make a request and read the first event
            async with client.stream("GET", metrics_url) as response:
                if response.status_code != 200:
                    print(f"❌ Failed to connect to metrics endpoint: {response.status_code}")
                    return False
                
                # Read first event
                initial_metrics = None
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        try:
                            event = json.loads(data_str)
                            if event.get("type") == "metrics":
                                initial_metrics = event.get("data", {}).get("global", {}).get("llm", {})
                                break
                        except json.JSONDecodeError:
                            continue
                
                if initial_metrics:
                    print(f"  Initial total calls: {initial_metrics.get('total_calls', 0)}")
                    print(f"  Initial total tokens: {initial_metrics.get('total_tokens', 0)}")
                    print(f"  Initial active sessions: {initial_metrics.get('active_sessions', 0)}")
                    print()
                else:
                    print("  ⚠️  Could not read initial metrics")
                    print()
        except Exception as e:
            print(f"  ⚠️  Error reading initial metrics: {e}")
            print()
        
        # Step 3: Send a message to trigger LLM call
        print("💬 Step 3: Sending message to trigger LLM call...")
        message_response = await client.post(
            f"{BASE_URL}/sessions/{session_id}/messages",
            json={"content": "What is 2+2? Just give me the number."},
        )
        
        if message_response.status_code != 202:
            print(f"❌ Failed to send message: {message_response.status_code}")
            print(message_response.text)
            return False
        
        print("✅ Message sent (202 Accepted)")
        print("   Waiting for processing...")
        
        # Wait a bit for processing
        await asyncio.sleep(3)
        print()
        
        # Step 4: Check metrics after LLM call
        print("📊 Step 4: Checking metrics after LLM call...")
        try:
            async with client.stream("GET", metrics_url) as response:
                if response.status_code != 200:
                    print(f"❌ Failed to connect to metrics endpoint: {response.status_code}")
                    return False
                
                # Read first metrics event
                final_metrics = None
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            event = json.loads(data_str)
                            if event.get("type") == "metrics":
                                final_metrics = event.get("data", {}).get("global", {}).get("llm", {})
                                break
                        except json.JSONDecodeError:
                            continue
                
                if final_metrics:
                    print(f"  Final total calls: {final_metrics.get('total_calls', 0)}")
                    print(f"  Final total tokens: {final_metrics.get('total_tokens', 0)}")
                    print(f"  Final active sessions: {final_metrics.get('active_sessions', 0)}")
                    print(f"  Models used: {final_metrics.get('models_used', [])}")
                    if final_metrics.get('model_avg_response_times'):
                        print(f"  Avg response times: {final_metrics['model_avg_response_times']}")
                    print()
                    
                    # Verify metrics were updated
                    if initial_metrics:
                        if final_metrics.get('total_calls', 0) > initial_metrics.get('total_calls', 0):
                            print("✅ SUCCESS: Metrics updated!")
                            return True
                        else:
                            print("❌ FAILURE: Metrics NOT updated!")
                            print(f"   Initial calls: {initial_metrics.get('total_calls', 0)}")
                            print(f"   Final calls: {final_metrics.get('total_calls', 0)}")
                            return False
                    else:
                        # If we couldn't read initial metrics, just check if we have any metrics now
                        if final_metrics.get('total_calls', 0) > 0:
                            print("✅ SUCCESS: Metrics are being tracked!")
                            return True
                        else:
                            print("❌ FAILURE: No metrics found!")
                            return False
                else:
                    print("❌ FAILURE: Could not read metrics!")
                    return False
                    
        except Exception as e:
            print(f"❌ ERROR reading metrics: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = asyncio.run(test_metrics_api())
    sys.exit(0 if success else 1)
