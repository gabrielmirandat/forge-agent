#!/usr/bin/env python3
"""Test script that simulates backend-frontend communication.

Tests:
1. Create session
2. Send message via POST API
3. Connect to WebSocket stream
4. Verify events are received
5. Verify tool calls are processed
"""

import asyncio
import json
import sys
import time
from agent.id import ascending
from typing import List, Dict, Any

import httpx
try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    print("⚠️  websockets library not found. Install with: pip install websockets")
    sys.exit(1)

# Base URL for API
BASE_URL = "http://localhost:8000"


async def create_session(client: httpx.AsyncClient) -> str:
    """Create a new session."""
    print("=" * 80)
    print("1. Creating session...")
    print("=" * 80)
    
    response = await client.post(
        f"{BASE_URL}/api/v1/sessions",
        json={}  # CreateSessionRequest doesn't require name, it's optional
    )
    response.raise_for_status()
    
    data = response.json()
    session_id = data["session_id"]
    
    print(f"✅ Session created: {session_id}")
    return session_id


async def send_message(client: httpx.AsyncClient, session_id: str, content: str) -> Dict[str, Any]:
    """Send a message to the session."""
    print("\n" + "=" * 80)
    print("2. Sending message...")
    print("=" * 80)
    print(f"Message: {content}")
    
    response = await client.post(
        f"{BASE_URL}/api/v1/sessions/{session_id}/messages",
        json={"content": content}
    )
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 202:
        data = response.json()
        message_id = data.get("message_id")
        print(f"✅ Message accepted (202): {message_id}")
        return data
    else:
        response.raise_for_status()
        return response.json()


async def listen_to_websocket(session_id: str, timeout: float = 30.0) -> List[Dict[str, Any]]:
    """Listen to WebSocket events for a session."""
    print("\n" + "=" * 80)
    print("3. Connecting to WebSocket stream...")
    print("=" * 80)
    
    events_received = []
    start_time = time.time()
    
    # Convert HTTP URL to WebSocket URL
    ws_url = BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/api/v1/events/ws"
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket connection established")
            print("Listening for events...\n")
            
            while True:
                if time.time() - start_time > timeout:
                    print(f"\n⏱️  Timeout reached ({timeout}s)")
                    break
                
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=1.0
                    )
                    
                    try:
                        event_data = json.loads(message)
                        event_type = event_data.get("type", "unknown")
                        properties = event_data.get("properties", {})
                        
                        # Collect all events
                        events_received.append(event_data)
                        
                        # Filter events for this session (if session_id is in properties)
                        event_session_id = properties.get("session_id")
                        request_id = properties.get("request_id")
                        
                        if event_session_id == session_id or event_type.startswith("server.") or not event_session_id:
                            # Print event
                            print(f"📨 Event: {event_type}")
                            if event_session_id:
                                print(f"   Session: {event_session_id[:8]}...")
                            if request_id:
                                print(f"   Request ID: {request_id[:8]}...")
                            if "tool" in properties:
                                print(f"   Tool: {properties.get('tool')}")
                            if "operation" in properties:
                                print(f"   Operation: {properties.get('operation')}")
                            if "response" in properties:
                                resp = properties.get("response", "")
                                if len(resp) > 100:
                                    resp = resp[:100] + "..."
                                print(f"   Response: {resp}")
                            if "error" in properties:
                                print(f"   Error: {properties.get('error')}")
                            print()
                            
                            # Stop if we get a final response for this session
                            if event_type == "llm.response" and properties.get("response") and event_session_id == session_id:
                                print("✅ Received final LLM response")
                                break
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Failed to parse event: {e}")
                        print(f"   Raw data: {message[:200]}")
                        
                except asyncio.TimeoutError:
                    # Continue loop to check overall timeout
                    continue
                    
    except ConnectionClosed:
        print("\n✅ WebSocket connection closed normally")
    except asyncio.TimeoutError:
        print(f"\n⏱️  Connection timeout after {timeout}s")
    except Exception as e:
        print(f"\n❌ Error in WebSocket stream: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    return events_received


async def get_session_messages(client: httpx.AsyncClient, session_id: str) -> Dict[str, Any]:
    """Get session with messages."""
    print("\n" + "=" * 80)
    print("4. Fetching session messages...")
    print("=" * 80)
    
    response = await client.get(f"{BASE_URL}/api/v1/sessions/{session_id}")
    response.raise_for_status()
    
    data = response.json()
    messages = data.get("messages", [])
    
    print(f"✅ Session has {len(messages)} messages")
    for idx, msg in enumerate(messages[-3:], 1):  # Last 3 messages
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if len(content) > 150:
            content = content[:150] + "..."
        print(f"   {idx}. [{role}] {content}")
    
    return data


def analyze_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze received events."""
    print("\n" + "=" * 80)
    print("5. Analyzing events...")
    print("=" * 80)
    
    analysis = {
        "total_events": len(events),
        "event_types": {},
        "tool_calls": [],
        "tool_results": [],
        "llm_responses": [],
        "errors": [],
    }
    
    for event in events:
        event_type = event.get("type", "unknown")
        properties = event.get("properties", {})
        
        # Count event types
        analysis["event_types"][event_type] = analysis["event_types"].get(event_type, 0) + 1
        
        # Track tool calls
        if event_type == "tool.called":
            analysis["tool_calls"].append({
                "tool": properties.get("tool"),
                "operation": properties.get("operation"),
            })
        
        # Track tool results
        if event_type == "tool.result":
            analysis["tool_results"].append({
                "tool": properties.get("tool"),
                "success": properties.get("success"),
            })
        
        # Track LLM responses
        if event_type == "llm.response":
            analysis["llm_responses"].append({
                "has_response": bool(properties.get("response")),
                "response_length": len(properties.get("response", "")),
            })
        
        # Track errors
        if event_type in ["error", "execution.error"]:
            analysis["errors"].append(properties)
    
    # Print analysis
    print(f"Total events: {analysis['total_events']}")
    print(f"\nEvent types:")
    for event_type, count in sorted(analysis["event_types"].items()):
        print(f"  - {event_type}: {count}")
    
    print(f"\nTool calls: {len(analysis['tool_calls'])}")
    for tc in analysis["tool_calls"]:
        print(f"  - {tc['tool']}.{tc['operation']}")
    
    print(f"\nTool results: {len(analysis['tool_results'])}")
    for tr in analysis["tool_results"]:
        status = "✅" if tr.get("success") else "❌"
        print(f"  {status} {tr['tool']}")
    
    print(f"\nLLM responses: {len(analysis['llm_responses'])}")
    for lr in analysis["llm_responses"]:
        if lr["has_response"]:
            print(f"  ✅ Response length: {lr['response_length']} chars")
        else:
            print(f"  ⚠️  Empty response")
    
    if analysis["errors"]:
        print(f"\n❌ Errors: {len(analysis['errors'])}")
        for err in analysis["errors"]:
            print(f"  - {err}")
    
    return analysis


async def main():
    """Main test function."""
    print("=" * 80)
    print("BACKEND-FRONTEND COMMUNICATION TEST")
    print("=" * 80)
    print(f"\nAPI Base URL: {BASE_URL}")
    print(f"Test will:")
    print("  1. Create a session")
    print("  2. Send a message requesting tool usage")
    print("  3. Listen to WebSocket events")
    print("  4. Verify tool calls are processed")
    print("  5. Analyze results")
    print("\n" + "=" * 80)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Step 1: Create session
            session_id = await create_session(client)
            
            # Step 2: Send message
            message_content = "List the files in the current directory"
            message_response = await send_message(client, session_id, message_content)
            
            # Step 3: Listen to WebSocket events (in parallel)
            # Start WebSocket listener before message is processed
            ws_task = asyncio.create_task(listen_to_websocket(session_id, timeout=30.0))
            
            # Wait a bit for processing to start
            await asyncio.sleep(1)
            
            # Wait for WebSocket to finish or timeout
            events = await ws_task
            
            # Step 4: Get final session state
            session_data = await get_session_messages(client, session_id)
            
            # Step 5: Analyze events
            analysis = analyze_events(events)
            
            # Final summary
            print("\n" + "=" * 80)
            print("TEST SUMMARY")
            print("=" * 80)
            
            if analysis["tool_calls"]:
                print("✅ Tool calls detected!")
            else:
                print("⚠️  No tool calls detected")
            
            if analysis["llm_responses"] and any(lr["has_response"] for lr in analysis["llm_responses"]):
                print("✅ LLM responses received")
            else:
                print("⚠️  No LLM responses or empty responses")
            
            if analysis["errors"]:
                print(f"❌ {len(analysis['errors'])} errors detected")
            else:
                print("✅ No errors")
            
            print("\n" + "=" * 80)
            
        except httpx.HTTPStatusError as e:
            print(f"\n❌ HTTP Error: {e.response.status_code}")
            print(f"   Response: {e.response.text[:500]}")
        except Exception as e:
            print(f"\n❌ Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
