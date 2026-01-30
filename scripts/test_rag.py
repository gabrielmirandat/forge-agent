#!/usr/bin/env python3
"""Test script for RAG tool - sends a simple query to test RAG functionality."""

import asyncio
import httpx
import json
import time

BASE_URL = "http://localhost:8000/api/v1"


async def test_rag():
    """Test RAG tool by creating a session and sending a query."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("=" * 80)
        print("Testing RAG Tool")
        print("=" * 80)
        
        # Step 1: Create a session
        print("\n1. Creating session...")
        response = await client.post(
            f"{BASE_URL}/sessions",
            json={"title": "RAG Test"}
        )
        response.raise_for_status()
        session_data = response.json()
        session_id = session_data["session_id"]
        print(f"✅ Session created: {session_id}")
        
        # Step 2: Send a message that should trigger RAG
        print("\n2. Sending query to test RAG...")
        query = "What is LangChain? Search the documentation for information about LangChain."
        print(f"Query: {query}")
        
        response = await client.post(
            f"{BASE_URL}/sessions/{session_id}/messages",
            json={"content": query}
        )
        response.raise_for_status()
        print(f"✅ Message sent (status: {response.status_code})")
        
        # Step 3: Wait a bit and check the session for response
        print("\n3. Waiting for response...")
        await asyncio.sleep(10)  # Wait for processing
        
        # Step 4: Get session to see messages
        print("\n4. Checking session messages...")
        response = await client.get(f"{BASE_URL}/sessions/{session_id}")
        response.raise_for_status()
        session = response.json()
        
        print(f"\n📊 Session has {len(session.get('messages', []))} messages:")
        for msg in session.get("messages", []):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # First 200 chars
            print(f"\n  [{role.upper()}]")
            print(f"  {content}...")
        
        print("\n" + "=" * 80)
        print("Test completed!")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_rag())
