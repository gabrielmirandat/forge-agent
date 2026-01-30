#!/usr/bin/env python3
"""Test RAG tool directly without going through the API."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.tools.rag_tool import RAGTool


async def test_rag_direct():
    """Test RAG tool directly."""
    print("=" * 80)
    print("Testing RAG Tool Directly")
    print("=" * 80)
    
    # Create RAG tool
    print("\n1. Creating RAG tool...")
    config = {
        "docs_path": "_docs",
        "enabled": True
    }
    rag_tool = RAGTool(config)
    print(f"✅ RAG tool created: {rag_tool.name}")
    print(f"   Description: {rag_tool.description}")
    
    # Test search
    print("\n2. Testing search...")
    query = "What is LangChain?"
    print(f"Query: {query}")
    
    try:
        result = await rag_tool.execute(
            operation="search",
            arguments={"query": query, "k": 3}
        )
        
        if result.success:
            print(f"\n✅ Search successful!")
            print(f"   Found {result.output['count']} results")
            
            for i, res in enumerate(result.output['results'], 1):
                print(f"\n   Result {i}:")
                print(f"   - Source: {res['metadata'].get('filename', 'unknown')}")
                print(f"   - Score: {res['score']:.4f}")
                print(f"   - Content preview: {res['content'][:200]}...")
        else:
            print(f"\n❌ Search failed: {result.error}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_rag_direct())
