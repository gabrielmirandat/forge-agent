# ToolStrategy uses artificial tool calling to generate structured output. 
# This works with any model that supports tool calling. ToolStrategy should be used 
# when provider-native structured output (via ProviderStrategy) is not available or reliable.

# As of langchain 1.0, simply passing a schema (e.g., response_format=ContactInfo) will default to ProviderStrategy 
# if the model supports native structured output. It will fall back to ToolStrategy otherwise.

from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool

# Define a simple tool for demonstration (optional - ToolStrategy creates its own tool)
@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

# Define structured output schema
class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

# Use open-source model with Ollama
model = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.0,
)

# Create agent with ToolStrategy for structured output
agent = create_agent(
    model=model,
    tools=[search_tool],  # Optional: can be empty list if only using structured output
    response_format=ToolStrategy(ContactInfo)
)

# Example usage
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    print("=" * 60)
    print("Testing ToolStrategy Structured Output")
    print("=" * 60)
    
    result = agent.invoke({
        "messages": [HumanMessage(content="Extract contact info from: John Doe, john@example.com, (555) 123-4567")]
    })
    
    # Try to access structured_response
    if isinstance(result, dict):
        if "structured_response" in result:
            print(f"\n✅ Structured Response: {result['structured_response']}")
        elif "messages" in result:
            # Extract from messages if structured_response not available
            messages = result["messages"]
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    print(f"\n✅ Response: {msg.content}")
                    break
        else:
            print(f"\nResult keys: {list(result.keys())}")
            print(f"Full result: {result}")
    else:
        print(f"\nResult: {result}")