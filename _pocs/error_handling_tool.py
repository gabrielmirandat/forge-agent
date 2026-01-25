from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain.tools import tool

# Define tools as specified (with error conditions for testing)
@tool
def search(query: str) -> str:
    """Search for information."""
    # Generate error if query contains "error" or "fail"
    if "error" in query.lower() or "fail" in query.lower():
        raise ValueError(f"Search service unavailable for query: {query}")
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    # Generate error if location is "ERROR" or "FAIL"
    if location.upper() in ["ERROR", "FAIL", "INVALID"]:
        raise ValueError(f"Weather service error: Cannot get weather for {location}")
    return f"Weather in {location}: Sunny, 72°F"

# Use open-source model with Ollama
model = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.0,
)

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        # This allows the model to understand the error and potentially retry or adjust
        error_msg = f"Tool error: {str(e)}. Please check your input and try again."
        print(f"⚠️ Tool error caught: {error_msg}")
        return ToolMessage(
            content=error_msg,
            tool_call_id=request.tool_call["id"]
        )

# Create agent with error handling middleware
agent = create_agent(
    model=model,
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)

# Example usage
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    print("=" * 60)
    print("Testing Tool Error Handling Middleware")
    print("=" * 60)
    
    # Test 1: Normal tool call (should work)
    print("\n1. Testing normal tool call:")
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in San Francisco?")]
    })
    
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                print(f"✅ Response: {msg.content}")
                break
    
    # Test 2: Search tool call
    print("\n2. Testing search tool:")
    result = agent.invoke({
        "messages": [HumanMessage(content="Search for Python tutorials")]
    })
    
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                print(f"✅ Response: {msg.content}")
                break
    
    # Test 3: Force error in search tool (should be handled gracefully)
    print("\n3. Testing error handling - search with 'error' keyword:")
    result = agent.invoke({
        "messages": [HumanMessage(content="Search for error handling")]
    })
    
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                print(f"✅ Response: {msg.content}")
                break
    
    # Test 4: Force error in weather tool (should be handled gracefully)
    print("\n4. Testing error handling - weather with invalid location:")
    result = agent.invoke({
        "messages": [HumanMessage(content="Get weather for ERROR")]
    })
    
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                print(f"✅ Response: {msg.content}")
                break