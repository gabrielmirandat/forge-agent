# Middleware
# Middleware provides powerful extensibility for customizing agent behavior at different stages of execution. You can use middleware to:

# Process state before the model is called (e.g., message trimming, context injection)
# Modify or validate the model’s response (e.g., guardrails, content filtering)
# Handle tool execution errors with custom logic
# Implement dynamic model selection based on state or context
# Add custom logging, monitoring, or analytics
# Middleware integrates seamlessly into the agent’s execution, allowing you to intercept and modify data flow at key points without changing the core agent logic.
# or comprehensive middleware documentation including decorators like @before_model, @after_model, and @wrap_tool_call, see Middleware.

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.tools import tool

# Define a simple tool for demonstration
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Use open-source models with Ollama
# Basic model: smaller, faster (qwen3:8b)
basic_model = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.0,
)

# Advanced model: larger, more capable (qwen2.5:14b or devstral)
# For longer/complex conversations, use a more powerful model
advanced_model = ChatOllama(
    model="qwen2.5:14b",  # Or use "devstral" for best coding performance
    base_url="http://localhost:11434",
    temperature=0.0,
)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state.get("messages", []))

    if message_count > 10:
        # Use an advanced model for longer conversations
        print(f"🔄 Switching to advanced model (qwen2.5:14b) - {message_count} messages")
        model = advanced_model
    else:
        # Use basic model for shorter conversations
        print(f"⚡ Using basic model (qwen3:8b) - {message_count} messages")
        model = basic_model

    return handler(request.override(model=model))

# Create agent with dynamic model selection middleware
agent = create_agent(
    model=basic_model,  # Default model (will be overridden by middleware)
    tools=[get_weather],
    middleware=[dynamic_model_selection]
)

# Example usage
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    print("=" * 60)
    print("Testing Dynamic Model Selection Middleware")
    print("=" * 60)
    
    # Test with a simple query (will use basic model)
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in SF?")]
    })
    
    # Extract and print the final response
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        # Get the last AI message (final response)
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                print(f"\n✅ Final Response: {msg.content}")
                break
        
        # Show tool calls if any
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"🔧 Tool called: {tc.get('name', 'unknown')}({tc.get('args', {})})")
    else:
        print(f"\nResult: {result}")