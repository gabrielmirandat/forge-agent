from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import tool


class Context(TypedDict):
    user_role: str

# Define a simple web search tool for demonstration
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

# Use open-source model with Ollama
model = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.0,
)

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        prompt = f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        prompt = f"{base_prompt} Explain concepts simply and avoid jargon."
    else:
        prompt = base_prompt
    
    # Debug: print which prompt is being used
    print(f"📝 Dynamic prompt for role '{user_role}': {prompt}")
    return prompt

# Create agent with dynamic system prompt middleware
agent = create_agent(
    model=model,
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# Example usage
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    print("=" * 60)
    print("Testing Dynamic System Prompt Middleware")
    print("=" * 60)
    
    # Test 1: Expert role (should get detailed technical response)
    print("\n1. Testing with 'expert' role:")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Explain machine learning")]},
        context=Context(user_role="expert")
    )
    
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                print(f"✅ Expert Response:\n{msg.content}\n")
                break
    
    # Test 2: Beginner role (should get simple explanation)
    print("\n2. Testing with 'beginner' role:")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Explain machine learning")]},
        context=Context(user_role="beginner")
    )
    
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                print(f"✅ Beginner Response:\n{msg.content}\n")
                break
    
    # Test 3: Default role (should get standard response)
    print("\n3. Testing with default role:")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Explain machine learning")]},
        context=Context(user_role="user")
    )
    
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                print(f"✅ Default Response:\n{msg.content}\n")
                break