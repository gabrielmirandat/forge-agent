# pip install -qU langchain langchain-ollama langgraph
import json
import re
from langchain.agents import create_agent, AgentState
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def extract_tool_call_from_xml(content: str):
    """Extract tool call from XML format like <tools>{"name": "...", "arguments": {...}}</tools>"""
    # Try to find JSON in <tools> tags
    match = re.search(r'<tools>\s*(\{.*?\})\s*</tools>', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object directly
    match = re.search(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\{[^{}]*\}[^{}]*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None

# Use Ollama with an open-source model (requires Ollama running locally)
# Make sure to pull the model first: ollama pull hhao/qwen2.5-coder-tools
llm = ChatOllama(
    model="hhao/qwen2.5-coder-tools",
    base_url="http://localhost:11434",
    temperature=0.0,
)

# Convert function to tool and bind to LLM
tools = [get_weather]
tool_dict = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# Create agent with state management
checkpointer = MemorySaver()
agent = create_agent(
    model=llm_with_tools,
    tools=tools,
    system_prompt="You are a helpful assistant. When asked about weather, you MUST use the get_weather tool.",
    state_schema=AgentState,
    checkpointer=checkpointer,
)

# Run the agent with manual tool execution loop
print("Invoking agent...")
messages = [HumanMessage(content="what is the weather in sf")]
thread_id = "weather_test"
max_iterations = 5

for iteration in range(max_iterations):
    result = agent.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}},
    )
    
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        
        # Check the last message
        last_msg = messages[-1] if messages else None
        
        if isinstance(last_msg, AIMessage):
            # Check if it has tool calls
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                # Execute tools
                for tool_call in last_msg.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    print(f"🔧 Executing tool: {tool_name}({tool_args})")
                    
                    if tool_name in tool_dict:
                        tool_result = tool_dict[tool_name].invoke(tool_args)
                        print(f"   Result: {tool_result}")
                        # Add tool message to continue the conversation
                        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call.get("id", "")))
                continue
            
            # Check if content contains XML/JSON tool call
            if last_msg.content:
                tool_call_data = extract_tool_call_from_xml(str(last_msg.content))
                if tool_call_data:
                    tool_name = tool_call_data.get("name", "")
                    tool_args = tool_call_data.get("arguments", {})
                    print(f"🔧 Found tool call in XML format: {tool_name}({tool_args})")
                    
                    if tool_name in tool_dict:
                        tool_result = tool_dict[tool_name].invoke(tool_args)
                        print(f"   Result: {tool_result}")
                        # Add tool message and continue
                        messages.append(ToolMessage(content=str(tool_result), tool_call_id="manual_1"))
                        continue
            
            # No tool call, this is the final response
            print(f"\n{'='*60}")
            print(f"Final response: {last_msg.content}")
            print(f"{'='*60}")
            break

if iteration == max_iterations - 1:
    print("⚠️ Reached max iterations without final response")