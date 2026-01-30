# Models can request to call tools that perform tasks such as fetching data from a database, searching the web, 
#     or running code. Tools are pairings of:
# A schema, including the name of the tool, a description, and/or argument definitions (often a JSON schema)
# A function or coroutine to execute.

# To make tools that you have defined available for use by a model, you must bind them using bind_tools. 
# In subsequent invocations, the model can choose to call any of the bound tools as needed.
# Some model providers offer built-in tools that can be enabled via model or invocation parameters 
# (e.g. ChatOpenAI, ChatAnthropic). Check the respective provider reference for details.

# When binding user-defined tools, the model’s response includes a request to execute a tool. 
# When using a model separately from an agent, it is up to you to execute the requested tool and 
# return the result back to the model for use in subsequent reasoning. When using an agent, 
# the agent loop will handle the tool execution loop for you.

## Binding user tools
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


model_with_tools = model.bind_tools([get_weather])  

response = model_with_tools.invoke("What's the weather like in Boston?")
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")

## Tool execution loop
# Each ToolMessage returned by the tool includes a tool_call_id that matches the original tool call, helping the model correlate results with requests.

# Bind (potentially multiple) tools to the model
model_with_tools = model.bind_tools([get_weather])

# Step 1: Model generates tool calls
messages = [{"role": "user", "content": "What's the weather in Boston?"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# Step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
    # Execute the tool with the generated arguments
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# Step 3: Pass results back to model for final response
final_response = model_with_tools.invoke(messages)
print(final_response.text)
# "The current weather in Boston is 72°F and sunny."

## Forcing tool calls
# By default, the model has the freedom to choose which bound tool to use based on the user’s input. 
# However, you might want to force choosing a tool, ensuring the model uses either a particular tool or any tool from a given list:

model_with_tools = model.bind_tools([tool_1], tool_choice="any")

## Parallel tool calls
# Many models support calling multiple tools in parallel when appropriate. 
# This allows the model to gather information from different sources simultaneously.
# The model intelligently determines when parallel execution is appropriate based on the independence of the requested operations.
# Most models supporting tool calling enable parallel tool calls by default. Some (including OpenAI and Anthropic) 
# allow you to disable this feature. To do this, set parallel_tool_calls=False:

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke(
    "What's the weather in Boston and Tokyo?"
)


# The model may generate multiple tool calls
print(response.tool_calls)
# [
#   {'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': 'call_1'},
#   {'name': 'get_weather', 'args': {'location': 'Tokyo'}, 'id': 'call_2'},
# ]


# Execute all tools (can be done in parallel with async)
results = []
for tool_call in response.tool_calls:
    if tool_call['name'] == 'get_weather':
        result = get_weather.invoke(tool_call)
    ...
    results.append(result)

## Streaming tool calls
# When streaming responses, tool calls are progressively built through ToolCallChunk. This allows you to 
# see tool calls as they’re being generated rather than waiting for the complete response.

for chunk in model_with_tools.stream(
    "What's the weather in Boston and Tokyo?"
):
    # Tool call chunks arrive progressively
    for tool_chunk in chunk.tool_call_chunks:
        if name := tool_chunk.get("name"):
            print(f"Tool: {name}")
        if id_ := tool_chunk.get("id"):
            print(f"ID: {id_}")
        if args := tool_chunk.get("args"):
            print(f"Args: {args}")

# Output:
# Tool: get_weather
# ID: call_SvMlU1TVIZugrFLckFE2ceRE
# Args: {"lo
# Args: catio
# Args: n": "B
# Args: osto
# Args: n"}
# Tool: get_weather
# ID: call_QMZdy6qInx13oWKE7KhuhOLR
# Args: {"lo
# Args: catio
# Args: n": "T
# Args: okyo
# Args: "}

# You can accumulate chunks to build complete tool calls:
gathered = None
for chunk in model_with_tools.stream("What's the weather in Boston?"):
    gathered = chunk if gathered is None else gathered + chunk
    print(gathered.tool_calls)
