# When a model returns tool calls, you need to execute the tools and pass the results back to the model. 
# This creates a conversation loop where the model can use tool results to generate its final response. 
# LangChain includes agent abstractions that handle this orchestration for you.
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