[Skip to content](https://reference.langchain.com/python/langchain/agents/#langchain.agents)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain/agents.md "Edit this page")

# Agents

Reference docs

This page contains **reference documentation** for Agents. See [the docs](https://docs.langchain.com/oss/python/langchain/agents) for conceptual guides, tutorials, and examples on using Agents.

## ``agents [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents "Copy anchor link to this section for reference")

Entrypoint to building [Agents](https://docs.langchain.com/oss/python/langchain/agents) with LangChain.

### ``create\_agent [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent "Copy anchor link to this section for reference")

```
create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT]
    | type[ResponseT]
    | dict[str, Any]
    | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
) -> CompiledStateGraph[\
    AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]\
]
```

Creates an agent graph that calls tools in a loop until a stopping condition is met.

For more details on using `create_agent`,
visit the [Agents](https://docs.langchain.com/oss/python/langchain/agents) docs.

| PARAMETER | DESCRIPTION |
| --- | --- |
| #### `model` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(model) "Copy anchor link to this section for reference") | The language model for the agent.<br>Can be a string identifier (e.g., `"openai:gpt-4"`) or a direct chat model<br>instance (e.g., [`ChatOpenAI`](https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI/#langchain_openai.ChatOpenAI "<code class=\"doc-symbol doc-symbol-heading doc-symbol-class\"></code>            <span class=\"doc doc-object-name doc-class-name\">ChatOpenAI</span>") or other another<br>[LangChain chat model](https://docs.langchain.com/oss/python/integrations/chat)).<br>For a full list of supported model strings, see<br>[`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model(model_provider) "<code>model_provider</code>").<br>See the [Models](https://docs.langchain.com/oss/python/langchain/models)<br>docs for more information.<br>**TYPE:**`str | BaseChatModel` |
| #### `tools` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(tools) "Copy anchor link to this section for reference") | A list of tools, `dict`, or `Callable`.<br>If `None` or an empty list, the agent will consist of a model node without a<br>tool calling loop.<br>See the [Tools](https://docs.langchain.com/oss/python/langchain/tools)<br>docs for more information.<br>**TYPE:**`Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None`**DEFAULT:**`None` |
| #### `system_prompt` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(system_prompt) "Copy anchor link to this section for reference") | An optional system prompt for the LLM.<br>Can be a `str` (which will be converted to a `SystemMessage`) or a<br>`SystemMessage` instance directly. The system message is added to the<br>beginning of the message list when calling the model.<br>**TYPE:**`str | SystemMessage | None`**DEFAULT:**`None` |
| #### `middleware` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(middleware) "Copy anchor link to this section for reference") | A sequence of middleware instances to apply to the agent.<br>Middleware can intercept and modify agent behavior at various stages.<br>See the [Middleware](https://docs.langchain.com/oss/python/langchain/middleware)<br>docs for more information.<br>**TYPE:**`Sequence[AgentMiddleware[StateT_co, ContextT]]`**DEFAULT:**`()` |
| #### `response_format` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(response_format) "Copy anchor link to this section for reference") | An optional configuration for structured responses.<br>Can be a `ToolStrategy`, `ProviderStrategy`, or a Pydantic model class.<br>If provided, the agent will handle structured output during the<br>conversation flow.<br>Raw schemas will be wrapped in an appropriate strategy based on model<br>capabilities.<br>See the [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)<br>docs for more information.<br>**TYPE:**`ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None`**DEFAULT:**`None` |
| #### `state_schema` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(state_schema) "Copy anchor link to this section for reference") | An optional `TypedDict` schema that extends `AgentState`.<br>When provided, this schema is used instead of `AgentState` as the base<br>schema for merging with middleware state schemas. This allows users to<br>add custom state fields without needing to create custom middleware.<br>Generally, it's recommended to use `state_schema` extensions via middleware<br>to keep relevant extensions scoped to corresponding hooks / tools.<br>**TYPE:**`type[AgentState[ResponseT]] | None`**DEFAULT:**`None` |
| #### `context_schema` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(context_schema) "Copy anchor link to this section for reference") | An optional schema for runtime context.<br>**TYPE:**`type[ContextT] | None`**DEFAULT:**`None` |
| #### `checkpointer` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(checkpointer) "Copy anchor link to this section for reference") | An optional checkpoint saver object.<br>Used for persisting the state of the graph (e.g., as chat memory) for a<br>single thread (e.g., a single conversation).<br>**TYPE:**`Checkpointer | None`**DEFAULT:**`None` |
| #### `store` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(store) "Copy anchor link to this section for reference") | An optional store object.<br>Used for persisting data across multiple threads (e.g., multiple<br>conversations / users).<br>**TYPE:**`BaseStore | None`**DEFAULT:**`None` |
| #### `interrupt_before` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(interrupt_before) "Copy anchor link to this section for reference") | An optional list of node names to interrupt before.<br>Useful if you want to add a user confirmation or other interrupt<br>before taking an action.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| #### `interrupt_after` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(interrupt_after) "Copy anchor link to this section for reference") | An optional list of node names to interrupt after.<br>Useful if you want to return directly or run additional processing<br>on an output.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| #### `debug` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(debug) "Copy anchor link to this section for reference") | Whether to enable verbose logging for graph execution.<br>When enabled, prints detailed information about each node execution, state<br>updates, and transitions during agent runtime. Useful for debugging<br>middleware behavior and understanding agent execution flow.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| #### `name` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(name) "Copy anchor link to this section for reference") | An optional name for the `CompiledStateGraph`.<br>This name will be automatically used when adding the agent graph to<br>another graph as a subgraph node - particularly useful for building<br>multi-agent systems.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| #### `cache` [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.create_agent(cache) "Copy anchor link to this section for reference") | An optional `BaseCache` instance to enable caching of graph execution.<br>**TYPE:**`BaseCache[Any] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `CompiledStateGraph[AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]` | A compiled `StateGraph` that can be used for chat interactions. |

| RAISES | DESCRIPTION |
| --- | --- |
| `AssertionError` | If duplicate middleware instances are provided. |

The agent node calls the language model with the messages list (after applying
the system prompt). If the resulting [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage "<code class=\"doc-symbol doc-symbol-heading doc-symbol-class\"></code>            <span class=\"doc doc-object-name doc-class-name\">AIMessage</span>")
contains `tool_calls`, the graph will then call the tools. The tools node executes
the tools and adds the responses to the messages list as
[`ToolMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ToolMessage "<code class=\"doc-symbol doc-symbol-heading doc-symbol-class\"></code>            <span class=\"doc doc-object-name doc-class-name\">ToolMessage</span>") objects. The agent node then calls
the language model again. The process repeats until no more `tool_calls` are present
in the response. The agent then returns the full list of messages.

Example

```
from langchain.agents import create_agent

def check_weather(location: str) -> str:
    '''Return the weather forecast for the specified location.'''
    return f"It's always sunny in {location}"

graph = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[check_weather],
    system_prompt="You are a helpful assistant",
)
inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)
```

### ``AgentState [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.AgentState "Copy anchor link to this section for reference")

Bases: `TypedDict`, `Generic[ResponseT]`

State schema for the agent.

* * *

## Structured output [¶](https://reference.langchain.com/python/langchain/agents/\#structured-output "Copy anchor link to this section for reference")

## ``ResponseFormat`module-attribute`[¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.structured_output.ResponseFormat "Copy anchor link to this section for reference")

```
ResponseFormat = (
    ToolStrategy[SchemaT] | ProviderStrategy[SchemaT] | AutoStrategy[SchemaT]
)
```

Union type for all supported response format strategies.

## ``ToolStrategy`dataclass`[¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.structured_output.ToolStrategy "Copy anchor link to this section for reference")

```
ToolStrategy(
    schema: type[SchemaT] | UnionType | dict[str, Any],
    *,
    tool_message_content: str | None = None,
    handle_errors: bool
    | str
    | type[Exception]
    | tuple[type[Exception], ...]
    | Callable[[Exception], str] = True,
)
```

Bases: `Generic[SchemaT]`

Use a tool calling strategy for model responses.

Initialize `ToolStrategy`.

Initialize `ToolStrategy` with schemas, tool message content, and error handling
strategy.

### ``tool\_message\_content`instance-attribute`[¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.structured_output.ToolStrategy.tool_message_content "Copy anchor link to this section for reference")

```
tool_message_content: str | None = tool_message_content
```

The content of the tool message to be returned when the model calls
an artificial structured output tool.

### ``handle\_errors`instance-attribute`[¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.structured_output.ToolStrategy.handle_errors "Copy anchor link to this section for reference")

```
handle_errors: (
    bool
    | str
    | type[Exception]
    | tuple[type[Exception], ...]
    | Callable[[Exception], str]
) = handle_errors
```

Error handling strategy for structured output via `ToolStrategy`.

- `True`: Catch all errors with default error template
- `str`: Catch all errors with this custom message
- `type[Exception]`: Only catch this exception type with default message
- `tuple[type[Exception], ...]`: Only catch these exception types with default
message
- `Callable[[Exception], str]`: Custom function that returns error message
- `False`: No retry, let exceptions propagate

## ``ProviderStrategy`dataclass`[¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.structured_output.ProviderStrategy "Copy anchor link to this section for reference")

```
ProviderStrategy(schema: type[SchemaT] | dict[str, Any], *, strict: bool | None = None)
```

Bases: `Generic[SchemaT]`

Use the model provider's native structured output method.

Initialize `ProviderStrategy` with schema.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `schema` | Schema to enforce via the provider's native structured output.<br>**TYPE:**`type[SchemaT] | dict[str, Any]` |
| `strict` | Whether to request strict provider-side schema enforcement.<br>**TYPE:**`bool | None`**DEFAULT:**`None` |

| METHOD | DESCRIPTION |
| --- | --- |
| `to_model_kwargs` | Convert to kwargs to bind to a model to force structured output. |

### ``to\_model\_kwargs [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.structured_output.ProviderStrategy.to_model_kwargs "Copy anchor link to this section for reference")

```
to_model_kwargs() -> dict[str, Any]
```

Convert to kwargs to bind to a model to force structured output.

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, Any]` | The kwargs to bind to a model. |

## ``AutoStrategy [¶](https://reference.langchain.com/python/langchain/agents/\#langchain.agents.structured_output.AutoStrategy "Copy anchor link to this section for reference")

```
AutoStrategy(schema: type[SchemaT] | dict[str, Any])
```

Bases: `Generic[SchemaT]`

Automatically select the best strategy for structured output.

Initialize `AutoStrategy` with schema.

Back to top