[Skip to content](https://reference.langchain.com/python/langchain/tools/#langchain.tools.tool)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain/tools.md "Edit this page")

# Tools

Reference docs

This page contains **reference documentation** for Tools. See [the docs](https://docs.langchain.com/oss/python/langchain/tools) for conceptual guides, tutorials, and examples on using Tools.

## ``tool [¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.tool "Copy anchor link to this section for reference")

```
tool(
    name_or_callable: str | Callable | None = None,
    runnable: Runnable | None = None,
    *args: Any,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> BaseTool | Callable[[Callable | Runnable], BaseTool]
```

Convert Python functions and `Runnables` to LangChain tools.

Can be used as a decorator with or without arguments to create tools from functions.

Functions can have any signature - the tool will automatically infer input schemas
unless disabled.

Requirements

- Functions must have type hints for proper schema inference
- When `infer_schema=False`, functions must be `(str) -> str` and have
docstrings
- When using with `Runnable`, a string name must be provided

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name_or_callable` | Optional name of the tool or the `Callable` to be<br>converted to a tool. Overrides the function's name.<br>Must be provided as a positional argument.<br>**TYPE:**`str | Callable | None`**DEFAULT:**`None` |
| `runnable` | Optional `Runnable` to convert to a tool.<br>Must be provided as a positional argument.<br>**TYPE:**`Runnable | None`**DEFAULT:**`None` |
| `description` | Optional description for the tool.<br>Precedence for the tool description value is as follows:<br>- This `description` argument<br>(used even if docstring and/or `args_schema` are provided)<br>- Tool function docstring<br>(used even if `args_schema` is provided)<br>- `args_schema` description<br>(used only if `description` and docstring are not provided)<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `*args` | Extra positional arguments. Must be empty.<br>**TYPE:**`Any`**DEFAULT:**`()` |
| `return_direct` | Whether to return directly from the tool rather than continuing<br>the agent loop.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `args_schema` | Optional argument schema for user to specify.<br>**TYPE:**`ArgsSchema | None`**DEFAULT:**`None` |
| `infer_schema` | Whether to infer the schema of the arguments from the function's<br>signature. This also makes the resultant tool accept a dictionary input to<br>its `run()` function.<br>**TYPE:**`bool`**DEFAULT:**`True` |
| `response_format` | The tool response format.<br>If `'content'`, then the output of the tool is interpreted as the contents<br>of a `ToolMessage`.<br>If `'content_and_artifact'`, then the output is expected to be a two-tuple<br>corresponding to the `(content, artifact)` of a `ToolMessage`.<br>**TYPE:**`Literal['content', 'content_and_artifact']`**DEFAULT:**`'content'` |
| `parse_docstring` | If `infer_schema` and `parse_docstring`, will attempt to<br>parse parameter descriptions from Google Style function docstrings.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `error_on_invalid_docstring` | If `parse_docstring` is provided, configure<br>whether to raise `ValueError` on invalid Google Style docstrings.<br>**TYPE:**`bool`**DEFAULT:**`True` |
| `extras` | Optional provider-specific extra fields for the tool.<br>Used to pass configuration that doesn't fit into standard tool fields.<br>Chat models should process known extras when constructing model payloads.<br>Example<br>For example, Anthropic-specific fields like `cache_control`,<br>`defer_loading`, or `input_examples`.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If too many positional arguments are provided (e.g. violating the<br>`*args` constraint). |
| `ValueError` | If a `Runnable` is provided without a string name. When using `tool`<br>with a `Runnable`, a `str` name must be provided as the `name_or_callable`. |
| `ValueError` | If the first argument is not a string or callable with<br>a `__name__` attribute. |
| `ValueError` | If the function does not have a docstring and description<br>is not provided and `infer_schema` is `False`. |
| `ValueError` | If `parse_docstring` is `True` and the function has an invalid<br>Google-style docstring and `error_on_invalid_docstring` is True. |
| `ValueError` | If a `Runnable` is provided that does not have an object schema. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `BaseTool | Callable[[Callable | Runnable], BaseTool]` | The tool. |

Examples:

```
@tool
def search_api(query: str) -> str:
    # Searches the API for the query.
    return

@tool("search", return_direct=True)
def search_api(query: str) -> str:
    # Searches the API for the query.
    return

@tool(response_format="content_and_artifact")
def search_api(query: str) -> tuple[str, dict]:
    return "partial json of results", {"full": "object of results"}
```

Parse Google-style docstrings:

```
@tool(parse_docstring=True)
def foo(bar: str, baz: int) -> str:
    """The foo.

    Args:
        bar: The bar.
        baz: The baz.
    """
    return bar

foo.args_schema.model_json_schema()
```

```
{
    "title": "foo",
    "description": "The foo.",
    "type": "object",
    "properties": {
        "bar": {
            "title": "Bar",
            "description": "The bar.",
            "type": "string",
        },
        "baz": {
            "title": "Baz",
            "description": "The baz.",
            "type": "integer",
        },
    },
    "required": ["bar", "baz"],
}
```

Note that parsing by default will raise `ValueError` if the docstring
is considered invalid. A docstring is considered invalid if it contains
arguments not in the function signature, or is unable to be parsed into
a summary and `"Args:"` blocks. Examples below:

```
# No args section
def invalid_docstring_1(bar: str, baz: int) -> str:
    """The foo."""
    return bar

# Improper whitespace between summary and args section
def invalid_docstring_2(bar: str, baz: int) -> str:
    """The foo.
    Args:
        bar: The bar.
        baz: The baz.
    """
    return bar

# Documented args absent from function signature
def invalid_docstring_3(bar: str, baz: int) -> str:
    """The foo.

    Args:
        banana: The bar.
        monkey: The baz.
    """
    return bar
```

## ``BaseTool [¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.BaseTool "Copy anchor link to this section for reference")

Bases: `RunnableSerializable[str | dict | ToolCall, Any]`

Base class for all LangChain tools.

This abstract class defines the interface that all LangChain tools must implement.

Tools are components that can be called by agents to perform specific actions.

| METHOD | DESCRIPTION |
| --- | --- |
| `invoke` | Transform a single input into an output. |
| `ainvoke` | Transform a single input into an output. |
| `get_input_schema` | The tool's input schema. |
| `get_output_schema` | Get a Pydantic model that can be used to validate output to the `Runnable`. |

### ``name`instance-attribute`[¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.BaseTool.name "Copy anchor link to this section for reference")

```
name: str
```

The unique name of the tool that clearly communicates its purpose.

### ``description`instance-attribute`[¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.BaseTool.description "Copy anchor link to this section for reference")

```
description: str
```

Used to tell the model how/when/why to use the tool.

You can provide few-shot examples as a part of the description.

### ``response\_format`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.BaseTool.response_format "Copy anchor link to this section for reference")

```
response_format: Literal['content', 'content_and_artifact'] = 'content'
```

The tool response format.

If `'content'` then the output of the tool is interpreted as the contents of a
`ToolMessage`. If `'content_and_artifact'` then the output is expected to be a
two-tuple corresponding to the `(content, artifact)` of a `ToolMessage`.

### ``args\_schema`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.BaseTool.args_schema "Copy anchor link to this section for reference")

```
args_schema: Annotated[ArgsSchema | None, SkipValidation()] = Field(
    default=None, description="The tool schema."
)
```

Pydantic model class to validate and parse the tool's input arguments.

Args schema should be either:

- A subclass of `pydantic.BaseModel`.
- A subclass of `pydantic.v1.BaseModel` if accessing v1 namespace in pydantic 2
- A JSON schema dict

### ``return\_direct`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.BaseTool.return_direct "Copy anchor link to this section for reference")

```
return_direct: bool = False
```

Whether to return the tool's output directly.

Setting this to `True` means that after the tool is called, the `AgentExecutor` will
stop looping.

### ``extras`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.BaseTool.extras "Copy anchor link to this section for reference")

```
extras: dict[str, Any] | None = None
```

Optional provider-specific extra fields for the tool.

This is used to pass provider-specific configuration that doesn't fit into
standard tool fields.

Example

Anthropic-specific fields like [`cache_control`](https://docs.langchain.com/oss/python/integrations/chat/anthropic#prompt-caching),
[`defer_loading`](https://docs.langchain.com/oss/python/integrations/chat/anthropic#tool-search),
or `input_examples`.

```
@tool(extras={"defer_loading": True, "cache_control": {"type": "ephemeral"}})
def my_tool(x: str) -> str:
    return x
```

### ``invoke [¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.BaseTool.invoke "Copy anchor link to this section for reference")

```
invoke(
    input: str | dict | ToolCall, config: RunnableConfig | None = None, **kwargs: Any
) -> Any
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``ainvoke`async`[¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.BaseTool.ainvoke "Copy anchor link to this section for reference")

```
ainvoke(
    input: str | dict | ToolCall, config: RunnableConfig | None = None, **kwargs: Any
) -> Any
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``get\_input\_schema [¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.BaseTool.get_input_schema "Copy anchor link to this section for reference")

```
get_input_schema(config: RunnableConfig | None = None) -> type[BaseModel]
```

The tool's input schema.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | The configuration for the tool.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `type[BaseModel]` | The input schema for the tool. |

### ``get\_output\_schema [¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.BaseTool.get_output_schema "Copy anchor link to this section for reference")

```
get_output_schema(config: RunnableConfig | None = None) -> type[BaseModel]
```

Get a Pydantic model that can be used to validate output to the `Runnable`.

`Runnable` objects that leverage the `configurable_fields` and
`configurable_alternatives` methods will have a dynamic output schema that
depends on which configuration the `Runnable` is invoked with.

This method allows to get an output schema for a specific configuration.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | A config to use when generating the schema.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `type[BaseModel]` | A Pydantic model that can be used to validate output. |

## ``InjectedState [¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.InjectedState "Copy anchor link to this section for reference")

Bases: `InjectedToolArg`

Annotation for injecting graph state into tool arguments.

This annotation enables tools to access graph state without exposing state
management details to the language model. Tools annotated with `InjectedState`
receive state data automatically during execution while remaining invisible
to the model's tool-calling interface.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `field` | Optional key to extract from the state dictionary. If `None`, the entire<br>state is injected. If specified, only that field's value is injected.<br>This allows tools to request specific state components rather than<br>processing the full state structure.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

Example

```
from typing import List
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import BaseMessage, AIMessage
from langchain.tools import InjectedState, ToolNode, tool

class AgentState(TypedDict):
    messages: List[BaseMessage]
    foo: str

@tool
def state_tool(x: int, state: Annotated[dict, InjectedState]) -> str:
    '''Do something with state.'''
    if len(state["messages"]) > 2:
        return state["foo"] + str(x)
    else:
        return "not enough messages"

@tool
def foo_tool(x: int, foo: Annotated[str, InjectedState("foo")]) -> str:
    '''Do something else with state.'''
    return foo + str(x + 1)

node = ToolNode([state_tool, foo_tool])

tool_call1 = {"name": "state_tool", "args": {"x": 1}, "id": "1", "type": "tool_call"}
tool_call2 = {"name": "foo_tool", "args": {"x": 1}, "id": "2", "type": "tool_call"}
state = {
    "messages": [AIMessage("", tool_calls=[tool_call1, tool_call2])],
    "foo": "bar",
}
node.invoke(state)
```

```
[\
    ToolMessage(content="not enough messages", name="state_tool", tool_call_id="1"),\
    ToolMessage(content="bar2", name="foo_tool", tool_call_id="2"),\
]
```

Note

- `InjectedState` arguments are automatically excluded from tool schemas
presented to language models
- `ToolNode` handles the injection process during execution
- Tools can mix regular arguments (controlled by the model) with injected
arguments (controlled by the system)
- State injection occurs after the model generates tool calls but before
tool execution

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Initialize the `InjectedState` annotation. |

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.InjectedState.__init__ "Copy anchor link to this section for reference")

```
__init__(field: str | None = None) -> None
```

Initialize the `InjectedState` annotation.

## ``InjectedStore [¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.InjectedStore "Copy anchor link to this section for reference")

Bases: `InjectedToolArg`

Annotation for injecting persistent store into tool arguments.

This annotation enables tools to access LangGraph's persistent storage system
without exposing storage details to the language model. Tools annotated with
`InjectedStore` receive the store instance automatically during execution while
remaining invisible to the model's tool-calling interface.

The store provides persistent, cross-session data storage that tools can use
for maintaining context, user preferences, or any other data that needs to
persist beyond individual workflow executions.

Warning

`InjectedStore` annotation requires `langchain-core >= 0.3.8`

Example

```
from typing_extensions import Annotated
from langgraph.store.memory import InMemoryStore
from langchain.tools import InjectedStore, ToolNode, tool

@tool
def save_preference(
    key: str,
    value: str,
    store: Annotated[Any, InjectedStore()]
) -> str:
    """Save user preference to persistent storage."""
    store.put(("preferences",), key, value)
    return f"Saved {key} = {value}"

@tool
def get_preference(
    key: str,
    store: Annotated[Any, InjectedStore()]
) -> str:
    """Retrieve user preference from persistent storage."""
    result = store.get(("preferences",), key)
    return result.value if result else "Not found"
```

Usage with `ToolNode` and graph compilation:

```
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
tool_node = ToolNode([save_preference, get_preference])

graph = StateGraph(State)
graph.add_node("tools", tool_node)
compiled_graph = graph.compile(store=store)  # Store is injected automatically
```

Cross-session persistence:

```
# First session
result1 = graph.invoke({"messages": [HumanMessage("Save my favorite color as blue")]})

# Later session - data persists
result2 = graph.invoke({"messages": [HumanMessage("What's my favorite color?")]})
```

Note

- `InjectedStore` arguments are automatically excluded from tool schemas
presented to language models
- The store instance is automatically injected by `ToolNode` during execution
- Tools can access namespaced storage using the store's get/put methods
- Store injection requires the graph to be compiled with a store instance
- Multiple tools can share the same store instance for data consistency

## ``InjectedToolArg [¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.InjectedToolArg "Copy anchor link to this section for reference")

Annotation for tool arguments that are injected at runtime.

Tool arguments annotated with this class are not included in the tool
schema sent to language models and are instead injected during execution.

## ``InjectedToolCallId [¶](https://reference.langchain.com/python/langchain/tools/\#langchain.tools.InjectedToolCallId "Copy anchor link to this section for reference")

Bases: `InjectedToolArg`

Annotation for injecting the tool call ID.

This annotation is used to mark a tool parameter that should receive
the tool call ID at runtime.

```
from typing import Annotated
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId

@tool
def foo(
    x: int, tool_call_id: Annotated[str, InjectedToolCallId]
) -> ToolMessage:
    """Return x."""
    return ToolMessage(
        str(x),
        artifact=x,
        name="foo",
        tool_call_id=tool_call_id
    )
```

Back to top