[Skip to content](https://reference.langchain.com/python/langchain_mcp_adapters/#langchain-mcp-adapters)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_mcp_adapters/index.md "Edit this page")

# `langchain-mcp-adapters` [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain-mcp-adapters "Copy anchor link to this section for reference")

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-mcp-adapters?label=%20)](https://pypi.org/project/langchain-mcp-adapters/#history)[![PyPI - License](https://img.shields.io/pypi/l/langchain-mcp-adapters)](https://opensource.org/licenses/MIT)[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-mcp-adapters)](https://pypistats.org/packages/langchain-mcp-adapters)

Reference documentation for the [`langchain-mcp-adapters`](https://pypi.org/project/langchain-mcp-adapters/) package.

## ``client [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.client "Copy anchor link to this section for reference")

Client for connecting to multiple MCP servers and loading LC tools/resources.

This module provides the `MultiServerMCPClient` class for managing connections
to multiple MCP servers and loading tools, prompts, and resources from them.

### ``MultiServerMCPClient [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.client.MultiServerMCPClient "Copy anchor link to this section for reference")

Client for connecting to multiple MCP servers.

Loads LangChain-compatible tools, prompts and resources from MCP servers.

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Initialize a `MultiServerMCPClient` with MCP servers connections. |
| `session` | Connect to an MCP server and initialize a session. |
| `get_tools` | Get a list of all tools from all connected servers. |
| `get_prompt` | Get a prompt from a given MCP server. |
| `get_resources` | Get resources from MCP server(s). |
| `__aenter__` | Async context manager entry point. |
| `__aexit__` | Async context manager exit point. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.client.MultiServerMCPClient.__init__ "Copy anchor link to this section for reference")

```
__init__(
    connections: dict[str, Connection] | None = None,
    *,
    callbacks: Callbacks | None = None,
    tool_interceptors: list[ToolCallInterceptor] | None = None,
    tool_name_prefix: bool = False,
) -> None
```

Initialize a `MultiServerMCPClient` with MCP servers connections.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `connections` | A `dict` mapping server names to connection configurations. If<br>`None`, no initial connections are established.<br>**TYPE:**`dict[str, Connection] | None`**DEFAULT:**`None` |
| `callbacks` | Optional callbacks for handling notifications and events.<br>**TYPE:**`Callbacks | None`**DEFAULT:**`None` |
| `tool_interceptors` | Optional list of tool call interceptors for modifying<br>requests and responses.<br>**TYPE:**`list[ToolCallInterceptor] | None`**DEFAULT:**`None` |
| `tool_name_prefix` | If `True`, tool names are prefixed with the server name<br>using an underscore separator (e.g., `"weather_search"` instead of<br>`"search"`). This helps avoid conflicts when multiple servers have tools<br>with the same name. Defaults to `False`.<br>**TYPE:**`bool`**DEFAULT:**`False` |

Basic usage (starting a new session on each tool call)

```
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Make sure to update to the full absolute path to your
            # math_server.py file
            "args": ["/path/to/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            # Make sure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp",
            "transport": "http",
        }
    }
)
all_tools = await client.get_tools()
```

Explicitly starting a session

```
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

client = MultiServerMCPClient({...})
async with client.session("math") as session:
    tools = await load_mcp_tools(session)
```

#### ``session`async`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.client.MultiServerMCPClient.session "Copy anchor link to this section for reference")

```
session(
    server_name: str, *, auto_initialize: bool = True
) -> AsyncIterator[ClientSession]
```

Connect to an MCP server and initialize a session.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `server_name` | Name to identify this server connection<br>**TYPE:**`str` |
| `auto_initialize` | Whether to automatically initialize the session<br>**TYPE:**`bool`**DEFAULT:**`True` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the server name is not found in the connections |

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[ClientSession]` | An initialized `ClientSession` |

#### ``get\_tools`async`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.client.MultiServerMCPClient.get_tools "Copy anchor link to this section for reference")

```
get_tools(*, server_name: str | None = None) -> list[BaseTool]
```

Get a list of all tools from all connected servers.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `server_name` | Optional name of the server to get tools from.<br>If `None`, all tools from all servers will be returned.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

Note

A new session will be created for each tool call

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[BaseTool]` | A list of LangChain [tools](https://docs.langchain.com/oss/python/langchain/tools) |

#### ``get\_prompt`async`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.client.MultiServerMCPClient.get_prompt "Copy anchor link to this section for reference")

```
get_prompt(
    server_name: str, prompt_name: str, *, arguments: dict[str, Any] | None = None
) -> list[HumanMessage | AIMessage]
```

Get a prompt from a given MCP server.

#### ``get\_resources`async`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.client.MultiServerMCPClient.get_resources "Copy anchor link to this section for reference")

```
get_resources(
    server_name: str | None = None, *, uris: str | list[str] | None = None
) -> list[Blob]
```

Get resources from MCP server(s).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `server_name` | Optional name of the server to get resources from.<br>If `None`, all resources from all servers will be returned.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `uris` | Optional resource URI or list of URIs to load. If not provided,<br>all resources will be loaded.<br>**TYPE:**`str | list[str] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Blob]` | A list of LangChain [Blob](https://reference.langchain.com/python/langchain_core/documents/#langchain_core.documents.base.Blob "<code class=\"doc-symbol doc-symbol-heading doc-symbol-class\"></code>            <span class=\"doc doc-object-name doc-class-name\">Blob</span>") objects. |

#### ``\_\_aenter\_\_`async`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.client.MultiServerMCPClient.__aenter__ "Copy anchor link to this section for reference")

```
__aenter__() -> MultiServerMCPClient
```

Async context manager entry point.

| RAISES | DESCRIPTION |
| --- | --- |
| `NotImplementedError` | Context manager support has been removed. |

#### ``\_\_aexit\_\_ [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.client.MultiServerMCPClient.__aexit__ "Copy anchor link to this section for reference")

```
__aexit__(
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
) -> None
```

Async context manager exit point.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `exc_type` | Exception type if an exception occurred.<br>**TYPE:**`type[BaseException] | None` |
| `exc_val` | Exception value if an exception occurred.<br>**TYPE:**`BaseException | None` |
| `exc_tb` | Exception traceback if an exception occurred.<br>**TYPE:**`TracebackType | None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `NotImplementedError` | Context manager support has been removed. |

## ``tools [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.tools "Copy anchor link to this section for reference")

Tools adapter for converting MCP tools to LangChain tools.

This module provides functionality to convert MCP tools into LangChain-compatible
tools, handle tool execution, and manage tool conversion between the two formats.

| FUNCTION | DESCRIPTION |
| --- | --- |
| `load_mcp_tools` | Load all available MCP tools and convert them to LangChain [tools](https://docs.langchain.com/oss/python/langchain/tools). |

### ``MCPToolArtifact [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.tools.MCPToolArtifact "Copy anchor link to this section for reference")

Bases: `TypedDict`

Artifact returned from MCP tool calls.

This TypedDict wraps the structured content from MCP tool calls,
allowing for future extension if MCP adds more fields to tool results.

| ATTRIBUTE | DESCRIPTION |
| --- | --- |
| `structured_content` | The structured content returned by the MCP tool,<br>corresponding to the structuredContent field in CallToolResult.<br>**TYPE:**`dict[str, Any]` |

### ``load\_mcp\_tools`async`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.tools.load_mcp_tools "Copy anchor link to this section for reference")

```
load_mcp_tools(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
    callbacks: Callbacks | None = None,
    tool_interceptors: list[ToolCallInterceptor] | None = None,
    server_name: str | None = None,
    tool_name_prefix: bool = False,
) -> list[BaseTool]
```

Load all available MCP tools and convert them to LangChain [tools](https://docs.langchain.com/oss/python/langchain/tools).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `session` | The MCP client session. If `None`, connection must be provided.<br>**TYPE:**`ClientSession | None` |
| `connection` | Connection config to create a new session if session is `None`.<br>**TYPE:**`Connection | None`**DEFAULT:**`None` |
| `callbacks` | Optional `Callbacks` for handling notifications and events.<br>**TYPE:**`Callbacks | None`**DEFAULT:**`None` |
| `tool_interceptors` | Optional list of interceptors for tool call processing.<br>**TYPE:**`list[ToolCallInterceptor] | None`**DEFAULT:**`None` |
| `server_name` | Name of the server these tools belong to.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `tool_name_prefix` | If `True` and `server_name` is provided, tool names will be<br>prefixed w/ server name (e.g., `"weather_search"` instead of `"search"`).<br>**TYPE:**`bool`**DEFAULT:**`False` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[BaseTool]` | List of LangChain [tools](https://docs.langchain.com/oss/python/langchain/tools).<br>Tool annotations are returned as part of the tool metadata object. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If neither session nor connection is provided. |

## ``prompts [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.prompts "Copy anchor link to this section for reference")

Prompts adapter for converting MCP prompts to LangChain [messages](https://docs.langchain.com/oss/python/langchain/messages).

This module provides functionality to convert MCP prompt messages into LangChain
message objects, handling both user and assistant message types.

| FUNCTION | DESCRIPTION |
| --- | --- |
| `load_mcp_prompt` | Load MCP prompt and convert to LangChain [messages](https://docs.langchain.com/oss/python/langchain/messages). |

### ``load\_mcp\_prompt`async`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.prompts.load_mcp_prompt "Copy anchor link to this section for reference")

```
load_mcp_prompt(
    session: ClientSession, name: str, *, arguments: dict[str, Any] | None = None
) -> list[HumanMessage | AIMessage]
```

Load MCP prompt and convert to LangChain [messages](https://docs.langchain.com/oss/python/langchain/messages).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `session` | The MCP client session.<br>**TYPE:**`ClientSession` |
| `name` | Name of the prompt to load.<br>**TYPE:**`str` |
| `arguments` | Optional arguments to pass to the prompt.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[HumanMessage | AIMessage]` | A list of LangChain [messages](https://docs.langchain.com/oss/python/langchain/messages)<br>converted from the MCP prompt. |

## ``resources [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.resources "Copy anchor link to this section for reference")

Resources adapter for converting MCP resources to LangChain [Blob objects](https://reference.langchain.com/python/langchain_core/documents/#langchain_core.documents.base.Blob "<code class=\"doc-symbol doc-symbol-heading doc-symbol-class\"></code>            <span class=\"doc doc-object-name doc-class-name\">Blob</span>").

This module provides functionality to convert MCP resources into LangChain Blob
objects, handling both text and binary resource content types.

| FUNCTION | DESCRIPTION |
| --- | --- |
| `load_mcp_resources` | Load MCP resources and convert them to LangChain [Blob objects](https://reference.langchain.com/python/langchain_core/documents/#langchain_core.documents.base.Blob "<code class=\"doc-symbol doc-symbol-heading doc-symbol-class\"></code>            <span class=\"doc doc-object-name doc-class-name\">Blob</span>"). |

### ``load\_mcp\_resources`async`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.resources.load_mcp_resources "Copy anchor link to this section for reference")

```
load_mcp_resources(
    session: ClientSession, *, uris: str | list[str] | None = None
) -> list[Blob]
```

Load MCP resources and convert them to LangChain [Blob objects](https://reference.langchain.com/python/langchain_core/documents/#langchain_core.documents.base.Blob "<code class=\"doc-symbol doc-symbol-heading doc-symbol-class\"></code>            <span class=\"doc doc-object-name doc-class-name\">Blob</span>").

| PARAMETER | DESCRIPTION |
| --- | --- |
| `session` | MCP client session.<br>**TYPE:**`ClientSession` |
| `uris` | List of URIs to load. If `None`, all resources will be loaded.<br>Note<br>Dynamic resources will NOT be loaded when `None` is specified,<br>as they require parameters and are ignored by the MCP SDK's<br>`session.list_resources()` method.<br>**TYPE:**`str | list[str] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Blob]` | A list of LangChain [Blob](https://reference.langchain.com/python/langchain_core/documents/#langchain_core.documents.base.Blob "<code class=\"doc-symbol doc-symbol-heading doc-symbol-class\"></code>            <span class=\"doc doc-object-name doc-class-name\">Blob</span>") objects. |

| RAISES | DESCRIPTION |
| --- | --- |
| `RuntimeError` | If an error occurs while fetching a resource. |

## ``interceptors [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.interceptors "Copy anchor link to this section for reference")

Interceptor interfaces and types for MCP client tool call lifecycle management.

This module provides an interceptor interface for wrapping and controlling
MCP tool call execution with a handler callback pattern.

In the future, we might add more interceptors for other parts of the
request / result lifecycle, for example to support elicitation.

### ``ToolCallInterceptor [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.interceptors.ToolCallInterceptor "Copy anchor link to this section for reference")

Bases: `Protocol`

Protocol for tool call interceptors using handler callback pattern.

Interceptors wrap tool execution to enable request/response modification,
retry logic, caching, rate limiting, and other cross-cutting concerns.
Multiple interceptors compose in "onion" pattern (first is outermost).

The handler can be called multiple times (retry), skipped (caching/short-circuit),
or wrapped with error handling. Each handler call is independent.

Similar to LangChain's middleware pattern but adapted for MCP remote tools.

| METHOD | DESCRIPTION |
| --- | --- |
| `__call__` | Intercept tool execution with control over handler invocation. |

#### ``\_\_call\_\_`async`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.interceptors.ToolCallInterceptor.__call__ "Copy anchor link to this section for reference")

```
__call__(
    request: MCPToolCallRequest,
    handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
) -> MCPToolCallResult
```

Intercept tool execution with control over handler invocation.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `request` | Tool call request containing name, args, headers, and context<br>(server\_name, runtime). Access context fields like request.server\_name.<br>**TYPE:**`MCPToolCallRequest` |
| `handler` | Async callable executing the tool. Can be called multiple<br>times, skipped, or wrapped for error handling.<br>**TYPE:**`Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `MCPToolCallResult` | Final MCPToolCallResult from tool execution or interceptor logic. |

## ``callbacks [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.callbacks "Copy anchor link to this section for reference")

Types for callbacks.

### ``CallbackContext`dataclass`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.callbacks.CallbackContext "Copy anchor link to this section for reference")

LangChain MCP client callback context.

### ``Callbacks`dataclass`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.callbacks.Callbacks "Copy anchor link to this section for reference")

Callbacks for the LangChain MCP client.

| METHOD | DESCRIPTION |
| --- | --- |
| `to_mcp_format` | Convert the LangChain MCP client callbacks to MCP SDK callbacks. |

#### ``to\_mcp\_format [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.callbacks.Callbacks.to_mcp_format "Copy anchor link to this section for reference")

```
to_mcp_format(*, context: CallbackContext) -> _MCPCallbacks
```

Convert the LangChain MCP client callbacks to MCP SDK callbacks.

Injects the LangChain CallbackContext as the last argument.

## ``sessions [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions "Copy anchor link to this section for reference")

Session management for different MCP transport types.

This module provides connection configurations and session management for various
MCP transport types including stdio, SSE, WebSocket, and streamable HTTP.

### ``Connection`module-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.Connection "Copy anchor link to this section for reference")

```
Connection = (
    StdioConnection | SSEConnection | StreamableHttpConnection | WebsocketConnection
)
```

### ``SSEConnection [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.SSEConnection "Copy anchor link to this section for reference")

Bases: `TypedDict`

Configuration for Server-Sent Events (SSE) transport connections to MCP.

#### ``url`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.SSEConnection.url "Copy anchor link to this section for reference")

```
url: str
```

The URL of the SSE endpoint to connect to.

#### ``headers`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.SSEConnection.headers "Copy anchor link to this section for reference")

```
headers: NotRequired[dict[str, Any] | None]
```

HTTP headers to send to the SSE endpoint.

#### ``timeout`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.SSEConnection.timeout "Copy anchor link to this section for reference")

```
timeout: NotRequired[float]
```

HTTP timeout.

Default is 5 seconds. If the server takes longer to respond,
you can increase this value.

#### ``sse\_read\_timeout`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.SSEConnection.sse_read_timeout "Copy anchor link to this section for reference")

```
sse_read_timeout: NotRequired[float]
```

SSE read timeout.

Default is 300 seconds (5 minutes). This is how long the client will
wait for a new event before disconnecting.

#### ``session\_kwargs`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.SSEConnection.session_kwargs "Copy anchor link to this section for reference")

```
session_kwargs: NotRequired[dict[str, Any] | None]
```

Additional keyword arguments to pass to the ClientSession.

#### ``httpx\_client\_factory`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.SSEConnection.httpx_client_factory "Copy anchor link to this section for reference")

```
httpx_client_factory: NotRequired[McpHttpClientFactory | None]
```

Custom factory for httpx.AsyncClient (optional).

#### ``auth`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.SSEConnection.auth "Copy anchor link to this section for reference")

```
auth: NotRequired[Auth]
```

Optional authentication for the HTTP client.

### ``StdioConnection [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StdioConnection "Copy anchor link to this section for reference")

Bases: `TypedDict`

Configuration for stdio transport connections to MCP servers.

#### ``command`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StdioConnection.command "Copy anchor link to this section for reference")

```
command: str
```

The executable to run to start the server.

#### ``args`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StdioConnection.args "Copy anchor link to this section for reference")

```
args: list[str]
```

Command line arguments to pass to the executable.

#### ``env`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StdioConnection.env "Copy anchor link to this section for reference")

```
env: NotRequired[dict[str, str] | None]
```

The environment to use when spawning the process.

If not specified or set to None, a subset of the default environment
variables from the current process will be used.

Please refer to the MCP SDK documentation for details on which
environment variables are included by default. The behavior
varies by operating system.

[https://github.com/modelcontextprotocol/python-sdk/blob/c47c767ff437ee88a19e6b9001e2472cb6f7d5ed/src/mcp/client/stdio/\_\_init\_\_.py#L51](https://github.com/modelcontextprotocol/python-sdk/blob/c47c767ff437ee88a19e6b9001e2472cb6f7d5ed/src/mcp/client/stdio/__init__.py#L51)

#### ``cwd`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StdioConnection.cwd "Copy anchor link to this section for reference")

```
cwd: NotRequired[str | Path | None]
```

The working directory to use when spawning the process.

#### ``encoding`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StdioConnection.encoding "Copy anchor link to this section for reference")

```
encoding: NotRequired[str]
```

The text encoding used when sending/receiving messages to the server.

Default is 'utf-8'.

#### ``encoding\_error\_handler`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StdioConnection.encoding_error_handler "Copy anchor link to this section for reference")

```
encoding_error_handler: NotRequired[EncodingErrorHandler]
```

The text encoding error handler.

See [https://docs.python.org/3/library/codecs.html#codec-base-classes](https://docs.python.org/3/library/codecs.html#codec-base-classes) for
explanations of possible values.

Default is 'strict', which raises an error on encoding/decoding errors.

#### ``session\_kwargs`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StdioConnection.session_kwargs "Copy anchor link to this section for reference")

```
session_kwargs: NotRequired[dict[str, Any] | None]
```

Additional keyword arguments to pass to the ClientSession.

### ``StreamableHttpConnection [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StreamableHttpConnection "Copy anchor link to this section for reference")

Bases: `TypedDict`

Connection configuration for Streamable HTTP transport.

#### ``url`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StreamableHttpConnection.url "Copy anchor link to this section for reference")

```
url: str
```

The URL of the endpoint to connect to.

#### ``headers`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StreamableHttpConnection.headers "Copy anchor link to this section for reference")

```
headers: NotRequired[dict[str, Any] | None]
```

HTTP headers to send to the endpoint.

#### ``timeout`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StreamableHttpConnection.timeout "Copy anchor link to this section for reference")

```
timeout: NotRequired[timedelta]
```

HTTP timeout.

#### ``sse\_read\_timeout`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StreamableHttpConnection.sse_read_timeout "Copy anchor link to this section for reference")

```
sse_read_timeout: NotRequired[timedelta]
```

How long (in seconds) the client will wait for a new event before disconnecting.
All other HTTP operations are controlled by `timeout`.

#### ``terminate\_on\_close`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StreamableHttpConnection.terminate_on_close "Copy anchor link to this section for reference")

```
terminate_on_close: NotRequired[bool]
```

Whether to terminate the session on close.

#### ``session\_kwargs`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StreamableHttpConnection.session_kwargs "Copy anchor link to this section for reference")

```
session_kwargs: NotRequired[dict[str, Any] | None]
```

Additional keyword arguments to pass to the ClientSession.

#### ``httpx\_client\_factory`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StreamableHttpConnection.httpx_client_factory "Copy anchor link to this section for reference")

```
httpx_client_factory: NotRequired[McpHttpClientFactory | None]
```

Custom factory for httpx.AsyncClient (optional).

#### ``auth`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.StreamableHttpConnection.auth "Copy anchor link to this section for reference")

```
auth: NotRequired[Auth]
```

Optional authentication for the HTTP client.

### ``WebsocketConnection [¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.WebsocketConnection "Copy anchor link to this section for reference")

Bases: `TypedDict`

Configuration for WebSocket transport connections to MCP servers.

#### ``url`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.WebsocketConnection.url "Copy anchor link to this section for reference")

```
url: str
```

The URL of the Websocket endpoint to connect to.

#### ``session\_kwargs`instance-attribute`[¶](https://reference.langchain.com/python/langchain_mcp_adapters/\#langchain_mcp_adapters.sessions.WebsocketConnection.session_kwargs "Copy anchor link to this section for reference")

```
session_kwargs: NotRequired[dict[str, Any] | None]
```

Additional keyword arguments to pass to the ClientSession

Back to top