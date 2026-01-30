[Skip to content](https://reference.langchain.com/python/integrations/langchain_ollama/#langchain-ollama)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/integrations/langchain_ollama.md "Edit this page")

# `langchain-ollama` [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain-ollama "Copy anchor link to this section for reference")

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-ollama?label=%20)](https://pypi.org/project/langchain-ollama/#history)[![PyPI - License](https://img.shields.io/pypi/l/langchain-ollama)](https://opensource.org/licenses/MIT)[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-ollama)](https://pypistats.org/packages/langchain-ollama)

Reference docs

This page contains **reference documentation** for Ollama. See [the docs](https://docs.langchain.com/oss/python/integrations/providers/ollama) for conceptual guides, tutorials, and examples on using Ollama modules.

## ``langchain\_ollama [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama "Copy anchor link to this section for reference")

This is the langchain\_ollama package.

Provides infrastructure for interacting with the [Ollama](https://ollama.com/)
service.

Note

**Newly added in 0.3.4:**`validate_model_on_init` param on all models.
This parameter allows you to validate the model exists in Ollama locally on
initialization. If set to `True`, it will raise an error if the model does not
exist locally. This is useful for ensuring that the model is available before
attempting to use it, especially in environments where models may not be
pre-downloaded.

### ``ChatOllama [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama "Copy anchor link to this section for reference")

Bases: `BaseChatModel`

Ollama chat model integration.

Setup

Install `langchain-ollama` and download any models you want to use from ollama.

```
ollama pull gpt-oss:20b
pip install -U langchain-ollama
```

Key init args — completion params:
model: str
Name of Ollama model to use.
reasoning: bool \| None
Controls the reasoning/thinking mode for
[supported models](https://ollama.com/search?c=thinking).

```
    - `True`: Enables reasoning mode. The model's reasoning process will be
        captured and returned separately in the `additional_kwargs` of the
        response message, under `reasoning_content`. The main response
        content will not include the reasoning tags.
    - `False`: Disables reasoning mode. The model will not perform any reasoning,
        and the response will not include any reasoning content.
    - `None` (Default): The model will use its default reasoning behavior. Note
        however, if the model's default behavior *is* to perform reasoning, think tags
        (`<think>` and `</think>`) will be present within the main response content
        unless you set `reasoning` to `True`.
temperature: float
    Sampling temperature. Ranges from `0.0` to `1.0`.
num_predict: int | None
    Max number of tokens to generate.
```

See full list of supported init args and their descriptions in the params section.

Instantiate

```
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="gpt-oss:20b",
    validate_model_on_init=True,
    temperature=0.8,
    num_predict=256,
    # other params ...
)
```

Invoke

```
messages = [\
    ("system", "You are a helpful translator. Translate the user sentence to French."),\
    ("human", "I love programming."),\
]
model.invoke(messages)
```

```
AIMessage(content='J'adore le programmation. (Note: "programming" can also refer to the act of writing code, so if you meant that, I could translate it as "J'adore programmer". But since you didn\'t specify, I assumed you were talking about the activity itself, which is what "le programmation" usually refers to.)', response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:37:50.182604Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 3576619666, 'load_duration': 788524916, 'prompt_eval_count': 32, 'prompt_eval_duration': 128125000, 'eval_count': 71, 'eval_duration': 2656556000}, id='run-ba48f958-6402-41a5-b461-5e250a4ebd36-0')
```

Stream

```
for chunk in model.stream("Return the words Hello World!"):
    print(chunk.text, end="")
```

```
content='Hello' id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
content=' World' id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
content='!' id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
content='' response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:39:42.274449Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 411875125, 'load_duration': 1898166, 'prompt_eval_count': 14, 'prompt_eval_duration': 297320000, 'eval_count': 4, 'eval_duration': 111099000} id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
```

```
stream = model.stream(messages)
full = next(stream)
for chunk in stream:
    full += chunk
full
```

```
AIMessageChunk(
    content='Je adore le programmation.(Note: "programmation" is the formal way to say "programming" in French, but informally, people might use the phrase "le développement logiciel" or simply "le code")',
    response_metadata={
        "model": "llama3",
        "created_at": "2024-07-04T03:38:54.933154Z",
        "message": {"role": "assistant", "content": ""},
        "done_reason": "stop",
        "done": True,
        "total_duration": 1977300042,
        "load_duration": 1345709,
        "prompt_eval_duration": 159343000,
        "eval_count": 47,
        "eval_duration": 1815123000,
    },
    id="run-3c81a3ed-3e79-4dd3-a796-04064d804890",
)
```

Async

```
await model.ainvoke("Hello how are you!")
```

```
AIMessage(
    content="Hi there! I'm just an AI, so I don't have feelings or emotions like humans do. But I'm functioning properly and ready to help with any questions or tasks you may have! How can I assist you today?",
    response_metadata={
        "model": "llama3",
        "created_at": "2024-07-04T03:52:08.165478Z",
        "message": {"role": "assistant", "content": ""},
        "done_reason": "stop",
        "done": True,
        "total_duration": 2138492875,
        "load_duration": 1364000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 297081000,
        "eval_count": 47,
        "eval_duration": 1838524000,
    },
    id="run-29c510ae-49a4-4cdd-8f23-b972bfab1c49-0",
)
```

```
async for chunk in model.astream("Say hello world!"):
    print(chunk.content)
```

```
HEL
LO
WORLD
!
```

```
messages = [("human", "Say hello world!"), ("human", "Say goodbye world!")]
await model.abatch(messages)
```

```
[\
    AIMessage(\
        content="HELLO, WORLD!",\
        response_metadata={\
            "model": "llama3",\
            "created_at": "2024-07-04T03:55:07.315396Z",\
            "message": {"role": "assistant", "content": ""},\
            "done_reason": "stop",\
            "done": True,\
            "total_duration": 1696745458,\
            "load_duration": 1505000,\
            "prompt_eval_count": 8,\
            "prompt_eval_duration": 111627000,\
            "eval_count": 6,\
            "eval_duration": 185181000,\
        },\
        id="run-da6c7562-e25a-4a44-987a-2c83cd8c2686-0",\
    ),\
    AIMessage(\
        content="It's been a blast chatting with you! Say goodbye to the world for me, and don't forget to come back and visit us again soon!",\
        response_metadata={\
            "model": "llama3",\
            "created_at": "2024-07-04T03:55:07.018076Z",\
            "message": {"role": "assistant", "content": ""},\
            "done_reason": "stop",\
            "done": True,\
            "total_duration": 1399391083,\
            "load_duration": 1187417,\
            "prompt_eval_count": 20,\
            "prompt_eval_duration": 230349000,\
            "eval_count": 31,\
            "eval_duration": 1166047000,\
        },\
        id="run-96cad530-6f3e-4cf9-86b4-e0f8abba4cdb-0",\
    ),\
]
```

JSON mode

```
json_model = ChatOllama(format="json")
json_model.invoke(
    "Return a query for the weather in a random location and time of day with two keys: location and time_of_day. "
    "Respond using JSON only."
).content
```

```
'{"location": "Pune, India", "time_of_day": "morning"}'
```

Tool Calling

```
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

class Multiply(BaseModel):
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

ans = await chat.invoke("What is 45*67")
ans.tool_calls
```

```
[\
    {\
        "name": "Multiply",\
        "args": {"a": 45, "b": 67},\
        "id": "420c3f3b-df10-4188-945f-eb3abdb40622",\
        "type": "tool_call",\
    }\
]
```

Thinking / Reasoning:
You can enable reasoning mode for models that support it by setting
the `reasoning` parameter to `True` in either the constructor or
the `invoke`/`stream` methods. This will enable the model to think
through the problem and return the reasoning process separately in the
`additional_kwargs` of the response message, under `reasoning_content`.

````
If `reasoning` is set to `None`, the model will use its default reasoning
behavior, and any reasoning content will *not* be captured under the
`reasoning_content` key, but will be present within the main response content
as think tags (`<think>` and `</think>`).

!!! note
    This feature is only available for [models that support reasoning](https://ollama.com/search?c=thinking).

```python
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="deepseek-r1:8b",
    validate_model_on_init=True,
    reasoning=True,
)

model.invoke("how many r in the word strawberry?")

# or, on an invocation basis:

model.invoke("how many r in the word strawberry?", reasoning=True)
# or model.stream("how many r in the word strawberry?", reasoning=True)

# If not provided, the invocation will default to the ChatOllama reasoning
# param provided (None by default).
```

```python
AIMessage(content='The word "strawberry" contains **three \'r\' letters**. Here\'s a breakdown for clarity:\n\n- The spelling of "strawberry" has two parts ... be 3.\n\nTo be thorough, let\'s confirm with an online source or common knowledge.\n\nI can recall that "strawberry" has: s-t-r-a-w-b-e-r-r-y — yes, three r\'s.\n\nPerhaps it\'s misspelled by some, but standard is correct.\n\nSo I think the response should be 3.\n'}, response_metadata={'model': 'deepseek-r1:8b', 'created_at': '2025-07-08T19:33:55.891269Z', 'done': True, 'done_reason': 'stop', 'total_duration': 98232561292, 'load_duration': 28036792, 'prompt_eval_count': 10, 'prompt_eval_duration': 40171834, 'eval_count': 3615, 'eval_duration': 98163832416, 'model_name': 'deepseek-r1:8b'}, id='run--18f8269f-6a35-4a7c-826d-b89d52c753b3-0', usage_metadata={'input_tokens': 10, 'output_tokens': 3615, 'total_tokens': 3625})

```
````

| METHOD | DESCRIPTION |
| --- | --- |
| `get_name` | Get the name of the `Runnable`. |
| `get_input_schema` | Get a Pydantic model that can be used to validate input to the `Runnable`. |
| `get_input_jsonschema` | Get a JSON schema that represents the input to the `Runnable`. |
| `get_output_schema` | Get a Pydantic model that can be used to validate output to the `Runnable`. |
| `get_output_jsonschema` | Get a JSON schema that represents the output of the `Runnable`. |
| `config_schema` | The type of config this `Runnable` accepts specified as a Pydantic model. |
| `get_config_jsonschema` | Get a JSON schema that represents the config of the `Runnable`. |
| `get_graph` | Return a graph representation of this `Runnable`. |
| `get_prompts` | Return a list of prompts used by this `Runnable`. |
| `__or__` | Runnable "or" operator. |
| `__ror__` | Runnable "reverse-or" operator. |
| `pipe` | Pipe `Runnable` objects. |
| `pick` | Pick keys from the output `dict` of this `Runnable`. |
| `assign` | Assigns new fields to the `dict` output of this `Runnable`. |
| `invoke` | Transform a single input into an output. |
| `ainvoke` | Transform a single input into an output. |
| `batch` | Default implementation runs invoke in parallel using a thread pool executor. |
| `batch_as_completed` | Run `invoke` in parallel on a list of inputs. |
| `abatch` | Default implementation runs `ainvoke` in parallel using `asyncio.gather`. |
| `abatch_as_completed` | Run `ainvoke` in parallel on a list of inputs. |
| `stream` | Default implementation of `stream`, which calls `invoke`. |
| `astream` | Default implementation of `astream`, which calls `ainvoke`. |
| `astream_log` | Stream all output from a `Runnable`, as reported to the callback system. |
| `astream_events` | Generate a stream of events. |
| `transform` | Transform inputs to outputs. |
| `atransform` | Transform inputs to outputs. |
| `bind` | Bind arguments to a `Runnable`, returning a new `Runnable`. |
| `with_config` | Bind config to a `Runnable`, returning a new `Runnable`. |
| `with_listeners` | Bind lifecycle listeners to a `Runnable`, returning a new `Runnable`. |
| `with_alisteners` | Bind async lifecycle listeners to a `Runnable`. |
| `with_types` | Bind input and output types to a `Runnable`, returning a new `Runnable`. |
| `with_retry` | Create a new `Runnable` that retries the original `Runnable` on exceptions. |
| `map` | Return a new `Runnable` that maps a list of inputs to a list of outputs. |
| `with_fallbacks` | Add fallbacks to a `Runnable`, returning a new `Runnable`. |
| `as_tool` | Create a `BaseTool` from a `Runnable`. |
| `__init__` |  |
| `is_lc_serializable` | Is this class serializable? |
| `get_lc_namespace` | Get the namespace of the LangChain object. |
| `lc_id` | Return a unique identifier for this class for serialization purposes. |
| `to_json` | Serialize the `Runnable` to JSON. |
| `to_json_not_implemented` | Serialize a "not implemented" object. |
| `configurable_fields` | Configure particular `Runnable` fields at runtime. |
| `configurable_alternatives` | Configure alternatives for `Runnable` objects that can be set at runtime. |
| `set_verbose` | If verbose is `None`, set it. |
| `generate_prompt` | Pass a sequence of prompts to the model and return model generations. |
| `agenerate_prompt` | Asynchronously pass a sequence of prompts and return model generations. |
| `get_token_ids` | Return the ordered IDs of the tokens in a text. |
| `get_num_tokens` | Get the number of tokens present in the text. |
| `get_num_tokens_from_messages` | Get the number of tokens in the messages. |
| `generate` | Pass a sequence of prompts to the model and return model generations. |
| `agenerate` | Asynchronously pass a sequence of prompts to a model and return generations. |
| `dict` | Return a dictionary of the LLM. |
| `bind_tools` | Bind tool-like objects to this chat model. |
| `with_structured_output` | Model wrapper that returns outputs formatted to match the given schema. |

#### ``name`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.name "Copy anchor link to this section for reference")

```
name: str | None = None
```

The name of the `Runnable`. Used for debugging and tracing.

#### ``InputType`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.InputType "Copy anchor link to this section for reference")

```
InputType: TypeAlias
```

Get the input type for this `Runnable`.

#### ``OutputType`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.OutputType "Copy anchor link to this section for reference")

```
OutputType: Any
```

Get the output type for this `Runnable`.

#### ``input\_schema`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.input_schema "Copy anchor link to this section for reference")

```
input_schema: type[BaseModel]
```

The type of input this `Runnable` accepts specified as a Pydantic model.

#### ``output\_schema`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.output_schema "Copy anchor link to this section for reference")

```
output_schema: type[BaseModel]
```

Output schema.

The type of output this `Runnable` produces specified as a Pydantic model.

#### ``config\_specs`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.config_specs "Copy anchor link to this section for reference")

```
config_specs: list[ConfigurableFieldSpec]
```

List configurable fields for this `Runnable`.

#### ``lc\_secrets`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.lc_secrets "Copy anchor link to this section for reference")

```
lc_secrets: dict[str, str]
```

A map of constructor argument names to secret ids.

For example, `{"openai_api_key": "OPENAI_API_KEY"}`

#### ``lc\_attributes`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.lc_attributes "Copy anchor link to this section for reference")

```
lc_attributes: dict
```

List of attribute names that should be included in the serialized kwargs.

These attributes must be accepted by the constructor.

Default is an empty dictionary.

#### ``cache`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.cache "Copy anchor link to this section for reference")

```
cache: BaseCache | bool | None = Field(default=None, exclude=True)
```

Whether to cache the response.

- If `True`, will use the global cache.
- If `False`, will not use a cache
- If `None`, will use the global cache if it's set, otherwise no cache.
- If instance of `BaseCache`, will use the provided cache.

Caching is not currently supported for streaming methods of models.

#### ``verbose`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.verbose "Copy anchor link to this section for reference")

```
verbose: bool = Field(default_factory=_get_verbosity, exclude=True, repr=False)
```

Whether to print out response text.

#### ``callbacks`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.callbacks "Copy anchor link to this section for reference")

```
callbacks: Callbacks = Field(default=None, exclude=True)
```

Callbacks to add to the run trace.

#### ``tags`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.tags "Copy anchor link to this section for reference")

```
tags: list[str] | None = Field(default=None, exclude=True)
```

Tags to add to the run trace.

#### ``metadata`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.metadata "Copy anchor link to this section for reference")

```
metadata: dict[str, Any] | None = Field(default=None, exclude=True)
```

Metadata to add to the run trace.

#### ``custom\_get\_token\_ids`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.custom_get_token_ids "Copy anchor link to this section for reference")

```
custom_get_token_ids: Callable[[str], list[int]] | None = Field(
    default=None, exclude=True
)
```

Optional encoder to use for counting tokens.

#### ``rate\_limiter`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.rate_limiter "Copy anchor link to this section for reference")

```
rate_limiter: BaseRateLimiter | None = Field(default=None, exclude=True)
```

An optional rate limiter to use for limiting the number of requests.

#### ``disable\_streaming`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.disable_streaming "Copy anchor link to this section for reference")

```
disable_streaming: bool | Literal['tool_calling'] = False
```

Whether to disable streaming for this model.

If streaming is bypassed, then `stream`/`astream`/`astream_events` will
defer to `invoke`/`ainvoke`.

- If `True`, will always bypass streaming case.
- If `'tool_calling'`, will bypass streaming case only when the model is called
with a `tools` keyword argument. In other words, LangChain will automatically
switch to non-streaming behavior (`invoke`) only when the tools argument is
provided. This offers the best of both worlds.
- If `False` (Default), will always use streaming case if available.

The main reason for this flag is that code might be written using `stream` and
a user may want to swap out a given model for another model whose the implementation
does not properly support streaming.

#### ``output\_version`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.output_version "Copy anchor link to this section for reference")

```
output_version: str | None = Field(
    default_factory=from_env("LC_OUTPUT_VERSION", default=None)
)
```

Version of `AIMessage` output format to store in message content.

`AIMessage.content_blocks` will lazily parse the contents of `content` into a
standard format. This flag can be used to additionally store the standard format
in message content, e.g., for serialization purposes.

Supported values:

- `'v0'`: provider-specific format in content (can lazily-parse with
`content_blocks`)
- `'v1'`: standardized format in content (consistent with `content_blocks`)

Partner packages (e.g.,
[`langchain-openai`](https://pypi.org/project/langchain-openai)) can also use this
field to roll out new content formats in a backward-compatible way.

Added in `langchain-core` 1.0.0

#### ``profile`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.profile "Copy anchor link to this section for reference")

```
profile: ModelProfile | None = Field(default=None, exclude=True)
```

Profile detailing model capabilities.

Beta feature

This is a beta feature. The format of model profiles is subject to change.

If not specified, automatically loaded from the provider package on initialization
if data is available.

Example profile data includes context window sizes, supported modalities, or support
for tool calling, structured output, and other features.

Added in `langchain-core` 1.1.0

#### ``model`instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.model "Copy anchor link to this section for reference")

```
model: str
```

Model name to use.

#### ``reasoning`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.reasoning "Copy anchor link to this section for reference")

```
reasoning: bool | str | None = None
```

Controls the reasoning/thinking mode for [supported models](https://ollama.com/search?c=thinking).

- `True`: Enables reasoning mode. The model's reasoning process will be
captured and returned separately in the `additional_kwargs` of the
response message, under `reasoning_content`. The main response
content will not include the reasoning tags.
- `False`: Disables reasoning mode. The model will not perform any reasoning,
and the response will not include any reasoning content.
- `None` (Default): The model will use its default reasoning behavior. Note
however, if the model's default behavior _is_ to perform reasoning, think tags
(`<think>` and `</think>`) will be present within the main response content
unless you set `reasoning` to `True`.
- `str`: e.g. `'low'`, `'medium'`, `'high'`. Enables reasoning with a custom
intensity level. Currently, this is only supported `gpt-oss`. See the
[Ollama docs](https://github.com/ollama/ollama-python/blob/da79e987f0ac0a4986bf396f043b36ef840370bc/ollama/_types.py#L210)
for more information.

#### ``validate\_model\_on\_init`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.validate_model_on_init "Copy anchor link to this section for reference")

```
validate_model_on_init: bool = False
```

Whether to validate the model exists in Ollama locally on initialization.

Added in `langchain-ollama` 0.3.4

#### ``mirostat`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.mirostat "Copy anchor link to this section for reference")

```
mirostat: int | None = None
```

Enable Mirostat sampling for controlling perplexity.

(Default: `0`, `0` = disabled, `1` = Mirostat, `2` = Mirostat 2.0)

#### ``mirostat\_eta`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.mirostat_eta "Copy anchor link to this section for reference")

```
mirostat_eta: float | None = None
```

Influences how quickly the algorithm responds to feedback from generated text.

A lower learning rate will result in slower adjustments, while a higher learning
rate will make the algorithm more responsive.

(Default: `0.1`)

#### ``mirostat\_tau`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.mirostat_tau "Copy anchor link to this section for reference")

```
mirostat_tau: float | None = None
```

Controls the balance between coherence and diversity of the output.

A lower value will result in more focused and coherent text.

(Default: `5.0`)

#### ``num\_ctx`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.num_ctx "Copy anchor link to this section for reference")

```
num_ctx: int | None = None
```

Sets the size of the context window used to generate the next token.

(Default: `2048`)

#### ``num\_gpu`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.num_gpu "Copy anchor link to this section for reference")

```
num_gpu: int | None = None
```

The number of GPUs to use.

On macOS it defaults to `1` to enable metal support, `0` to disable.

#### ``num\_thread`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.num_thread "Copy anchor link to this section for reference")

```
num_thread: int | None = None
```

Sets the number of threads to use during computation.

By default, Ollama will detect this for optimal performance. It is recommended to
set this value to the number of physical CPU cores your system has (as opposed to
the logical number of cores).

#### ``num\_predict`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.num_predict "Copy anchor link to this section for reference")

```
num_predict: int | None = None
```

Maximum number of tokens to predict when generating text.

(Default: `128`, `-1` = infinite generation, `-2` = fill context)

#### ``repeat\_last\_n`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.repeat_last_n "Copy anchor link to this section for reference")

```
repeat_last_n: int | None = None
```

Sets how far back for the model to look back to prevent repetition.

(Default: `64`, `0` = disabled, `-1` = `num_ctx`)

#### ``repeat\_penalty`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.repeat_penalty "Copy anchor link to this section for reference")

```
repeat_penalty: float | None = None
```

Sets how strongly to penalize repetitions.

A higher value (e.g., `1.5`) will penalize repetitions more strongly, while a
lower value (e.g., `0.9`) will be more lenient. (Default: `1.1`)

#### ``temperature`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.temperature "Copy anchor link to this section for reference")

```
temperature: float | None = None
```

The temperature of the model.

Increasing the temperature will make the model answer more creatively.

(Default: `0.8`)

#### ``seed`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.seed "Copy anchor link to this section for reference")

```
seed: int | None = None
```

Sets the random number seed to use for generation.

Setting this to a specific number will make the model generate the same text for the
same prompt.

#### ``stop`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.stop "Copy anchor link to this section for reference")

```
stop: list[str] | None = None
```

Sets the stop tokens to use.

#### ``tfs\_z`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.tfs_z "Copy anchor link to this section for reference")

```
tfs_z: float | None = None
```

Tail free sampling.

Used to reduce the impact of less probable tokens from the output.

A higher value (e.g., `2.0`) will reduce the impact more, while a value of `1.0`
disables this setting.

(Default: `1`)

#### ``top\_k`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.top_k "Copy anchor link to this section for reference")

```
top_k: int | None = None
```

Reduces the probability of generating nonsense.

A higher value (e.g. `100`) will give more diverse answers, while a lower value
(e.g. `10`) will be more conservative.

(Default: `40`)

#### ``top\_p`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.top_p "Copy anchor link to this section for reference")

```
top_p: float | None = None
```

Works together with top-k.

A higher value (e.g., `0.95`) will lead to more diverse text, while a lower value
(e.g., `0.5`) will generate more focused and conservative text.

(Default: `0.9`)

#### ``format`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.format "Copy anchor link to this section for reference")

```
format: Literal['', 'json'] | JsonSchemaValue | None = None
```

Specify the format of the output (options: `'json'`, JSON schema).

#### ``keep\_alive`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.keep_alive "Copy anchor link to this section for reference")

```
keep_alive: int | str | None = None
```

How long the model will stay loaded into memory.

#### ``base\_url`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.base_url "Copy anchor link to this section for reference")

```
base_url: str | None = None
```

Base url the model is hosted under.

If none, defaults to the Ollama client default.

Supports `userinfo` auth in the format `http://username:password@localhost:11434`.
Useful if your Ollama server is behind a proxy.

Warning

`userinfo` is not secure and should only be used for local testing or
in secure environments. Avoid using it in production or over unsecured
networks.

Note

If using `userinfo`, ensure that the Ollama server is configured to
accept and validate these credentials.

Note

`userinfo` headers are passed to both sync and async clients.

#### ``client\_kwargs`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.client_kwargs "Copy anchor link to this section for reference")

```
client_kwargs: dict | None = {}
```

Additional kwargs to pass to the httpx clients. Pass headers in here.

These arguments are passed to both synchronous and async clients.

Use `sync_client_kwargs` and `async_client_kwargs` to pass different arguments
to synchronous and asynchronous clients.

#### ``async\_client\_kwargs`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.async_client_kwargs "Copy anchor link to this section for reference")

```
async_client_kwargs: dict | None = {}
```

Additional kwargs to merge with `client_kwargs` before passing to httpx client.

These are clients unique to the async client; for shared args use `client_kwargs`.

For a full list of the params, see the [httpx documentation](https://www.python-httpx.org/api/#asyncclient).

#### ``sync\_client\_kwargs`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.sync_client_kwargs "Copy anchor link to this section for reference")

```
sync_client_kwargs: dict | None = {}
```

Additional kwargs to merge with `client_kwargs` before passing to httpx client.

These are clients unique to the sync client; for shared args use `client_kwargs`.

For a full list of the params, see the [httpx documentation](https://www.python-httpx.org/api/#client).

#### ``get\_name [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.get_name "Copy anchor link to this section for reference")

```
get_name(suffix: str | None = None, *, name: str | None = None) -> str
```

Get the name of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `suffix` | An optional suffix to append to the name.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `name` | An optional name to use instead of the `Runnable`'s name.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `str` | The name of the `Runnable`. |

#### ``get\_input\_schema [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.get_input_schema "Copy anchor link to this section for reference")

```
get_input_schema(config: RunnableConfig | None = None) -> type[BaseModel]
```

Get a Pydantic model that can be used to validate input to the `Runnable`.

`Runnable` objects that leverage the `configurable_fields` and
`configurable_alternatives` methods will have a dynamic input schema that
depends on which configuration the `Runnable` is invoked with.

This method allows to get an input schema for a specific configuration.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | A config to use when generating the schema.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `type[BaseModel]` | A Pydantic model that can be used to validate input. |

#### ``get\_input\_jsonschema [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.get_input_jsonschema "Copy anchor link to this section for reference")

```
get_input_jsonschema(config: RunnableConfig | None = None) -> dict[str, Any]
```

Get a JSON schema that represents the input to the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | A config to use when generating the schema.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, Any]` | A JSON schema that represents the input to the `Runnable`. |

Example

```
from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

runnable = RunnableLambda(add_one)

print(runnable.get_input_jsonschema())
```

Added in `langchain-core` 0.3.0

#### ``get\_output\_schema [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.get_output_schema "Copy anchor link to this section for reference")

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

#### ``get\_output\_jsonschema [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.get_output_jsonschema "Copy anchor link to this section for reference")

```
get_output_jsonschema(config: RunnableConfig | None = None) -> dict[str, Any]
```

Get a JSON schema that represents the output of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | A config to use when generating the schema.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, Any]` | A JSON schema that represents the output of the `Runnable`. |

Example

```
from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

runnable = RunnableLambda(add_one)

print(runnable.get_output_jsonschema())
```

Added in `langchain-core` 0.3.0

#### ``config\_schema [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.config_schema "Copy anchor link to this section for reference")

```
config_schema(*, include: Sequence[str] | None = None) -> type[BaseModel]
```

The type of config this `Runnable` accepts specified as a Pydantic model.

To mark a field as configurable, see the `configurable_fields`
and `configurable_alternatives` methods.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `include` | A list of fields to include in the config schema.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `type[BaseModel]` | A Pydantic model that can be used to validate config. |

#### ``get\_config\_jsonschema [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.get_config_jsonschema "Copy anchor link to this section for reference")

```
get_config_jsonschema(*, include: Sequence[str] | None = None) -> dict[str, Any]
```

Get a JSON schema that represents the config of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `include` | A list of fields to include in the config schema.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, Any]` | A JSON schema that represents the config of the `Runnable`. |

Added in `langchain-core` 0.3.0

#### ``get\_graph [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.get_graph "Copy anchor link to this section for reference")

```
get_graph(config: RunnableConfig | None = None) -> Graph
```

Return a graph representation of this `Runnable`.

#### ``get\_prompts [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.get_prompts "Copy anchor link to this section for reference")

```
get_prompts(config: RunnableConfig | None = None) -> list[BasePromptTemplate]
```

Return a list of prompts used by this `Runnable`.

#### ``\_\_or\_\_ [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.__or__ "Copy anchor link to this section for reference")

```
__or__(
    other: Runnable[Any, Other]
    | Callable[[Iterator[Any]], Iterator[Other]]
    | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
    | Callable[[Any], Other]
    | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
) -> RunnableSerializable[Input, Other]
```

Runnable "or" operator.

Compose this `Runnable` with another object to create a
`RunnableSequence`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `other` | Another `Runnable` or a `Runnable`-like object.<br>**TYPE:**`Runnable[Any, Other] | Callable[[Iterator[Any]], Iterator[Other]] | Callable[[AsyncIterator[Any]], AsyncIterator[Other]] | Callable[[Any], Other] | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Input, Other]` | A new `Runnable`. |

#### ``\_\_ror\_\_ [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.__ror__ "Copy anchor link to this section for reference")

```
__ror__(
    other: Runnable[Other, Any]
    | Callable[[Iterator[Other]], Iterator[Any]]
    | Callable[[AsyncIterator[Other]], AsyncIterator[Any]]
    | Callable[[Other], Any]
    | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any] | Any],
) -> RunnableSerializable[Other, Output]
```

Runnable "reverse-or" operator.

Compose this `Runnable` with another object to create a
`RunnableSequence`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `other` | Another `Runnable` or a `Runnable`-like object.<br>**TYPE:**`Runnable[Other, Any] | Callable[[Iterator[Other]], Iterator[Any]] | Callable[[AsyncIterator[Other]], AsyncIterator[Any]] | Callable[[Other], Any] | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any] | Any]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Other, Output]` | A new `Runnable`. |

#### ``pipe [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.pipe "Copy anchor link to this section for reference")

```
pipe(
    *others: Runnable[Any, Other] | Callable[[Any], Other], name: str | None = None
) -> RunnableSerializable[Input, Other]
```

Pipe `Runnable` objects.

Compose this `Runnable` with `Runnable`-like objects to make a
`RunnableSequence`.

Equivalent to `RunnableSequence(self, *others)` or `self | others[0] | ...`

Example

```
from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

def mul_two(x: int) -> int:
    return x * 2

runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)
sequence = runnable_1.pipe(runnable_2)
# Or equivalently:
# sequence = runnable_1 | runnable_2
# sequence = RunnableSequence(first=runnable_1, last=runnable_2)
sequence.invoke(1)
await sequence.ainvoke(1)
# -> 4

sequence.batch([1, 2, 3])
await sequence.abatch([1, 2, 3])
# -> [4, 6, 8]
```

| PARAMETER | DESCRIPTION |
| --- | --- |
| `*others` | Other `Runnable` or `Runnable`-like objects to compose<br>**TYPE:**`Runnable[Any, Other] | Callable[[Any], Other]`**DEFAULT:**`()` |
| `name` | An optional name for the resulting `RunnableSequence`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Input, Other]` | A new `Runnable`. |

#### ``pick [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.pick "Copy anchor link to this section for reference")

```
pick(keys: str | list[str]) -> RunnableSerializable[Any, Any]
```

Pick keys from the output `dict` of this `Runnable`.

Pick a single key

```
import json

from langchain_core.runnables import RunnableLambda, RunnableMap

as_str = RunnableLambda(str)
as_json = RunnableLambda(json.loads)
chain = RunnableMap(str=as_str, json=as_json)

chain.invoke("[1, 2, 3]")
# -> {"str": "[1, 2, 3]", "json": [1, 2, 3]}

json_only_chain = chain.pick("json")
json_only_chain.invoke("[1, 2, 3]")
# -> [1, 2, 3]
```

Pick a list of keys

```
from typing import Any

import json

from langchain_core.runnables import RunnableLambda, RunnableMap

as_str = RunnableLambda(str)
as_json = RunnableLambda(json.loads)

def as_bytes(x: Any) -> bytes:
    return bytes(x, "utf-8")

chain = RunnableMap(
    str=as_str, json=as_json, bytes=RunnableLambda(as_bytes)
)

chain.invoke("[1, 2, 3]")
# -> {"str": "[1, 2, 3]", "json": [1, 2, 3], "bytes": b"[1, 2, 3]"}

json_and_bytes_chain = chain.pick(["json", "bytes"])
json_and_bytes_chain.invoke("[1, 2, 3]")
# -> {"json": [1, 2, 3], "bytes": b"[1, 2, 3]"}
```

| PARAMETER | DESCRIPTION |
| --- | --- |
| `keys` | A key or list of keys to pick from the output dict.<br>**TYPE:**`str | list[str]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Any, Any]` | a new `Runnable`. |

#### ``assign [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.assign "Copy anchor link to this section for reference")

```
assign(
    **kwargs: Runnable[dict[str, Any], Any]
    | Callable[[dict[str, Any]], Any]
    | Mapping[str, Runnable[dict[str, Any], Any] | Callable[[dict[str, Any]], Any]],
) -> RunnableSerializable[Any, Any]
```

Assigns new fields to the `dict` output of this `Runnable`.

```
from langchain_core.language_models.fake import FakeStreamingListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import Runnable
from operator import itemgetter

prompt = (
    SystemMessagePromptTemplate.from_template("You are a nice assistant.")
    + "{question}"
)
model = FakeStreamingListLLM(responses=["foo-lish"])

chain: Runnable = prompt | model | {"str": StrOutputParser()}

chain_with_assign = chain.assign(hello=itemgetter("str") | model)

print(chain_with_assign.input_schema.model_json_schema())
# {'title': 'PromptInput', 'type': 'object', 'properties':
{'question': {'title': 'Question', 'type': 'string'}}}
print(chain_with_assign.output_schema.model_json_schema())
# {'title': 'RunnableSequenceOutput', 'type': 'object', 'properties':
{'str': {'title': 'Str',
'type': 'string'}, 'hello': {'title': 'Hello', 'type': 'string'}}}
```

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | A mapping of keys to `Runnable` or `Runnable`-like objects<br>that will be invoked with the entire output dict of this `Runnable`.<br>**TYPE:**`Runnable[dict[str, Any], Any] | Callable[[dict[str, Any]], Any] | Mapping[str, Runnable[dict[str, Any], Any] | Callable[[dict[str, Any]], Any]]`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Any, Any]` | A new `Runnable`. |

#### ``invoke [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.invoke "Copy anchor link to this section for reference")

```
invoke(
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AIMessage
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

#### ``ainvoke`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.ainvoke "Copy anchor link to this section for reference")

```
ainvoke(
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AIMessage
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

#### ``batch [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.batch "Copy anchor link to this section for reference")

```
batch(
    inputs: list[Input],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> list[Output]
```

Default implementation runs invoke in parallel using a thread pool executor.

The default implementation of batch works well for IO bound runnables.

Subclasses must override this method if they can batch more efficiently;
e.g., if the underlying `Runnable` uses an API which supports a batch mode.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `inputs` | A list of inputs to the `Runnable`.<br>**TYPE:**`list[Input]` |
| `config` | A config to use when invoking the `Runnable`. The config supports<br>standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work<br>to do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | list[RunnableConfig] | None`**DEFAULT:**`None` |
| `return_exceptions` | Whether to return exceptions instead of raising them.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Output]` | A list of outputs from the `Runnable`. |

#### ``batch\_as\_completed [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.batch_as_completed "Copy anchor link to this section for reference")

```
batch_as_completed(
    inputs: Sequence[Input],
    config: RunnableConfig | Sequence[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> Iterator[tuple[int, Output | Exception]]
```

Run `invoke` in parallel on a list of inputs.

Yields results as they complete.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `inputs` | A list of inputs to the `Runnable`.<br>**TYPE:**`Sequence[Input]` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | Sequence[RunnableConfig] | None`**DEFAULT:**`None` |
| `return_exceptions` | Whether to return exceptions instead of raising them.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `tuple[int, Output | Exception]` | Tuples of the index of the input and the output from the `Runnable`. |

#### ``abatch`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.abatch "Copy anchor link to this section for reference")

```
abatch(
    inputs: list[Input],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> list[Output]
```

Default implementation runs `ainvoke` in parallel using `asyncio.gather`.

The default implementation of `batch` works well for IO bound runnables.

Subclasses must override this method if they can batch more efficiently;
e.g., if the underlying `Runnable` uses an API which supports a batch mode.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `inputs` | A list of inputs to the `Runnable`.<br>**TYPE:**`list[Input]` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | list[RunnableConfig] | None`**DEFAULT:**`None` |
| `return_exceptions` | Whether to return exceptions instead of raising them.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Output]` | A list of outputs from the `Runnable`. |

#### ``abatch\_as\_completed`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.abatch_as_completed "Copy anchor link to this section for reference")

```
abatch_as_completed(
    inputs: Sequence[Input],
    config: RunnableConfig | Sequence[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> AsyncIterator[tuple[int, Output | Exception]]
```

Run `ainvoke` in parallel on a list of inputs.

Yields results as they complete.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `inputs` | A list of inputs to the `Runnable`.<br>**TYPE:**`Sequence[Input]` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | Sequence[RunnableConfig] | None`**DEFAULT:**`None` |
| `return_exceptions` | Whether to return exceptions instead of raising them.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[tuple[int, Output | Exception]]` | A tuple of the index of the input and the output from the `Runnable`. |

#### ``stream [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.stream "Copy anchor link to this section for reference")

```
stream(
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> Iterator[AIMessageChunk]
```

Default implementation of `stream`, which calls `invoke`.

Subclasses must override this method if they support streaming output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | The config to use for the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

#### ``astream`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.astream "Copy anchor link to this section for reference")

```
astream(
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[AIMessageChunk]
```

Default implementation of `astream`, which calls `ainvoke`.

Subclasses must override this method if they support streaming output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | The config to use for the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[Output]` | The output of the `Runnable`. |

#### ``astream\_log`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.astream_log "Copy anchor link to this section for reference")

```
astream_log(
    input: Any,
    config: RunnableConfig | None = None,
    *,
    diff: bool = True,
    with_streamed_output_list: bool = True,
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]
```

Stream all output from a `Runnable`, as reported to the callback system.

This includes all inner runs of LLMs, Retrievers, Tools, etc.

Output is streamed as Log objects, which include a list of
Jsonpatch ops that describe how the state of the run has changed in each
step, and the final state of the run.

The Jsonpatch ops can be applied in order to construct state.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Any` |
| `config` | The config to use for the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `diff` | Whether to yield diffs between each step or the current state.<br>**TYPE:**`bool`**DEFAULT:**`True` |
| `with_streamed_output_list` | Whether to yield the `streamed_output` list.<br>**TYPE:**`bool`**DEFAULT:**`True` |
| `include_names` | Only include logs with these names.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `include_types` | Only include logs with these types.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `include_tags` | Only include logs with these tags.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `exclude_names` | Exclude logs with these names.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `exclude_types` | Exclude logs with these types.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `exclude_tags` | Exclude logs with these tags.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]` | A `RunLogPatch` or `RunLog` object. |

#### ``astream\_events`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.astream_events "Copy anchor link to this section for reference")

```
astream_events(
    input: Any,
    config: RunnableConfig | None = None,
    *,
    version: Literal["v1", "v2"] = "v2",
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]
```

Generate a stream of events.

Use to create an iterator over `StreamEvent` that provide real-time information
about the progress of the `Runnable`, including `StreamEvent` from intermediate
results.

A `StreamEvent` is a dictionary with the following schema:

- `event`: Event names are of the format:
`on_[runnable_type]_(start|stream|end)`.
- `name`: The name of the `Runnable` that generated the event.
- `run_id`: Randomly generated ID associated with the given execution of the
`Runnable` that emitted the event. A child `Runnable` that gets invoked as
part of the execution of a parent `Runnable` is assigned its own unique ID.
- `parent_ids`: The IDs of the parent runnables that generated the event. The
root `Runnable` will have an empty list. The order of the parent IDs is from
the root to the immediate parent. Only available for v2 version of the API.
The v1 version of the API will return an empty list.
- `tags`: The tags of the `Runnable` that generated the event.
- `metadata`: The metadata of the `Runnable` that generated the event.
- `data`: The data associated with the event. The contents of this field
depend on the type of event. See the table below for more details.

Below is a table that illustrates some events that might be emitted by various
chains. Metadata fields have been omitted from the table for brevity.
Chain definitions have been included after the table.

Note

This reference table is for the v2 version of the schema.

| event | name | chunk | input | output |
| --- | --- | --- | --- | --- |
| `on_chat_model_start` | `'[model name]'` |  | `{"messages": [[SystemMessage, HumanMessage]]}` |  |
| `on_chat_model_stream` | `'[model name]'` | `AIMessageChunk(content="hello")` |  |  |
| `on_chat_model_end` | `'[model name]'` |  | `{"messages": [[SystemMessage, HumanMessage]]}` | `AIMessageChunk(content="hello world")` |
| `on_llm_start` | `'[model name]'` |  | `{'input': 'hello'}` |  |
| `on_llm_stream` | `'[model name]'` | `'Hello'` |  |  |
| `on_llm_end` | `'[model name]'` |  | `'Hello human!'` |  |
| `on_chain_start` | `'format_docs'` |  |  |  |
| `on_chain_stream` | `'format_docs'` | `'hello world!, goodbye world!'` |  |  |
| `on_chain_end` | `'format_docs'` |  | `[Document(...)]` | `'hello world!, goodbye world!'` |
| `on_tool_start` | `'some_tool'` |  | `{"x": 1, "y": "2"}` |  |
| `on_tool_end` | `'some_tool'` |  |  | `{"x": 1, "y": "2"}` |
| `on_retriever_start` | `'[retriever name]'` |  | `{"query": "hello"}` |  |
| `on_retriever_end` | `'[retriever name]'` |  | `{"query": "hello"}` | `[Document(...), ..]` |
| `on_prompt_start` | `'[template_name]'` |  | `{"question": "hello"}` |  |
| `on_prompt_end` | `'[template_name]'` |  | `{"question": "hello"}` | `ChatPromptValue(messages: [SystemMessage, ...])` |

In addition to the standard events, users can also dispatch custom events (see example below).

Custom events will be only be surfaced with in the v2 version of the API!

A custom event has following format:

| Attribute | Type | Description |
| --- | --- | --- |
| `name` | `str` | A user defined name for the event. |
| `data` | `Any` | The data associated with the event. This can be anything, though we suggest making it JSON serializable. |

Here are declarations associated with the standard events shown above:

`format_docs`:

```
def format_docs(docs: list[Document]) -> str:
    '''Format the docs.'''
    return ", ".join([doc.page_content for doc in docs])

format_docs = RunnableLambda(format_docs)
```

`some_tool`:

```
@tool
def some_tool(x: int, y: str) -> dict:
    '''Some_tool.'''
    return {"x": x, "y": y}
```

`prompt`:

```
template = ChatPromptTemplate.from_messages(
    [\
        ("system", "You are Cat Agent 007"),\
        ("human", "{question}"),\
    ]
).with_config({"run_name": "my_template", "tags": ["my_template"]})
```

Example

```
from langchain_core.runnables import RunnableLambda

async def reverse(s: str) -> str:
    return s[::-1]

chain = RunnableLambda(func=reverse)

events = [\
    event async for event in chain.astream_events("hello", version="v2")\
]

# Will produce the following events
# (run_id, and parent_ids has been omitted for brevity):
[\
    {\
        "data": {"input": "hello"},\
        "event": "on_chain_start",\
        "metadata": {},\
        "name": "reverse",\
        "tags": [],\
    },\
    {\
        "data": {"chunk": "olleh"},\
        "event": "on_chain_stream",\
        "metadata": {},\
        "name": "reverse",\
        "tags": [],\
    },\
    {\
        "data": {"output": "olleh"},\
        "event": "on_chain_end",\
        "metadata": {},\
        "name": "reverse",\
        "tags": [],\
    },\
]
```

Dispatch custom event

```
from langchain_core.callbacks.manager import (
    adispatch_custom_event,
)
from langchain_core.runnables import RunnableLambda, RunnableConfig
import asyncio

async def slow_thing(some_input: str, config: RunnableConfig) -> str:
    """Do something that takes a long time."""
    await asyncio.sleep(1) # Placeholder for some slow operation
    await adispatch_custom_event(
        "progress_event",
        {"message": "Finished step 1 of 3"},
        config=config # Must be included for python < 3.10
    )
    await asyncio.sleep(1) # Placeholder for some slow operation
    await adispatch_custom_event(
        "progress_event",
        {"message": "Finished step 2 of 3"},
        config=config # Must be included for python < 3.10
    )
    await asyncio.sleep(1) # Placeholder for some slow operation
    return "Done"

slow_thing = RunnableLambda(slow_thing)

async for event in slow_thing.astream_events("some_input", version="v2"):
    print(event)
```

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Any` |
| `config` | The config to use for the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `version` | The version of the schema to use, either `'v2'` or `'v1'`.<br>Users should use `'v2'`.<br>`'v1'` is for backwards compatibility and will be deprecated<br>in `0.4.0`.<br>No default will be assigned until the API is stabilized.<br>custom events will only be surfaced in `'v2'`.<br>**TYPE:**`Literal['v1', 'v2']`**DEFAULT:**`'v2'` |
| `include_names` | Only include events from `Runnable` objects with matching names.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `include_types` | Only include events from `Runnable` objects with matching types.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `include_tags` | Only include events from `Runnable` objects with matching tags.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `exclude_names` | Exclude events from `Runnable` objects with matching names.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `exclude_types` | Exclude events from `Runnable` objects with matching types.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `exclude_tags` | Exclude events from `Runnable` objects with matching tags.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>These will be passed to `astream_log` as this implementation<br>of `astream_events` is built on top of `astream_log`.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[StreamEvent]` | An async stream of `StreamEvent`. |

| RAISES | DESCRIPTION |
| --- | --- |
| `NotImplementedError` | If the version is not `'v1'` or `'v2'`. |

#### ``transform [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.transform "Copy anchor link to this section for reference")

```
transform(
    input: Iterator[Input], config: RunnableConfig | None = None, **kwargs: Any | None
) -> Iterator[Output]
```

Transform inputs to outputs.

Default implementation of transform, which buffers input and calls `astream`.

Subclasses must override this method if they can start producing output while
input is still being generated.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | An iterator of inputs to the `Runnable`.<br>**TYPE:**`Iterator[Input]` |
| `config` | The config to use for the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

#### ``atransform`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.atransform "Copy anchor link to this section for reference")

```
atransform(
    input: AsyncIterator[Input],
    config: RunnableConfig | None = None,
    **kwargs: Any | None,
) -> AsyncIterator[Output]
```

Transform inputs to outputs.

Default implementation of atransform, which buffers input and calls `astream`.

Subclasses must override this method if they can start producing output while
input is still being generated.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | An async iterator of inputs to the `Runnable`.<br>**TYPE:**`AsyncIterator[Input]` |
| `config` | The config to use for the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[Output]` | The output of the `Runnable`. |

#### ``bind [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.bind "Copy anchor link to this section for reference")

```
bind(**kwargs: Any) -> Runnable[Input, Output]
```

Bind arguments to a `Runnable`, returning a new `Runnable`.

Useful when a `Runnable` in a chain requires an argument that is not
in the output of the previous `Runnable` or included in the user input.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | The arguments to bind to the `Runnable`.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` with the arguments bound. |

Example

```
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3.1")

# Without bind
chain = model | StrOutputParser()

chain.invoke("Repeat quoted words exactly: 'One two three four five.'")
# Output is 'One two three four five.'

# With bind
chain = model.bind(stop=["three"]) | StrOutputParser()

chain.invoke("Repeat quoted words exactly: 'One two three four five.'")
# Output is 'One two'
```

#### ``with\_config [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.with_config "Copy anchor link to this section for reference")

```
with_config(
    config: RunnableConfig | None = None, **kwargs: Any
) -> Runnable[Input, Output]
```

Bind config to a `Runnable`, returning a new `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | The config to bind to the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` with the config bound. |

#### ``with\_listeners [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.with_listeners "Copy anchor link to this section for reference")

```
with_listeners(
    *,
    on_start: Callable[[Run], None]
    | Callable[[Run, RunnableConfig], None]
    | None = None,
    on_end: Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None = None,
    on_error: Callable[[Run], None]
    | Callable[[Run, RunnableConfig], None]
    | None = None,
) -> Runnable[Input, Output]
```

Bind lifecycle listeners to a `Runnable`, returning a new `Runnable`.

The Run object contains information about the run, including its `id`,
`type`, `input`, `output`, `error`, `start_time`, `end_time`, and
any tags or metadata added to the run.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `on_start` | Called before the `Runnable` starts running, with the `Run`<br>object.<br>**TYPE:**`Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None`**DEFAULT:**`None` |
| `on_end` | Called after the `Runnable` finishes running, with the `Run`<br>object.<br>**TYPE:**`Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None`**DEFAULT:**`None` |
| `on_error` | Called if the `Runnable` throws an error, with the `Run`<br>object.<br>**TYPE:**`Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` with the listeners bound. |

Example

```
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.schemas import Run

import time

def test_runnable(time_to_sleep: int):
    time.sleep(time_to_sleep)

def fn_start(run_obj: Run):
    print("start_time:", run_obj.start_time)

def fn_end(run_obj: Run):
    print("end_time:", run_obj.end_time)

chain = RunnableLambda(test_runnable).with_listeners(
    on_start=fn_start, on_end=fn_end
)
chain.invoke(2)
```

#### ``with\_alisteners [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.with_alisteners "Copy anchor link to this section for reference")

```
with_alisteners(
    *,
    on_start: AsyncListener | None = None,
    on_end: AsyncListener | None = None,
    on_error: AsyncListener | None = None,
) -> Runnable[Input, Output]
```

Bind async lifecycle listeners to a `Runnable`.

Returns a new `Runnable`.

The Run object contains information about the run, including its `id`,
`type`, `input`, `output`, `error`, `start_time`, `end_time`, and
any tags or metadata added to the run.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `on_start` | Called asynchronously before the `Runnable` starts running,<br>with the `Run` object.<br>**TYPE:**`AsyncListener | None`**DEFAULT:**`None` |
| `on_end` | Called asynchronously after the `Runnable` finishes running,<br>with the `Run` object.<br>**TYPE:**`AsyncListener | None`**DEFAULT:**`None` |
| `on_error` | Called asynchronously if the `Runnable` throws an error,<br>with the `Run` object.<br>**TYPE:**`AsyncListener | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` with the listeners bound. |

Example

```
from langchain_core.runnables import RunnableLambda, Runnable
from datetime import datetime, timezone
import time
import asyncio

def format_t(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

async def test_runnable(time_to_sleep: int):
    print(f"Runnable[{time_to_sleep}s]: starts at {format_t(time.time())}")
    await asyncio.sleep(time_to_sleep)
    print(f"Runnable[{time_to_sleep}s]: ends at {format_t(time.time())}")

async def fn_start(run_obj: Runnable):
    print(f"on start callback starts at {format_t(time.time())}")
    await asyncio.sleep(3)
    print(f"on start callback ends at {format_t(time.time())}")

async def fn_end(run_obj: Runnable):
    print(f"on end callback starts at {format_t(time.time())}")
    await asyncio.sleep(2)
    print(f"on end callback ends at {format_t(time.time())}")

runnable = RunnableLambda(test_runnable).with_alisteners(
    on_start=fn_start, on_end=fn_end
)

async def concurrent_runs():
    await asyncio.gather(runnable.ainvoke(2), runnable.ainvoke(3))

asyncio.run(concurrent_runs())
# Result:
# on start callback starts at 2025-03-01T07:05:22.875378+00:00
# on start callback starts at 2025-03-01T07:05:22.875495+00:00
# on start callback ends at 2025-03-01T07:05:25.878862+00:00
# on start callback ends at 2025-03-01T07:05:25.878947+00:00
# Runnable[2s]: starts at 2025-03-01T07:05:25.879392+00:00
# Runnable[3s]: starts at 2025-03-01T07:05:25.879804+00:00
# Runnable[2s]: ends at 2025-03-01T07:05:27.881998+00:00
# on end callback starts at 2025-03-01T07:05:27.882360+00:00
# Runnable[3s]: ends at 2025-03-01T07:05:28.881737+00:00
# on end callback starts at 2025-03-01T07:05:28.882428+00:00
# on end callback ends at 2025-03-01T07:05:29.883893+00:00
# on end callback ends at 2025-03-01T07:05:30.884831+00:00
```

#### ``with\_types [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.with_types "Copy anchor link to this section for reference")

```
with_types(
    *, input_type: type[Input] | None = None, output_type: type[Output] | None = None
) -> Runnable[Input, Output]
```

Bind input and output types to a `Runnable`, returning a new `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input_type` | The input type to bind to the `Runnable`.<br>**TYPE:**`type[Input] | None`**DEFAULT:**`None` |
| `output_type` | The output type to bind to the `Runnable`.<br>**TYPE:**`type[Output] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` with the types bound. |

#### ``with\_retry [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.with_retry "Copy anchor link to this section for reference")

```
with_retry(
    *,
    retry_if_exception_type: tuple[type[BaseException], ...] = (Exception,),
    wait_exponential_jitter: bool = True,
    exponential_jitter_params: ExponentialJitterParams | None = None,
    stop_after_attempt: int = 3,
) -> Runnable[Input, Output]
```

Create a new `Runnable` that retries the original `Runnable` on exceptions.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `retry_if_exception_type` | A tuple of exception types to retry on.<br>**TYPE:**`tuple[type[BaseException], ...]`**DEFAULT:**`(Exception,)` |
| `wait_exponential_jitter` | Whether to add jitter to the wait<br>time between retries.<br>**TYPE:**`bool`**DEFAULT:**`True` |
| `stop_after_attempt` | The maximum number of attempts to make before<br>giving up.<br>**TYPE:**`int`**DEFAULT:**`3` |
| `exponential_jitter_params` | Parameters for<br>`tenacity.wait_exponential_jitter`. Namely: `initial`, `max`,<br>`exp_base`, and `jitter` (all `float` values).<br>**TYPE:**`ExponentialJitterParams | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` that retries the original `Runnable` on exceptions. |

Example

```
from langchain_core.runnables import RunnableLambda

count = 0

def _lambda(x: int) -> None:
    global count
    count = count + 1
    if x == 1:
        raise ValueError("x is 1")
    else:
        pass

runnable = RunnableLambda(_lambda)
try:
    runnable.with_retry(
        stop_after_attempt=2,
        retry_if_exception_type=(ValueError,),
    ).invoke(1)
except ValueError:
    pass

assert count == 2
```

#### ``map [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.map "Copy anchor link to this section for reference")

```
map() -> Runnable[list[Input], list[Output]]
```

Return a new `Runnable` that maps a list of inputs to a list of outputs.

Calls `invoke` with each input.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[list[Input], list[Output]]` | A new `Runnable` that maps a list of inputs to a list of outputs. |

Example

```
from langchain_core.runnables import RunnableLambda

def _lambda(x: int) -> int:
    return x + 1

runnable = RunnableLambda(_lambda)
print(runnable.map().invoke([1, 2, 3]))  # [2, 3, 4]
```

#### ``with\_fallbacks [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.with_fallbacks "Copy anchor link to this section for reference")

```
with_fallbacks(
    fallbacks: Sequence[Runnable[Input, Output]],
    *,
    exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,),
    exception_key: str | None = None,
) -> RunnableWithFallbacks[Input, Output]
```

Add fallbacks to a `Runnable`, returning a new `Runnable`.

The new `Runnable` will try the original `Runnable`, and then each fallback
in order, upon failures.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `fallbacks` | A sequence of runnables to try if the original `Runnable`<br>fails.<br>**TYPE:**`Sequence[Runnable[Input, Output]]` |
| `exceptions_to_handle` | A tuple of exception types to handle.<br>**TYPE:**`tuple[type[BaseException], ...]`**DEFAULT:**`(Exception,)` |
| `exception_key` | If `string` is specified then handled exceptions will be<br>passed to fallbacks as part of the input under the specified key.<br>If `None`, exceptions will not be passed to fallbacks.<br>If used, the base `Runnable` and its fallbacks must accept a<br>dictionary as input.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableWithFallbacks[Input, Output]` | A new `Runnable` that will try the original `Runnable`, and then each<br>Fallback in order, upon failures. |

Example

```
from typing import Iterator

from langchain_core.runnables import RunnableGenerator

def _generate_immediate_error(input: Iterator) -> Iterator[str]:
    raise ValueError()
    yield ""

def _generate(input: Iterator) -> Iterator[str]:
    yield from "foo bar"

runnable = RunnableGenerator(_generate_immediate_error).with_fallbacks(
    [RunnableGenerator(_generate)]
)
print("".join(runnable.stream({})))  # foo bar
```

| PARAMETER | DESCRIPTION |
| --- | --- |
| `fallbacks` | A sequence of runnables to try if the original `Runnable`<br>fails.<br>**TYPE:**`Sequence[Runnable[Input, Output]]` |
| `exceptions_to_handle` | A tuple of exception types to handle.<br>**TYPE:**`tuple[type[BaseException], ...]`**DEFAULT:**`(Exception,)` |
| `exception_key` | If `string` is specified then handled exceptions will be<br>passed to fallbacks as part of the input under the specified key.<br>If `None`, exceptions will not be passed to fallbacks.<br>If used, the base `Runnable` and its fallbacks must accept a<br>dictionary as input.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableWithFallbacks[Input, Output]` | A new `Runnable` that will try the original `Runnable`, and then each<br>Fallback in order, upon failures. |

#### ``as\_tool [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.as_tool "Copy anchor link to this section for reference")

```
as_tool(
    args_schema: type[BaseModel] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    arg_types: dict[str, type] | None = None,
) -> BaseTool
```

Create a `BaseTool` from a `Runnable`.

`as_tool` will instantiate a `BaseTool` with a name, description, and
`args_schema` from a `Runnable`. Where possible, schemas are inferred
from `runnable.get_input_schema`.

Alternatively (e.g., if the `Runnable` takes a dict as input and the specific
`dict` keys are not typed), the schema can be specified directly with
`args_schema`.

You can also pass `arg_types` to just specify the required arguments and their
types.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `args_schema` | The schema for the tool.<br>**TYPE:**`type[BaseModel] | None`**DEFAULT:**`None` |
| `name` | The name of the tool.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `description` | The description of the tool.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `arg_types` | A dictionary of argument names to types.<br>**TYPE:**`dict[str, type] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `BaseTool` | A `BaseTool` instance. |

`TypedDict` input

```
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableLambda

class Args(TypedDict):
    a: int
    b: list[int]

def f(x: Args) -> str:
    return str(x["a"] * max(x["b"]))

runnable = RunnableLambda(f)
as_tool = runnable.as_tool()
as_tool.invoke({"a": 3, "b": [1, 2]})
```

`dict` input, specifying schema via `args_schema`

```
from typing import Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

def f(x: dict[str, Any]) -> str:
    return str(x["a"] * max(x["b"]))

class FSchema(BaseModel):
    """Apply a function to an integer and list of integers."""

    a: int = Field(..., description="Integer")
    b: list[int] = Field(..., description="List of ints")

runnable = RunnableLambda(f)
as_tool = runnable.as_tool(FSchema)
as_tool.invoke({"a": 3, "b": [1, 2]})
```

`dict` input, specifying schema via `arg_types`

```
from typing import Any
from langchain_core.runnables import RunnableLambda

def f(x: dict[str, Any]) -> str:
    return str(x["a"] * max(x["b"]))

runnable = RunnableLambda(f)
as_tool = runnable.as_tool(arg_types={"a": int, "b": list[int]})
as_tool.invoke({"a": 3, "b": [1, 2]})
```

`str` input

```
from langchain_core.runnables import RunnableLambda

def f(x: str) -> str:
    return x + "a"

def g(x: str) -> str:
    return x + "z"

runnable = RunnableLambda(f) | g
as_tool = runnable.as_tool()
as_tool.invoke("b")
```

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.__init__ "Copy anchor link to this section for reference")

```
__init__(*args: Any, **kwargs: Any) -> None
```

#### ``is\_lc\_serializable`classmethod`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.is_lc_serializable "Copy anchor link to this section for reference")

```
is_lc_serializable() -> bool
```

Is this class serializable?

By design, even if a class inherits from `Serializable`, it is not serializable
by default. This is to prevent accidental serialization of objects that should
not be serialized.

| RETURNS | DESCRIPTION |
| --- | --- |
| `bool` | Whether the class is serializable. Default is `False`. |

#### ``get\_lc\_namespace`classmethod`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.get_lc_namespace "Copy anchor link to this section for reference")

```
get_lc_namespace() -> list[str]
```

Get the namespace of the LangChain object.

For example, if the class is
[`langchain.llms.openai.OpenAI`](https://reference.langchain.com/python/integrations/langchain_openai/OpenAI/#langchain_openai.OpenAI "<code class=\"doc-symbol doc-symbol-heading doc-symbol-class\"></code>            <span class=\"doc doc-object-name doc-class-name\">OpenAI</span>"), then the namespace is
`["langchain", "llms", "openai"]`

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | The namespace. |

#### ``lc\_id`classmethod`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.lc_id "Copy anchor link to this section for reference")

```
lc_id() -> list[str]
```

Return a unique identifier for this class for serialization purposes.

The unique identifier is a list of strings that describes the path
to the object.

For example, for the class `langchain.llms.openai.OpenAI`, the id is
`["langchain", "llms", "openai", "OpenAI"]`.

#### ``to\_json [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.to_json "Copy anchor link to this section for reference")

```
to_json() -> SerializedConstructor | SerializedNotImplemented
```

Serialize the `Runnable` to JSON.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedConstructor | SerializedNotImplemented` | A JSON-serializable representation of the `Runnable`. |

#### ``to\_json\_not\_implemented [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.to_json_not_implemented "Copy anchor link to this section for reference")

```
to_json_not_implemented() -> SerializedNotImplemented
```

Serialize a "not implemented" object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedNotImplemented` | `SerializedNotImplemented`. |

#### ``configurable\_fields [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.configurable_fields "Copy anchor link to this section for reference")

```
configurable_fields(
    **kwargs: AnyConfigurableField,
) -> RunnableSerializable[Input, Output]
```

Configure particular `Runnable` fields at runtime.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | A dictionary of `ConfigurableField` instances to configure.<br>**TYPE:**`AnyConfigurableField`**DEFAULT:**`{}` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If a configuration key is not found in the `Runnable`. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Input, Output]` | A new `Runnable` with the fields configured. |

Example

```
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatOpenAI(max_tokens=20).configurable_fields(
    max_tokens=ConfigurableField(
        id="output_token_number",
        name="Max tokens in the output",
        description="The maximum number of tokens in the output",
    )
)

# max_tokens = 20
print(
    "max_tokens_20: ", model.invoke("tell me something about chess").content
)

# max_tokens = 200
print(
    "max_tokens_200: ",
    model.with_config(configurable={"output_token_number": 200})
    .invoke("tell me something about chess")
    .content,
)
```

#### ``configurable\_alternatives [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.configurable_alternatives "Copy anchor link to this section for reference")

```
configurable_alternatives(
    which: ConfigurableField,
    *,
    default_key: str = "default",
    prefix_keys: bool = False,
    **kwargs: Runnable[Input, Output] | Callable[[], Runnable[Input, Output]],
) -> RunnableSerializable[Input, Output]
```

Configure alternatives for `Runnable` objects that can be set at runtime.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `which` | The `ConfigurableField` instance that will be used to select the<br>alternative.<br>**TYPE:**`ConfigurableField` |
| `default_key` | The default key to use if no alternative is selected.<br>**TYPE:**`str`**DEFAULT:**`'default'` |
| `prefix_keys` | Whether to prefix the keys with the `ConfigurableField` id.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `**kwargs` | A dictionary of keys to `Runnable` instances or callables that<br>return `Runnable` instances.<br>**TYPE:**`Runnable[Input, Output] | Callable[[], Runnable[Input, Output]]`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Input, Output]` | A new `Runnable` with the alternatives configured. |

Example

```
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables.utils import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatAnthropic(
    model_name="claude-sonnet-4-5-20250929"
).configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="anthropic",
    openai=ChatOpenAI(),
)

# uses the default model ChatAnthropic
print(model.invoke("which organization created you?").content)

# uses ChatOpenAI
print(
    model.with_config(configurable={"llm": "openai"})
    .invoke("which organization created you?")
    .content
)
```

#### ``set\_verbose [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.set_verbose "Copy anchor link to this section for reference")

```
set_verbose(verbose: bool | None) -> bool
```

If verbose is `None`, set it.

This allows users to pass in `None` as verbose to access the global setting.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `verbose` | The verbosity setting to use.<br>**TYPE:**`bool | None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `bool` | The verbosity setting to use. |

#### ``generate\_prompt [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.generate_prompt "Copy anchor link to this section for reference")

```
generate_prompt(
    prompts: list[PromptValue],
    stop: list[str] | None = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> LLMResult
```

Pass a sequence of prompts to the model and return model generations.

This method should make use of batched calls for models that expose a batched
API.

Use this method when you want to:

1. Take advantage of batched calls,
2. Need more output from the model than just the top generated value,
3. Are building chains that are agnostic to the underlying language model
    type (e.g., pure text completion models vs chat models).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompts` | List of `PromptValue` objects.<br>A `PromptValue` is an object that can be converted to match the format<br>of any language model (string for pure text generation models and<br>`BaseMessage` objects for chat models).<br>**TYPE:**`list[PromptValue]` |
| `stop` | Stop words to use when generating.<br>Model output is cut off at the first occurrence of any of these<br>substrings.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `callbacks` | `Callbacks` to pass through.<br>Used for executing additional functionality, such as logging or<br>streaming, throughout generation.<br>**TYPE:**`Callbacks`**DEFAULT:**`None` |
| `**kwargs` | Arbitrary additional keyword arguments.<br>These are usually passed to the model provider API call.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `LLMResult` | An `LLMResult`, which contains a list of candidate `Generation` objects for<br>each input prompt and additional model provider-specific output. |

#### ``agenerate\_prompt`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.agenerate_prompt "Copy anchor link to this section for reference")

```
agenerate_prompt(
    prompts: list[PromptValue],
    stop: list[str] | None = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> LLMResult
```

Asynchronously pass a sequence of prompts and return model generations.

This method should make use of batched calls for models that expose a batched
API.

Use this method when you want to:

1. Take advantage of batched calls,
2. Need more output from the model than just the top generated value,
3. Are building chains that are agnostic to the underlying language model
    type (e.g., pure text completion models vs chat models).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompts` | List of `PromptValue` objects.<br>A `PromptValue` is an object that can be converted to match the format<br>of any language model (string for pure text generation models and<br>`BaseMessage` objects for chat models).<br>**TYPE:**`list[PromptValue]` |
| `stop` | Stop words to use when generating.<br>Model output is cut off at the first occurrence of any of these<br>substrings.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `callbacks` | `Callbacks` to pass through.<br>Used for executing additional functionality, such as logging or<br>streaming, throughout generation.<br>**TYPE:**`Callbacks`**DEFAULT:**`None` |
| `**kwargs` | Arbitrary additional keyword arguments.<br>These are usually passed to the model provider API call.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `LLMResult` | An `LLMResult`, which contains a list of candidate `Generation` objects for<br>each input prompt and additional model provider-specific output. |

#### ``get\_token\_ids [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.get_token_ids "Copy anchor link to this section for reference")

```
get_token_ids(text: str) -> list[int]
```

Return the ordered IDs of the tokens in a text.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The string input to tokenize.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[int]` | A list of IDs corresponding to the tokens in the text, in order they occur<br>in the text. |

#### ``get\_num\_tokens [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.get_num_tokens "Copy anchor link to this section for reference")

```
get_num_tokens(text: str) -> int
```

Get the number of tokens present in the text.

Useful for checking if an input fits in a model's context window.

This should be overridden by model-specific implementations to provide accurate
token counts via model-specific tokenizers.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The string input to tokenize.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `int` | The integer number of tokens in the text. |

#### ``get\_num\_tokens\_from\_messages [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.get_num_tokens_from_messages "Copy anchor link to this section for reference")

```
get_num_tokens_from_messages(
    messages: list[BaseMessage], tools: Sequence | None = None
) -> int
```

Get the number of tokens in the messages.

Useful for checking if an input fits in a model's context window.

This should be overridden by model-specific implementations to provide accurate
token counts via model-specific tokenizers.

Note

- The base implementation of `get_num_tokens_from_messages` ignores tool
schemas.
- The base implementation of `get_num_tokens_from_messages` adds additional
prefixes to messages in represent user roles, which will add to the
overall token count. Model-specific implementations may choose to
handle this differently.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `messages` | The message inputs to tokenize.<br>**TYPE:**`list[BaseMessage]` |
| `tools` | If provided, sequence of dict, `BaseModel`, function, or<br>`BaseTool` objects to be converted to tool schemas.<br>**TYPE:**`Sequence | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `int` | The sum of the number of tokens across the messages. |

#### ``generate [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.generate "Copy anchor link to this section for reference")

```
generate(
    messages: list[list[BaseMessage]],
    stop: list[str] | None = None,
    callbacks: Callbacks = None,
    *,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    run_name: str | None = None,
    run_id: UUID | None = None,
    **kwargs: Any,
) -> LLMResult
```

Pass a sequence of prompts to the model and return model generations.

This method should make use of batched calls for models that expose a batched
API.

Use this method when you want to:

1. Take advantage of batched calls,
2. Need more output from the model than just the top generated value,
3. Are building chains that are agnostic to the underlying language model
    type (e.g., pure text completion models vs chat models).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `messages` | List of list of messages.<br>**TYPE:**`list[list[BaseMessage]]` |
| `stop` | Stop words to use when generating.<br>Model output is cut off at the first occurrence of any of these<br>substrings.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `callbacks` | `Callbacks` to pass through.<br>Used for executing additional functionality, such as logging or<br>streaming, throughout generation.<br>**TYPE:**`Callbacks`**DEFAULT:**`None` |
| `tags` | The tags to apply.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata to apply.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `run_name` | The name of the run.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Arbitrary additional keyword arguments.<br>These are usually passed to the model provider API call.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `LLMResult` | An `LLMResult`, which contains a list of candidate `Generations` for each<br>input prompt and additional model provider-specific output. |

#### ``agenerate`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.agenerate "Copy anchor link to this section for reference")

```
agenerate(
    messages: list[list[BaseMessage]],
    stop: list[str] | None = None,
    callbacks: Callbacks = None,
    *,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    run_name: str | None = None,
    run_id: UUID | None = None,
    **kwargs: Any,
) -> LLMResult
```

Asynchronously pass a sequence of prompts to a model and return generations.

This method should make use of batched calls for models that expose a batched
API.

Use this method when you want to:

1. Take advantage of batched calls,
2. Need more output from the model than just the top generated value,
3. Are building chains that are agnostic to the underlying language model
    type (e.g., pure text completion models vs chat models).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `messages` | List of list of messages.<br>**TYPE:**`list[list[BaseMessage]]` |
| `stop` | Stop words to use when generating.<br>Model output is cut off at the first occurrence of any of these<br>substrings.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `callbacks` | `Callbacks` to pass through.<br>Used for executing additional functionality, such as logging or<br>streaming, throughout generation.<br>**TYPE:**`Callbacks`**DEFAULT:**`None` |
| `tags` | The tags to apply.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata to apply.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `run_name` | The name of the run.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Arbitrary additional keyword arguments.<br>These are usually passed to the model provider API call.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `LLMResult` | An `LLMResult`, which contains a list of candidate `Generations` for each<br>input prompt and additional model provider-specific output. |

#### ``dict [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.dict "Copy anchor link to this section for reference")

```
dict(**kwargs: Any) -> dict
```

Return a dictionary of the LLM.

#### ``bind\_tools [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.bind_tools "Copy anchor link to this section for reference")

```
bind_tools(
    tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
    *,
    tool_choice: dict | str | Literal["auto", "any"] | bool | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]
```

Bind tool-like objects to this chat model.

Assumes model is compatible with OpenAI tool-calling API.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tools` | A list of tool definitions to bind to this chat model.<br>Supports any tool definition handled by [`convert_to_openai_tool`](https://reference.langchain.com/python/langchain_core/utils/#langchain_core.utils.function_calling.convert_to_openai_tool "<code class=\"doc-symbol doc-symbol-heading doc-symbol-function\"></code>            <span class=\"doc doc-object-name doc-function-name\">convert_to_openai_tool</span>").<br>**TYPE:**`Sequence[dict[str, Any] | type | Callable | BaseTool]` |
| `tool_choice` | If provided, which tool for model to call. **This parameter**<br>**is currently ignored as it is not supported by Ollama.**<br>**TYPE:**`dict | str | Literal['auto', 'any'] | bool | None`**DEFAULT:**`None` |
| `kwargs` | Any additional parameters are passed directly to<br>`self.bind(**kwargs)`.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

#### ``with\_structured\_output [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.ChatOllama.with_structured_output "Copy anchor link to this section for reference")

```
with_structured_output(
    schema: dict | type,
    *,
    method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, dict | BaseModel]
```

Model wrapper that returns outputs formatted to match the given schema.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `schema` | The output schema. Can be passed in as:<br>- An OpenAI function/tool schema.<br>- A JSON Schema,<br>- A `TypedDict` class,<br>- Or a Pydantic class.<br>If `schema` is a Pydantic class then the model output will be a<br>Pydantic instance of that class, and the model-generated fields will be<br>validated by the Pydantic class. Otherwise the model output will be a<br>dict and will not be validated.<br>See `langchain_core.utils.function_calling.convert_to_openai_tool` for<br>more on how to properly specify types and descriptions of schema fields<br>when specifying a Pydantic or `TypedDict` class.<br>**TYPE:**`dict | type` |
| `method` | The method for steering model generation, one of:<br>- `'json_schema'`:<br>Uses Ollama's [structured output API](https://ollama.com/blog/structured-outputs)<br>- `'function_calling'`:<br>Uses Ollama's tool-calling API<br>- `'json_mode'`:<br>Specifies `format='json'`. Note that if using JSON mode then you<br>must include instructions for formatting the output into the<br>desired schema into the model call.<br>**TYPE:**`Literal['function_calling', 'json_mode', 'json_schema']`**DEFAULT:**`'json_schema'` |
| `include_raw` | If `False` then only the parsed structured output is returned.<br>If an error occurs during model output parsing it will be raised.<br>If `True` then both the raw model response (a `BaseMessage`) and the<br>parsed model response will be returned.<br>If an error occurs during output parsing it will be caught and returned<br>as well.<br>The final output is always a `dict` with keys `'raw'`, `'parsed'`, and<br>`'parsing_error'`.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `kwargs` | Additional keyword args aren't supported.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[LanguageModelInput, dict | BaseModel]` | A `Runnable` that takes same inputs as a<br>`langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is<br>`False` and `schema` is a Pydantic class, `Runnable` outputs an instance<br>of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is<br>`False` then `Runnable` outputs a `dict`.<br>If `include_raw` is `True`, then `Runnable` outputs a `dict` with keys:<br>- `'raw'`: `BaseMessage`<br>- `'parsed'`: `None` if there was a parsing error, otherwise the type<br>depends on the `schema` as described above.<br>- `'parsing_error'`: `BaseException | None` |

Behavior changed in `langchain-ollama` 0.2.2

Added support for structured output API via `format` parameter.

Behavior changed in `langchain-ollama` 0.3.0

Updated default `method` to `'json_schema'`.

Example: `schema=Pydantic` class, `method='json_schema'`, `include_raw=False`

```
from typing import Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''

    answer: str
    justification: str | None = Field(
        default=...,
        description="A justification for the answer.",
    )

model = ChatOllama(model="llama3.1", temperature=0)
structured_model = model.with_structured_output(AnswerWithJustification)

structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")

# -> AnswerWithJustification(
#     answer='They weigh the same',
#     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
# )
```

Example: `schema=Pydantic` class, `method='json_schema'`, `include_raw=True`

```
from langchain_ollama import ChatOllama
from pydantic import BaseModel

class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''

    answer: str
    justification: str

model = ChatOllama(model="llama3.1", temperature=0)
structured_model = model.with_structured_output(
    AnswerWithJustification,
    include_raw=True,
)

structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")
# -> {
#     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
#     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
#     'parsing_error': None
# }
```

Example: `schema=Pydantic` class, `method='function_calling'`, `include_raw=False`

```
from typing import Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''

    answer: str
    justification: str | None = Field(
        default=...,
        description="A justification for the answer.",
    )

model = ChatOllama(model="llama3.1", temperature=0)
structured_model = model.with_structured_output(
    AnswerWithJustification,
    method="function_calling",
)

structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")

# -> AnswerWithJustification(
#     answer='They weigh the same',
#     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
# )
```

Example: `schema=TypedDict` class, `method='function_calling'`, `include_raw=False`

```
from typing_extensions import Annotated, TypedDict

from langchain_ollama import ChatOllama

class AnswerWithJustification(TypedDict):
    '''An answer to the user question along with justification for the answer.'''

    answer: str
    justification: Annotated[str | None, None, "A justification for the answer."]

model = ChatOllama(model="llama3.1", temperature=0)
structured_model = model.with_structured_output(AnswerWithJustification)

structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")
# -> {
#     'answer': 'They weigh the same',
#     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
# }
```

Example: `schema=OpenAI` function schema, `method='function_calling'`, `include_raw=False`

```
from langchain_ollama import ChatOllama

oai_schema = {
    'name': 'AnswerWithJustification',
    'description': 'An answer to the user question along with justification for the answer.',
    'parameters': {
        'type': 'object',
        'properties': {
            'answer': {'type': 'string'},
            'justification': {'description': 'A justification for the answer.', 'type': 'string'}
        },
        'required': ['answer']
    }

    model = ChatOllama(model="llama3.1", temperature=0)
    structured_model = model.with_structured_output(oai_schema)

    structured_model.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    # -> {
    #     'answer': 'They weigh the same',
    #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
    # }
```

Example: `schema=Pydantic` class, `method='json_mode'`, `include_raw=True`

```
from langchain_ollama import ChatOllama
from pydantic import BaseModel

class AnswerWithJustification(BaseModel):
    answer: str
    justification: str

model = ChatOllama(model="llama3.1", temperature=0)
structured_model = model.with_structured_output(
    AnswerWithJustification, method="json_mode", include_raw=True
)

structured_model.invoke(
    "Answer the following question. "
    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\\n\\n"
    "What's heavier a pound of bricks or a pound of feathers?"
)
# -> {
#     'raw': AIMessage(content='{\\n    "answer": "They are both the same weight.",\\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \\n}'),
#     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
#     'parsing_error': None
# }
```

### ``OllamaEmbeddings [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings "Copy anchor link to this section for reference")

Bases: `BaseModel`, `Embeddings`

Ollama embedding model integration.

Set up a local Ollama instance

[Install the Ollama package](https://github.com/ollama/ollama) and set up a
local Ollama instance.

You will need to choose a model to serve.

You can view a list of available models via [the model library](https://ollama.com/library).

To fetch a model from the Ollama model library use `ollama pull <name-of-model>`.

For example, to pull the llama3 model:

```
ollama pull llama3
```

This will download the default tagged version of the model.
Typically, the default points to the latest, smallest sized-parameter model.

- On Mac, the models will be downloaded to `~/.ollama/models`
- On Linux (or WSL), the models will be stored at `/usr/share/ollama/.ollama/models`

You can specify the exact version of the model of interest
as such `ollama pull vicuna:13b-v1.5-16k-q4_0`.

To view pulled models:

```
ollama list
```

To start serving:

```
ollama serve
```

View the Ollama documentation for more commands.

```
ollama help
```

Install the `langchain-ollama` integration package:


```
pip install -U langchain_ollama
```

Key init args — completion params:
model: str
Name of Ollama model to use.
base\_url: str \| None
Base url the model is hosted under.

See full list of supported init args and their descriptions in the params section.

Instantiate

```
from langchain_ollama import OllamaEmbeddings

embed = OllamaEmbeddings(model="llama3")
```

Embed single text

```
input_text = "The meaning of life is 42"
vector = embed.embed_query(input_text)
print(vector[:3])
```

```
[-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
```

Embed multiple texts

```
input_texts = ["Document 1...", "Document 2..."]
vectors = embed.embed_documents(input_texts)
print(len(vectors))
# The first 3 coordinates for the first vector
print(vectors[0][:3])
```

```
2
[-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
```

Async

```
vector = await embed.aembed_query(input_text)
print(vector[:3])

# multiple:
# await embed.aembed_documents(input_texts)
```

```
[-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
```

| METHOD | DESCRIPTION |
| --- | --- |
| `embed_documents` | Embed search docs. |
| `embed_query` | Embed query text. |
| `aembed_documents` | Embed search docs. |
| `aembed_query` | Embed query text. |

#### ``model`instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.model "Copy anchor link to this section for reference")

```
model: str
```

Model name to use.

#### ``validate\_model\_on\_init`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.validate_model_on_init "Copy anchor link to this section for reference")

```
validate_model_on_init: bool = False
```

Whether to validate the model exists in ollama locally on initialization.

Added in `langchain-ollama` 0.3.4

#### ``base\_url`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.base_url "Copy anchor link to this section for reference")

```
base_url: str | None = None
```

Base url the model is hosted under.

If none, defaults to the Ollama client default.

Supports `userinfo` auth in the format `http://username:password@localhost:11434`.
Useful if your Ollama server is behind a proxy.

Warning

`userinfo` is not secure and should only be used for local testing or
in secure environments. Avoid using it in production or over unsecured
networks.

Note

If using `userinfo`, ensure that the Ollama server is configured to
accept and validate these credentials.

Note

`userinfo` headers are passed to both sync and async clients.

#### ``client\_kwargs`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.client_kwargs "Copy anchor link to this section for reference")

```
client_kwargs: dict | None = {}
```

Additional kwargs to pass to the httpx clients. Pass headers in here.

These arguments are passed to both synchronous and async clients.

Use `sync_client_kwargs` and `async_client_kwargs` to pass different arguments
to synchronous and asynchronous clients.

#### ``async\_client\_kwargs`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.async_client_kwargs "Copy anchor link to this section for reference")

```
async_client_kwargs: dict | None = {}
```

Additional kwargs to merge with `client_kwargs` before passing to httpx client.

These are clients unique to the async client; for shared args use `client_kwargs`.

For a full list of the params, see the [httpx documentation](https://www.python-httpx.org/api/#asyncclient).

#### ``sync\_client\_kwargs`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.sync_client_kwargs "Copy anchor link to this section for reference")

```
sync_client_kwargs: dict | None = {}
```

Additional kwargs to merge with `client_kwargs` before passing to httpx client.

These are clients unique to the sync client; for shared args use `client_kwargs`.

For a full list of the params, see the [httpx documentation](https://www.python-httpx.org/api/#client).

#### ``mirostat`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.mirostat "Copy anchor link to this section for reference")

```
mirostat: int | None = None
```

Enable Mirostat sampling for controlling perplexity.
(default: `0`, `0` = disabled, `1` = Mirostat, `2` = Mirostat 2.0)

#### ``mirostat\_eta`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.mirostat_eta "Copy anchor link to this section for reference")

```
mirostat_eta: float | None = None
```

Influences how quickly the algorithm responds to feedback
from the generated text. A lower learning rate will result in
slower adjustments, while a higher learning rate will make
the algorithm more responsive. (Default: `0.1`)

#### ``mirostat\_tau`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.mirostat_tau "Copy anchor link to this section for reference")

```
mirostat_tau: float | None = None
```

Controls the balance between coherence and diversity
of the output. A lower value will result in more focused and
coherent text. (Default: `5.0`)

#### ``num\_ctx`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.num_ctx "Copy anchor link to this section for reference")

```
num_ctx: int | None = None
```

Sets the size of the context window used to generate the
next token. (Default: `2048`)

#### ``num\_gpu`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.num_gpu "Copy anchor link to this section for reference")

```
num_gpu: int | None = None
```

The number of GPUs to use. On macOS it defaults to `1` to
enable metal support, `0` to disable.

#### ``keep\_alive`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.keep_alive "Copy anchor link to this section for reference")

```
keep_alive: int | None = None
```

Controls how long the model will stay loaded into memory
following the request (default: `5m`)

#### ``num\_thread`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.num_thread "Copy anchor link to this section for reference")

```
num_thread: int | None = None
```

Sets the number of threads to use during computation.
By default, Ollama will detect this for optimal performance.
It is recommended to set this value to the number of physical
CPU cores your system has (as opposed to the logical number of cores).

#### ``repeat\_last\_n`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.repeat_last_n "Copy anchor link to this section for reference")

```
repeat_last_n: int | None = None
```

Sets how far back for the model to look back to prevent
repetition. (Default: `64`, `0` = disabled, `-1` = `num_ctx`)

#### ``repeat\_penalty`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.repeat_penalty "Copy anchor link to this section for reference")

```
repeat_penalty: float | None = None
```

Sets how strongly to penalize repetitions. A higher value (e.g., `1.5`)
will penalize repetitions more strongly, while a lower value (e.g., `0.9`)
will be more lenient. (Default: `1.1`)

#### ``temperature`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.temperature "Copy anchor link to this section for reference")

```
temperature: float | None = None
```

The temperature of the model. Increasing the temperature will
make the model answer more creatively. (Default: `0.8`)

#### ``stop`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.stop "Copy anchor link to this section for reference")

```
stop: list[str] | None = None
```

Sets the stop tokens to use.

#### ``tfs\_z`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.tfs_z "Copy anchor link to this section for reference")

```
tfs_z: float | None = None
```

Tail free sampling is used to reduce the impact of less probable
tokens from the output. A higher value (e.g., `2.0`) will reduce the
impact more, while a value of `1.0` disables this setting. (default: `1`)

#### ``top\_k`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.top_k "Copy anchor link to this section for reference")

```
top_k: int | None = None
```

Reduces the probability of generating nonsense. A higher value (e.g. `100`)
will give more diverse answers, while a lower value (e.g. `10`)
will be more conservative. (Default: `40`)

#### ``top\_p`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.top_p "Copy anchor link to this section for reference")

```
top_p: float | None = None
```

Works together with top-k. A higher value (e.g., `0.95`) will lead
to more diverse text, while a lower value (e.g., `0.5`) will
generate more focused and conservative text. (Default: `0.9`)

#### ``embed\_documents [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.embed_documents "Copy anchor link to this section for reference")

```
embed_documents(texts: list[str]) -> list[list[float]]
```

Embed search docs.

#### ``embed\_query [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.embed_query "Copy anchor link to this section for reference")

```
embed_query(text: str) -> list[float]
```

Embed query text.

#### ``aembed\_documents`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.aembed_documents "Copy anchor link to this section for reference")

```
aembed_documents(texts: list[str]) -> list[list[float]]
```

Embed search docs.

#### ``aembed\_query`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaEmbeddings.aembed_query "Copy anchor link to this section for reference")

```
aembed_query(text: str) -> list[float]
```

Embed query text.

### ``OllamaLLM [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM "Copy anchor link to this section for reference")

Bases: `BaseLLM`

Ollama large language models.

Setup

Install `langchain-ollama` and install/run the Ollama server locally:

```
pip install -U langchain-ollama
# Visit https://ollama.com/download to download and install Ollama
# (Linux users): start the server with `ollama serve`
```

Download a model to use:

```
ollama pull llama3.1
```

Key init args — generation params:
model: str
Name of the Ollama model to use (e.g. `'llama4'`).
temperature: float \| None
Sampling temperature. Higher values make output more creative.
num\_predict: int \| None
Maximum number of tokens to predict.
top\_k: int \| None
Limits the next token selection to the K most probable tokens.
top\_p: float \| None
Nucleus sampling parameter. Higher values lead to more diverse text.
mirostat: int \| None
Enable Mirostat sampling for controlling perplexity.
seed: int \| None
Random number seed for generation reproducibility.

Key init args — client params:
base\_url:
Base URL where Ollama server is hosted.
keep\_alive:
How long the model stays loaded into memory.
format:
Specify the format of the output.

See full list of supported init args and their descriptions in the params section.

Instantiate

```
from langchain_ollama import OllamaLLM

model = OllamaLLM(
    model="llama3.1",
    temperature=0.7,
    num_predict=256,
    # base_url="http://localhost:11434",
    # other params...
)
```

Invoke

```
input_text = "The meaning of life is "
response = model.invoke(input_text)
print(response)
```

```
"a philosophical question that has been contemplated by humans for
centuries..."
```

Stream

```
for chunk in model.stream(input_text):
    print(chunk, end="")
```

```
a philosophical question that has been contemplated by humans for
centuries...
```

Async

```
response = await model.ainvoke(input_text)

# stream:
# async for chunk in model.astream(input_text):
#     print(chunk, end="")
```

| METHOD | DESCRIPTION |
| --- | --- |
| `get_name` | Get the name of the `Runnable`. |
| `get_input_schema` | Get a Pydantic model that can be used to validate input to the `Runnable`. |
| `get_input_jsonschema` | Get a JSON schema that represents the input to the `Runnable`. |
| `get_output_schema` | Get a Pydantic model that can be used to validate output to the `Runnable`. |
| `get_output_jsonschema` | Get a JSON schema that represents the output of the `Runnable`. |
| `config_schema` | The type of config this `Runnable` accepts specified as a Pydantic model. |
| `get_config_jsonschema` | Get a JSON schema that represents the config of the `Runnable`. |
| `get_graph` | Return a graph representation of this `Runnable`. |
| `get_prompts` | Return a list of prompts used by this `Runnable`. |
| `__or__` | Runnable "or" operator. |
| `__ror__` | Runnable "reverse-or" operator. |
| `pipe` | Pipe `Runnable` objects. |
| `pick` | Pick keys from the output `dict` of this `Runnable`. |
| `assign` | Assigns new fields to the `dict` output of this `Runnable`. |
| `invoke` | Transform a single input into an output. |
| `ainvoke` | Transform a single input into an output. |
| `batch` | Default implementation runs invoke in parallel using a thread pool executor. |
| `batch_as_completed` | Run `invoke` in parallel on a list of inputs. |
| `abatch` | Default implementation runs `ainvoke` in parallel using `asyncio.gather`. |
| `abatch_as_completed` | Run `ainvoke` in parallel on a list of inputs. |
| `stream` | Default implementation of `stream`, which calls `invoke`. |
| `astream` | Default implementation of `astream`, which calls `ainvoke`. |
| `astream_log` | Stream all output from a `Runnable`, as reported to the callback system. |
| `astream_events` | Generate a stream of events. |
| `transform` | Transform inputs to outputs. |
| `atransform` | Transform inputs to outputs. |
| `bind` | Bind arguments to a `Runnable`, returning a new `Runnable`. |
| `with_config` | Bind config to a `Runnable`, returning a new `Runnable`. |
| `with_listeners` | Bind lifecycle listeners to a `Runnable`, returning a new `Runnable`. |
| `with_alisteners` | Bind async lifecycle listeners to a `Runnable`. |
| `with_types` | Bind input and output types to a `Runnable`, returning a new `Runnable`. |
| `with_retry` | Create a new `Runnable` that retries the original `Runnable` on exceptions. |
| `map` | Return a new `Runnable` that maps a list of inputs to a list of outputs. |
| `with_fallbacks` | Add fallbacks to a `Runnable`, returning a new `Runnable`. |
| `as_tool` | Create a `BaseTool` from a `Runnable`. |
| `__init__` |  |
| `is_lc_serializable` | Is this class serializable? |
| `get_lc_namespace` | Get the namespace of the LangChain object. |
| `lc_id` | Return a unique identifier for this class for serialization purposes. |
| `to_json` | Serialize the `Runnable` to JSON. |
| `to_json_not_implemented` | Serialize a "not implemented" object. |
| `configurable_fields` | Configure particular `Runnable` fields at runtime. |
| `configurable_alternatives` | Configure alternatives for `Runnable` objects that can be set at runtime. |
| `set_verbose` | If verbose is `None`, set it. |
| `generate_prompt` | Pass a sequence of prompts to the model and return model generations. |
| `agenerate_prompt` | Asynchronously pass a sequence of prompts and return model generations. |
| `with_structured_output` | Not implemented on this class. |
| `get_token_ids` | Return the ordered IDs of the tokens in a text. |
| `get_num_tokens` | Get the number of tokens present in the text. |
| `get_num_tokens_from_messages` | Get the number of tokens in the messages. |
| `generate` | Pass a sequence of prompts to a model and return generations. |
| `agenerate` | Asynchronously pass a sequence of prompts to a model and return generations. |
| `__str__` | Return a string representation of the object for printing. |
| `dict` | Return a dictionary of the LLM. |
| `save` | Save the LLM. |

#### ``name`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.name "Copy anchor link to this section for reference")

```
name: str | None = None
```

The name of the `Runnable`. Used for debugging and tracing.

#### ``InputType`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.InputType "Copy anchor link to this section for reference")

```
InputType: TypeAlias
```

Get the input type for this `Runnable`.

#### ``OutputType`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.OutputType "Copy anchor link to this section for reference")

```
OutputType: type[str]
```

Get the input type for this `Runnable`.

#### ``input\_schema`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.input_schema "Copy anchor link to this section for reference")

```
input_schema: type[BaseModel]
```

The type of input this `Runnable` accepts specified as a Pydantic model.

#### ``output\_schema`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.output_schema "Copy anchor link to this section for reference")

```
output_schema: type[BaseModel]
```

Output schema.

The type of output this `Runnable` produces specified as a Pydantic model.

#### ``config\_specs`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.config_specs "Copy anchor link to this section for reference")

```
config_specs: list[ConfigurableFieldSpec]
```

List configurable fields for this `Runnable`.

#### ``lc\_secrets`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.lc_secrets "Copy anchor link to this section for reference")

```
lc_secrets: dict[str, str]
```

A map of constructor argument names to secret ids.

For example, `{"openai_api_key": "OPENAI_API_KEY"}`

#### ``lc\_attributes`property`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.lc_attributes "Copy anchor link to this section for reference")

```
lc_attributes: dict
```

List of attribute names that should be included in the serialized kwargs.

These attributes must be accepted by the constructor.

Default is an empty dictionary.

#### ``cache`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.cache "Copy anchor link to this section for reference")

```
cache: BaseCache | bool | None = Field(default=None, exclude=True)
```

Whether to cache the response.

- If `True`, will use the global cache.
- If `False`, will not use a cache
- If `None`, will use the global cache if it's set, otherwise no cache.
- If instance of `BaseCache`, will use the provided cache.

Caching is not currently supported for streaming methods of models.

#### ``verbose`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.verbose "Copy anchor link to this section for reference")

```
verbose: bool = Field(default_factory=_get_verbosity, exclude=True, repr=False)
```

Whether to print out response text.

#### ``callbacks`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.callbacks "Copy anchor link to this section for reference")

```
callbacks: Callbacks = Field(default=None, exclude=True)
```

Callbacks to add to the run trace.

#### ``tags`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.tags "Copy anchor link to this section for reference")

```
tags: list[str] | None = Field(default=None, exclude=True)
```

Tags to add to the run trace.

#### ``metadata`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.metadata "Copy anchor link to this section for reference")

```
metadata: dict[str, Any] | None = Field(default=None, exclude=True)
```

Metadata to add to the run trace.

#### ``custom\_get\_token\_ids`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.custom_get_token_ids "Copy anchor link to this section for reference")

```
custom_get_token_ids: Callable[[str], list[int]] | None = Field(
    default=None, exclude=True
)
```

Optional encoder to use for counting tokens.

#### ``model`instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.model "Copy anchor link to this section for reference")

```
model: str
```

Model name to use.

#### ``reasoning`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.reasoning "Copy anchor link to this section for reference")

```
reasoning: bool | None = None
```

Controls the reasoning/thinking mode for
[supported models](https://ollama.com/search?c=thinking).

- `True`: Enables reasoning mode. The model's reasoning process will be
captured and returned separately in the `additional_kwargs` of the
response message, under `reasoning_content`. The main response
content will not include the reasoning tags.
- `False`: Disables reasoning mode. The model will not perform any reasoning,
and the response will not include any reasoning content.
- `None` (Default): The model will use its default reasoning behavior. If
the model performs reasoning, the `<think>` and `</think>` tags will
be present directly within the main response content.

#### ``validate\_model\_on\_init`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.validate_model_on_init "Copy anchor link to this section for reference")

```
validate_model_on_init: bool = False
```

Whether to validate the model exists in ollama locally on initialization.

Added in `langchain-ollama` 0.3.4

#### ``mirostat`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.mirostat "Copy anchor link to this section for reference")

```
mirostat: int | None = None
```

Enable Mirostat sampling for controlling perplexity.
(default: `0`, `0` = disabled, `1` = Mirostat, `2` = Mirostat 2.0)

#### ``mirostat\_eta`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.mirostat_eta "Copy anchor link to this section for reference")

```
mirostat_eta: float | None = None
```

Influences how quickly the algorithm responds to feedback
from the generated text. A lower learning rate will result in
slower adjustments, while a higher learning rate will make
the algorithm more responsive. (Default: `0.1`)

#### ``mirostat\_tau`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.mirostat_tau "Copy anchor link to this section for reference")

```
mirostat_tau: float | None = None
```

Controls the balance between coherence and diversity
of the output. A lower value will result in more focused and
coherent text. (Default: `5.0`)

#### ``num\_ctx`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.num_ctx "Copy anchor link to this section for reference")

```
num_ctx: int | None = None
```

Sets the size of the context window used to generate the
next token. (Default: `2048`)

#### ``num\_gpu`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.num_gpu "Copy anchor link to this section for reference")

```
num_gpu: int | None = None
```

The number of GPUs to use. On macOS it defaults to `1` to
enable metal support, `0` to disable.

#### ``num\_thread`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.num_thread "Copy anchor link to this section for reference")

```
num_thread: int | None = None
```

Sets the number of threads to use during computation.
By default, Ollama will detect this for optimal performance.
It is recommended to set this value to the number of physical
CPU cores your system has (as opposed to the logical number of cores).

#### ``num\_predict`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.num_predict "Copy anchor link to this section for reference")

```
num_predict: int | None = None
```

Maximum number of tokens to predict when generating text.
(Default: `128`, `-1` = infinite generation, `-2` = fill context)

#### ``repeat\_last\_n`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.repeat_last_n "Copy anchor link to this section for reference")

```
repeat_last_n: int | None = None
```

Sets how far back for the model to look back to prevent
repetition. (Default: `64`, `0` = disabled, `-1` = `num_ctx`)

#### ``repeat\_penalty`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.repeat_penalty "Copy anchor link to this section for reference")

```
repeat_penalty: float | None = None
```

Sets how strongly to penalize repetitions. A higher value (e.g., `1.5`)
will penalize repetitions more strongly, while a lower value (e.g., `0.9`)
will be more lenient. (Default: `1.1`)

#### ``temperature`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.temperature "Copy anchor link to this section for reference")

```
temperature: float | None = None
```

The temperature of the model. Increasing the temperature will
make the model answer more creatively. (Default: `0.8`)

#### ``seed`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.seed "Copy anchor link to this section for reference")

```
seed: int | None = None
```

Sets the random number seed to use for generation. Setting this
to a specific number will make the model generate the same text for
the same prompt.

#### ``stop`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.stop "Copy anchor link to this section for reference")

```
stop: list[str] | None = None
```

Sets the stop tokens to use.

#### ``tfs\_z`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.tfs_z "Copy anchor link to this section for reference")

```
tfs_z: float | None = None
```

Tail free sampling is used to reduce the impact of less probable
tokens from the output. A higher value (e.g., `2.0`) will reduce the
impact more, while a value of 1.0 disables this setting. (default: `1`)

#### ``top\_k`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.top_k "Copy anchor link to this section for reference")

```
top_k: int | None = None
```

Reduces the probability of generating nonsense. A higher value (e.g. `100`)
will give more diverse answers, while a lower value (e.g. `10`)
will be more conservative. (Default: `40`)

#### ``top\_p`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.top_p "Copy anchor link to this section for reference")

```
top_p: float | None = None
```

Works together with top-k. A higher value (e.g., `0.95`) will lead
to more diverse text, while a lower value (e.g., `0.5`) will
generate more focused and conservative text. (Default: `0.9`)

#### ``format`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.format "Copy anchor link to this section for reference")

```
format: Literal['', 'json'] = ''
```

Specify the format of the output (options: `'json'`)

#### ``keep\_alive`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.keep_alive "Copy anchor link to this section for reference")

```
keep_alive: int | str | None = None
```

How long the model will stay loaded into memory.

#### ``base\_url`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.base_url "Copy anchor link to this section for reference")

```
base_url: str | None = None
```

Base url the model is hosted under.

If none, defaults to the Ollama client default.

Supports `userinfo` auth in the format `http://username:password@localhost:11434`.
Useful if your Ollama server is behind a proxy.

Warning

`userinfo` is not secure and should only be used for local testing or
in secure environments. Avoid using it in production or over unsecured
networks.

Note

If using `userinfo`, ensure that the Ollama server is configured to
accept and validate these credentials.

Note

`userinfo` headers are passed to both sync and async clients.

#### ``client\_kwargs`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.client_kwargs "Copy anchor link to this section for reference")

```
client_kwargs: dict | None = {}
```

Additional kwargs to pass to the httpx clients. Pass headers in here.

These arguments are passed to both synchronous and async clients.

Use `sync_client_kwargs` and `async_client_kwargs` to pass different arguments
to synchronous and asynchronous clients.

#### ``async\_client\_kwargs`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.async_client_kwargs "Copy anchor link to this section for reference")

```
async_client_kwargs: dict | None = {}
```

Additional kwargs to merge with `client_kwargs` before passing to httpx client.

These are clients unique to the async client; for shared args use `client_kwargs`.

For a full list of the params, see the [httpx documentation](https://www.python-httpx.org/api/#asyncclient).

#### ``sync\_client\_kwargs`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.sync_client_kwargs "Copy anchor link to this section for reference")

```
sync_client_kwargs: dict | None = {}
```

Additional kwargs to merge with `client_kwargs` before passing to httpx client.

These are clients unique to the sync client; for shared args use `client_kwargs`.

For a full list of the params, see the [httpx documentation](https://www.python-httpx.org/api/#client).

#### ``get\_name [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.get_name "Copy anchor link to this section for reference")

```
get_name(suffix: str | None = None, *, name: str | None = None) -> str
```

Get the name of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `suffix` | An optional suffix to append to the name.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `name` | An optional name to use instead of the `Runnable`'s name.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `str` | The name of the `Runnable`. |

#### ``get\_input\_schema [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.get_input_schema "Copy anchor link to this section for reference")

```
get_input_schema(config: RunnableConfig | None = None) -> type[BaseModel]
```

Get a Pydantic model that can be used to validate input to the `Runnable`.

`Runnable` objects that leverage the `configurable_fields` and
`configurable_alternatives` methods will have a dynamic input schema that
depends on which configuration the `Runnable` is invoked with.

This method allows to get an input schema for a specific configuration.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | A config to use when generating the schema.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `type[BaseModel]` | A Pydantic model that can be used to validate input. |

#### ``get\_input\_jsonschema [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.get_input_jsonschema "Copy anchor link to this section for reference")

```
get_input_jsonschema(config: RunnableConfig | None = None) -> dict[str, Any]
```

Get a JSON schema that represents the input to the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | A config to use when generating the schema.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, Any]` | A JSON schema that represents the input to the `Runnable`. |

Example

```
from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

runnable = RunnableLambda(add_one)

print(runnable.get_input_jsonschema())
```

Added in `langchain-core` 0.3.0

#### ``get\_output\_schema [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.get_output_schema "Copy anchor link to this section for reference")

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

#### ``get\_output\_jsonschema [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.get_output_jsonschema "Copy anchor link to this section for reference")

```
get_output_jsonschema(config: RunnableConfig | None = None) -> dict[str, Any]
```

Get a JSON schema that represents the output of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | A config to use when generating the schema.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, Any]` | A JSON schema that represents the output of the `Runnable`. |

Example

```
from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

runnable = RunnableLambda(add_one)

print(runnable.get_output_jsonschema())
```

Added in `langchain-core` 0.3.0

#### ``config\_schema [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.config_schema "Copy anchor link to this section for reference")

```
config_schema(*, include: Sequence[str] | None = None) -> type[BaseModel]
```

The type of config this `Runnable` accepts specified as a Pydantic model.

To mark a field as configurable, see the `configurable_fields`
and `configurable_alternatives` methods.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `include` | A list of fields to include in the config schema.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `type[BaseModel]` | A Pydantic model that can be used to validate config. |

#### ``get\_config\_jsonschema [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.get_config_jsonschema "Copy anchor link to this section for reference")

```
get_config_jsonschema(*, include: Sequence[str] | None = None) -> dict[str, Any]
```

Get a JSON schema that represents the config of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `include` | A list of fields to include in the config schema.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, Any]` | A JSON schema that represents the config of the `Runnable`. |

Added in `langchain-core` 0.3.0

#### ``get\_graph [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.get_graph "Copy anchor link to this section for reference")

```
get_graph(config: RunnableConfig | None = None) -> Graph
```

Return a graph representation of this `Runnable`.

#### ``get\_prompts [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.get_prompts "Copy anchor link to this section for reference")

```
get_prompts(config: RunnableConfig | None = None) -> list[BasePromptTemplate]
```

Return a list of prompts used by this `Runnable`.

#### ``\_\_or\_\_ [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.__or__ "Copy anchor link to this section for reference")

```
__or__(
    other: Runnable[Any, Other]
    | Callable[[Iterator[Any]], Iterator[Other]]
    | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
    | Callable[[Any], Other]
    | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
) -> RunnableSerializable[Input, Other]
```

Runnable "or" operator.

Compose this `Runnable` with another object to create a
`RunnableSequence`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `other` | Another `Runnable` or a `Runnable`-like object.<br>**TYPE:**`Runnable[Any, Other] | Callable[[Iterator[Any]], Iterator[Other]] | Callable[[AsyncIterator[Any]], AsyncIterator[Other]] | Callable[[Any], Other] | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Input, Other]` | A new `Runnable`. |

#### ``\_\_ror\_\_ [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.__ror__ "Copy anchor link to this section for reference")

```
__ror__(
    other: Runnable[Other, Any]
    | Callable[[Iterator[Other]], Iterator[Any]]
    | Callable[[AsyncIterator[Other]], AsyncIterator[Any]]
    | Callable[[Other], Any]
    | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any] | Any],
) -> RunnableSerializable[Other, Output]
```

Runnable "reverse-or" operator.

Compose this `Runnable` with another object to create a
`RunnableSequence`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `other` | Another `Runnable` or a `Runnable`-like object.<br>**TYPE:**`Runnable[Other, Any] | Callable[[Iterator[Other]], Iterator[Any]] | Callable[[AsyncIterator[Other]], AsyncIterator[Any]] | Callable[[Other], Any] | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any] | Any]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Other, Output]` | A new `Runnable`. |

#### ``pipe [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.pipe "Copy anchor link to this section for reference")

```
pipe(
    *others: Runnable[Any, Other] | Callable[[Any], Other], name: str | None = None
) -> RunnableSerializable[Input, Other]
```

Pipe `Runnable` objects.

Compose this `Runnable` with `Runnable`-like objects to make a
`RunnableSequence`.

Equivalent to `RunnableSequence(self, *others)` or `self | others[0] | ...`

Example

```
from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

def mul_two(x: int) -> int:
    return x * 2

runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)
sequence = runnable_1.pipe(runnable_2)
# Or equivalently:
# sequence = runnable_1 | runnable_2
# sequence = RunnableSequence(first=runnable_1, last=runnable_2)
sequence.invoke(1)
await sequence.ainvoke(1)
# -> 4

sequence.batch([1, 2, 3])
await sequence.abatch([1, 2, 3])
# -> [4, 6, 8]
```

| PARAMETER | DESCRIPTION |
| --- | --- |
| `*others` | Other `Runnable` or `Runnable`-like objects to compose<br>**TYPE:**`Runnable[Any, Other] | Callable[[Any], Other]`**DEFAULT:**`()` |
| `name` | An optional name for the resulting `RunnableSequence`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Input, Other]` | A new `Runnable`. |

#### ``pick [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.pick "Copy anchor link to this section for reference")

```
pick(keys: str | list[str]) -> RunnableSerializable[Any, Any]
```

Pick keys from the output `dict` of this `Runnable`.

Pick a single key

```
import json

from langchain_core.runnables import RunnableLambda, RunnableMap

as_str = RunnableLambda(str)
as_json = RunnableLambda(json.loads)
chain = RunnableMap(str=as_str, json=as_json)

chain.invoke("[1, 2, 3]")
# -> {"str": "[1, 2, 3]", "json": [1, 2, 3]}

json_only_chain = chain.pick("json")
json_only_chain.invoke("[1, 2, 3]")
# -> [1, 2, 3]
```

Pick a list of keys

```
from typing import Any

import json

from langchain_core.runnables import RunnableLambda, RunnableMap

as_str = RunnableLambda(str)
as_json = RunnableLambda(json.loads)

def as_bytes(x: Any) -> bytes:
    return bytes(x, "utf-8")

chain = RunnableMap(
    str=as_str, json=as_json, bytes=RunnableLambda(as_bytes)
)

chain.invoke("[1, 2, 3]")
# -> {"str": "[1, 2, 3]", "json": [1, 2, 3], "bytes": b"[1, 2, 3]"}

json_and_bytes_chain = chain.pick(["json", "bytes"])
json_and_bytes_chain.invoke("[1, 2, 3]")
# -> {"json": [1, 2, 3], "bytes": b"[1, 2, 3]"}
```

| PARAMETER | DESCRIPTION |
| --- | --- |
| `keys` | A key or list of keys to pick from the output dict.<br>**TYPE:**`str | list[str]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Any, Any]` | a new `Runnable`. |

#### ``assign [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.assign "Copy anchor link to this section for reference")

```
assign(
    **kwargs: Runnable[dict[str, Any], Any]
    | Callable[[dict[str, Any]], Any]
    | Mapping[str, Runnable[dict[str, Any], Any] | Callable[[dict[str, Any]], Any]],
) -> RunnableSerializable[Any, Any]
```

Assigns new fields to the `dict` output of this `Runnable`.

```
from langchain_core.language_models.fake import FakeStreamingListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import Runnable
from operator import itemgetter

prompt = (
    SystemMessagePromptTemplate.from_template("You are a nice assistant.")
    + "{question}"
)
model = FakeStreamingListLLM(responses=["foo-lish"])

chain: Runnable = prompt | model | {"str": StrOutputParser()}

chain_with_assign = chain.assign(hello=itemgetter("str") | model)

print(chain_with_assign.input_schema.model_json_schema())
# {'title': 'PromptInput', 'type': 'object', 'properties':
{'question': {'title': 'Question', 'type': 'string'}}}
print(chain_with_assign.output_schema.model_json_schema())
# {'title': 'RunnableSequenceOutput', 'type': 'object', 'properties':
{'str': {'title': 'Str',
'type': 'string'}, 'hello': {'title': 'Hello', 'type': 'string'}}}
```

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | A mapping of keys to `Runnable` or `Runnable`-like objects<br>that will be invoked with the entire output dict of this `Runnable`.<br>**TYPE:**`Runnable[dict[str, Any], Any] | Callable[[dict[str, Any]], Any] | Mapping[str, Runnable[dict[str, Any], Any] | Callable[[dict[str, Any]], Any]]`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Any, Any]` | A new `Runnable`. |

#### ``invoke [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.invoke "Copy anchor link to this section for reference")

```
invoke(
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> str
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

#### ``ainvoke`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.ainvoke "Copy anchor link to this section for reference")

```
ainvoke(
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> str
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

#### ``batch [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.batch "Copy anchor link to this section for reference")

```
batch(
    inputs: list[LanguageModelInput],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[str]
```

Default implementation runs invoke in parallel using a thread pool executor.

The default implementation of batch works well for IO bound runnables.

Subclasses must override this method if they can batch more efficiently;
e.g., if the underlying `Runnable` uses an API which supports a batch mode.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `inputs` | A list of inputs to the `Runnable`.<br>**TYPE:**`list[Input]` |
| `config` | A config to use when invoking the `Runnable`. The config supports<br>standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work<br>to do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | list[RunnableConfig] | None`**DEFAULT:**`None` |
| `return_exceptions` | Whether to return exceptions instead of raising them.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Output]` | A list of outputs from the `Runnable`. |

#### ``batch\_as\_completed [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.batch_as_completed "Copy anchor link to this section for reference")

```
batch_as_completed(
    inputs: Sequence[Input],
    config: RunnableConfig | Sequence[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> Iterator[tuple[int, Output | Exception]]
```

Run `invoke` in parallel on a list of inputs.

Yields results as they complete.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `inputs` | A list of inputs to the `Runnable`.<br>**TYPE:**`Sequence[Input]` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | Sequence[RunnableConfig] | None`**DEFAULT:**`None` |
| `return_exceptions` | Whether to return exceptions instead of raising them.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `tuple[int, Output | Exception]` | Tuples of the index of the input and the output from the `Runnable`. |

#### ``abatch`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.abatch "Copy anchor link to this section for reference")

```
abatch(
    inputs: list[LanguageModelInput],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[str]
```

Default implementation runs `ainvoke` in parallel using `asyncio.gather`.

The default implementation of `batch` works well for IO bound runnables.

Subclasses must override this method if they can batch more efficiently;
e.g., if the underlying `Runnable` uses an API which supports a batch mode.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `inputs` | A list of inputs to the `Runnable`.<br>**TYPE:**`list[Input]` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | list[RunnableConfig] | None`**DEFAULT:**`None` |
| `return_exceptions` | Whether to return exceptions instead of raising them.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Output]` | A list of outputs from the `Runnable`. |

#### ``abatch\_as\_completed`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.abatch_as_completed "Copy anchor link to this section for reference")

```
abatch_as_completed(
    inputs: Sequence[Input],
    config: RunnableConfig | Sequence[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> AsyncIterator[tuple[int, Output | Exception]]
```

Run `ainvoke` in parallel on a list of inputs.

Yields results as they complete.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `inputs` | A list of inputs to the `Runnable`.<br>**TYPE:**`Sequence[Input]` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | Sequence[RunnableConfig] | None`**DEFAULT:**`None` |
| `return_exceptions` | Whether to return exceptions instead of raising them.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[tuple[int, Output | Exception]]` | A tuple of the index of the input and the output from the `Runnable`. |

#### ``stream [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.stream "Copy anchor link to this section for reference")

```
stream(
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> Iterator[str]
```

Default implementation of `stream`, which calls `invoke`.

Subclasses must override this method if they support streaming output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | The config to use for the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

#### ``astream`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.astream "Copy anchor link to this section for reference")

```
astream(
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[str]
```

Default implementation of `astream`, which calls `ainvoke`.

Subclasses must override this method if they support streaming output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | The config to use for the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[Output]` | The output of the `Runnable`. |

#### ``astream\_log`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.astream_log "Copy anchor link to this section for reference")

```
astream_log(
    input: Any,
    config: RunnableConfig | None = None,
    *,
    diff: bool = True,
    with_streamed_output_list: bool = True,
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]
```

Stream all output from a `Runnable`, as reported to the callback system.

This includes all inner runs of LLMs, Retrievers, Tools, etc.

Output is streamed as Log objects, which include a list of
Jsonpatch ops that describe how the state of the run has changed in each
step, and the final state of the run.

The Jsonpatch ops can be applied in order to construct state.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Any` |
| `config` | The config to use for the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `diff` | Whether to yield diffs between each step or the current state.<br>**TYPE:**`bool`**DEFAULT:**`True` |
| `with_streamed_output_list` | Whether to yield the `streamed_output` list.<br>**TYPE:**`bool`**DEFAULT:**`True` |
| `include_names` | Only include logs with these names.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `include_types` | Only include logs with these types.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `include_tags` | Only include logs with these tags.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `exclude_names` | Exclude logs with these names.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `exclude_types` | Exclude logs with these types.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `exclude_tags` | Exclude logs with these tags.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]` | A `RunLogPatch` or `RunLog` object. |

#### ``astream\_events`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.astream_events "Copy anchor link to this section for reference")

```
astream_events(
    input: Any,
    config: RunnableConfig | None = None,
    *,
    version: Literal["v1", "v2"] = "v2",
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]
```

Generate a stream of events.

Use to create an iterator over `StreamEvent` that provide real-time information
about the progress of the `Runnable`, including `StreamEvent` from intermediate
results.

A `StreamEvent` is a dictionary with the following schema:

- `event`: Event names are of the format:
`on_[runnable_type]_(start|stream|end)`.
- `name`: The name of the `Runnable` that generated the event.
- `run_id`: Randomly generated ID associated with the given execution of the
`Runnable` that emitted the event. A child `Runnable` that gets invoked as
part of the execution of a parent `Runnable` is assigned its own unique ID.
- `parent_ids`: The IDs of the parent runnables that generated the event. The
root `Runnable` will have an empty list. The order of the parent IDs is from
the root to the immediate parent. Only available for v2 version of the API.
The v1 version of the API will return an empty list.
- `tags`: The tags of the `Runnable` that generated the event.
- `metadata`: The metadata of the `Runnable` that generated the event.
- `data`: The data associated with the event. The contents of this field
depend on the type of event. See the table below for more details.

Below is a table that illustrates some events that might be emitted by various
chains. Metadata fields have been omitted from the table for brevity.
Chain definitions have been included after the table.

Note

This reference table is for the v2 version of the schema.

| event | name | chunk | input | output |
| --- | --- | --- | --- | --- |
| `on_chat_model_start` | `'[model name]'` |  | `{"messages": [[SystemMessage, HumanMessage]]}` |  |
| `on_chat_model_stream` | `'[model name]'` | `AIMessageChunk(content="hello")` |  |  |
| `on_chat_model_end` | `'[model name]'` |  | `{"messages": [[SystemMessage, HumanMessage]]}` | `AIMessageChunk(content="hello world")` |
| `on_llm_start` | `'[model name]'` |  | `{'input': 'hello'}` |  |
| `on_llm_stream` | `'[model name]'` | `'Hello'` |  |  |
| `on_llm_end` | `'[model name]'` |  | `'Hello human!'` |  |
| `on_chain_start` | `'format_docs'` |  |  |  |
| `on_chain_stream` | `'format_docs'` | `'hello world!, goodbye world!'` |  |  |
| `on_chain_end` | `'format_docs'` |  | `[Document(...)]` | `'hello world!, goodbye world!'` |
| `on_tool_start` | `'some_tool'` |  | `{"x": 1, "y": "2"}` |  |
| `on_tool_end` | `'some_tool'` |  |  | `{"x": 1, "y": "2"}` |
| `on_retriever_start` | `'[retriever name]'` |  | `{"query": "hello"}` |  |
| `on_retriever_end` | `'[retriever name]'` |  | `{"query": "hello"}` | `[Document(...), ..]` |
| `on_prompt_start` | `'[template_name]'` |  | `{"question": "hello"}` |  |
| `on_prompt_end` | `'[template_name]'` |  | `{"question": "hello"}` | `ChatPromptValue(messages: [SystemMessage, ...])` |

In addition to the standard events, users can also dispatch custom events (see example below).

Custom events will be only be surfaced with in the v2 version of the API!

A custom event has following format:

| Attribute | Type | Description |
| --- | --- | --- |
| `name` | `str` | A user defined name for the event. |
| `data` | `Any` | The data associated with the event. This can be anything, though we suggest making it JSON serializable. |

Here are declarations associated with the standard events shown above:

`format_docs`:

```
def format_docs(docs: list[Document]) -> str:
    '''Format the docs.'''
    return ", ".join([doc.page_content for doc in docs])

format_docs = RunnableLambda(format_docs)
```

`some_tool`:

```
@tool
def some_tool(x: int, y: str) -> dict:
    '''Some_tool.'''
    return {"x": x, "y": y}
```

`prompt`:

```
template = ChatPromptTemplate.from_messages(
    [\
        ("system", "You are Cat Agent 007"),\
        ("human", "{question}"),\
    ]
).with_config({"run_name": "my_template", "tags": ["my_template"]})
```

Example

```
from langchain_core.runnables import RunnableLambda

async def reverse(s: str) -> str:
    return s[::-1]

chain = RunnableLambda(func=reverse)

events = [\
    event async for event in chain.astream_events("hello", version="v2")\
]

# Will produce the following events
# (run_id, and parent_ids has been omitted for brevity):
[\
    {\
        "data": {"input": "hello"},\
        "event": "on_chain_start",\
        "metadata": {},\
        "name": "reverse",\
        "tags": [],\
    },\
    {\
        "data": {"chunk": "olleh"},\
        "event": "on_chain_stream",\
        "metadata": {},\
        "name": "reverse",\
        "tags": [],\
    },\
    {\
        "data": {"output": "olleh"},\
        "event": "on_chain_end",\
        "metadata": {},\
        "name": "reverse",\
        "tags": [],\
    },\
]
```

Dispatch custom event

```
from langchain_core.callbacks.manager import (
    adispatch_custom_event,
)
from langchain_core.runnables import RunnableLambda, RunnableConfig
import asyncio

async def slow_thing(some_input: str, config: RunnableConfig) -> str:
    """Do something that takes a long time."""
    await asyncio.sleep(1) # Placeholder for some slow operation
    await adispatch_custom_event(
        "progress_event",
        {"message": "Finished step 1 of 3"},
        config=config # Must be included for python < 3.10
    )
    await asyncio.sleep(1) # Placeholder for some slow operation
    await adispatch_custom_event(
        "progress_event",
        {"message": "Finished step 2 of 3"},
        config=config # Must be included for python < 3.10
    )
    await asyncio.sleep(1) # Placeholder for some slow operation
    return "Done"

slow_thing = RunnableLambda(slow_thing)

async for event in slow_thing.astream_events("some_input", version="v2"):
    print(event)
```

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Any` |
| `config` | The config to use for the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `version` | The version of the schema to use, either `'v2'` or `'v1'`.<br>Users should use `'v2'`.<br>`'v1'` is for backwards compatibility and will be deprecated<br>in `0.4.0`.<br>No default will be assigned until the API is stabilized.<br>custom events will only be surfaced in `'v2'`.<br>**TYPE:**`Literal['v1', 'v2']`**DEFAULT:**`'v2'` |
| `include_names` | Only include events from `Runnable` objects with matching names.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `include_types` | Only include events from `Runnable` objects with matching types.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `include_tags` | Only include events from `Runnable` objects with matching tags.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `exclude_names` | Exclude events from `Runnable` objects with matching names.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `exclude_types` | Exclude events from `Runnable` objects with matching types.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `exclude_tags` | Exclude events from `Runnable` objects with matching tags.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>These will be passed to `astream_log` as this implementation<br>of `astream_events` is built on top of `astream_log`.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[StreamEvent]` | An async stream of `StreamEvent`. |

| RAISES | DESCRIPTION |
| --- | --- |
| `NotImplementedError` | If the version is not `'v1'` or `'v2'`. |

#### ``transform [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.transform "Copy anchor link to this section for reference")

```
transform(
    input: Iterator[Input], config: RunnableConfig | None = None, **kwargs: Any | None
) -> Iterator[Output]
```

Transform inputs to outputs.

Default implementation of transform, which buffers input and calls `astream`.

Subclasses must override this method if they can start producing output while
input is still being generated.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | An iterator of inputs to the `Runnable`.<br>**TYPE:**`Iterator[Input]` |
| `config` | The config to use for the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

#### ``atransform`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.atransform "Copy anchor link to this section for reference")

```
atransform(
    input: AsyncIterator[Input],
    config: RunnableConfig | None = None,
    **kwargs: Any | None,
) -> AsyncIterator[Output]
```

Transform inputs to outputs.

Default implementation of atransform, which buffers input and calls `astream`.

Subclasses must override this method if they can start producing output while
input is still being generated.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | An async iterator of inputs to the `Runnable`.<br>**TYPE:**`AsyncIterator[Input]` |
| `config` | The config to use for the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[Output]` | The output of the `Runnable`. |

#### ``bind [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.bind "Copy anchor link to this section for reference")

```
bind(**kwargs: Any) -> Runnable[Input, Output]
```

Bind arguments to a `Runnable`, returning a new `Runnable`.

Useful when a `Runnable` in a chain requires an argument that is not
in the output of the previous `Runnable` or included in the user input.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | The arguments to bind to the `Runnable`.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` with the arguments bound. |

Example

```
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3.1")

# Without bind
chain = model | StrOutputParser()

chain.invoke("Repeat quoted words exactly: 'One two three four five.'")
# Output is 'One two three four five.'

# With bind
chain = model.bind(stop=["three"]) | StrOutputParser()

chain.invoke("Repeat quoted words exactly: 'One two three four five.'")
# Output is 'One two'
```

#### ``with\_config [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.with_config "Copy anchor link to this section for reference")

```
with_config(
    config: RunnableConfig | None = None, **kwargs: Any
) -> Runnable[Input, Output]
```

Bind config to a `Runnable`, returning a new `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | The config to bind to the `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments to pass to the `Runnable`.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` with the config bound. |

#### ``with\_listeners [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.with_listeners "Copy anchor link to this section for reference")

```
with_listeners(
    *,
    on_start: Callable[[Run], None]
    | Callable[[Run, RunnableConfig], None]
    | None = None,
    on_end: Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None = None,
    on_error: Callable[[Run], None]
    | Callable[[Run, RunnableConfig], None]
    | None = None,
) -> Runnable[Input, Output]
```

Bind lifecycle listeners to a `Runnable`, returning a new `Runnable`.

The Run object contains information about the run, including its `id`,
`type`, `input`, `output`, `error`, `start_time`, `end_time`, and
any tags or metadata added to the run.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `on_start` | Called before the `Runnable` starts running, with the `Run`<br>object.<br>**TYPE:**`Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None`**DEFAULT:**`None` |
| `on_end` | Called after the `Runnable` finishes running, with the `Run`<br>object.<br>**TYPE:**`Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None`**DEFAULT:**`None` |
| `on_error` | Called if the `Runnable` throws an error, with the `Run`<br>object.<br>**TYPE:**`Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` with the listeners bound. |

Example

```
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.schemas import Run

import time

def test_runnable(time_to_sleep: int):
    time.sleep(time_to_sleep)

def fn_start(run_obj: Run):
    print("start_time:", run_obj.start_time)

def fn_end(run_obj: Run):
    print("end_time:", run_obj.end_time)

chain = RunnableLambda(test_runnable).with_listeners(
    on_start=fn_start, on_end=fn_end
)
chain.invoke(2)
```

#### ``with\_alisteners [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.with_alisteners "Copy anchor link to this section for reference")

```
with_alisteners(
    *,
    on_start: AsyncListener | None = None,
    on_end: AsyncListener | None = None,
    on_error: AsyncListener | None = None,
) -> Runnable[Input, Output]
```

Bind async lifecycle listeners to a `Runnable`.

Returns a new `Runnable`.

The Run object contains information about the run, including its `id`,
`type`, `input`, `output`, `error`, `start_time`, `end_time`, and
any tags or metadata added to the run.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `on_start` | Called asynchronously before the `Runnable` starts running,<br>with the `Run` object.<br>**TYPE:**`AsyncListener | None`**DEFAULT:**`None` |
| `on_end` | Called asynchronously after the `Runnable` finishes running,<br>with the `Run` object.<br>**TYPE:**`AsyncListener | None`**DEFAULT:**`None` |
| `on_error` | Called asynchronously if the `Runnable` throws an error,<br>with the `Run` object.<br>**TYPE:**`AsyncListener | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` with the listeners bound. |

Example

```
from langchain_core.runnables import RunnableLambda, Runnable
from datetime import datetime, timezone
import time
import asyncio

def format_t(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

async def test_runnable(time_to_sleep: int):
    print(f"Runnable[{time_to_sleep}s]: starts at {format_t(time.time())}")
    await asyncio.sleep(time_to_sleep)
    print(f"Runnable[{time_to_sleep}s]: ends at {format_t(time.time())}")

async def fn_start(run_obj: Runnable):
    print(f"on start callback starts at {format_t(time.time())}")
    await asyncio.sleep(3)
    print(f"on start callback ends at {format_t(time.time())}")

async def fn_end(run_obj: Runnable):
    print(f"on end callback starts at {format_t(time.time())}")
    await asyncio.sleep(2)
    print(f"on end callback ends at {format_t(time.time())}")

runnable = RunnableLambda(test_runnable).with_alisteners(
    on_start=fn_start, on_end=fn_end
)

async def concurrent_runs():
    await asyncio.gather(runnable.ainvoke(2), runnable.ainvoke(3))

asyncio.run(concurrent_runs())
# Result:
# on start callback starts at 2025-03-01T07:05:22.875378+00:00
# on start callback starts at 2025-03-01T07:05:22.875495+00:00
# on start callback ends at 2025-03-01T07:05:25.878862+00:00
# on start callback ends at 2025-03-01T07:05:25.878947+00:00
# Runnable[2s]: starts at 2025-03-01T07:05:25.879392+00:00
# Runnable[3s]: starts at 2025-03-01T07:05:25.879804+00:00
# Runnable[2s]: ends at 2025-03-01T07:05:27.881998+00:00
# on end callback starts at 2025-03-01T07:05:27.882360+00:00
# Runnable[3s]: ends at 2025-03-01T07:05:28.881737+00:00
# on end callback starts at 2025-03-01T07:05:28.882428+00:00
# on end callback ends at 2025-03-01T07:05:29.883893+00:00
# on end callback ends at 2025-03-01T07:05:30.884831+00:00
```

#### ``with\_types [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.with_types "Copy anchor link to this section for reference")

```
with_types(
    *, input_type: type[Input] | None = None, output_type: type[Output] | None = None
) -> Runnable[Input, Output]
```

Bind input and output types to a `Runnable`, returning a new `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input_type` | The input type to bind to the `Runnable`.<br>**TYPE:**`type[Input] | None`**DEFAULT:**`None` |
| `output_type` | The output type to bind to the `Runnable`.<br>**TYPE:**`type[Output] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` with the types bound. |

#### ``with\_retry [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.with_retry "Copy anchor link to this section for reference")

```
with_retry(
    *,
    retry_if_exception_type: tuple[type[BaseException], ...] = (Exception,),
    wait_exponential_jitter: bool = True,
    exponential_jitter_params: ExponentialJitterParams | None = None,
    stop_after_attempt: int = 3,
) -> Runnable[Input, Output]
```

Create a new `Runnable` that retries the original `Runnable` on exceptions.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `retry_if_exception_type` | A tuple of exception types to retry on.<br>**TYPE:**`tuple[type[BaseException], ...]`**DEFAULT:**`(Exception,)` |
| `wait_exponential_jitter` | Whether to add jitter to the wait<br>time between retries.<br>**TYPE:**`bool`**DEFAULT:**`True` |
| `stop_after_attempt` | The maximum number of attempts to make before<br>giving up.<br>**TYPE:**`int`**DEFAULT:**`3` |
| `exponential_jitter_params` | Parameters for<br>`tenacity.wait_exponential_jitter`. Namely: `initial`, `max`,<br>`exp_base`, and `jitter` (all `float` values).<br>**TYPE:**`ExponentialJitterParams | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` that retries the original `Runnable` on exceptions. |

Example

```
from langchain_core.runnables import RunnableLambda

count = 0

def _lambda(x: int) -> None:
    global count
    count = count + 1
    if x == 1:
        raise ValueError("x is 1")
    else:
        pass

runnable = RunnableLambda(_lambda)
try:
    runnable.with_retry(
        stop_after_attempt=2,
        retry_if_exception_type=(ValueError,),
    ).invoke(1)
except ValueError:
    pass

assert count == 2
```

#### ``map [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.map "Copy anchor link to this section for reference")

```
map() -> Runnable[list[Input], list[Output]]
```

Return a new `Runnable` that maps a list of inputs to a list of outputs.

Calls `invoke` with each input.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[list[Input], list[Output]]` | A new `Runnable` that maps a list of inputs to a list of outputs. |

Example

```
from langchain_core.runnables import RunnableLambda

def _lambda(x: int) -> int:
    return x + 1

runnable = RunnableLambda(_lambda)
print(runnable.map().invoke([1, 2, 3]))  # [2, 3, 4]
```

#### ``with\_fallbacks [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.with_fallbacks "Copy anchor link to this section for reference")

```
with_fallbacks(
    fallbacks: Sequence[Runnable[Input, Output]],
    *,
    exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,),
    exception_key: str | None = None,
) -> RunnableWithFallbacks[Input, Output]
```

Add fallbacks to a `Runnable`, returning a new `Runnable`.

The new `Runnable` will try the original `Runnable`, and then each fallback
in order, upon failures.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `fallbacks` | A sequence of runnables to try if the original `Runnable`<br>fails.<br>**TYPE:**`Sequence[Runnable[Input, Output]]` |
| `exceptions_to_handle` | A tuple of exception types to handle.<br>**TYPE:**`tuple[type[BaseException], ...]`**DEFAULT:**`(Exception,)` |
| `exception_key` | If `string` is specified then handled exceptions will be<br>passed to fallbacks as part of the input under the specified key.<br>If `None`, exceptions will not be passed to fallbacks.<br>If used, the base `Runnable` and its fallbacks must accept a<br>dictionary as input.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableWithFallbacks[Input, Output]` | A new `Runnable` that will try the original `Runnable`, and then each<br>Fallback in order, upon failures. |

Example

```
from typing import Iterator

from langchain_core.runnables import RunnableGenerator

def _generate_immediate_error(input: Iterator) -> Iterator[str]:
    raise ValueError()
    yield ""

def _generate(input: Iterator) -> Iterator[str]:
    yield from "foo bar"

runnable = RunnableGenerator(_generate_immediate_error).with_fallbacks(
    [RunnableGenerator(_generate)]
)
print("".join(runnable.stream({})))  # foo bar
```

| PARAMETER | DESCRIPTION |
| --- | --- |
| `fallbacks` | A sequence of runnables to try if the original `Runnable`<br>fails.<br>**TYPE:**`Sequence[Runnable[Input, Output]]` |
| `exceptions_to_handle` | A tuple of exception types to handle.<br>**TYPE:**`tuple[type[BaseException], ...]`**DEFAULT:**`(Exception,)` |
| `exception_key` | If `string` is specified then handled exceptions will be<br>passed to fallbacks as part of the input under the specified key.<br>If `None`, exceptions will not be passed to fallbacks.<br>If used, the base `Runnable` and its fallbacks must accept a<br>dictionary as input.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableWithFallbacks[Input, Output]` | A new `Runnable` that will try the original `Runnable`, and then each<br>Fallback in order, upon failures. |

#### ``as\_tool [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.as_tool "Copy anchor link to this section for reference")

```
as_tool(
    args_schema: type[BaseModel] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    arg_types: dict[str, type] | None = None,
) -> BaseTool
```

Create a `BaseTool` from a `Runnable`.

`as_tool` will instantiate a `BaseTool` with a name, description, and
`args_schema` from a `Runnable`. Where possible, schemas are inferred
from `runnable.get_input_schema`.

Alternatively (e.g., if the `Runnable` takes a dict as input and the specific
`dict` keys are not typed), the schema can be specified directly with
`args_schema`.

You can also pass `arg_types` to just specify the required arguments and their
types.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `args_schema` | The schema for the tool.<br>**TYPE:**`type[BaseModel] | None`**DEFAULT:**`None` |
| `name` | The name of the tool.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `description` | The description of the tool.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `arg_types` | A dictionary of argument names to types.<br>**TYPE:**`dict[str, type] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `BaseTool` | A `BaseTool` instance. |

`TypedDict` input

```
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableLambda

class Args(TypedDict):
    a: int
    b: list[int]

def f(x: Args) -> str:
    return str(x["a"] * max(x["b"]))

runnable = RunnableLambda(f)
as_tool = runnable.as_tool()
as_tool.invoke({"a": 3, "b": [1, 2]})
```

`dict` input, specifying schema via `args_schema`

```
from typing import Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

def f(x: dict[str, Any]) -> str:
    return str(x["a"] * max(x["b"]))

class FSchema(BaseModel):
    """Apply a function to an integer and list of integers."""

    a: int = Field(..., description="Integer")
    b: list[int] = Field(..., description="List of ints")

runnable = RunnableLambda(f)
as_tool = runnable.as_tool(FSchema)
as_tool.invoke({"a": 3, "b": [1, 2]})
```

`dict` input, specifying schema via `arg_types`

```
from typing import Any
from langchain_core.runnables import RunnableLambda

def f(x: dict[str, Any]) -> str:
    return str(x["a"] * max(x["b"]))

runnable = RunnableLambda(f)
as_tool = runnable.as_tool(arg_types={"a": int, "b": list[int]})
as_tool.invoke({"a": 3, "b": [1, 2]})
```

`str` input

```
from langchain_core.runnables import RunnableLambda

def f(x: str) -> str:
    return x + "a"

def g(x: str) -> str:
    return x + "z"

runnable = RunnableLambda(f) | g
as_tool = runnable.as_tool()
as_tool.invoke("b")
```

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.__init__ "Copy anchor link to this section for reference")

```
__init__(*args: Any, **kwargs: Any) -> None
```

#### ``is\_lc\_serializable`classmethod`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.is_lc_serializable "Copy anchor link to this section for reference")

```
is_lc_serializable() -> bool
```

Is this class serializable?

By design, even if a class inherits from `Serializable`, it is not serializable
by default. This is to prevent accidental serialization of objects that should
not be serialized.

| RETURNS | DESCRIPTION |
| --- | --- |
| `bool` | Whether the class is serializable. Default is `False`. |

#### ``get\_lc\_namespace`classmethod`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.get_lc_namespace "Copy anchor link to this section for reference")

```
get_lc_namespace() -> list[str]
```

Get the namespace of the LangChain object.

For example, if the class is
[`langchain.llms.openai.OpenAI`](https://reference.langchain.com/python/integrations/langchain_openai/OpenAI/#langchain_openai.OpenAI "<code class=\"doc-symbol doc-symbol-heading doc-symbol-class\"></code>            <span class=\"doc doc-object-name doc-class-name\">OpenAI</span>"), then the namespace is
`["langchain", "llms", "openai"]`

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | The namespace. |

#### ``lc\_id`classmethod`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.lc_id "Copy anchor link to this section for reference")

```
lc_id() -> list[str]
```

Return a unique identifier for this class for serialization purposes.

The unique identifier is a list of strings that describes the path
to the object.

For example, for the class `langchain.llms.openai.OpenAI`, the id is
`["langchain", "llms", "openai", "OpenAI"]`.

#### ``to\_json [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.to_json "Copy anchor link to this section for reference")

```
to_json() -> SerializedConstructor | SerializedNotImplemented
```

Serialize the `Runnable` to JSON.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedConstructor | SerializedNotImplemented` | A JSON-serializable representation of the `Runnable`. |

#### ``to\_json\_not\_implemented [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.to_json_not_implemented "Copy anchor link to this section for reference")

```
to_json_not_implemented() -> SerializedNotImplemented
```

Serialize a "not implemented" object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedNotImplemented` | `SerializedNotImplemented`. |

#### ``configurable\_fields [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.configurable_fields "Copy anchor link to this section for reference")

```
configurable_fields(
    **kwargs: AnyConfigurableField,
) -> RunnableSerializable[Input, Output]
```

Configure particular `Runnable` fields at runtime.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | A dictionary of `ConfigurableField` instances to configure.<br>**TYPE:**`AnyConfigurableField`**DEFAULT:**`{}` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If a configuration key is not found in the `Runnable`. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Input, Output]` | A new `Runnable` with the fields configured. |

Example

```
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatOpenAI(max_tokens=20).configurable_fields(
    max_tokens=ConfigurableField(
        id="output_token_number",
        name="Max tokens in the output",
        description="The maximum number of tokens in the output",
    )
)

# max_tokens = 20
print(
    "max_tokens_20: ", model.invoke("tell me something about chess").content
)

# max_tokens = 200
print(
    "max_tokens_200: ",
    model.with_config(configurable={"output_token_number": 200})
    .invoke("tell me something about chess")
    .content,
)
```

#### ``configurable\_alternatives [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.configurable_alternatives "Copy anchor link to this section for reference")

```
configurable_alternatives(
    which: ConfigurableField,
    *,
    default_key: str = "default",
    prefix_keys: bool = False,
    **kwargs: Runnable[Input, Output] | Callable[[], Runnable[Input, Output]],
) -> RunnableSerializable[Input, Output]
```

Configure alternatives for `Runnable` objects that can be set at runtime.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `which` | The `ConfigurableField` instance that will be used to select the<br>alternative.<br>**TYPE:**`ConfigurableField` |
| `default_key` | The default key to use if no alternative is selected.<br>**TYPE:**`str`**DEFAULT:**`'default'` |
| `prefix_keys` | Whether to prefix the keys with the `ConfigurableField` id.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `**kwargs` | A dictionary of keys to `Runnable` instances or callables that<br>return `Runnable` instances.<br>**TYPE:**`Runnable[Input, Output] | Callable[[], Runnable[Input, Output]]`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RunnableSerializable[Input, Output]` | A new `Runnable` with the alternatives configured. |

Example

```
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables.utils import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatAnthropic(
    model_name="claude-sonnet-4-5-20250929"
).configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="anthropic",
    openai=ChatOpenAI(),
)

# uses the default model ChatAnthropic
print(model.invoke("which organization created you?").content)

# uses ChatOpenAI
print(
    model.with_config(configurable={"llm": "openai"})
    .invoke("which organization created you?")
    .content
)
```

#### ``set\_verbose [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.set_verbose "Copy anchor link to this section for reference")

```
set_verbose(verbose: bool | None) -> bool
```

If verbose is `None`, set it.

This allows users to pass in `None` as verbose to access the global setting.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `verbose` | The verbosity setting to use.<br>**TYPE:**`bool | None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `bool` | The verbosity setting to use. |

#### ``generate\_prompt [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.generate_prompt "Copy anchor link to this section for reference")

```
generate_prompt(
    prompts: list[PromptValue],
    stop: list[str] | None = None,
    callbacks: Callbacks | list[Callbacks] | None = None,
    **kwargs: Any,
) -> LLMResult
```

Pass a sequence of prompts to the model and return model generations.

This method should make use of batched calls for models that expose a batched
API.

Use this method when you want to:

1. Take advantage of batched calls,
2. Need more output from the model than just the top generated value,
3. Are building chains that are agnostic to the underlying language model
    type (e.g., pure text completion models vs chat models).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompts` | List of `PromptValue` objects.<br>A `PromptValue` is an object that can be converted to match the format<br>of any language model (string for pure text generation models and<br>`BaseMessage` objects for chat models).<br>**TYPE:**`list[PromptValue]` |
| `stop` | Stop words to use when generating.<br>Model output is cut off at the first occurrence of any of these<br>substrings.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `callbacks` | `Callbacks` to pass through.<br>Used for executing additional functionality, such as logging or<br>streaming, throughout generation.<br>**TYPE:**`Callbacks`**DEFAULT:**`None` |
| `**kwargs` | Arbitrary additional keyword arguments.<br>These are usually passed to the model provider API call.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `LLMResult` | An `LLMResult`, which contains a list of candidate `Generation` objects for<br>each input prompt and additional model provider-specific output. |

#### ``agenerate\_prompt`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.agenerate_prompt "Copy anchor link to this section for reference")

```
agenerate_prompt(
    prompts: list[PromptValue],
    stop: list[str] | None = None,
    callbacks: Callbacks | list[Callbacks] | None = None,
    **kwargs: Any,
) -> LLMResult
```

Asynchronously pass a sequence of prompts and return model generations.

This method should make use of batched calls for models that expose a batched
API.

Use this method when you want to:

1. Take advantage of batched calls,
2. Need more output from the model than just the top generated value,
3. Are building chains that are agnostic to the underlying language model
    type (e.g., pure text completion models vs chat models).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompts` | List of `PromptValue` objects.<br>A `PromptValue` is an object that can be converted to match the format<br>of any language model (string for pure text generation models and<br>`BaseMessage` objects for chat models).<br>**TYPE:**`list[PromptValue]` |
| `stop` | Stop words to use when generating.<br>Model output is cut off at the first occurrence of any of these<br>substrings.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `callbacks` | `Callbacks` to pass through.<br>Used for executing additional functionality, such as logging or<br>streaming, throughout generation.<br>**TYPE:**`Callbacks`**DEFAULT:**`None` |
| `**kwargs` | Arbitrary additional keyword arguments.<br>These are usually passed to the model provider API call.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `LLMResult` | An `LLMResult`, which contains a list of candidate `Generation` objects for<br>each input prompt and additional model provider-specific output. |

#### ``with\_structured\_output [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.with_structured_output "Copy anchor link to this section for reference")

```
with_structured_output(
    schema: dict | type, **kwargs: Any
) -> Runnable[LanguageModelInput, dict | BaseModel]
```

Not implemented on this class.

#### ``get\_token\_ids [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.get_token_ids "Copy anchor link to this section for reference")

```
get_token_ids(text: str) -> list[int]
```

Return the ordered IDs of the tokens in a text.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The string input to tokenize.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[int]` | A list of IDs corresponding to the tokens in the text, in order they occur<br>in the text. |

#### ``get\_num\_tokens [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.get_num_tokens "Copy anchor link to this section for reference")

```
get_num_tokens(text: str) -> int
```

Get the number of tokens present in the text.

Useful for checking if an input fits in a model's context window.

This should be overridden by model-specific implementations to provide accurate
token counts via model-specific tokenizers.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The string input to tokenize.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `int` | The integer number of tokens in the text. |

#### ``get\_num\_tokens\_from\_messages [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.get_num_tokens_from_messages "Copy anchor link to this section for reference")

```
get_num_tokens_from_messages(
    messages: list[BaseMessage], tools: Sequence | None = None
) -> int
```

Get the number of tokens in the messages.

Useful for checking if an input fits in a model's context window.

This should be overridden by model-specific implementations to provide accurate
token counts via model-specific tokenizers.

Note

- The base implementation of `get_num_tokens_from_messages` ignores tool
schemas.
- The base implementation of `get_num_tokens_from_messages` adds additional
prefixes to messages in represent user roles, which will add to the
overall token count. Model-specific implementations may choose to
handle this differently.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `messages` | The message inputs to tokenize.<br>**TYPE:**`list[BaseMessage]` |
| `tools` | If provided, sequence of dict, `BaseModel`, function, or<br>`BaseTool` objects to be converted to tool schemas.<br>**TYPE:**`Sequence | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `int` | The sum of the number of tokens across the messages. |

#### ``generate [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.generate "Copy anchor link to this section for reference")

```
generate(
    prompts: list[str],
    stop: list[str] | None = None,
    callbacks: Callbacks | list[Callbacks] | None = None,
    *,
    tags: list[str] | list[list[str]] | None = None,
    metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
    run_name: str | list[str] | None = None,
    run_id: UUID | list[UUID | None] | None = None,
    **kwargs: Any,
) -> LLMResult
```

Pass a sequence of prompts to a model and return generations.

This method should make use of batched calls for models that expose a batched
API.

Use this method when you want to:

1. Take advantage of batched calls,
2. Need more output from the model than just the top generated value,
3. Are building chains that are agnostic to the underlying language model
    type (e.g., pure text completion models vs chat models).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompts` | List of string prompts.<br>**TYPE:**`list[str]` |
| `stop` | Stop words to use when generating.<br>Model output is cut off at the first occurrence of any of these<br>substrings.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `callbacks` | `Callbacks` to pass through.<br>Used for executing additional functionality, such as logging or<br>streaming, throughout generation.<br>**TYPE:**`Callbacks | list[Callbacks] | None`**DEFAULT:**`None` |
| `tags` | List of tags to associate with each prompt. If provided, the length<br>of the list must match the length of the prompts list.<br>**TYPE:**`list[str] | list[list[str]] | None`**DEFAULT:**`None` |
| `metadata` | List of metadata dictionaries to associate with each prompt. If<br>provided, the length of the list must match the length of the prompts<br>list.<br>**TYPE:**`dict[str, Any] | list[dict[str, Any]] | None`**DEFAULT:**`None` |
| `run_name` | List of run names to associate with each prompt. If provided, the<br>length of the list must match the length of the prompts list.<br>**TYPE:**`str | list[str] | None`**DEFAULT:**`None` |
| `run_id` | List of run IDs to associate with each prompt. If provided, the<br>length of the list must match the length of the prompts list.<br>**TYPE:**`UUID | list[UUID | None] | None`**DEFAULT:**`None` |
| `**kwargs` | Arbitrary additional keyword arguments.<br>These are usually passed to the model provider API call.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If prompts is not a list. |
| `ValueError` | If the length of `callbacks`, `tags`, `metadata`, or<br>`run_name` (if provided) does not match the length of prompts. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `LLMResult` | An `LLMResult`, which contains a list of candidate `Generations` for each<br>input prompt and additional model provider-specific output. |

#### ``agenerate`async`[¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.agenerate "Copy anchor link to this section for reference")

```
agenerate(
    prompts: list[str],
    stop: list[str] | None = None,
    callbacks: Callbacks | list[Callbacks] | None = None,
    *,
    tags: list[str] | list[list[str]] | None = None,
    metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
    run_name: str | list[str] | None = None,
    run_id: UUID | list[UUID | None] | None = None,
    **kwargs: Any,
) -> LLMResult
```

Asynchronously pass a sequence of prompts to a model and return generations.

This method should make use of batched calls for models that expose a batched
API.

Use this method when you want to:

1. Take advantage of batched calls,
2. Need more output from the model than just the top generated value,
3. Are building chains that are agnostic to the underlying language model
    type (e.g., pure text completion models vs chat models).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompts` | List of string prompts.<br>**TYPE:**`list[str]` |
| `stop` | Stop words to use when generating.<br>Model output is cut off at the first occurrence of any of these<br>substrings.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `callbacks` | `Callbacks` to pass through.<br>Used for executing additional functionality, such as logging or<br>streaming, throughout generation.<br>**TYPE:**`Callbacks | list[Callbacks] | None`**DEFAULT:**`None` |
| `tags` | List of tags to associate with each prompt. If provided, the length<br>of the list must match the length of the prompts list.<br>**TYPE:**`list[str] | list[list[str]] | None`**DEFAULT:**`None` |
| `metadata` | List of metadata dictionaries to associate with each prompt. If<br>provided, the length of the list must match the length of the prompts<br>list.<br>**TYPE:**`dict[str, Any] | list[dict[str, Any]] | None`**DEFAULT:**`None` |
| `run_name` | List of run names to associate with each prompt. If provided, the<br>length of the list must match the length of the prompts list.<br>**TYPE:**`str | list[str] | None`**DEFAULT:**`None` |
| `run_id` | List of run IDs to associate with each prompt. If provided, the<br>length of the list must match the length of the prompts list.<br>**TYPE:**`UUID | list[UUID | None] | None`**DEFAULT:**`None` |
| `**kwargs` | Arbitrary additional keyword arguments.<br>These are usually passed to the model provider API call.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the length of `callbacks`, `tags`, `metadata`, or<br>`run_name` (if provided) does not match the length of prompts. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `LLMResult` | An `LLMResult`, which contains a list of candidate `Generations` for each<br>input prompt and additional model provider-specific output. |

#### ``\_\_str\_\_ [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.__str__ "Copy anchor link to this section for reference")

```
__str__() -> str
```

Return a string representation of the object for printing.

#### ``dict [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.dict "Copy anchor link to this section for reference")

```
dict(**kwargs: Any) -> dict
```

Return a dictionary of the LLM.

#### ``save [¶](https://reference.langchain.com/python/integrations/langchain_ollama/\#langchain_ollama.OllamaLLM.save "Copy anchor link to this section for reference")

```
save(file_path: Path | str) -> None
```

Save the LLM.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `file_path` | Path to file to save the LLM to.<br>**TYPE:**`Path | str` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the file path is not a string or Path object. |

Example

```
llm.save(file_path="path/llm.yaml")
```

Back to top