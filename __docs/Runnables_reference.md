[Skip to content](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.base.Runnable)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_core/runnables.md "Edit this page")

# Runnables

## ``Runnable [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable "Copy anchor link to this section for reference")

Bases: `ABC`, `Generic[Input, Output]`

A unit of work that can be invoked, batched, streamed, transformed and composed.

### Key Methods [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable--key-methods "Copy anchor link to this section for reference")

- `invoke`/`ainvoke`: Transforms a single input into an output.
- `batch`/`abatch`: Efficiently transforms multiple inputs into outputs.
- `stream`/`astream`: Streams output from a single input as it's produced.
- `astream_log`: Streams output and selected intermediate results from an
input.

Built-in optimizations:

- **Batch**: By default, batch runs invoke() in parallel using a thread pool
executor. Override to optimize batching.

- **Async**: Methods with `'a'` suffix are asynchronous. By default, they execute
the sync counterpart using asyncio's thread pool.
Override for native async.


All methods accept an optional config argument, which can be used to configure
execution, add tags and metadata for tracing and debugging etc.

Runnables expose schematic information about their input, output and config via
the `input_schema` property, the `output_schema` property and `config_schema`
method.

### Composition [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable--composition "Copy anchor link to this section for reference")

Runnable objects can be composed together to create chains in a declarative way.

Any chain constructed this way will automatically have sync, async, batch, and
streaming support.

The main composition primitives are `RunnableSequence` and `RunnableParallel`.

**`RunnableSequence`** invokes a series of runnables sequentially, with
one Runnable's output serving as the next's input. Construct using
the `|` operator or by passing a list of runnables to `RunnableSequence`.

**`RunnableParallel`** invokes runnables concurrently, providing the same input
to each. Construct it using a dict literal within a sequence or by passing a
dict to `RunnableParallel`.

For example,

```
from langchain_core.runnables import RunnableLambda

# A RunnableSequence constructed using the `|` operator
sequence = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
sequence.invoke(1)  # 4
sequence.batch([1, 2, 3])  # [4, 6, 8]

# A sequence that contains a RunnableParallel constructed using a dict literal
sequence = RunnableLambda(lambda x: x + 1) | {
    "mul_2": RunnableLambda(lambda x: x * 2),
    "mul_5": RunnableLambda(lambda x: x * 5),
}
sequence.invoke(1)  # {'mul_2': 4, 'mul_5': 10}
```

### Standard Methods [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable--standard-methods "Copy anchor link to this section for reference")

All `Runnable`s expose additional methods that can be used to modify their
behavior (e.g., add a retry policy, add lifecycle listeners, make them
configurable, etc.).

These methods will work on any `Runnable`, including `Runnable` chains
constructed by composing other `Runnable`s.
See the individual methods for details.

For example,

```
from langchain_core.runnables import RunnableLambda

import random

def add_one(x: int) -> int:
    return x + 1

def buggy_double(y: int) -> int:
    """Buggy code that will fail 70% of the time"""
    if random.random() > 0.3:
        print('This code failed, and will probably be retried!')  # noqa: T201
        raise ValueError('Triggered buggy code')
    return y * 2

sequence = (
    RunnableLambda(add_one) |
    RunnableLambda(buggy_double).with_retry( # Retry on failure
        stop_after_attempt=10,
        wait_exponential_jitter=False
    )
)

print(sequence.input_schema.model_json_schema()) # Show inferred input schema
print(sequence.output_schema.model_json_schema()) # Show inferred output schema
print(sequence.invoke(2)) # invoke the sequence (note the retry above!!)
```

### Debugging and tracing [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable--debugging-and-tracing "Copy anchor link to this section for reference")

As the chains get longer, it can be useful to be able to see intermediate results
to debug and trace the chain.

You can set the global debug flag to True to enable debug output for all chains:

```
from langchain_core.globals import set_debug

set_debug(True)
```

Alternatively, you can pass existing or custom callbacks to any given chain:

```
from langchain_core.tracers import ConsoleCallbackHandler

chain.invoke(..., config={"callbacks": [ConsoleCallbackHandler()]})
```

For a UI (and much more) checkout [LangSmith](https://docs.langchain.com/langsmith/home).

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

### ``name`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.name "Copy anchor link to this section for reference")

```
name: str | None
```

The name of the `Runnable`. Used for debugging and tracing.

### ``InputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.InputType "Copy anchor link to this section for reference")

```
InputType: type[Input]
```

Input type.

The type of input this `Runnable` accepts specified as a type annotation.

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the input type cannot be inferred. |

### ``OutputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.OutputType "Copy anchor link to this section for reference")

```
OutputType: type[Output]
```

Output Type.

The type of output this `Runnable` produces specified as a type annotation.

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the output type cannot be inferred. |

### ``input\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.input_schema "Copy anchor link to this section for reference")

```
input_schema: type[BaseModel]
```

The type of input this `Runnable` accepts specified as a Pydantic model.

### ``output\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.output_schema "Copy anchor link to this section for reference")

```
output_schema: type[BaseModel]
```

Output schema.

The type of output this `Runnable` produces specified as a Pydantic model.

### ``config\_specs`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.config_specs "Copy anchor link to this section for reference")

```
config_specs: list[ConfigurableFieldSpec]
```

List configurable fields for this `Runnable`.

### ``get\_name [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.get_name "Copy anchor link to this section for reference")

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

### ``get\_input\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.get_input_schema "Copy anchor link to this section for reference")

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

### ``get\_input\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.get_input_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_output\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.get_output_schema "Copy anchor link to this section for reference")

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

### ``get\_output\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.get_output_jsonschema "Copy anchor link to this section for reference")

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

### ``config\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.config_schema "Copy anchor link to this section for reference")

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

### ``get\_config\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.get_config_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_graph [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.get_graph "Copy anchor link to this section for reference")

```
get_graph(config: RunnableConfig | None = None) -> Graph
```

Return a graph representation of this `Runnable`.

### ``get\_prompts [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.get_prompts "Copy anchor link to this section for reference")

```
get_prompts(config: RunnableConfig | None = None) -> list[BasePromptTemplate]
```

Return a list of prompts used by this `Runnable`.

### ``\_\_or\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.__or__ "Copy anchor link to this section for reference")

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

### ``\_\_ror\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.__ror__ "Copy anchor link to this section for reference")

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

### ``pipe [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.pipe "Copy anchor link to this section for reference")

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

### ``pick [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.pick "Copy anchor link to this section for reference")

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

### ``assign [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.assign "Copy anchor link to this section for reference")

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

### ``invoke`abstractmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.invoke "Copy anchor link to this section for reference")

```
invoke(input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Output
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``ainvoke`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.ainvoke "Copy anchor link to this section for reference")

```
ainvoke(input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Output
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``batch [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.batch "Copy anchor link to this section for reference")

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

### ``batch\_as\_completed [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.batch_as_completed "Copy anchor link to this section for reference")

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

### ``abatch`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.abatch "Copy anchor link to this section for reference")

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

### ``abatch\_as\_completed`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.abatch_as_completed "Copy anchor link to this section for reference")

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

### ``stream [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.stream "Copy anchor link to this section for reference")

```
stream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> Iterator[Output]
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

### ``astream`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.astream "Copy anchor link to this section for reference")

```
astream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> AsyncIterator[Output]
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

### ``astream\_log`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.astream_log "Copy anchor link to this section for reference")

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

### ``astream\_events`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.astream_events "Copy anchor link to this section for reference")

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

### ``transform [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.transform "Copy anchor link to this section for reference")

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

### ``atransform`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.atransform "Copy anchor link to this section for reference")

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

### ``bind [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.bind "Copy anchor link to this section for reference")

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

### ``with\_config [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.with_config "Copy anchor link to this section for reference")

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

### ``with\_listeners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.with_listeners "Copy anchor link to this section for reference")

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

### ``with\_alisteners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.with_alisteners "Copy anchor link to this section for reference")

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

### ``with\_types [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.with_types "Copy anchor link to this section for reference")

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

### ``with\_retry [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.with_retry "Copy anchor link to this section for reference")

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

### ``map [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.map "Copy anchor link to this section for reference")

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

### ``with\_fallbacks [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.with_fallbacks "Copy anchor link to this section for reference")

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

### ``as\_tool [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.Runnable.as_tool "Copy anchor link to this section for reference")

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

## ``RunnableBinding [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding "Copy anchor link to this section for reference")

Bases: `RunnableBindingBase[Input, Output]`

Wrap a `Runnable` with additional functionality.

A `RunnableBinding` can be thought of as a "runnable decorator" that
preserves the essential features of `Runnable`; i.e., batching, streaming,
and async support, while adding additional functionality.

Any class that inherits from `Runnable` can be bound to a `RunnableBinding`.
Runnables expose a standard set of methods for creating `RunnableBindings`
or sub-classes of `RunnableBindings` (e.g., `RunnableRetry`,
`RunnableWithFallbacks`) that add additional functionality.

These methods include:

- `bind`: Bind kwargs to pass to the underlying `Runnable` when running it.
- `with_config`: Bind config to pass to the underlying `Runnable` when running
it.
- `with_listeners`: Bind lifecycle listeners to the underlying `Runnable`.
- `with_types`: Override the input and output types of the underlying
`Runnable`.
- `with_retry`: Bind a retry policy to the underlying `Runnable`.
- `with_fallbacks`: Bind a fallback policy to the underlying `Runnable`.

Example:
`bind`: Bind kwargs to pass to the underlying `Runnable` when running it.

````
```python
# Create a Runnable binding that invokes the chat model with the
# additional kwarg `stop=['-']` when running it.
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
model.invoke('Say "Parrot-MAGIC"', stop=["-"])  # Should return `Parrot`
# Using it the easy way via `bind` method which returns a new
# RunnableBinding
runnable_binding = model.bind(stop=["-"])
runnable_binding.invoke('Say "Parrot-MAGIC"')  # Should return `Parrot`
```
Can also be done by instantiating a `RunnableBinding` directly (not
recommended):

```python
from langchain_core.runnables import RunnableBinding

runnable_binding = RunnableBinding(
    bound=model,
    kwargs={"stop": ["-"]},  # <-- Note the additional kwargs
)
runnable_binding.invoke('Say "Parrot-MAGIC"')  # Should return `Parrot`
```
````

| METHOD | DESCRIPTION |
| --- | --- |
| `bind` | Bind additional kwargs to a `Runnable`, returning a new `Runnable`. |
| `with_config` | Bind config to a `Runnable`, returning a new `Runnable`. |
| `with_listeners` | Bind lifecycle listeners to a `Runnable`, returning a new `Runnable`. |
| `with_types` | Bind input and output types to a `Runnable`, returning a new `Runnable`. |
| `with_retry` | Create a new `Runnable` that retries the original `Runnable` on exceptions. |
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
| `with_alisteners` | Bind async lifecycle listeners to a `Runnable`. |
| `map` | Return a new `Runnable` that maps a list of inputs to a list of outputs. |
| `with_fallbacks` | Add fallbacks to a `Runnable`, returning a new `Runnable`. |
| `as_tool` | Create a `BaseTool` from a `Runnable`. |
| `__init__` | Create a `RunnableBinding` from a `Runnable` and kwargs. |
| `is_lc_serializable` | Return `True` as this class is serializable. |
| `get_lc_namespace` | Get the namespace of the LangChain object. |
| `lc_id` | Return a unique identifier for this class for serialization purposes. |
| `to_json` | Serialize the `Runnable` to JSON. |
| `to_json_not_implemented` | Serialize a "not implemented" object. |
| `configurable_fields` | Configure particular `Runnable` fields at runtime. |
| `configurable_alternatives` | Configure alternatives for `Runnable` objects that can be set at runtime. |

### ``name`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.name "Copy anchor link to this section for reference")

```
name: str | None = None
```

The name of the `Runnable`. Used for debugging and tracing.

### ``InputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.InputType "Copy anchor link to this section for reference")

```
InputType: type[Input]
```

Input type.

The type of input this `Runnable` accepts specified as a type annotation.

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the input type cannot be inferred. |

### ``OutputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.OutputType "Copy anchor link to this section for reference")

```
OutputType: type[Output]
```

Output Type.

The type of output this `Runnable` produces specified as a type annotation.

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the output type cannot be inferred. |

### ``input\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.input_schema "Copy anchor link to this section for reference")

```
input_schema: type[BaseModel]
```

The type of input this `Runnable` accepts specified as a Pydantic model.

### ``output\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.output_schema "Copy anchor link to this section for reference")

```
output_schema: type[BaseModel]
```

Output schema.

The type of output this `Runnable` produces specified as a Pydantic model.

### ``config\_specs`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.config_specs "Copy anchor link to this section for reference")

```
config_specs: list[ConfigurableFieldSpec]
```

List configurable fields for this `Runnable`.

### ``lc\_secrets`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.lc_secrets "Copy anchor link to this section for reference")

```
lc_secrets: dict[str, str]
```

A map of constructor argument names to secret ids.

For example, `{"openai_api_key": "OPENAI_API_KEY"}`

### ``lc\_attributes`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.lc_attributes "Copy anchor link to this section for reference")

```
lc_attributes: dict
```

List of attribute names that should be included in the serialized kwargs.

These attributes must be accepted by the constructor.

Default is an empty dictionary.

### ``bound`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.bound "Copy anchor link to this section for reference")

```
bound: Runnable[Input, Output]
```

The underlying `Runnable` that this `Runnable` delegates to.

### ``kwargs`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.kwargs "Copy anchor link to this section for reference")

```
kwargs: Mapping[str, Any] = Field(default_factory=dict)
```

kwargs to pass to the underlying `Runnable` when running.

For example, when the `Runnable` binding is invoked the underlying
`Runnable` will be invoked with the same input but with these additional
kwargs.

### ``config`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.config "Copy anchor link to this section for reference")

```
config: RunnableConfig = config or {}
```

The config to bind to the underlying `Runnable`.

### ``config\_factories`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.config_factories "Copy anchor link to this section for reference")

```
config_factories: list[Callable[[RunnableConfig], RunnableConfig]] = Field(
    default_factory=list
)
```

The config factories to bind to the underlying `Runnable`.

### ``custom\_input\_type`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.custom_input_type "Copy anchor link to this section for reference")

```
custom_input_type: Any | None = None
```

Override the input type of the underlying `Runnable` with a custom type.

The type can be a Pydantic model, or a type annotation (e.g., `list[str]`).

### ``custom\_output\_type`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.custom_output_type "Copy anchor link to this section for reference")

```
custom_output_type: Any | None = None
```

Override the output type of the underlying `Runnable` with a custom type.

The type can be a Pydantic model, or a type annotation (e.g., `list[str]`).

### ``bind [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.bind "Copy anchor link to this section for reference")

```
bind(**kwargs: Any) -> Runnable[Input, Output]
```

Bind additional kwargs to a `Runnable`, returning a new `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | The kwargs to bind to the `Runnable`.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Runnable[Input, Output]` | A new `Runnable` with the same type and config as the original, |
| `Runnable[Input, Output]` | but with the additional kwargs bound. |

### ``with\_config [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.with_config "Copy anchor link to this section for reference")

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

### ``with\_listeners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.with_listeners "Copy anchor link to this section for reference")

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

The `Run` object contains information about the run, including its `id`,
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

### ``with\_types [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.with_types "Copy anchor link to this section for reference")

```
with_types(
    input_type: type[Input] | BaseModel | None = None,
    output_type: type[Output] | BaseModel | None = None,
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

### ``with\_retry [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.with_retry "Copy anchor link to this section for reference")

```
with_retry(**kwargs: Any) -> Runnable[Input, Output]
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

### ``get\_name [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.get_name "Copy anchor link to this section for reference")

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

### ``get\_input\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.get_input_schema "Copy anchor link to this section for reference")

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

### ``get\_input\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.get_input_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_output\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.get_output_schema "Copy anchor link to this section for reference")

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

### ``get\_output\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.get_output_jsonschema "Copy anchor link to this section for reference")

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

### ``config\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.config_schema "Copy anchor link to this section for reference")

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

### ``get\_config\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.get_config_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_graph [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.get_graph "Copy anchor link to this section for reference")

```
get_graph(config: RunnableConfig | None = None) -> Graph
```

Return a graph representation of this `Runnable`.

### ``get\_prompts [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.get_prompts "Copy anchor link to this section for reference")

```
get_prompts(config: RunnableConfig | None = None) -> list[BasePromptTemplate]
```

Return a list of prompts used by this `Runnable`.

### ``\_\_or\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.__or__ "Copy anchor link to this section for reference")

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

### ``\_\_ror\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.__ror__ "Copy anchor link to this section for reference")

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

### ``pipe [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.pipe "Copy anchor link to this section for reference")

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

### ``pick [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.pick "Copy anchor link to this section for reference")

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

### ``assign [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.assign "Copy anchor link to this section for reference")

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

### ``invoke [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.invoke "Copy anchor link to this section for reference")

```
invoke(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> Output
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``ainvoke`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.ainvoke "Copy anchor link to this section for reference")

```
ainvoke(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> Output
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``batch [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.batch "Copy anchor link to this section for reference")

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

### ``batch\_as\_completed [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.batch_as_completed "Copy anchor link to this section for reference")

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

### ``abatch`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.abatch "Copy anchor link to this section for reference")

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

### ``abatch\_as\_completed`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.abatch_as_completed "Copy anchor link to this section for reference")

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

### ``stream [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.stream "Copy anchor link to this section for reference")

```
stream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> Iterator[Output]
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

### ``astream`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.astream "Copy anchor link to this section for reference")

```
astream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> AsyncIterator[Output]
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

### ``astream\_log`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.astream_log "Copy anchor link to this section for reference")

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

### ``astream\_events`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.astream_events "Copy anchor link to this section for reference")

```
astream_events(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
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

### ``transform [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.transform "Copy anchor link to this section for reference")

```
transform(
    input: Iterator[Input], config: RunnableConfig | None = None, **kwargs: Any
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

### ``atransform`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.atransform "Copy anchor link to this section for reference")

```
atransform(
    input: AsyncIterator[Input], config: RunnableConfig | None = None, **kwargs: Any
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

### ``with\_alisteners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.with_alisteners "Copy anchor link to this section for reference")

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

### ``map [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.map "Copy anchor link to this section for reference")

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

### ``with\_fallbacks [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.with_fallbacks "Copy anchor link to this section for reference")

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

### ``as\_tool [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.as_tool "Copy anchor link to this section for reference")

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

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.__init__ "Copy anchor link to this section for reference")

```
__init__(
    *,
    bound: Runnable[Input, Output],
    kwargs: Mapping[str, Any] | None = None,
    config: RunnableConfig | None = None,
    config_factories: list[Callable[[RunnableConfig], RunnableConfig]] | None = None,
    custom_input_type: type[Input] | BaseModel | None = None,
    custom_output_type: type[Output] | BaseModel | None = None,
    **other_kwargs: Any,
) -> None
```

Create a `RunnableBinding` from a `Runnable` and kwargs.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `bound` | The underlying `Runnable` that this `Runnable` delegates calls<br>to.<br>**TYPE:**`Runnable[Input, Output]` |
| `kwargs` | optional kwargs to pass to the underlying `Runnable`, when running<br>the underlying `Runnable` (e.g., via `invoke`, `batch`,<br>`transform`, or `stream` or async variants)<br>**TYPE:**`Mapping[str, Any] | None`**DEFAULT:**`None` |
| `config` | optional config to bind to the underlying `Runnable`.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `config_factories` | optional list of config factories to apply to the<br>config before binding to the underlying `Runnable`.<br>**TYPE:**`list[Callable[[RunnableConfig], RunnableConfig]] | None`**DEFAULT:**`None` |
| `custom_input_type` | Specify to override the input type of the underlying<br>`Runnable` with a custom type.<br>**TYPE:**`type[Input] | BaseModel | None`**DEFAULT:**`None` |
| `custom_output_type` | Specify to override the output type of the underlying<br>`Runnable` with a custom type.<br>**TYPE:**`type[Output] | BaseModel | None`**DEFAULT:**`None` |
| `**other_kwargs` | Unpacked into the base class.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``is\_lc\_serializable`classmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.is_lc_serializable "Copy anchor link to this section for reference")

```
is_lc_serializable() -> bool
```

Return `True` as this class is serializable.

### ``get\_lc\_namespace`classmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.get_lc_namespace "Copy anchor link to this section for reference")

```
get_lc_namespace() -> list[str]
```

Get the namespace of the LangChain object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | `["langchain", "schema", "runnable"]` |

### ``lc\_id`classmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.lc_id "Copy anchor link to this section for reference")

```
lc_id() -> list[str]
```

Return a unique identifier for this class for serialization purposes.

The unique identifier is a list of strings that describes the path
to the object.

For example, for the class `langchain.llms.openai.OpenAI`, the id is
`["langchain", "llms", "openai", "OpenAI"]`.

### ``to\_json [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.to_json "Copy anchor link to this section for reference")

```
to_json() -> SerializedConstructor | SerializedNotImplemented
```

Serialize the `Runnable` to JSON.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedConstructor | SerializedNotImplemented` | A JSON-serializable representation of the `Runnable`. |

### ``to\_json\_not\_implemented [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.to_json_not_implemented "Copy anchor link to this section for reference")

```
to_json_not_implemented() -> SerializedNotImplemented
```

Serialize a "not implemented" object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedNotImplemented` | `SerializedNotImplemented`. |

### ``configurable\_fields [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.configurable_fields "Copy anchor link to this section for reference")

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

### ``configurable\_alternatives [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableBinding.configurable_alternatives "Copy anchor link to this section for reference")

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

## ``RunnableGenerator [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator "Copy anchor link to this section for reference")

Bases: `Runnable[Input, Output]`

`Runnable` that runs a generator function.

`RunnableGenerator`s can be instantiated directly or by using a generator within
a sequence.

`RunnableGenerator`s can be used to implement custom behavior, such as custom
output parsers, while preserving streaming capabilities. Given a generator function
with a signature `Iterator[A] -> Iterator[B]`, wrapping it in a
`RunnableGenerator` allows it to emit output chunks as soon as they are streamed
in from the previous step.

Note

If a generator function has a `signature A -> Iterator[B]`, such that it
requires its input from the previous step to be completed before emitting chunks
(e.g., most LLMs need the entire prompt available to start generating), it can
instead be wrapped in a `RunnableLambda`.

Here is an example to show the basic mechanics of a `RunnableGenerator`:

````
```python
from typing import Any, AsyncIterator, Iterator

from langchain_core.runnables import RunnableGenerator

def gen(input: Iterator[Any]) -> Iterator[str]:
    for token in ["Have", " a", " nice", " day"]:
        yield token

runnable = RunnableGenerator(gen)
runnable.invoke(None)  # "Have a nice day"
list(runnable.stream(None))  # ["Have", " a", " nice", " day"]
runnable.batch([None, None])  # ["Have a nice day", "Have a nice day"]

# Async version:
async def agen(input: AsyncIterator[Any]) -> AsyncIterator[str]:
    for token in ["Have", " a", " nice", " day"]:
        yield token

runnable = RunnableGenerator(agen)
await runnable.ainvoke(None)  # "Have a nice day"
[p async for p in runnable.astream(None)]  # ["Have", " a", " nice", " day"]
```
````

`RunnableGenerator` makes it easy to implement custom behavior within a streaming
context. Below we show an example:

````
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableGenerator, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI()
chant_chain = (
    ChatPromptTemplate.from_template("Give me a 3 word chant about {topic}")
    | model
    | StrOutputParser()
)

def character_generator(input: Iterator[str]) -> Iterator[str]:
    for token in input:
        if "," in token or "." in token:
            yield "👏" + token
        else:
            yield token

runnable = chant_chain | character_generator
assert type(runnable.last) is RunnableGenerator
"".join(runnable.stream({"topic": "waste"}))  # Reduce👏, Reuse👏, Recycle👏.

# Note that RunnableLambda can be used to delay streaming of one step in a
# sequence until the previous step is finished:
def reverse_generator(input: str) -> Iterator[str]:
    # Yield characters of input in reverse order.
    for character in input[::-1]:
        yield character

runnable = chant_chain | RunnableLambda(reverse_generator)
"".join(runnable.stream({"topic": "waste"}))  # ".elcycer ,esuer ,ecudeR"
```
````

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Initialize a `RunnableGenerator`. |
| `get_input_schema` | Get a Pydantic model that can be used to validate input to the `Runnable`. |
| `get_output_schema` | Get a Pydantic model that can be used to validate output to the `Runnable`. |
| `transform` | Transform inputs to outputs. |
| `stream` | Default implementation of `stream`, which calls `invoke`. |
| `invoke` | Transform a single input into an output. |
| `atransform` | Transform inputs to outputs. |
| `astream` | Default implementation of `astream`, which calls `ainvoke`. |
| `ainvoke` | Transform a single input into an output. |
| `get_name` | Get the name of the `Runnable`. |
| `get_input_jsonschema` | Get a JSON schema that represents the input to the `Runnable`. |
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
| `batch` | Default implementation runs invoke in parallel using a thread pool executor. |
| `batch_as_completed` | Run `invoke` in parallel on a list of inputs. |
| `abatch` | Default implementation runs `ainvoke` in parallel using `asyncio.gather`. |
| `abatch_as_completed` | Run `ainvoke` in parallel on a list of inputs. |
| `astream_log` | Stream all output from a `Runnable`, as reported to the callback system. |
| `astream_events` | Generate a stream of events. |
| `bind` | Bind arguments to a `Runnable`, returning a new `Runnable`. |
| `with_config` | Bind config to a `Runnable`, returning a new `Runnable`. |
| `with_listeners` | Bind lifecycle listeners to a `Runnable`, returning a new `Runnable`. |
| `with_alisteners` | Bind async lifecycle listeners to a `Runnable`. |
| `with_types` | Bind input and output types to a `Runnable`, returning a new `Runnable`. |
| `with_retry` | Create a new `Runnable` that retries the original `Runnable` on exceptions. |
| `map` | Return a new `Runnable` that maps a list of inputs to a list of outputs. |
| `with_fallbacks` | Add fallbacks to a `Runnable`, returning a new `Runnable`. |
| `as_tool` | Create a `BaseTool` from a `Runnable`. |

### ``name`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.name "Copy anchor link to this section for reference")

```
name = name or __name__
```

The name of the `Runnable`. Used for debugging and tracing.

### ``InputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.InputType "Copy anchor link to this section for reference")

```
InputType: Any
```

Input type.

The type of input this `Runnable` accepts specified as a type annotation.

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the input type cannot be inferred. |

### ``OutputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.OutputType "Copy anchor link to this section for reference")

```
OutputType: Any
```

Output Type.

The type of output this `Runnable` produces specified as a type annotation.

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the output type cannot be inferred. |

### ``input\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.input_schema "Copy anchor link to this section for reference")

```
input_schema: type[BaseModel]
```

The type of input this `Runnable` accepts specified as a Pydantic model.

### ``output\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.output_schema "Copy anchor link to this section for reference")

```
output_schema: type[BaseModel]
```

Output schema.

The type of output this `Runnable` produces specified as a Pydantic model.

### ``config\_specs`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.config_specs "Copy anchor link to this section for reference")

```
config_specs: list[ConfigurableFieldSpec]
```

List configurable fields for this `Runnable`.

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.__init__ "Copy anchor link to this section for reference")

```
__init__(
    transform: Callable[[Iterator[Input]], Iterator[Output]]
    | Callable[[AsyncIterator[Input]], AsyncIterator[Output]],
    atransform: Callable[[AsyncIterator[Input]], AsyncIterator[Output]] | None = None,
    *,
    name: str | None = None,
) -> None
```

Initialize a `RunnableGenerator`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `transform` | The transform function.<br>**TYPE:**`Callable[[Iterator[Input]], Iterator[Output]] | Callable[[AsyncIterator[Input]], AsyncIterator[Output]]` |
| `atransform` | The async transform function.<br>**TYPE:**`Callable[[AsyncIterator[Input]], AsyncIterator[Output]] | None`**DEFAULT:**`None` |
| `name` | The name of the `Runnable`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the transform is not a generator function. |

### ``get\_input\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.get_input_schema "Copy anchor link to this section for reference")

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

### ``get\_output\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.get_output_schema "Copy anchor link to this section for reference")

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

### ``transform [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.transform "Copy anchor link to this section for reference")

```
transform(
    input: Iterator[Input], config: RunnableConfig | None = None, **kwargs: Any
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

### ``stream [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.stream "Copy anchor link to this section for reference")

```
stream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any
) -> Iterator[Output]
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

### ``invoke [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.invoke "Copy anchor link to this section for reference")

```
invoke(input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Output
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``atransform [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.atransform "Copy anchor link to this section for reference")

```
atransform(
    input: AsyncIterator[Input], config: RunnableConfig | None = None, **kwargs: Any
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

### ``astream [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.astream "Copy anchor link to this section for reference")

```
astream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any
) -> AsyncIterator[Output]
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

### ``ainvoke`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.ainvoke "Copy anchor link to this section for reference")

```
ainvoke(input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Output
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``get\_name [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.get_name "Copy anchor link to this section for reference")

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

### ``get\_input\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.get_input_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_output\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.get_output_jsonschema "Copy anchor link to this section for reference")

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

### ``config\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.config_schema "Copy anchor link to this section for reference")

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

### ``get\_config\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.get_config_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_graph [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.get_graph "Copy anchor link to this section for reference")

```
get_graph(config: RunnableConfig | None = None) -> Graph
```

Return a graph representation of this `Runnable`.

### ``get\_prompts [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.get_prompts "Copy anchor link to this section for reference")

```
get_prompts(config: RunnableConfig | None = None) -> list[BasePromptTemplate]
```

Return a list of prompts used by this `Runnable`.

### ``\_\_or\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.__or__ "Copy anchor link to this section for reference")

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

### ``\_\_ror\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.__ror__ "Copy anchor link to this section for reference")

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

### ``pipe [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.pipe "Copy anchor link to this section for reference")

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

### ``pick [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.pick "Copy anchor link to this section for reference")

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

### ``assign [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.assign "Copy anchor link to this section for reference")

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

### ``batch [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.batch "Copy anchor link to this section for reference")

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

### ``batch\_as\_completed [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.batch_as_completed "Copy anchor link to this section for reference")

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

### ``abatch`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.abatch "Copy anchor link to this section for reference")

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

### ``abatch\_as\_completed`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.abatch_as_completed "Copy anchor link to this section for reference")

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

### ``astream\_log`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.astream_log "Copy anchor link to this section for reference")

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

### ``astream\_events`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.astream_events "Copy anchor link to this section for reference")

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

### ``bind [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.bind "Copy anchor link to this section for reference")

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

### ``with\_config [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.with_config "Copy anchor link to this section for reference")

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

### ``with\_listeners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.with_listeners "Copy anchor link to this section for reference")

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

### ``with\_alisteners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.with_alisteners "Copy anchor link to this section for reference")

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

### ``with\_types [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.with_types "Copy anchor link to this section for reference")

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

### ``with\_retry [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.with_retry "Copy anchor link to this section for reference")

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

### ``map [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.map "Copy anchor link to this section for reference")

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

### ``with\_fallbacks [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.with_fallbacks "Copy anchor link to this section for reference")

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

### ``as\_tool [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableGenerator.as_tool "Copy anchor link to this section for reference")

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

## ``RunnableLambda [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda "Copy anchor link to this section for reference")

Bases: `Runnable[Input, Output]`

`RunnableLambda` converts a python callable into a `Runnable`.

Wrapping a callable in a `RunnableLambda` makes the callable usable
within either a sync or async context.

`RunnableLambda` can be composed as any other `Runnable` and provides
seamless integration with LangChain tracing.

`RunnableLambda` is best suited for code that does not need to support
streaming. If you need to support streaming (i.e., be able to operate
on chunks of inputs and yield chunks of outputs), use `RunnableGenerator`
instead.

Note that if a `RunnableLambda` returns an instance of `Runnable`, that
instance is invoked (or streamed) during execution.

Examples:

```
# This is a RunnableLambda
from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

runnable = RunnableLambda(add_one)

runnable.invoke(1)  # returns 2
runnable.batch([1, 2, 3])  # returns [2, 3, 4]

# Async is supported by default by delegating to the sync implementation
await runnable.ainvoke(1)  # returns 2
await runnable.abatch([1, 2, 3])  # returns [2, 3, 4]

# Alternatively, can provide both synd and sync implementations
async def add_one_async(x: int) -> int:
    return x + 1

runnable = RunnableLambda(add_one, afunc=add_one_async)
runnable.invoke(1)  # Uses add_one
await runnable.ainvoke(1)  # Uses add_one_async
```

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Create a `RunnableLambda` from a callable, and async callable or both. |
| `get_input_schema` | The Pydantic schema for the input to this `Runnable`. |
| `get_output_schema` | Get a Pydantic model that can be used to validate output to the `Runnable`. |
| `get_graph` | Return a graph representation of this `Runnable`. |
| `__repr__` | Return a string representation of this `Runnable`. |
| `invoke` | Invoke this `Runnable` synchronously. |
| `ainvoke` | Invoke this `Runnable` asynchronously. |
| `transform` | Transform inputs to outputs. |
| `stream` | Default implementation of `stream`, which calls `invoke`. |
| `atransform` | Transform inputs to outputs. |
| `astream` | Default implementation of `astream`, which calls `ainvoke`. |
| `get_name` | Get the name of the `Runnable`. |
| `get_input_jsonschema` | Get a JSON schema that represents the input to the `Runnable`. |
| `get_output_jsonschema` | Get a JSON schema that represents the output of the `Runnable`. |
| `config_schema` | The type of config this `Runnable` accepts specified as a Pydantic model. |
| `get_config_jsonschema` | Get a JSON schema that represents the config of the `Runnable`. |
| `get_prompts` | Return a list of prompts used by this `Runnable`. |
| `__or__` | Runnable "or" operator. |
| `__ror__` | Runnable "reverse-or" operator. |
| `pipe` | Pipe `Runnable` objects. |
| `pick` | Pick keys from the output `dict` of this `Runnable`. |
| `assign` | Assigns new fields to the `dict` output of this `Runnable`. |
| `batch` | Default implementation runs invoke in parallel using a thread pool executor. |
| `batch_as_completed` | Run `invoke` in parallel on a list of inputs. |
| `abatch` | Default implementation runs `ainvoke` in parallel using `asyncio.gather`. |
| `abatch_as_completed` | Run `ainvoke` in parallel on a list of inputs. |
| `astream_log` | Stream all output from a `Runnable`, as reported to the callback system. |
| `astream_events` | Generate a stream of events. |
| `bind` | Bind arguments to a `Runnable`, returning a new `Runnable`. |
| `with_config` | Bind config to a `Runnable`, returning a new `Runnable`. |
| `with_listeners` | Bind lifecycle listeners to a `Runnable`, returning a new `Runnable`. |
| `with_alisteners` | Bind async lifecycle listeners to a `Runnable`. |
| `with_types` | Bind input and output types to a `Runnable`, returning a new `Runnable`. |
| `with_retry` | Create a new `Runnable` that retries the original `Runnable` on exceptions. |
| `map` | Return a new `Runnable` that maps a list of inputs to a list of outputs. |
| `with_fallbacks` | Add fallbacks to a `Runnable`, returning a new `Runnable`. |
| `as_tool` | Create a `BaseTool` from a `Runnable`. |

### ``name`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.name "Copy anchor link to this section for reference")

```
name = name
```

The name of the `Runnable`. Used for debugging and tracing.

### ``InputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.InputType "Copy anchor link to this section for reference")

```
InputType: Any
```

The type of the input to this `Runnable`.

### ``OutputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.OutputType "Copy anchor link to this section for reference")

```
OutputType: Any
```

The type of the output of this `Runnable` as a type annotation.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Any` | The type of the output of this `Runnable`. |

### ``deps`cached``property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.deps "Copy anchor link to this section for reference")

```
deps: list[Runnable]
```

The dependencies of this `Runnable`.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Runnable]` | The dependencies of this `Runnable`. If the function has nonlocal |
| `list[Runnable]` | variables that are `Runnable`s, they are considered dependencies. |

### ``config\_specs`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.config_specs "Copy anchor link to this section for reference")

```
config_specs: list[ConfigurableFieldSpec]
```

List configurable fields for this `Runnable`.

### ``input\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.input_schema "Copy anchor link to this section for reference")

```
input_schema: type[BaseModel]
```

The type of input this `Runnable` accepts specified as a Pydantic model.

### ``output\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.output_schema "Copy anchor link to this section for reference")

```
output_schema: type[BaseModel]
```

Output schema.

The type of output this `Runnable` produces specified as a Pydantic model.

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.__init__ "Copy anchor link to this section for reference")

```
__init__(
    func: Callable[[Input], Iterator[Output]]
    | Callable[[Input], Runnable[Input, Output]]
    | Callable[[Input], Output]
    | Callable[[Input, RunnableConfig], Output]
    | Callable[[Input, CallbackManagerForChainRun], Output]
    | Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output]
    | Callable[[Input], Awaitable[Output]]
    | Callable[[Input], AsyncIterator[Output]]
    | Callable[[Input, RunnableConfig], Awaitable[Output]]
    | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
    | Callable[\
        [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]\
    ],
    afunc: Callable[[Input], Awaitable[Output]]
    | Callable[[Input], AsyncIterator[Output]]
    | Callable[[Input, RunnableConfig], Awaitable[Output]]
    | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
    | Callable[\
        [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]\
    ]
    | None = None,
    name: str | None = None,
) -> None
```

Create a `RunnableLambda` from a callable, and async callable or both.

Accepts both sync and async variants to allow providing efficient
implementations for sync and async execution.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `func` | Either sync or async callable<br>**TYPE:**`Callable[[Input], Iterator[Output]] | Callable[[Input], Runnable[Input, Output]] | Callable[[Input], Output] | Callable[[Input, RunnableConfig], Output] | Callable[[Input, CallbackManagerForChainRun], Output] | Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output] | Callable[[Input], Awaitable[Output]] | Callable[[Input], AsyncIterator[Output]] | Callable[[Input, RunnableConfig], Awaitable[Output]] | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]] | Callable[[Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]]` |
| `afunc` | An async callable that takes an input and returns an output.<br>**TYPE:**`Callable[[Input], Awaitable[Output]] | Callable[[Input], AsyncIterator[Output]] | Callable[[Input, RunnableConfig], Awaitable[Output]] | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]] | Callable[[Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]] | None`**DEFAULT:**`None` |
| `name` | The name of the `Runnable`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the `func` is not a callable type. |
| `TypeError` | If both `func` and `afunc` are provided. |

### ``get\_input\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.get_input_schema "Copy anchor link to this section for reference")

```
get_input_schema(config: RunnableConfig | None = None) -> type[BaseModel]
```

The Pydantic schema for the input to this `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | The config to use.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `type[BaseModel]` | The input schema for this `Runnable`. |

### ``get\_output\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.get_output_schema "Copy anchor link to this section for reference")

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

### ``get\_graph [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.get_graph "Copy anchor link to this section for reference")

```
get_graph(config: RunnableConfig | None = None) -> Graph
```

Return a graph representation of this `Runnable`.

### ``\_\_repr\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.__repr__ "Copy anchor link to this section for reference")

```
__repr__() -> str
```

Return a string representation of this `Runnable`.

### ``invoke [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.invoke "Copy anchor link to this section for reference")

```
invoke(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> Output
```

Invoke this `Runnable` synchronously.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to this `Runnable`.<br>**TYPE:**`Input` |
| `config` | The config to use.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of this `Runnable`. |

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the `Runnable` is a coroutine function. |

### ``ainvoke`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.ainvoke "Copy anchor link to this section for reference")

```
ainvoke(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> Output
```

Invoke this `Runnable` asynchronously.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to this `Runnable`.<br>**TYPE:**`Input` |
| `config` | The config to use.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any | None`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of this `Runnable`. |

### ``transform [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.transform "Copy anchor link to this section for reference")

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

### ``stream [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.stream "Copy anchor link to this section for reference")

```
stream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> Iterator[Output]
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

### ``atransform`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.atransform "Copy anchor link to this section for reference")

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

### ``astream`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.astream "Copy anchor link to this section for reference")

```
astream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> AsyncIterator[Output]
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

### ``get\_name [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.get_name "Copy anchor link to this section for reference")

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

### ``get\_input\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.get_input_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_output\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.get_output_jsonschema "Copy anchor link to this section for reference")

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

### ``config\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.config_schema "Copy anchor link to this section for reference")

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

### ``get\_config\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.get_config_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_prompts [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.get_prompts "Copy anchor link to this section for reference")

```
get_prompts(config: RunnableConfig | None = None) -> list[BasePromptTemplate]
```

Return a list of prompts used by this `Runnable`.

### ``\_\_or\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.__or__ "Copy anchor link to this section for reference")

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

### ``\_\_ror\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.__ror__ "Copy anchor link to this section for reference")

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

### ``pipe [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.pipe "Copy anchor link to this section for reference")

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

### ``pick [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.pick "Copy anchor link to this section for reference")

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

### ``assign [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.assign "Copy anchor link to this section for reference")

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

### ``batch [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.batch "Copy anchor link to this section for reference")

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

### ``batch\_as\_completed [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.batch_as_completed "Copy anchor link to this section for reference")

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

### ``abatch`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.abatch "Copy anchor link to this section for reference")

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

### ``abatch\_as\_completed`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.abatch_as_completed "Copy anchor link to this section for reference")

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

### ``astream\_log`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.astream_log "Copy anchor link to this section for reference")

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

### ``astream\_events`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.astream_events "Copy anchor link to this section for reference")

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

### ``bind [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.bind "Copy anchor link to this section for reference")

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

### ``with\_config [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.with_config "Copy anchor link to this section for reference")

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

### ``with\_listeners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.with_listeners "Copy anchor link to this section for reference")

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

### ``with\_alisteners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.with_alisteners "Copy anchor link to this section for reference")

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

### ``with\_types [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.with_types "Copy anchor link to this section for reference")

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

### ``with\_retry [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.with_retry "Copy anchor link to this section for reference")

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

### ``map [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.map "Copy anchor link to this section for reference")

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

### ``with\_fallbacks [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.with_fallbacks "Copy anchor link to this section for reference")

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

### ``as\_tool [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableLambda.as_tool "Copy anchor link to this section for reference")

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

## ``RunnableParallel [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel "Copy anchor link to this section for reference")

Bases: `RunnableSerializable[Input, dict[str, Any]]`

Runnable that runs a mapping of `Runnable`s in parallel.

Returns a mapping of their outputs.

`RunnableParallel` is one of the two main composition primitives,
alongside `RunnableSequence`. It invokes `Runnable`s concurrently, providing the
same input to each.

A `RunnableParallel` can be instantiated directly or by using a dict literal
within a sequence.

Here is a simple example that uses functions to illustrate the use of
`RunnableParallel`:

````
```python
from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

def mul_two(x: int) -> int:
    return x * 2

def mul_three(x: int) -> int:
    return x * 3

runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)
runnable_3 = RunnableLambda(mul_three)

sequence = runnable_1 | {  # this dict is coerced to a RunnableParallel
    "mul_two": runnable_2,
    "mul_three": runnable_3,
}
# Or equivalently:
# sequence = runnable_1 | RunnableParallel(
#     {"mul_two": runnable_2, "mul_three": runnable_3}
# )
# Also equivalently:
# sequence = runnable_1 | RunnableParallel(
#     mul_two=runnable_2,
#     mul_three=runnable_3,
# )

sequence.invoke(1)
await sequence.ainvoke(1)

sequence.batch([1, 2, 3])
await sequence.abatch([1, 2, 3])
```
````

`RunnableParallel` makes it easy to run `Runnable`s in parallel. In the below
example, we simultaneously stream output from two different `Runnable` objects:

````
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
joke_chain = (
    ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
)
poem_chain = (
    ChatPromptTemplate.from_template("write a 2-line poem about {topic}")
    | model
)

runnable = RunnableParallel(joke=joke_chain, poem=poem_chain)

# Display stream
output = {key: "" for key, _ in runnable.output_schema()}
for chunk in runnable.stream({"topic": "bear"}):
    for key in chunk:
        output[key] = output[key] + chunk[key].content
    print(output)  # noqa: T201
```
````

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Create a `RunnableParallel`. |
| `is_lc_serializable` | Return `True` as this class is serializable. |
| `get_lc_namespace` | Get the namespace of the LangChain object. |
| `get_name` | Get the name of the `Runnable`. |
| `get_input_schema` | Get the input schema of the `Runnable`. |
| `get_output_schema` | Get the output schema of the `Runnable`. |
| `get_graph` | Get the graph representation of the `Runnable`. |
| `invoke` | Transform a single input into an output. |
| `ainvoke` | Transform a single input into an output. |
| `transform` | Transform inputs to outputs. |
| `stream` | Default implementation of `stream`, which calls `invoke`. |
| `atransform` | Transform inputs to outputs. |
| `astream` | Default implementation of `astream`, which calls `ainvoke`. |
| `get_input_jsonschema` | Get a JSON schema that represents the input to the `Runnable`. |
| `get_output_jsonschema` | Get a JSON schema that represents the output of the `Runnable`. |
| `config_schema` | The type of config this `Runnable` accepts specified as a Pydantic model. |
| `get_config_jsonschema` | Get a JSON schema that represents the config of the `Runnable`. |
| `get_prompts` | Return a list of prompts used by this `Runnable`. |
| `__or__` | Runnable "or" operator. |
| `__ror__` | Runnable "reverse-or" operator. |
| `pipe` | Pipe `Runnable` objects. |
| `pick` | Pick keys from the output `dict` of this `Runnable`. |
| `assign` | Assigns new fields to the `dict` output of this `Runnable`. |
| `batch` | Default implementation runs invoke in parallel using a thread pool executor. |
| `batch_as_completed` | Run `invoke` in parallel on a list of inputs. |
| `abatch` | Default implementation runs `ainvoke` in parallel using `asyncio.gather`. |
| `abatch_as_completed` | Run `ainvoke` in parallel on a list of inputs. |
| `astream_log` | Stream all output from a `Runnable`, as reported to the callback system. |
| `astream_events` | Generate a stream of events. |
| `bind` | Bind arguments to a `Runnable`, returning a new `Runnable`. |
| `with_config` | Bind config to a `Runnable`, returning a new `Runnable`. |
| `with_listeners` | Bind lifecycle listeners to a `Runnable`, returning a new `Runnable`. |
| `with_alisteners` | Bind async lifecycle listeners to a `Runnable`. |
| `with_types` | Bind input and output types to a `Runnable`, returning a new `Runnable`. |
| `with_retry` | Create a new `Runnable` that retries the original `Runnable` on exceptions. |
| `map` | Return a new `Runnable` that maps a list of inputs to a list of outputs. |
| `with_fallbacks` | Add fallbacks to a `Runnable`, returning a new `Runnable`. |
| `as_tool` | Create a `BaseTool` from a `Runnable`. |
| `lc_id` | Return a unique identifier for this class for serialization purposes. |
| `to_json` | Serialize the `Runnable` to JSON. |
| `to_json_not_implemented` | Serialize a "not implemented" object. |
| `configurable_fields` | Configure particular `Runnable` fields at runtime. |
| `configurable_alternatives` | Configure alternatives for `Runnable` objects that can be set at runtime. |

### ``InputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.InputType "Copy anchor link to this section for reference")

```
InputType: Any
```

The type of the input to the `Runnable`.

### ``config\_specs`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.config_specs "Copy anchor link to this section for reference")

```
config_specs: list[ConfigurableFieldSpec]
```

Get the config specs of the `Runnable`.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[ConfigurableFieldSpec]` | The config specs of the `Runnable`. |

### ``name`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.name "Copy anchor link to this section for reference")

```
name: str | None = None
```

The name of the `Runnable`. Used for debugging and tracing.

### ``OutputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.OutputType "Copy anchor link to this section for reference")

```
OutputType: type[Output]
```

Output Type.

The type of output this `Runnable` produces specified as a type annotation.

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the output type cannot be inferred. |

### ``input\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.input_schema "Copy anchor link to this section for reference")

```
input_schema: type[BaseModel]
```

The type of input this `Runnable` accepts specified as a Pydantic model.

### ``output\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.output_schema "Copy anchor link to this section for reference")

```
output_schema: type[BaseModel]
```

Output schema.

The type of output this `Runnable` produces specified as a Pydantic model.

### ``lc\_secrets`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.lc_secrets "Copy anchor link to this section for reference")

```
lc_secrets: dict[str, str]
```

A map of constructor argument names to secret ids.

For example, `{"openai_api_key": "OPENAI_API_KEY"}`

### ``lc\_attributes`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.lc_attributes "Copy anchor link to this section for reference")

```
lc_attributes: dict
```

List of attribute names that should be included in the serialized kwargs.

These attributes must be accepted by the constructor.

Default is an empty dictionary.

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.__init__ "Copy anchor link to this section for reference")

```
__init__(
    steps__: Mapping[\
        str,\
        Runnable[Input, Any]\
        | Callable[[Input], Any]\
        | Mapping[str, Runnable[Input, Any] | Callable[[Input], Any]],\
    ]
    | None = None,
    **kwargs: Runnable[Input, Any]
    | Callable[[Input], Any]
    | Mapping[str, Runnable[Input, Any] | Callable[[Input], Any]],
) -> None
```

Create a `RunnableParallel`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `steps__` | The steps to include.<br>**TYPE:**`Mapping[str, Runnable[Input, Any] | Callable[[Input], Any] | Mapping[str, Runnable[Input, Any] | Callable[[Input], Any]]] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional steps to include.<br>**TYPE:**`Runnable[Input, Any] | Callable[[Input], Any] | Mapping[str, Runnable[Input, Any] | Callable[[Input], Any]]`**DEFAULT:**`{}` |

### ``is\_lc\_serializable`classmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.is_lc_serializable "Copy anchor link to this section for reference")

```
is_lc_serializable() -> bool
```

Return `True` as this class is serializable.

### ``get\_lc\_namespace`classmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.get_lc_namespace "Copy anchor link to this section for reference")

```
get_lc_namespace() -> list[str]
```

Get the namespace of the LangChain object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | `["langchain", "schema", "runnable"]` |

### ``get\_name [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.get_name "Copy anchor link to this section for reference")

```
get_name(suffix: str | None = None, *, name: str | None = None) -> str
```

Get the name of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `suffix` | The suffix to use.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `name` | The name to use.<br>**TYPE:**`str | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `str` | The name of the `Runnable`. |

### ``get\_input\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.get_input_schema "Copy anchor link to this section for reference")

```
get_input_schema(config: RunnableConfig | None = None) -> type[BaseModel]
```

Get the input schema of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | The config to use.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `type[BaseModel]` | The input schema of the `Runnable`. |

### ``get\_output\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.get_output_schema "Copy anchor link to this section for reference")

```
get_output_schema(config: RunnableConfig | None = None) -> type[BaseModel]
```

Get the output schema of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | The config to use.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `type[BaseModel]` | The output schema of the `Runnable`. |

### ``get\_graph [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.get_graph "Copy anchor link to this section for reference")

```
get_graph(config: RunnableConfig | None = None) -> Graph
```

Get the graph representation of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | The config to use.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Graph` | The graph representation of the `Runnable`. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If a `Runnable` has no first or last node. |

### ``invoke [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.invoke "Copy anchor link to this section for reference")

```
invoke(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any
) -> dict[str, Any]
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``ainvoke`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.ainvoke "Copy anchor link to this section for reference")

```
ainvoke(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> dict[str, Any]
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``transform [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.transform "Copy anchor link to this section for reference")

```
transform(
    input: Iterator[Input], config: RunnableConfig | None = None, **kwargs: Any
) -> Iterator[dict[str, Any]]
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

### ``stream [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.stream "Copy anchor link to this section for reference")

```
stream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> Iterator[dict[str, Any]]
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

### ``atransform`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.atransform "Copy anchor link to this section for reference")

```
atransform(
    input: AsyncIterator[Input], config: RunnableConfig | None = None, **kwargs: Any
) -> AsyncIterator[dict[str, Any]]
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

### ``astream`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.astream "Copy anchor link to this section for reference")

```
astream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> AsyncIterator[dict[str, Any]]
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

### ``get\_input\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.get_input_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_output\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.get_output_jsonschema "Copy anchor link to this section for reference")

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

### ``config\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.config_schema "Copy anchor link to this section for reference")

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

### ``get\_config\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.get_config_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_prompts [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.get_prompts "Copy anchor link to this section for reference")

```
get_prompts(config: RunnableConfig | None = None) -> list[BasePromptTemplate]
```

Return a list of prompts used by this `Runnable`.

### ``\_\_or\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.__or__ "Copy anchor link to this section for reference")

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

### ``\_\_ror\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.__ror__ "Copy anchor link to this section for reference")

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

### ``pipe [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.pipe "Copy anchor link to this section for reference")

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

### ``pick [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.pick "Copy anchor link to this section for reference")

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

### ``assign [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.assign "Copy anchor link to this section for reference")

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

### ``batch [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.batch "Copy anchor link to this section for reference")

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

### ``batch\_as\_completed [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.batch_as_completed "Copy anchor link to this section for reference")

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

### ``abatch`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.abatch "Copy anchor link to this section for reference")

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

### ``abatch\_as\_completed`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.abatch_as_completed "Copy anchor link to this section for reference")

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

### ``astream\_log`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.astream_log "Copy anchor link to this section for reference")

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

### ``astream\_events`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.astream_events "Copy anchor link to this section for reference")

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

### ``bind [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.bind "Copy anchor link to this section for reference")

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

### ``with\_config [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.with_config "Copy anchor link to this section for reference")

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

### ``with\_listeners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.with_listeners "Copy anchor link to this section for reference")

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

### ``with\_alisteners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.with_alisteners "Copy anchor link to this section for reference")

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

### ``with\_types [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.with_types "Copy anchor link to this section for reference")

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

### ``with\_retry [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.with_retry "Copy anchor link to this section for reference")

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

### ``map [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.map "Copy anchor link to this section for reference")

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

### ``with\_fallbacks [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.with_fallbacks "Copy anchor link to this section for reference")

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

### ``as\_tool [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.as_tool "Copy anchor link to this section for reference")

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

### ``lc\_id`classmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.lc_id "Copy anchor link to this section for reference")

```
lc_id() -> list[str]
```

Return a unique identifier for this class for serialization purposes.

The unique identifier is a list of strings that describes the path
to the object.

For example, for the class `langchain.llms.openai.OpenAI`, the id is
`["langchain", "llms", "openai", "OpenAI"]`.

### ``to\_json [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.to_json "Copy anchor link to this section for reference")

```
to_json() -> SerializedConstructor | SerializedNotImplemented
```

Serialize the `Runnable` to JSON.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedConstructor | SerializedNotImplemented` | A JSON-serializable representation of the `Runnable`. |

### ``to\_json\_not\_implemented [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.to_json_not_implemented "Copy anchor link to this section for reference")

```
to_json_not_implemented() -> SerializedNotImplemented
```

Serialize a "not implemented" object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedNotImplemented` | `SerializedNotImplemented`. |

### ``configurable\_fields [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.configurable_fields "Copy anchor link to this section for reference")

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

### ``configurable\_alternatives [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableParallel.configurable_alternatives "Copy anchor link to this section for reference")

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

## ``RunnableSequence [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence "Copy anchor link to this section for reference")

Bases: `RunnableSerializable[Input, Output]`

Sequence of `Runnable` objects, where the output of one is the input of the next.

**`RunnableSequence`** is the most important composition operator in LangChain
as it is used in virtually every chain.

A `RunnableSequence` can be instantiated directly or more commonly by using the
`|` operator where either the left or right operands (or both) must be a
`Runnable`.

Any `RunnableSequence` automatically supports sync, async, batch.

The default implementations of `batch` and `abatch` utilize threadpools and
asyncio gather and will be faster than naive invocation of `invoke` or `ainvoke`
for IO bound `Runnable`s.

Batching is implemented by invoking the batch method on each component of the
`RunnableSequence` in order.

A `RunnableSequence` preserves the streaming properties of its components, so if
all components of the sequence implement a `transform` method -- which
is the method that implements the logic to map a streaming input to a streaming
output -- then the sequence will be able to stream input to output!

If any component of the sequence does not implement transform then the
streaming will only begin after this component is run. If there are
multiple blocking components, streaming begins after the last one.

Note

`RunnableLambdas` do not support `transform` by default! So if you need to
use a `RunnableLambdas` be careful about where you place them in a
`RunnableSequence` (if you need to use the `stream`/`astream` methods).

If you need arbitrary logic and need streaming, you can subclass
Runnable, and implement `transform` for whatever logic you need.

Here is a simple example that uses simple functions to illustrate the use of
`RunnableSequence`:

````
```python
from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

def mul_two(x: int) -> int:
    return x * 2

runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)
sequence = runnable_1 | runnable_2
# Or equivalently:
# sequence = RunnableSequence(first=runnable_1, last=runnable_2)
sequence.invoke(1)
await sequence.ainvoke(1)

sequence.batch([1, 2, 3])
await sequence.abatch([1, 2, 3])
```
````

Here's an example that uses streams JSON output generated by an LLM:

````
```python
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template(
    "In JSON format, give me a list of {topic} and their "
    "corresponding names in French, Spanish and in a "
    "Cat Language."
)

model = ChatOpenAI()
chain = prompt | model | SimpleJsonOutputParser()

async for chunk in chain.astream({"topic": "colors"}):
    print("-")  # noqa: T201
    print(chunk, sep="", flush=True)  # noqa: T201
```
````

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Create a new `RunnableSequence`. |
| `get_lc_namespace` | Get the namespace of the LangChain object. |
| `is_lc_serializable` | Return `True` as this class is serializable. |
| `get_input_schema` | Get the input schema of the `Runnable`. |
| `get_output_schema` | Get the output schema of the `Runnable`. |
| `get_graph` | Get the graph representation of the `Runnable`. |
| `__or__` | Runnable "or" operator. |
| `__ror__` | Runnable "reverse-or" operator. |
| `invoke` | Transform a single input into an output. |
| `ainvoke` | Transform a single input into an output. |
| `batch` | Default implementation runs invoke in parallel using a thread pool executor. |
| `abatch` | Default implementation runs `ainvoke` in parallel using `asyncio.gather`. |
| `transform` | Transform inputs to outputs. |
| `stream` | Default implementation of `stream`, which calls `invoke`. |
| `atransform` | Transform inputs to outputs. |
| `astream` | Default implementation of `astream`, which calls `ainvoke`. |
| `get_name` | Get the name of the `Runnable`. |
| `get_input_jsonschema` | Get a JSON schema that represents the input to the `Runnable`. |
| `get_output_jsonschema` | Get a JSON schema that represents the output of the `Runnable`. |
| `config_schema` | The type of config this `Runnable` accepts specified as a Pydantic model. |
| `get_config_jsonschema` | Get a JSON schema that represents the config of the `Runnable`. |
| `get_prompts` | Return a list of prompts used by this `Runnable`. |
| `pipe` | Pipe `Runnable` objects. |
| `pick` | Pick keys from the output `dict` of this `Runnable`. |
| `assign` | Assigns new fields to the `dict` output of this `Runnable`. |
| `batch_as_completed` | Run `invoke` in parallel on a list of inputs. |
| `abatch_as_completed` | Run `ainvoke` in parallel on a list of inputs. |
| `astream_log` | Stream all output from a `Runnable`, as reported to the callback system. |
| `astream_events` | Generate a stream of events. |
| `bind` | Bind arguments to a `Runnable`, returning a new `Runnable`. |
| `with_config` | Bind config to a `Runnable`, returning a new `Runnable`. |
| `with_listeners` | Bind lifecycle listeners to a `Runnable`, returning a new `Runnable`. |
| `with_alisteners` | Bind async lifecycle listeners to a `Runnable`. |
| `with_types` | Bind input and output types to a `Runnable`, returning a new `Runnable`. |
| `with_retry` | Create a new `Runnable` that retries the original `Runnable` on exceptions. |
| `map` | Return a new `Runnable` that maps a list of inputs to a list of outputs. |
| `with_fallbacks` | Add fallbacks to a `Runnable`, returning a new `Runnable`. |
| `as_tool` | Create a `BaseTool` from a `Runnable`. |
| `lc_id` | Return a unique identifier for this class for serialization purposes. |
| `to_json` | Serialize the `Runnable` to JSON. |
| `to_json_not_implemented` | Serialize a "not implemented" object. |
| `configurable_fields` | Configure particular `Runnable` fields at runtime. |
| `configurable_alternatives` | Configure alternatives for `Runnable` objects that can be set at runtime. |

### ``first`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.first "Copy anchor link to this section for reference")

```
first: Runnable[Input, Any]
```

The first `Runnable` in the sequence.

### ``middle`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.middle "Copy anchor link to this section for reference")

```
middle: list[Runnable[Any, Any]] = Field(default_factory=list)
```

The middle `Runnable` in the sequence.

### ``last`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.last "Copy anchor link to this section for reference")

```
last: Runnable[Any, Output]
```

The last `Runnable` in the sequence.

### ``steps`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.steps "Copy anchor link to this section for reference")

```
steps: list[Runnable[Any, Any]]
```

All the `Runnable`s that make up the sequence in order.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Runnable[Any, Any]]` | A list of `Runnable`s. |

### ``InputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.InputType "Copy anchor link to this section for reference")

```
InputType: type[Input]
```

The type of the input to the `Runnable`.

### ``OutputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.OutputType "Copy anchor link to this section for reference")

```
OutputType: type[Output]
```

The type of the output of the `Runnable`.

### ``config\_specs`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.config_specs "Copy anchor link to this section for reference")

```
config_specs: list[ConfigurableFieldSpec]
```

Get the config specs of the `Runnable`.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[ConfigurableFieldSpec]` | The config specs of the `Runnable`. |

### ``name`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.name "Copy anchor link to this section for reference")

```
name: str | None = None
```

The name of the `Runnable`. Used for debugging and tracing.

### ``input\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.input_schema "Copy anchor link to this section for reference")

```
input_schema: type[BaseModel]
```

The type of input this `Runnable` accepts specified as a Pydantic model.

### ``output\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.output_schema "Copy anchor link to this section for reference")

```
output_schema: type[BaseModel]
```

Output schema.

The type of output this `Runnable` produces specified as a Pydantic model.

### ``lc\_secrets`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.lc_secrets "Copy anchor link to this section for reference")

```
lc_secrets: dict[str, str]
```

A map of constructor argument names to secret ids.

For example, `{"openai_api_key": "OPENAI_API_KEY"}`

### ``lc\_attributes`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.lc_attributes "Copy anchor link to this section for reference")

```
lc_attributes: dict
```

List of attribute names that should be included in the serialized kwargs.

These attributes must be accepted by the constructor.

Default is an empty dictionary.

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.__init__ "Copy anchor link to this section for reference")

```
__init__(
    *steps: RunnableLike,
    name: str | None = None,
    first: Runnable[Any, Any] | None = None,
    middle: list[Runnable[Any, Any]] | None = None,
    last: Runnable[Any, Any] | None = None,
) -> None
```

Create a new `RunnableSequence`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `steps` | The steps to include in the sequence.<br>**TYPE:**`RunnableLike`**DEFAULT:**`()` |
| `name` | The name of the `Runnable`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `first` | The first `Runnable` in the sequence.<br>**TYPE:**`Runnable[Any, Any] | None`**DEFAULT:**`None` |
| `middle` | The middle `Runnable` objects in the sequence.<br>**TYPE:**`list[Runnable[Any, Any]] | None`**DEFAULT:**`None` |
| `last` | The last `Runnable` in the sequence.<br>**TYPE:**`Runnable[Any, Any] | None`**DEFAULT:**`None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the sequence has less than 2 steps. |

### ``get\_lc\_namespace`classmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.get_lc_namespace "Copy anchor link to this section for reference")

```
get_lc_namespace() -> list[str]
```

Get the namespace of the LangChain object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | `["langchain", "schema", "runnable"]` |

### ``is\_lc\_serializable`classmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.is_lc_serializable "Copy anchor link to this section for reference")

```
is_lc_serializable() -> bool
```

Return `True` as this class is serializable.

### ``get\_input\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.get_input_schema "Copy anchor link to this section for reference")

```
get_input_schema(config: RunnableConfig | None = None) -> type[BaseModel]
```

Get the input schema of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | The config to use.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `type[BaseModel]` | The input schema of the `Runnable`. |

### ``get\_output\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.get_output_schema "Copy anchor link to this section for reference")

```
get_output_schema(config: RunnableConfig | None = None) -> type[BaseModel]
```

Get the output schema of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | The config to use.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `type[BaseModel]` | The output schema of the `Runnable`. |

### ``get\_graph [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.get_graph "Copy anchor link to this section for reference")

```
get_graph(config: RunnableConfig | None = None) -> Graph
```

Get the graph representation of the `Runnable`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `config` | The config to use.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Graph` | The graph representation of the `Runnable`. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If a `Runnable` has no first or last node. |

### ``\_\_or\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.__or__ "Copy anchor link to this section for reference")

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

### ``\_\_ror\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.__ror__ "Copy anchor link to this section for reference")

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

### ``invoke [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.invoke "Copy anchor link to this section for reference")

```
invoke(input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Output
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``ainvoke`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.ainvoke "Copy anchor link to this section for reference")

```
ainvoke(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> Output
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``batch [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.batch "Copy anchor link to this section for reference")

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

### ``abatch`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.abatch "Copy anchor link to this section for reference")

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

### ``transform [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.transform "Copy anchor link to this section for reference")

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

### ``stream [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.stream "Copy anchor link to this section for reference")

```
stream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> Iterator[Output]
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

### ``atransform`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.atransform "Copy anchor link to this section for reference")

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

### ``astream`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.astream "Copy anchor link to this section for reference")

```
astream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> AsyncIterator[Output]
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

### ``get\_name [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.get_name "Copy anchor link to this section for reference")

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

### ``get\_input\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.get_input_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_output\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.get_output_jsonschema "Copy anchor link to this section for reference")

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

### ``config\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.config_schema "Copy anchor link to this section for reference")

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

### ``get\_config\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.get_config_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_prompts [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.get_prompts "Copy anchor link to this section for reference")

```
get_prompts(config: RunnableConfig | None = None) -> list[BasePromptTemplate]
```

Return a list of prompts used by this `Runnable`.

### ``pipe [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.pipe "Copy anchor link to this section for reference")

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

### ``pick [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.pick "Copy anchor link to this section for reference")

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

### ``assign [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.assign "Copy anchor link to this section for reference")

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

### ``batch\_as\_completed [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.batch_as_completed "Copy anchor link to this section for reference")

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

### ``abatch\_as\_completed`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.abatch_as_completed "Copy anchor link to this section for reference")

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

### ``astream\_log`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.astream_log "Copy anchor link to this section for reference")

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

### ``astream\_events`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.astream_events "Copy anchor link to this section for reference")

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

### ``bind [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.bind "Copy anchor link to this section for reference")

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

### ``with\_config [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.with_config "Copy anchor link to this section for reference")

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

### ``with\_listeners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.with_listeners "Copy anchor link to this section for reference")

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

### ``with\_alisteners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.with_alisteners "Copy anchor link to this section for reference")

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

### ``with\_types [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.with_types "Copy anchor link to this section for reference")

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

### ``with\_retry [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.with_retry "Copy anchor link to this section for reference")

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

### ``map [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.map "Copy anchor link to this section for reference")

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

### ``with\_fallbacks [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.with_fallbacks "Copy anchor link to this section for reference")

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

### ``as\_tool [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.as_tool "Copy anchor link to this section for reference")

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

### ``lc\_id`classmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.lc_id "Copy anchor link to this section for reference")

```
lc_id() -> list[str]
```

Return a unique identifier for this class for serialization purposes.

The unique identifier is a list of strings that describes the path
to the object.

For example, for the class `langchain.llms.openai.OpenAI`, the id is
`["langchain", "llms", "openai", "OpenAI"]`.

### ``to\_json [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.to_json "Copy anchor link to this section for reference")

```
to_json() -> SerializedConstructor | SerializedNotImplemented
```

Serialize the `Runnable` to JSON.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedConstructor | SerializedNotImplemented` | A JSON-serializable representation of the `Runnable`. |

### ``to\_json\_not\_implemented [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.to_json_not_implemented "Copy anchor link to this section for reference")

```
to_json_not_implemented() -> SerializedNotImplemented
```

Serialize a "not implemented" object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedNotImplemented` | `SerializedNotImplemented`. |

### ``configurable\_fields [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.configurable_fields "Copy anchor link to this section for reference")

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

### ``configurable\_alternatives [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSequence.configurable_alternatives "Copy anchor link to this section for reference")

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

## ``RunnableSerializable [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable "Copy anchor link to this section for reference")

Bases: `Serializable`, `Runnable[Input, Output]`

Runnable that can be serialized to JSON.

| METHOD | DESCRIPTION |
| --- | --- |
| `to_json` | Serialize the `Runnable` to JSON. |
| `configurable_fields` | Configure particular `Runnable` fields at runtime. |
| `configurable_alternatives` | Configure alternatives for `Runnable` objects that can be set at runtime. |
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
| `to_json_not_implemented` | Serialize a "not implemented" object. |

### ``name`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.name "Copy anchor link to this section for reference")

```
name: str | None = None
```

The name of the `Runnable`. Used for debugging and tracing.

### ``InputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.InputType "Copy anchor link to this section for reference")

```
InputType: type[Input]
```

Input type.

The type of input this `Runnable` accepts specified as a type annotation.

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the input type cannot be inferred. |

### ``OutputType`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.OutputType "Copy anchor link to this section for reference")

```
OutputType: type[Output]
```

Output Type.

The type of output this `Runnable` produces specified as a type annotation.

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the output type cannot be inferred. |

### ``input\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.input_schema "Copy anchor link to this section for reference")

```
input_schema: type[BaseModel]
```

The type of input this `Runnable` accepts specified as a Pydantic model.

### ``output\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.output_schema "Copy anchor link to this section for reference")

```
output_schema: type[BaseModel]
```

Output schema.

The type of output this `Runnable` produces specified as a Pydantic model.

### ``config\_specs`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.config_specs "Copy anchor link to this section for reference")

```
config_specs: list[ConfigurableFieldSpec]
```

List configurable fields for this `Runnable`.

### ``lc\_secrets`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.lc_secrets "Copy anchor link to this section for reference")

```
lc_secrets: dict[str, str]
```

A map of constructor argument names to secret ids.

For example, `{"openai_api_key": "OPENAI_API_KEY"}`

### ``lc\_attributes`property`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.lc_attributes "Copy anchor link to this section for reference")

```
lc_attributes: dict
```

List of attribute names that should be included in the serialized kwargs.

These attributes must be accepted by the constructor.

Default is an empty dictionary.

### ``to\_json [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.to_json "Copy anchor link to this section for reference")

```
to_json() -> SerializedConstructor | SerializedNotImplemented
```

Serialize the `Runnable` to JSON.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedConstructor | SerializedNotImplemented` | A JSON-serializable representation of the `Runnable`. |

### ``configurable\_fields [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.configurable_fields "Copy anchor link to this section for reference")

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

### ``configurable\_alternatives [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.configurable_alternatives "Copy anchor link to this section for reference")

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

### ``get\_name [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.get_name "Copy anchor link to this section for reference")

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

### ``get\_input\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.get_input_schema "Copy anchor link to this section for reference")

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

### ``get\_input\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.get_input_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_output\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.get_output_schema "Copy anchor link to this section for reference")

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

### ``get\_output\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.get_output_jsonschema "Copy anchor link to this section for reference")

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

### ``config\_schema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.config_schema "Copy anchor link to this section for reference")

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

### ``get\_config\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.get_config_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_graph [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.get_graph "Copy anchor link to this section for reference")

```
get_graph(config: RunnableConfig | None = None) -> Graph
```

Return a graph representation of this `Runnable`.

### ``get\_prompts [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.get_prompts "Copy anchor link to this section for reference")

```
get_prompts(config: RunnableConfig | None = None) -> list[BasePromptTemplate]
```

Return a list of prompts used by this `Runnable`.

### ``\_\_or\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.__or__ "Copy anchor link to this section for reference")

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

### ``\_\_ror\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.__ror__ "Copy anchor link to this section for reference")

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

### ``pipe [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.pipe "Copy anchor link to this section for reference")

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

### ``pick [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.pick "Copy anchor link to this section for reference")

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

### ``assign [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.assign "Copy anchor link to this section for reference")

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

### ``invoke`abstractmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.invoke "Copy anchor link to this section for reference")

```
invoke(input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Output
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``ainvoke`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.ainvoke "Copy anchor link to this section for reference")

```
ainvoke(input: Input, config: RunnableConfig | None = None, **kwargs: Any) -> Output
```

Transform a single input into an output.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The input to the `Runnable`.<br>**TYPE:**`Input` |
| `config` | A config to use when invoking the `Runnable`.<br>The config supports standard keys like `'tags'`, `'metadata'` for<br>tracing purposes, `'max_concurrency'` for controlling how much work to<br>do in parallel, and other keys.<br>Please refer to `RunnableConfig` for more details.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Output` | The output of the `Runnable`. |

### ``batch [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.batch "Copy anchor link to this section for reference")

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

### ``batch\_as\_completed [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.batch_as_completed "Copy anchor link to this section for reference")

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

### ``abatch`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.abatch "Copy anchor link to this section for reference")

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

### ``abatch\_as\_completed`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.abatch_as_completed "Copy anchor link to this section for reference")

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

### ``stream [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.stream "Copy anchor link to this section for reference")

```
stream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> Iterator[Output]
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

### ``astream`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.astream "Copy anchor link to this section for reference")

```
astream(
    input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
) -> AsyncIterator[Output]
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

### ``astream\_log`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.astream_log "Copy anchor link to this section for reference")

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

### ``astream\_events`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.astream_events "Copy anchor link to this section for reference")

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

### ``transform [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.transform "Copy anchor link to this section for reference")

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

### ``atransform`async`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.atransform "Copy anchor link to this section for reference")

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

### ``bind [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.bind "Copy anchor link to this section for reference")

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

### ``with\_config [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.with_config "Copy anchor link to this section for reference")

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

### ``with\_listeners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.with_listeners "Copy anchor link to this section for reference")

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

### ``with\_alisteners [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.with_alisteners "Copy anchor link to this section for reference")

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

### ``with\_types [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.with_types "Copy anchor link to this section for reference")

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

### ``with\_retry [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.with_retry "Copy anchor link to this section for reference")

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

### ``map [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.map "Copy anchor link to this section for reference")

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

### ``with\_fallbacks [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.with_fallbacks "Copy anchor link to this section for reference")

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

### ``as\_tool [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.as_tool "Copy anchor link to this section for reference")

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

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.__init__ "Copy anchor link to this section for reference")

```
__init__(*args: Any, **kwargs: Any) -> None
```

### ``is\_lc\_serializable`classmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.is_lc_serializable "Copy anchor link to this section for reference")

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

### ``get\_lc\_namespace`classmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.get_lc_namespace "Copy anchor link to this section for reference")

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

### ``lc\_id`classmethod`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.lc_id "Copy anchor link to this section for reference")

```
lc_id() -> list[str]
```

Return a unique identifier for this class for serialization purposes.

The unique identifier is a list of strings that describes the path
to the object.

For example, for the class `langchain.llms.openai.OpenAI`, the id is
`["langchain", "llms", "openai", "OpenAI"]`.

### ``to\_json\_not\_implemented [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.base.RunnableSerializable.to_json_not_implemented "Copy anchor link to this section for reference")

```
to_json_not_implemented() -> SerializedNotImplemented
```

Serialize a "not implemented" object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedNotImplemented` | `SerializedNotImplemented`. |

## ``RunnableConfig [¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.RunnableConfig "Copy anchor link to this section for reference")

Bases: `TypedDict`

Configuration for a `Runnable`.

Note

The `TypedDict` has `total=False` set intentionally to:

- Allow partial configs to be created and merged together via `merge_configs`
- Support config propagation from parent to child runnables via
`var_child_runnable_config` (a `ContextVar` that automatically passes
config down the call stack without explicit parameter passing), where
configs are merged rather than replaced

Example

```
# Parent sets tags
chain.invoke(input, config={"tags": ["parent"]})
# Child automatically inherits and can add:
# ensure_config({"tags": ["child"]}) -> {"tags": ["parent", "child"]}
```

### ``tags`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.RunnableConfig.tags "Copy anchor link to this section for reference")

```
tags: list[str]
```

Tags for this call and any sub-calls (e.g. a Chain calling an LLM).

You can use these to filter calls.

### ``metadata`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.RunnableConfig.metadata "Copy anchor link to this section for reference")

```
metadata: dict[str, Any]
```

Metadata for this call and any sub-calls (e.g. a Chain calling an LLM).

Keys should be strings, values should be JSON-serializable.

### ``callbacks`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.RunnableConfig.callbacks "Copy anchor link to this section for reference")

```
callbacks: Callbacks
```

Callbacks for this call and any sub-calls (e.g. a Chain calling an LLM).

Tags are passed to all callbacks, metadata is passed to handle\*Start callbacks.

### ``run\_name`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.RunnableConfig.run_name "Copy anchor link to this section for reference")

```
run_name: str
```

Name for the tracer run for this call.

Defaults to the name of the class.

### ``max\_concurrency`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.RunnableConfig.max_concurrency "Copy anchor link to this section for reference")

```
max_concurrency: int | None
```

Maximum number of parallel calls to make.

If not provided, defaults to `ThreadPoolExecutor`'s default.

### ``recursion\_limit`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.RunnableConfig.recursion_limit "Copy anchor link to this section for reference")

```
recursion_limit: int
```

Maximum number of times a call can recurse.

If not provided, defaults to `25`.

### ``configurable`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.RunnableConfig.configurable "Copy anchor link to this section for reference")

```
configurable: dict[str, Any]
```

Runtime values for attributes previously made configurable on this `Runnable`,
or sub-`Runnable` objects, through `configurable_fields` or
`configurable_alternatives`.

Check `output_schema` for a description of the attributes that have been made
configurable.

### ``run\_id`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/runnables/\#langchain_core.runnables.RunnableConfig.run_id "Copy anchor link to this section for reference")

```
run_id: UUID | None
```

Unique identifier for the tracer run for this call.

If not provided, a new UUID will be generated.

Back to top