![Revisit consent button](https://uploads-ssl.webflow.com/65ff950538088944d66126b3/662ef3209b872e92e41212f6_cookieicon.png)

![](https://cdn-cookieyes.com/assets/images/close.svg)

We value your privacy

We use cookies to improve your experience and to understand how our site is used. Some analytics tools may share limited data with our advertising partners. You can opt out at any time.

Do Not Sell or Share My Personal Information

Opt-out Preferences![](https://cdn-cookieyes.com/assets/images/close.svg)

We use cookies to improve your experience and to understand how our site is used. Some analytics tools may share limited data with our advertising partners. You can opt out of this sharing at any time by selecting **“Do Not Sell or Share My Personal Information”** and saving your preferences.

Do Not Sell or Share My Personal Information

CancelSave My Preferences

[Skip to content](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/#chat-model-integration-tests)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_tests/integration_tests/chat_models.md "Edit this page")

# Chat model integration tests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#chat-model-integration-tests "Copy anchor link to this section for reference")

## ``ChatModelIntegrationTests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests "Copy anchor link to this section for reference")

Bases: `ChatModelTests`

Base class for chat model integration tests.

Test subclasses must implement the `chat_model_class` and
`chat_model_params` properties to specify what model to test and its
initialization parameters.

```
from typing import Type

from langchain_tests.integration_tests import ChatModelIntegrationTests
from my_package.chat_models import MyChatModel

class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[MyChatModel]:
        # Return the chat model class to test here
        return MyChatModel

    @property
    def chat_model_params(self) -> dict:
        # Return initialization parameters for the model.
        return {"model": "model-001", "temperature": 0}
```

Note

API references for individual test methods include troubleshooting tips.

Test subclasses **must** implement the following two properties:

`chat_model_class`: The chat model class to test, e.g., `ChatParrotLink`.

```
@property
def chat_model_class(self) -> Type[ChatParrotLink]:
    return ChatParrotLink
```

`chat_model_params`: Initialization parameters for the chat model.

```
@property
def chat_model_params(self) -> dict:
    return {"model": "bird-brain-001", "temperature": 0}
```

In addition, test subclasses can control what features are tested (such as tool
calling or multi-modality) by selectively overriding the following properties.

Expand to see details:

`has_tool_calling`

Boolean property indicating whether the chat model supports tool calling.

By default, this is determined by whether the chat model's `bind_tools` method
is overridden. It typically does not need to be overridden on the test class.

```
@property
def has_tool_calling(self) -> bool:
    return True
```

`has_tool_choice`

Boolean property indicating whether the chat model supports forcing tool
calling via a `tool_choice` parameter.

By default, this is determined by whether the parameter is included in the
signature for the corresponding `bind_tools` method.

If `True`, the minimum requirement for this feature is that
`tool_choice='any'` will force a tool call, and `tool_choice=<tool name>`
will force a call to a specific tool.

```
@property
def has_tool_choice(self) -> bool:
    return False
```

`has_structured_output`

Boolean property indicating whether the chat model supports structured
output.

By default, this is determined by whether the chat model's
`with_structured_output` method is overridden. If the base implementation is
intended to be used, this method should be overridden.

See docs for [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output).

```
@property
def has_structured_output(self) -> bool:
    return True
```

`structured_output_kwargs`

Dict property specifying additional kwargs to pass to
`with_structured_output()` when running structured output tests.

Override this to customize how your model generates structured output.

The most common use case is specifying the `method` parameter:

- `'function_calling'`: Uses tool/function calling to enforce the schema.
- `'json_mode'`: Uses the model's JSON mode.
- `'json_schema'`: Uses native JSON schema support (e.g., OpenAI's structured
outputs).

```
@property
def structured_output_kwargs(self) -> dict:
    return {"method": "json_schema"}
```

`supports_json_mode`

Boolean property indicating whether the chat model supports
`method='json_mode'` in `with_structured_output`.

Defaults to `False`.

JSON mode constrains the model to output valid JSON without enforcing
a specific schema (unlike `'function_calling'` or `'json_schema'` methods).

When using JSON mode, you must prompt the model to output JSON in your
message.

Example

```
structured_llm = llm.with_structured_output(MySchema, method="json_mode")
structured_llm.invoke("... Return the result as JSON.")
```

See docs for [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output).

```
@property
def supports_json_mode(self) -> bool:
    return True
```

`supports_image_inputs`

Boolean property indicating whether the chat model supports image inputs.

Defaults to `False`.

If set to `True`, the chat model will be tested by inputting an
`ImageContentBlock` with the shape:

```
{
    "type": "image",
    "base64": "<base64 image data>",
    "mime_type": "image/jpeg",  # or appropriate MIME type
}
```

In addition to OpenAI-style content blocks:

```
{
    "type": "image_url",
    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
}
```

See docs for [Multimodality](https://docs.langchain.com/oss/python/langchain/models#multimodal).

```
@property
def supports_image_inputs(self) -> bool:
    return True
```

`supports_image_urls`

Boolean property indicating whether the chat model supports image inputs from
URLs.

Defaults to `False`.

If set to `True`, the chat model will be tested using content blocks of the
form

```
{
    "type": "image",
    "url": "https://...",
}
```

See docs for [Multimodality](https://docs.langchain.com/oss/python/langchain/models#multimodal).

```
@property
def supports_image_urls(self) -> bool:
    return True
```

`supports_image_tool_message`

Boolean property indicating whether the chat model supports a `ToolMessage`
that includes image content, e.g. in the OpenAI Chat Completions format.

Defaults to `False`.

```
ToolMessage(
    content=[\
        {\
            "type": "image_url",\
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},\
        },\
    ],
    tool_call_id="1",
    name="random_image",
)
```

...as well as the LangChain `ImageContentBlock` format:

```
ToolMessage(
    content=[\
        {\
            "type": "image",\
            "base64": image_data,\
            "mime_type": "image/jpeg",\
        },\
    ],
    tool_call_id="1",
    name="random_image",
)
```

If set to `True`, the chat model will be tested with message sequences that
include `ToolMessage` objects of this form.

```
@property
def supports_image_tool_message(self) -> bool:
    return True
```

`supports_pdf_inputs`

Boolean property indicating whether the chat model supports PDF inputs.

Defaults to `False`.

If set to `True`, the chat model will be tested by inputting a
`FileContentBlock` with the shape:

```
{
    "type": "file",
    "base64": "<base64 file data>",
    "mime_type": "application/pdf",
}
```

See docs for [Multimodality](https://docs.langchain.com/oss/python/langchain/models#multimodal).

```
@property
def supports_pdf_inputs(self) -> bool:
    return True
```

`supports_pdf_tool_message`

Boolean property indicating whether the chat model supports a `ToolMessage`
that includes PDF content using the LangChain `FileContentBlock` format.

Defaults to `False`.

```
ToolMessage(
    content=[\
        {\
            "type": "file",\
            "base64": pdf_data,\
            "mime_type": "application/pdf",\
        },\
    ],
    tool_call_id="1",
    name="random_pdf",
)
```

If set to `True`, the chat model will be tested with message sequences that
include `ToolMessage` objects of this form.

```
@property
def supports_pdf_tool_message(self) -> bool:
    return True
```

`supports_audio_inputs`

Boolean property indicating whether the chat model supports audio inputs.

Defaults to `False`.

If set to `True`, the chat model will be tested by inputting an
`AudioContentBlock` with the shape:

```
{
    "type": "audio",
    "base64": "<base64 audio data>",
    "mime_type": "audio/wav",  # or appropriate MIME type
}
```

See docs for [Multimodality](https://docs.langchain.com/oss/python/langchain/models#multimodal).

```
@property
def supports_audio_inputs(self) -> bool:
    return True
```

Warning

This test downloads audio data from wikimedia.org. You may need to set the
`LANGCHAIN_TESTS_USER_AGENT` environment variable to identify these tests,
e.g.,

```
export LANGCHAIN_TESTS_USER_AGENT="CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0"
```

Refer to the [Wikimedia Foundation User-Agent Policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy).

`supports_video_inputs`

Boolean property indicating whether the chat model supports image inputs.

Defaults to `False`.

No current tests are written for this feature.

`returns_usage_metadata`

Boolean property indicating whether the chat model returns usage metadata
on invoke and streaming responses.

Defaults to `True`.

`usage_metadata` is an optional dict attribute on `AIMessage` objects that track
input and output tokens.

[See more](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.messages.ai.UsageMetadata).

```
@property
def returns_usage_metadata(self) -> bool:
    return False
```

Models supporting `usage_metadata` should also return the name of the underlying
model in the `response_metadata` of the `AIMessage`.

`supports_anthropic_inputs`

Boolean property indicating whether the chat model supports Anthropic-style
inputs.

Defaults to `False`.

These inputs might feature "tool use" and "tool result" content blocks, e.g.,

```
[\
    {"type": "text", "text": "Hmm let me think about that"},\
    {\
        "type": "tool_use",\
        "input": {"fav_color": "green"},\
        "id": "foo",\
        "name": "color_picker",\
    },\
]
```

If set to `True`, the chat model will be tested using content blocks of this
form.

```
@property
def supports_anthropic_inputs(self) -> bool:
    return True
```

`supported_usage_metadata_details`

Property controlling what usage metadata details are emitted in both invoke
and stream.

Defaults to `{"invoke": [], "stream": []}`.

`usage_metadata` is an optional dict attribute on `AIMessage` objects that track
input and output tokens.

[See more](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.messages.ai.UsageMetadata).

It includes optional keys `input_token_details` and `output_token_details`
that can track usage details associated with special types of tokens, such as
cached, audio, or reasoning.

Only needs to be overridden if these details are supplied.

`enable_vcr_tests`

Property controlling whether to enable select tests that rely on
[VCR](https://vcrpy.readthedocs.io/en/latest/) caching of HTTP calls, such
as benchmarking tests.

Defaults to `False`.

To enable these tests, follow these steps:

1. Override the `enable_vcr_tests` property to return `True`:



```
@property
def enable_vcr_tests(self) -> bool:
       return True
```

2. Configure VCR to exclude sensitive headers and other information from
    cassettes.



Warning



VCR will by default record authentication headers and other sensitive
information in cassettes. Read below for how to configure what
information is recorded in cassettes.



To add configuration to VCR, add a `conftest.py` file to the `tests/`
directory and implement the `vcr_config` fixture there.

`langchain-tests` excludes the headers `'authorization'`,
`'x-api-key'`, and `'api-key'` from VCR cassettes. To pick up this
configuration, you will need to add `conftest.py` as shown below. You can
also exclude additional headers, override the default exclusions, or apply
other customizations to the VCR configuration. See example below:

tests/conftest.py

```
import pytest
from langchain_tests.conftest import base_vcr_config

_EXTRA_HEADERS = [\
       # Specify additional headers to redact\
       ("user-agent", "PLACEHOLDER"),\
]


def remove_response_headers(response: dict) -> dict:
       # If desired, remove or modify headers in the response.
       response["headers"] = {}
       return response


@pytest.fixture(scope="session")
def vcr_config() -> dict:
       """Extend the default configuration from langchain_tests."""
       config = base_vcr_config()
       config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
       config["before_record_response"] = remove_response_headers

       return config
```


Compressing cassettes

`langchain-tests` includes a custom VCR serializer that compresses
cassettes using gzip. To use it, register the `yaml.gz` serializer
to your VCR fixture and enable this serializer in the config. See
example below:



tests/conftest.py

```
import pytest
from langchain_tests.conftest import (
       CustomPersister,
       CustomSerializer,
)
from langchain_tests.conftest import base_vcr_config
from vcr import VCR

_EXTRA_HEADERS = [\
       # Specify additional headers to redact\
       ("user-agent", "PLACEHOLDER"),\
]


def remove_response_headers(response: dict) -> dict:
       # If desired, remove or modify headers in the response.
       response["headers"] = {}
       return response


@pytest.fixture(scope="session")
def vcr_config() -> dict:
       """Extend the default configuration from langchain_tests."""
       config = base_vcr_config()
       config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
       config["before_record_response"] = remove_response_headers
       # New: enable serializer and set file extension
       config["serializer"] = "yaml.gz"
       config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")

       return config


def pytest_recording_configure(config: dict, vcr: VCR) -> None:
       vcr.register_persister(CustomPersister())
       vcr.register_serializer("yaml.gz", CustomSerializer())
```





You can inspect the contents of the compressed cassettes (e.g., to
ensure no sensitive information is recorded) using





```
gunzip -k /path/to/tests/cassettes/TestClass_test.yaml.gz
```





...or by using the serializer:





```
from langchain_tests.conftest import (
       CustomPersister,
       CustomSerializer,
)

cassette_path = "/path/to/tests/cassettes/TestClass_test.yaml.gz"
requests, responses = CustomPersister().load_cassette(
       path, CustomSerializer()
)
```

3. Run tests to generate VCR cassettes.

Example

```
uv run python -m pytest tests/integration_tests/test_chat_models.py::TestMyModel::test_stream_time
```



This will generate a VCR cassette for the test in
`tests/integration_tests/cassettes/`.



Warning



You should inspect the generated cassette to ensure that it does not
contain sensitive information. If it does, you can modify the
`vcr_config` fixture to exclude headers or modify the response
before it is recorded.



You can then commit the cassette to your repository. Subsequent test runs
will use the cassette instead of making HTTP calls.


| METHOD | DESCRIPTION |
| --- | --- |
| `test_no_overrides_DO_NOT_OVERRIDE` | Test that no standard tests are overridden. |
| `model` | Model fixture. |
| `my_adder_tool` | Adder tool fixture. |
| `test_invoke` | Test to verify that `model.invoke(simple_message)` works. |
| `test_ainvoke` | Test to verify that `await model.ainvoke(simple_message)` works. |
| `test_stream` | Test to verify that `model.stream(simple_message)` works. |
| `test_astream` | Test to verify that `await model.astream(simple_message)` works. |
| `test_invoke_with_model_override` | Test that model name can be overridden at invoke time via kwargs. |
| `test_ainvoke_with_model_override` | Test that model name can be overridden at ainvoke time via kwargs. |
| `test_stream_with_model_override` | Test that model name can be overridden at stream time via kwargs. |
| `test_astream_with_model_override` | Test that model name can be overridden at astream time via kwargs. |
| `test_batch` | Test to verify that `model.batch([messages])` works. |
| `test_abatch` | Test to verify that `await model.abatch([messages])` works. |
| `test_conversation` | Test to verify that the model can handle multi-turn conversations. |
| `test_double_messages_conversation` | Test to verify that the model can handle double-message conversations. |
| `test_usage_metadata` | Test to verify that the model returns correct usage metadata. |
| `test_usage_metadata_streaming` | Test usage metadata in streaming mode. |
| `test_stop_sequence` | Test that model does not fail when invoked with the `stop` parameter. |
| `test_tool_calling` | Test that the model generates tool calls. |
| `test_tool_calling_async` | Test that the model generates tool calls. |
| `test_bind_runnables_as_tools` | Test bind runnables as tools. |
| `test_tool_message_histories_string_content` | Test that message histories are compatible with string tool contents. |
| `test_tool_message_histories_list_content` | Test that message histories are compatible with list tool contents. |
| `test_tool_choice` | Test `tool_choice` parameter. |
| `test_tool_calling_with_no_arguments` | Test that the model generates tool calls for tools with no arguments. |
| `test_tool_message_error_status` | Test that `ToolMessage` with `status="error"` can be handled. |
| `test_structured_few_shot_examples` | Test that the model can process few-shot examples with tool calls. |
| `test_structured_output` | Test to verify structured output is generated both on invoke and stream. |
| `test_structured_output_async` | Test to verify structured output is generated both on invoke and stream. |
| `test_structured_output_pydantic_2_v1` | Test structured output using pydantic.v1.BaseModel. |
| `test_structured_output_optional_param` | Test structured output with optional parameters. |
| `test_json_mode` | Test [structured output](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/(https://docs.langchain.com/oss/python/langchain/structured-output)) via JSON mode. |
| `test_pdf_inputs` | Test that the model can process PDF inputs. |
| `test_audio_inputs` | Test that the model can process audio inputs. |
| `test_image_inputs` | Test that the model can process image inputs. |
| `test_image_tool_message` | Test that the model can process `ToolMessage` objects with image inputs. |
| `test_pdf_tool_message` | Test that the model can process `ToolMessage` objects with PDF inputs. |
| `test_anthropic_inputs` | Test that model can process Anthropic-style message histories. |
| `test_message_with_name` | Test that `HumanMessage` with values for the `name` field can be handled. |
| `test_agent_loop` | Test that the model supports a simple ReAct agent loop. |
| `test_stream_time` | Test that streaming does not introduce undue overhead. |
| `invoke_with_audio_input` | Invoke with audio input. |
| `invoke_with_audio_output` | Invoke with audio output. |
| `invoke_with_reasoning_output` | Invoke with reasoning output. |
| `invoke_with_cache_read_input` | Invoke with cache read input. |
| `invoke_with_cache_creation_input` | Invoke with cache creation input. |
| `test_unicode_tool_call_integration` | Generic integration test for Unicode characters in tool calls. |

### ``chat\_model\_class`abstractmethod``property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.chat_model_class "Copy anchor link to this section for reference")

```
chat_model_class: type[BaseChatModel]
```

The chat model class to test, e.g., `ChatParrotLink`.

### ``chat\_model\_params`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.chat_model_params "Copy anchor link to this section for reference")

```
chat_model_params: dict[str, Any]
```

Initialization parameters for the chat model.

### ``has\_tool\_calling`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.has_tool_calling "Copy anchor link to this section for reference")

```
has_tool_calling: bool
```

Whether the model supports tool calling.

### ``has\_tool\_choice`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.has_tool_choice "Copy anchor link to this section for reference")

```
has_tool_choice: bool
```

Whether the model supports tool calling.

### ``has\_structured\_output`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.has_structured_output "Copy anchor link to this section for reference")

```
has_structured_output: bool
```

Whether the chat model supports structured output.

### ``structured\_output\_kwargs`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.structured_output_kwargs "Copy anchor link to this section for reference")

```
structured_output_kwargs: dict[str, Any]
```

Additional kwargs to pass to `with_structured_output()` in tests.

Override this property to customize how structured output is generated
for your model. The most common use case is specifying the `method`
parameter, which controls the mechanism used to enforce structured output:

- `'function_calling'`: Uses tool/function calling to enforce the schema.
- `'json_mode'`: Uses the model's JSON mode.
- `'json_schema'`: Uses native JSON schema support (e.g., OpenAI's
structured outputs).

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, Any]` | A dict of kwargs passed to `with_structured_output()`. |

Example

```
@property
def structured_output_kwargs(self) -> dict:
    return {"method": "json_schema"}
```

### ``supports\_json\_mode`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.supports_json_mode "Copy anchor link to this section for reference")

```
supports_json_mode: bool
```

Whether the chat model supports JSON mode.

### ``supports\_image\_inputs`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.supports_image_inputs "Copy anchor link to this section for reference")

```
supports_image_inputs: bool
```

Supports image inputs.

Whether the chat model supports image inputs, defaults to
`False`.

### ``supports\_image\_urls`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.supports_image_urls "Copy anchor link to this section for reference")

```
supports_image_urls: bool
```

Supports image inputs from URLs.

Whether the chat model supports image inputs from URLs, defaults to
`False`.

### ``supports\_pdf\_inputs`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.supports_pdf_inputs "Copy anchor link to this section for reference")

```
supports_pdf_inputs: bool
```

Whether the chat model supports PDF inputs, defaults to `False`.

### ``supports\_audio\_inputs`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.supports_audio_inputs "Copy anchor link to this section for reference")

```
supports_audio_inputs: bool
```

Supports audio inputs.

Whether the chat model supports audio inputs, defaults to `False`.

### ``supports\_video\_inputs`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.supports_video_inputs "Copy anchor link to this section for reference")

```
supports_video_inputs: bool
```

Supports video inputs.

Whether the chat model supports video inputs, defaults to `False`.

No current tests are written for this feature.

### ``returns\_usage\_metadata`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.returns_usage_metadata "Copy anchor link to this section for reference")

```
returns_usage_metadata: bool
```

Returns usage metadata.

Whether the chat model returns usage metadata on invoke and streaming
responses.

### ``supports\_anthropic\_inputs`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.supports_anthropic_inputs "Copy anchor link to this section for reference")

```
supports_anthropic_inputs: bool
```

Whether the chat model supports Anthropic-style inputs.

### ``supports\_image\_tool\_message`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.supports_image_tool_message "Copy anchor link to this section for reference")

```
supports_image_tool_message: bool
```

Supports image `ToolMessage` objects.

Whether the chat model supports `ToolMessage` objects that include image
content.

### ``supports\_pdf\_tool\_message`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.supports_pdf_tool_message "Copy anchor link to this section for reference")

```
supports_pdf_tool_message: bool
```

Supports PDF `ToolMessage` objects.

Whether the chat model supports `ToolMessage` objects that include PDF
content.

### ``enable\_vcr\_tests`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.enable_vcr_tests "Copy anchor link to this section for reference")

```
enable_vcr_tests: bool
```

Whether to enable VCR tests for the chat model.

Warning

See `enable_vcr_tests` dropdown `above <ChatModelTests>` for more
information.

### ``supported\_usage\_metadata\_details`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.supported_usage_metadata_details "Copy anchor link to this section for reference")

```
supported_usage_metadata_details: dict[\
    Literal["invoke", "stream"],\
    list[\
        Literal[\
            "audio_input",\
            "audio_output",\
            "reasoning_output",\
            "cache_read_input",\
            "cache_creation_input",\
        ]\
    ],\
]
```

Supported usage metadata details.

What usage metadata details are emitted in invoke and stream. Only needs to be
overridden if these details are returned by the model.

### ``supports\_model\_override`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.supports_model_override "Copy anchor link to this section for reference")

```
supports_model_override: bool
```

Whether the model supports overriding the model name at runtime.

Defaults to `True`.

If `True`, the model accepts a `model` kwarg in `invoke()`, `stream()`,
etc. that overrides the model specified at initialization.

This enables dynamic model selection without creating new instances.

### ``model\_override\_value`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.model_override_value "Copy anchor link to this section for reference")

```
model_override_value: str | None
```

Alternative model name to use when testing model override.

Should return a valid model name that differs from the default model.
Required if `supports_model_override` is `True`.

### ``standard\_chat\_model\_params`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.standard_chat_model_params "Copy anchor link to this section for reference")

```
standard_chat_model_params: dict[str, Any]
```

Standard parameters for chat model.

### ``test\_no\_overrides\_DO\_NOT\_OVERRIDE [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_no_overrides_DO_NOT_OVERRIDE "Copy anchor link to this section for reference")

```
test_no_overrides_DO_NOT_OVERRIDE() -> None
```

Test that no standard tests are overridden.

### ``model [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.model "Copy anchor link to this section for reference")

```
model(request: Any) -> BaseChatModel
```

Model fixture.

### ``my\_adder\_tool [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.my_adder_tool "Copy anchor link to this section for reference")

```
my_adder_tool() -> BaseTool
```

Adder tool fixture.

### ``test\_invoke [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_invoke "Copy anchor link to this section for reference")

```
test_invoke(model: BaseChatModel) -> None
```

Test to verify that `model.invoke(simple_message)` works.

This should pass for all integrations.

Troubleshooting

If this test fails, you should make sure your `_generate` method
does not raise any exceptions, and that it returns a valid
`langchain_core.outputs.chat_result.ChatResult` like so:

```
return ChatResult(
    generations=[ChatGeneration(message=AIMessage(content="Output text"))]
)
```

### ``test\_ainvoke`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_ainvoke "Copy anchor link to this section for reference")

```
test_ainvoke(model: BaseChatModel) -> None
```

Test to verify that `await model.ainvoke(simple_message)` works.

This should pass for all integrations. Passing this test does not indicate
a "natively async" implementation, but rather that the model can be used
in an async context.

Troubleshooting

First, debug
`langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`.
because `ainvoke` has a default implementation that calls `invoke` in an
async context.

If that test passes but not this one, you should make sure your `_agenerate`
method does not raise any exceptions, and that it returns a valid
`langchain_core.outputs.chat_result.ChatResult` like so:

```
return ChatResult(
    generations=[ChatGeneration(message=AIMessage(content="Output text"))]
)
```

### ``test\_stream [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_stream "Copy anchor link to this section for reference")

```
test_stream(model: BaseChatModel) -> None
```

Test to verify that `model.stream(simple_message)` works.

This should pass for all integrations. Passing this test does not indicate
a "streaming" implementation, but rather that the model can be used in a
streaming context.

Troubleshooting

First, debug
`langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`.
because `stream` has a default implementation that calls `invoke` and
yields the result as a single chunk.

If that test passes but not this one, you should make sure your `_stream`
method does not raise any exceptions, and that it yields valid
`langchain_core.outputs.chat_generation.ChatGenerationChunk`
objects like so:

```
yield ChatGenerationChunk(message=AIMessageChunk(content="chunk text"))
```

The final chunk must have `chunk_position='last'` to signal stream
completion. This enables proper parsing of `tool_call_chunks` into
`tool_calls` on the aggregated message:

```
for i, token in enumerate(tokens):
    is_last = i == len(tokens) - 1
    yield ChatGenerationChunk(
        message=AIMessageChunk(
            content=token,
            chunk_position="last" if is_last else None,
        )
    )
```

### ``test\_astream`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_astream "Copy anchor link to this section for reference")

```
test_astream(model: BaseChatModel) -> None
```

Test to verify that `await model.astream(simple_message)` works.

This should pass for all integrations. Passing this test does not indicate
a "natively async" or "streaming" implementation, but rather that the model can
be used in an async streaming context.

Troubleshooting

First, debug
`langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_stream`.
and
`langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_ainvoke`.
because `astream` has a default implementation that calls `_stream` in
an async context if it is implemented, or `ainvoke` and yields the result
as a single chunk if not.

If those tests pass but not this one, you should make sure your `_astream`
method does not raise any exceptions, and that it yields valid
`langchain_core.outputs.chat_generation.ChatGenerationChunk`
objects like so:

```
yield ChatGenerationChunk(message=AIMessageChunk(content="chunk text"))
```

See `test_stream` troubleshooting for `chunk_position` requirements.

### ``test\_invoke\_with\_model\_override [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_invoke_with_model_override "Copy anchor link to this section for reference")

```
test_invoke_with_model_override(model: BaseChatModel) -> None
```

Test that model name can be overridden at invoke time via kwargs.

This enables dynamic model selection without creating new instances,
which is useful for fallback strategies, A/B testing, or cost optimization.

Test is skipped if `supports_model_override` is `False`.

Troubleshooting

If this test fails, ensure that your `_generate` method passes
`**kwargs` through to the API request payload in a way that allows
the `model` parameter to be overridden.

For example:

```
def _get_request_payload(self, ..., **kwargs) -> dict:
    return {
        "model": self.model,
        ...
        **kwargs,  # kwargs should come last to allow overrides
    }
```

### ``test\_ainvoke\_with\_model\_override`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_ainvoke_with_model_override "Copy anchor link to this section for reference")

```
test_ainvoke_with_model_override(model: BaseChatModel) -> None
```

Test that model name can be overridden at ainvoke time via kwargs.

Test is skipped if `supports_model_override` is `False`.

Troubleshooting

See troubleshooting for `test_invoke_with_model_override`.

### ``test\_stream\_with\_model\_override [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_stream_with_model_override "Copy anchor link to this section for reference")

```
test_stream_with_model_override(model: BaseChatModel) -> None
```

Test that model name can be overridden at stream time via kwargs.

Test is skipped if `supports_model_override` is `False`.

Troubleshooting

See troubleshooting for `test_invoke_with_model_override`.

### ``test\_astream\_with\_model\_override`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_astream_with_model_override "Copy anchor link to this section for reference")

```
test_astream_with_model_override(model: BaseChatModel) -> None
```

Test that model name can be overridden at astream time via kwargs.

Test is skipped if `supports_model_override` is `False`.

Troubleshooting

See troubleshooting for `test_invoke_with_model_override`.

### ``test\_batch [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_batch "Copy anchor link to this section for reference")

```
test_batch(model: BaseChatModel) -> None
```

Test to verify that `model.batch([messages])` works.

This should pass for all integrations. Tests the model's ability to process
multiple prompts in a single batch.

Troubleshooting

First, debug
`langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`
because `batch` has a default implementation that calls `invoke` for
each message in the batch.

If that test passes but not this one, you should make sure your `batch`
method does not raise any exceptions, and that it returns a list of valid
`AIMessage` objects.

### ``test\_abatch`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_abatch "Copy anchor link to this section for reference")

```
test_abatch(model: BaseChatModel) -> None
```

Test to verify that `await model.abatch([messages])` works.

This should pass for all integrations. Tests the model's ability to process
multiple prompts in a single batch asynchronously.

Troubleshooting

First, debug
`langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_batch`
and
`langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_ainvoke`
because `abatch` has a default implementation that calls `ainvoke` for
each message in the batch.

If those tests pass but not this one, you should make sure your `abatch`
method does not raise any exceptions, and that it returns a list of valid
`AIMessage` objects.

### ``test\_conversation [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_conversation "Copy anchor link to this section for reference")

```
test_conversation(model: BaseChatModel) -> None
```

Test to verify that the model can handle multi-turn conversations.

This should pass for all integrations. Tests the model's ability to process
a sequence of alternating `HumanMessage` and `AIMessage` objects as context for
generating the next response.

Troubleshooting

First, debug
`langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`
because this test also uses `model.invoke`.

If that test passes but not this one, you should verify that:

1. Your model correctly processes the message history
2. The model maintains appropriate context from previous messages
3. The response is a valid `langchain_core.messages.AIMessage`

### ``test\_double\_messages\_conversation [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_double_messages_conversation "Copy anchor link to this section for reference")

```
test_double_messages_conversation(model: BaseChatModel) -> None
```

Test to verify that the model can handle double-message conversations.

This should pass for all integrations. Tests the model's ability to process
a sequence of double-system, double-human, and double-ai messages as context
for generating the next response.

Troubleshooting

First, debug
`langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_invoke`
because this test also uses `model.invoke`.

Second, debug
`langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_conversation`
because this test is the "basic case" without double messages.

If that test passes those but not this one, you should verify that:

1. Your model API can handle double messages, or the integration should
    merge messages before sending them to the API.
2. The response is a valid `langchain_core.messages.AIMessage`

### ``test\_usage\_metadata [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_usage_metadata "Copy anchor link to this section for reference")

```
test_usage_metadata(model: BaseChatModel) -> None
```

Test to verify that the model returns correct usage metadata.

This test is optional and should be skipped if the model does not return
usage metadata (see configuration below).

Behavior changed in `langchain-tests` 0.3.17

Additionally check for the presence of `model_name` in the response
metadata, which is needed for usage tracking in callback handlers.

Configuration

By default, this test is run.

To disable this feature, set `returns_usage_metadata` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def returns_usage_metadata(self) -> bool:
        return False
```

This test can also check the format of specific kinds of usage metadata
based on the `supported_usage_metadata_details` property.

This property should be configured as follows with the types of tokens that
the model supports tracking:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def supported_usage_metadata_details(self) -> dict:
        return {
            "invoke": [\
                "audio_input",\
                "audio_output",\
                "reasoning_output",\
                "cache_read_input",\
                "cache_creation_input",\
            ],
            "stream": [\
                "audio_input",\
                "audio_output",\
                "reasoning_output",\
                "cache_read_input",\
                "cache_creation_input",\
            ],
        }
```

Troubleshooting

If this test fails, first verify that your model returns
`langchain_core.messages.ai.UsageMetadata` dicts
attached to the returned `AIMessage` object in `_generate`:

```
return ChatResult(
    generations=[\
        ChatGeneration(\
            message=AIMessage(\
                content="Output text",\
                usage_metadata={\
                    "input_tokens": 350,\
                    "output_tokens": 240,\
                    "total_tokens": 590,\
                    "input_token_details": {\
                        "audio": 10,\
                        "cache_creation": 200,\
                        "cache_read": 100,\
                    },\
                    "output_token_details": {\
                        "audio": 10,\
                        "reasoning": 200,\
                    },\
                },\
            )\
        )\
    ]
)
```

Check also that the response includes a `model_name` key in its
`usage_metadata`.

### ``test\_usage\_metadata\_streaming [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_usage_metadata_streaming "Copy anchor link to this section for reference")

```
test_usage_metadata_streaming(model: BaseChatModel) -> None
```

Test usage metadata in streaming mode.

Test to verify that the model returns correct usage metadata in streaming mode.

Behavior changed in `langchain-tests` 0.3.17

Additionally check for the presence of `model_name` in the response
metadata, which is needed for usage tracking in callback handlers.

Configuration

By default, this test is run.
To disable this feature, set `returns_usage_metadata` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def returns_usage_metadata(self) -> bool:
        return False
```

This test can also check the format of specific kinds of usage metadata
based on the `supported_usage_metadata_details` property.

This property should be configured as follows with the types of tokens that
the model supports tracking:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def supported_usage_metadata_details(self) -> dict:
        return {
            "invoke": [\
                "audio_input",\
                "audio_output",\
                "reasoning_output",\
                "cache_read_input",\
                "cache_creation_input",\
            ],
            "stream": [\
                "audio_input",\
                "audio_output",\
                "reasoning_output",\
                "cache_read_input",\
                "cache_creation_input",\
            ],
        }
```

Troubleshooting

If this test fails, first verify that your model yields
`langchain_core.messages.ai.UsageMetadata` dicts
attached to the returned `AIMessage` object in `_stream`
that sum up to the total usage metadata.

Note that `input_tokens` should only be included on one of the chunks
(typically the first or the last chunk), and the rest should have `0` or
`None` to avoid counting input tokens multiple times.

`output_tokens` typically count the number of tokens in each chunk, not
the sum. This test will pass as long as the sum of `output_tokens` across
all chunks is not `0`.

```
yield ChatResult(
    generations=[\
        ChatGeneration(\
            message=AIMessage(\
                content="Output text",\
                usage_metadata={\
                    "input_tokens": (\
                        num_input_tokens if is_first_chunk else 0\
                    ),\
                    "output_tokens": 11,\
                    "total_tokens": (\
                        11 + num_input_tokens if is_first_chunk else 11\
                    ),\
                    "input_token_details": {\
                        "audio": 10,\
                        "cache_creation": 200,\
                        "cache_read": 100,\
                    },\
                    "output_token_details": {\
                        "audio": 10,\
                        "reasoning": 200,\
                    },\
                },\
            )\
        )\
    ]
)
```

Check also that the aggregated response includes a `model_name` key
in its `usage_metadata`.

### ``test\_stop\_sequence [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_stop_sequence "Copy anchor link to this section for reference")

```
test_stop_sequence(model: BaseChatModel) -> None
```

Test that model does not fail when invoked with the `stop` parameter.

The `stop` parameter is a standard parameter for stopping generation at a
certain token.

[More on standard parameters](https://python.langchain.com/docs/concepts/chat_models/#standard-parameters).

This should pass for all integrations.

Troubleshooting

If this test fails, check that the function signature for `_generate`
(as well as `_stream` and async variants) accepts the `stop` parameter:

```
def _generate(
    self,
    messages: List[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> ChatResult:
```

### ``test\_tool\_calling [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_tool_calling "Copy anchor link to this section for reference")

```
test_tool_calling(model: BaseChatModel) -> None
```

Test that the model generates tool calls.

This test is skipped if the `has_tool_calling` property on the test class is
set to `False`.

This test is optional and should be skipped if the model does not support
tool calling (see configuration below).

Configuration

To disable tool calling tests, set `has_tool_calling` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_tool_calling(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that `bind_tools` is implemented to correctly
translate LangChain tool objects into the appropriate schema for your
chat model.

This test may fail if the chat model does not support a `tool_choice`
parameter. This parameter can be used to force a tool call. If
`tool_choice` is not supported and the model consistently fails this
test, you can `xfail` the test:

```
@pytest.mark.xfail(reason=("Does not support tool_choice."))
def test_tool_calling(self, model: BaseChatModel) -> None:
    super().test_tool_calling(model)
```

Otherwise, in the case that only one tool is bound, ensure that
`tool_choice` supports the string `'any'` to force calling that tool.

### ``test\_tool\_calling\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_tool_calling_async "Copy anchor link to this section for reference")

```
test_tool_calling_async(model: BaseChatModel) -> None
```

Test that the model generates tool calls.

This test is skipped if the `has_tool_calling` property on the test class is
set to `False`.

This test is optional and should be skipped if the model does not support
tool calling (see configuration below).

Configuration

To disable tool calling tests, set `has_tool_calling` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_tool_calling(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that `bind_tools` is implemented to correctly
translate LangChain tool objects into the appropriate schema for your
chat model.

This test may fail if the chat model does not support a `tool_choice`
parameter. This parameter can be used to force a tool call. If
`tool_choice` is not supported and the model consistently fails this
test, you can `xfail` the test:

```
@pytest.mark.xfail(reason=("Does not support tool_choice."))
async def test_tool_calling_async(self, model: BaseChatModel) -> None:
    await super().test_tool_calling_async(model)
```

Otherwise, in the case that only one tool is bound, ensure that
`tool_choice` supports the string `'any'` to force calling that tool.

### ``test\_bind\_runnables\_as\_tools [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_bind_runnables_as_tools "Copy anchor link to this section for reference")

```
test_bind_runnables_as_tools(model: BaseChatModel) -> None
```

Test bind runnables as tools.

Test that the model generates tool calls for tools that are derived from
LangChain runnables. This test is skipped if the `has_tool_calling` property
on the test class is set to `False`.

This test is optional and should be skipped if the model does not support
tool calling (see configuration below).

Configuration

To disable tool calling tests, set `has_tool_calling` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_tool_calling(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that `bind_tools` is implemented to correctly
translate LangChain tool objects into the appropriate schema for your
chat model.

This test may fail if the chat model does not support a `tool_choice`
parameter. This parameter can be used to force a tool call. If
`tool_choice` is not supported, set `has_tool_choice` to `False` in
your test class:

```
@property
def has_tool_choice(self) -> bool:
    return False
```

### ``test\_tool\_message\_histories\_string\_content [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_tool_message_histories_string_content "Copy anchor link to this section for reference")

```
test_tool_message_histories_string_content(
    model: BaseChatModel, my_adder_tool: BaseTool
) -> None
```

Test that message histories are compatible with string tool contents.

For instance with OpenAI format contents.
If a model passes this test, it should be compatible
with messages generated from providers following OpenAI format.

This test should be skipped if the model does not support tool calling
(see configuration below).

Configuration

To disable tool calling tests, set `has_tool_calling` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_tool_calling(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that:

1. The model can correctly handle message histories that include
    `AIMessage` objects with `""` content.
2. The `tool_calls` attribute on `AIMessage` objects is correctly
    handled and passed to the model in an appropriate format.
3. The model can correctly handle `ToolMessage` objects with string
    content and arbitrary string values for `tool_call_id`.

You can `xfail` the test if tool calling is implemented but this format
is not supported.

```
@pytest.mark.xfail(reason=("Not implemented."))
def test_tool_message_histories_string_content(self, *args: Any) -> None:
    super().test_tool_message_histories_string_content(*args)
```

### ``test\_tool\_message\_histories\_list\_content [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_tool_message_histories_list_content "Copy anchor link to this section for reference")

```
test_tool_message_histories_list_content(
    model: BaseChatModel, my_adder_tool: BaseTool
) -> None
```

Test that message histories are compatible with list tool contents.

For instance with Anthropic format contents.

These message histories will include `AIMessage` objects with "tool use" and
content blocks, e.g.,

```
[\
    {"type": "text", "text": "Hmm let me think about that"},\
    {\
        "type": "tool_use",\
        "input": {"fav_color": "green"},\
        "id": "foo",\
        "name": "color_picker",\
    },\
]
```

This test should be skipped if the model does not support tool calling
(see configuration below).

Configuration

To disable tool calling tests, set `has_tool_calling` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_tool_calling(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that:

1. The model can correctly handle message histories that include
    `AIMessage` objects with list content.
2. The `tool_calls` attribute on `AIMessage` objects is correctly
    handled and passed to the model in an appropriate format.
3. The model can correctly handle ToolMessage objects with string content
    and arbitrary string values for `tool_call_id`.

You can `xfail` the test if tool calling is implemented but this format
is not supported.

```
@pytest.mark.xfail(reason=("Not implemented."))
def test_tool_message_histories_list_content(self, *args: Any) -> None:
    super().test_tool_message_histories_list_content(*args)
```

### ``test\_tool\_choice [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_tool_choice "Copy anchor link to this section for reference")

```
test_tool_choice(model: BaseChatModel) -> None
```

Test `tool_choice` parameter.

Test that the model can force tool calling via the `tool_choice`
parameter. This test is skipped if the `has_tool_choice` property on the
test class is set to `False`.

This test is optional and should be skipped if the model does not support
tool calling (see configuration below).

Configuration

To disable tool calling tests, set `has_tool_choice` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_tool_choice(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check whether the `test_tool_calling` test is passing.
If it is not, refer to the troubleshooting steps in that test first.

If `test_tool_calling` is passing, check that the underlying model
supports forced tool calling. If it does, `bind_tools` should accept a
`tool_choice` parameter that can be used to force a tool call.

It should accept (1) the string `'any'` to force calling the bound tool,
and (2) the string name of the tool to force calling that tool.

### ``test\_tool\_calling\_with\_no\_arguments [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_tool_calling_with_no_arguments "Copy anchor link to this section for reference")

```
test_tool_calling_with_no_arguments(model: BaseChatModel) -> None
```

Test that the model generates tool calls for tools with no arguments.

This test is skipped if the `has_tool_calling` property on the test class
is set to `False`.

This test is optional and should be skipped if the model does not support
tool calling (see configuration below).

Configuration

To disable tool calling tests, set `has_tool_calling` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_tool_calling(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that `bind_tools` is implemented to correctly
translate LangChain tool objects into the appropriate schema for your
chat model. It should correctly handle the case where a tool has no
arguments.

This test may fail if the chat model does not support a `tool_choice`
parameter. This parameter can be used to force a tool call. It may also
fail if a provider does not support this form of tool. In these cases,
you can `xfail` the test:

```
@pytest.mark.xfail(reason=("Does not support tool_choice."))
def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
    super().test_tool_calling_with_no_arguments(model)
```

Otherwise, in the case that only one tool is bound, ensure that
`tool_choice` supports the string `'any'` to force calling that tool.

### ``test\_tool\_message\_error\_status [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_tool_message_error_status "Copy anchor link to this section for reference")

```
test_tool_message_error_status(model: BaseChatModel, my_adder_tool: BaseTool) -> None
```

Test that `ToolMessage` with `status="error"` can be handled.

These messages may take the form:

```
ToolMessage(
    "Error: Missing required argument 'b'.",
    name="my_adder_tool",
    tool_call_id="abc123",
    status="error",
)
```

If possible, the `status` field should be parsed and passed appropriately
to the model.

This test is optional and should be skipped if the model does not support
tool calling (see configuration below).

Configuration

To disable tool calling tests, set `has_tool_calling` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_tool_calling(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that the `status` field on `ToolMessage`
objects is either ignored or passed to the model appropriately.

### ``test\_structured\_few\_shot\_examples [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_structured_few_shot_examples "Copy anchor link to this section for reference")

```
test_structured_few_shot_examples(
    model: BaseChatModel, my_adder_tool: BaseTool
) -> None
```

Test that the model can process few-shot examples with tool calls.

These are represented as a sequence of messages of the following form:

- `HumanMessage` with string content;
- `AIMessage` with the `tool_calls` attribute populated;
- `ToolMessage` with string content;
- `AIMessage` with string content (an answer);
- `HumanMessage` with string content (a follow-up question).

This test should be skipped if the model does not support tool calling
(see configuration below).

Configuration

To disable tool calling tests, set `has_tool_calling` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_tool_calling(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that the model can correctly handle this
sequence of messages.

You can `xfail` the test if tool calling is implemented but this format
is not supported.

```
@pytest.mark.xfail(reason=("Not implemented."))
def test_structured_few_shot_examples(self, *args: Any) -> None:
    super().test_structured_few_shot_examples(*args)
```

### ``test\_structured\_output [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_structured_output "Copy anchor link to this section for reference")

```
test_structured_output(
    model: BaseChatModel, schema_type: Literal["pydantic", "typeddict", "json_schema"]
) -> None
```

Test to verify structured output is generated both on invoke and stream.

This test is optional and should be skipped if the model does not support
structured output (see configuration below).

Configuration

To disable structured output tests, set `has_structured_output` to `False`
in your test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_structured_output(self) -> bool:
        return False
```

By default, `has_structured_output` is `True` if a model overrides the
`with_structured_output` or `bind_tools` methods.

Troubleshooting

If this test fails, ensure that the model's `bind_tools` method
properly handles both JSON Schema and Pydantic V2 models.

`langchain_core` implements a [utility function](https://reference.langchain.com/python/langchain_core/utils/?h=convert_to_op#langchain_core.utils.function_calling.convert_to_openai_tool).
that will accommodate most formats.

See [example implementation](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py).
of `with_structured_output`.

### ``test\_structured\_output\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_structured_output_async "Copy anchor link to this section for reference")

```
test_structured_output_async(
    model: BaseChatModel, schema_type: Literal["pydantic", "typeddict", "json_schema"]
) -> None
```

Test to verify structured output is generated both on invoke and stream.

This test is optional and should be skipped if the model does not support
structured output (see configuration below).

Configuration

To disable structured output tests, set `has_structured_output` to `False`
in your test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_structured_output(self) -> bool:
        return False
```

By default, `has_structured_output` is `True` if a model overrides the
`with_structured_output` or `bind_tools` methods.

Troubleshooting

If this test fails, ensure that the model's `bind_tools` method
properly handles both JSON Schema and Pydantic V2 models.

`langchain_core` implements a [utility function](https://reference.langchain.com/python/langchain_core/utils/?h=convert_to_op#langchain_core.utils.function_calling.convert_to_openai_tool).
that will accommodate most formats.

See [example implementation](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py).
of `with_structured_output`.

### ``test\_structured\_output\_pydantic\_2\_v1 [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_structured_output_pydantic_2_v1 "Copy anchor link to this section for reference")

```
test_structured_output_pydantic_2_v1(model: BaseChatModel) -> None
```

Test structured output using pydantic.v1.BaseModel.

Verify we can generate structured output using `pydantic.v1.BaseModel`.

`pydantic.v1.BaseModel` is available in the Pydantic 2 package.

This test is optional and should be skipped if the model does not support
structured output (see configuration below).

Configuration

To disable structured output tests, set `has_structured_output` to `False`
in your test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_structured_output(self) -> bool:
        return False
```

By default, `has_structured_output` is `True` if a model overrides the
`with_structured_output` or `bind_tools` methods.

Troubleshooting

If this test fails, ensure that the model's `bind_tools` method
properly handles both JSON Schema and Pydantic V1 models.

`langchain_core` implements a [utility function](https://reference.langchain.com/python/langchain_core/utils/?h=convert_to_op#langchain_core.utils.function_calling.convert_to_openai_tool).
that will accommodate most formats.

See [example implementation](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py).
of `with_structured_output`.

### ``test\_structured\_output\_optional\_param [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_structured_output_optional_param "Copy anchor link to this section for reference")

```
test_structured_output_optional_param(model: BaseChatModel) -> None
```

Test structured output with optional parameters.

Test to verify we can generate structured output that includes optional
parameters.

This test is optional and should be skipped if the model does not support
structured output (see configuration below).

Configuration

To disable structured output tests, set `has_structured_output` to `False`
in your test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_structured_output(self) -> bool:
        return False
```

By default, `has_structured_output` is True if a model overrides the
`with_structured_output` or `bind_tools` methods.

Troubleshooting

If this test fails, ensure that the model's `bind_tools` method
properly handles Pydantic V2 models with optional parameters.

`langchain_core` implements a [utility function](https://reference.langchain.com/python/langchain_core/utils/?h=convert_to_op#langchain_core.utils.function_calling.convert_to_openai_tool).
that will accommodate most formats.

See [example implementation](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py).
of `with_structured_output`.

### ``test\_json\_mode [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_json_mode "Copy anchor link to this section for reference")

```
test_json_mode(model: BaseChatModel) -> None
```

Test [structured output](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/(https://docs.langchain.com/oss/python/langchain/structured-output)) via JSON mode.

This test is optional and should be skipped if the model does not support
the JSON mode feature (see configuration below).

Configuration

To disable this test, set `supports_json_mode` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def supports_json_mode(self) -> bool:
        return False
```

Troubleshooting

See example implementation of `with_structured_output` here: [https://python.langchain.com/api\_reference/\_modules/langchain\_openai/chat\_models/base.html#BaseChatOpenAI.with\_structured\_output](https://python.langchain.com/api_reference/_modules/langchain_openai/chat_models/base.html#BaseChatOpenAI.with_structured_output)

### ``test\_pdf\_inputs [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_pdf_inputs "Copy anchor link to this section for reference")

```
test_pdf_inputs(model: BaseChatModel) -> None
```

Test that the model can process PDF inputs.

This test should be skipped (see configuration below) if the model does not
support PDF inputs. These will take the shape of the LangChain
`FileContentBlock`:

```
{
    "type": "image",
    "base64": "<base64 image data>",
    "mime_type": "application/pdf",
}
```

Furthermore, for backward-compatibility, we must also support OpenAI chat
completions file content blocks:

```
(
    {
        "type": "file",
        "file": {
            "filename": "test_file.pdf",
            "file_data": f"data:application/pdf;base64,{pdf_data}",
        },
    },
)
```

Configuration

To disable this test, set `supports_pdf_inputs` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def supports_pdf_inputs(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that the model can correctly handle messages
with pdf content blocks, including base64-encoded files. Otherwise, set
the `supports_pdf_inputs` property to `False`.

### ``test\_audio\_inputs [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_audio_inputs "Copy anchor link to this section for reference")

```
test_audio_inputs(model: BaseChatModel) -> None
```

Test that the model can process audio inputs.

This test should be skipped (see configuration below) if the model does not
support audio inputs. These will take the shape of the LangChain
`AudioContentBlock`:

```
{
    "type": "audio",
    "base64": "<base64 audio data>",
    "mime_type": "audio/wav",  # or appropriate MIME type
}
```

Furthermore, for backward-compatibility, we must also support OpenAI chat
completions audio content blocks:

```
{
    "type": "input_audio",
    "input_audio": {
        "data": "<base64 audio data>",
        "format": "wav",  # or appropriate format
    },
}
```

Note: this test downloads audio data from wikimedia.org. You may need to set
the `LANGCHAIN_TESTS_USER_AGENT` environment variable to identify these
requests, e.g.,

```
export LANGCHAIN_TESTS_USER_AGENT="CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0"
```

Refer to the [Wikimedia Foundation User-Agent Policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy).

Configuration

To disable this test, set `supports_audio_inputs` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def supports_audio_inputs(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that the model can correctly handle messages
with audio content blocks, specifically base64-encoded files. Otherwise,
set the `supports_audio_inputs` property to `False`.

### ``test\_image\_inputs [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_image_inputs "Copy anchor link to this section for reference")

```
test_image_inputs(model: BaseChatModel) -> None
```

Test that the model can process image inputs.

This test should be skipped (see configuration below) if the model does not
support image inputs. These will take the shape of the LangChain
`ImageContentBlock`:

```
{
    "type": "image",
    "base64": "<base64 image data>",
    "mime_type": "image/jpeg",  # or appropriate MIME type
}
```

For backward-compatibility, we must also support OpenAI chat completions
image content blocks containing base64-encoded images:

```
[\
    {"type": "text", "text": "describe the weather in this image"},\
    {\
        "type": "image_url",\
        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},\
    },\
]
```

See docs for [Multimodality](https://docs.langchain.com/oss/python/langchain/models#multimodal).

If the property `supports_image_urls` is set to `True`, the test will also
check that we can process content blocks of the form:

```
{
    "type": "image",
    "url": "<url>",
}
```

Configuration

To disable this test, set `supports_image_inputs` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def supports_image_inputs(self) -> bool:
        return False

    # Can also explicitly disable testing image URLs:
    @property
    def supports_image_urls(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that the model can correctly handle messages
with image content blocks, including base64-encoded images. Otherwise, set
the `supports_image_inputs` property to `False`.

### ``test\_image\_tool\_message [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_image_tool_message "Copy anchor link to this section for reference")

```
test_image_tool_message(model: BaseChatModel) -> None
```

Test that the model can process `ToolMessage` objects with image inputs.

This test should be skipped if the model does not support messages of the
Chat Completions `image_url` format:

```
ToolMessage(
    content=[\
        {\
            "type": "image_url",\
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},\
        },\
    ],
    tool_call_id="1",
    name="random_image",
)
```

In addition, models should support the standard LangChain `ImageContentBlock`
format:

```
ToolMessage(
    content=[\
        {\
            "type": "image",\
            "base64": image_data,\
            "mime_type": "image/jpeg",\
        },\
    ],
    tool_call_id="1",
    name="random_image",
)
```

This test can be skipped by setting the `supports_image_tool_message` property
to `False` (see configuration below).

Configuration

To disable this test, set `supports_image_tool_message` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def supports_image_tool_message(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that the model can correctly handle messages
with image content blocks in `ToolMessage` objects, including base64-encoded
images. Otherwise, set the `supports_image_tool_message` property to
`False`.

### ``test\_pdf\_tool\_message [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_pdf_tool_message "Copy anchor link to this section for reference")

```
test_pdf_tool_message(model: BaseChatModel) -> None
```

Test that the model can process `ToolMessage` objects with PDF inputs.

This test should be skipped if the model does not support messages of the
LangChain `FileContentBlock` format:

```
ToolMessage(
    content=[\
        {\
            "type": "file",\
            "base64": pdf_data,\
            "mime_type": "application/pdf",\
        },\
    ],
    tool_call_id="1",
    name="random_pdf",
)
```

This test can be skipped by setting the `supports_pdf_tool_message` property
to `False` (see configuration below).

Configuration

To disable this test, set `supports_pdf_tool_message` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def supports_pdf_tool_message(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that the model can correctly handle messages
with PDF content blocks in `ToolMessage` objects, specifically
base64-encoded PDFs. Otherwise, set the `supports_pdf_tool_message` property
to `False`.

### ``test\_anthropic\_inputs [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_anthropic_inputs "Copy anchor link to this section for reference")

```
test_anthropic_inputs(model: BaseChatModel) -> None
```

Test that model can process Anthropic-style message histories.

These message histories will include `AIMessage` objects with `tool_use`
content blocks, e.g.,

```
AIMessage(
    [\
        {"type": "text", "text": "Hmm let me think about that"},\
        {\
            "type": "tool_use",\
            "input": {"fav_color": "green"},\
            "id": "foo",\
            "name": "color_picker",\
        },\
    ]
)
```

...as well as `HumanMessage` objects containing `tool_result` content blocks:

```
HumanMessage(
    [\
        {\
            "type": "tool_result",\
            "tool_use_id": "foo",\
            "content": [\
                {\
                    "type": "text",\
                    "text": "green is a great pick! "\
                    "that's my sister's favorite color",\
                }\
            ],\
            "is_error": False,\
        },\
        {"type": "text", "text": "what's my sister's favorite color"},\
    ]
)
```

This test should be skipped if the model does not support messages of this
form (or doesn't support tool calling generally). See Configuration below.

Configuration

To disable this test, set `supports_anthropic_inputs` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def supports_anthropic_inputs(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that:

1. The model can correctly handle message histories that include message
    objects with list content.
2. The `tool_calls` attribute on AIMessage objects is correctly handled
    and passed to the model in an appropriate format.
3. `HumanMessage`s with "tool\_result" content blocks are correctly
    handled.

Otherwise, if Anthropic tool call and result formats are not supported,
set the `supports_anthropic_inputs` property to `False`.

### ``test\_message\_with\_name [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_message_with_name "Copy anchor link to this section for reference")

```
test_message_with_name(model: BaseChatModel) -> None
```

Test that `HumanMessage` with values for the `name` field can be handled.

These messages may take the form:

```
HumanMessage("hello", name="example_user")
```

If possible, the `name` field should be parsed and passed appropriately
to the model. Otherwise, it should be ignored.

Troubleshooting

If this test fails, check that the `name` field on `HumanMessage`
objects is either ignored or passed to the model appropriately.

### ``test\_agent\_loop [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_agent_loop "Copy anchor link to this section for reference")

```
test_agent_loop(model: BaseChatModel) -> None
```

Test that the model supports a simple ReAct agent loop.

This test is skipped if the `has_tool_calling` property on the test class is
set to `False`.

This test is optional and should be skipped if the model does not support
tool calling (see configuration below).

Configuration

To disable tool calling tests, set `has_tool_calling` to `False` in your
test class:

```
class TestMyChatModelIntegration(ChatModelIntegrationTests):
    @property
    def has_tool_calling(self) -> bool:
        return False
```

Troubleshooting

If this test fails, check that `bind_tools` is implemented to correctly
translate LangChain tool objects into the appropriate schema for your
chat model.

Check also that all required information (e.g., tool calling identifiers)
from `AIMessage` objects is propagated correctly to model payloads.

This test may fail if the chat model does not consistently generate tool
calls in response to an appropriate query. In these cases you can `xfail`
the test:

```
@pytest.mark.xfail(reason=("Does not support tool_choice."))
def test_agent_loop(self, model: BaseChatModel) -> None:
    super().test_agent_loop(model)
```

### ``test\_stream\_time [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_stream_time "Copy anchor link to this section for reference")

```
test_stream_time(
    model: BaseChatModel, benchmark: BenchmarkFixture, vcr: Cassette
) -> None
```

Test that streaming does not introduce undue overhead.

See `enable_vcr_tests` dropdown `above <ChatModelIntegrationTests>`
for more information.

Configuration

This test can be enabled or disabled using the `enable_vcr_tests`
property. For example, to disable the test, set this property to `False`:

```
@property
def enable_vcr_tests(self) -> bool:
    return False
```

Warning

VCR will by default record authentication headers and other sensitive
information in cassettes. See `enable_vcr_tests` dropdown
`above <ChatModelIntegrationTests>` for how to configure what
information is recorded in cassettes.

### ``invoke\_with\_audio\_input [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.invoke_with_audio_input "Copy anchor link to this section for reference")

```
invoke_with_audio_input(*, stream: bool = False) -> AIMessage
```

Invoke with audio input.

### ``invoke\_with\_audio\_output [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.invoke_with_audio_output "Copy anchor link to this section for reference")

```
invoke_with_audio_output(*, stream: bool = False) -> AIMessage
```

Invoke with audio output.

### ``invoke\_with\_reasoning\_output [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.invoke_with_reasoning_output "Copy anchor link to this section for reference")

```
invoke_with_reasoning_output(*, stream: bool = False) -> AIMessage
```

Invoke with reasoning output.

### ``invoke\_with\_cache\_read\_input [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.invoke_with_cache_read_input "Copy anchor link to this section for reference")

```
invoke_with_cache_read_input(*, stream: bool = False) -> AIMessage
```

Invoke with cache read input.

### ``invoke\_with\_cache\_creation\_input [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.invoke_with_cache_creation_input "Copy anchor link to this section for reference")

```
invoke_with_cache_creation_input(*, stream: bool = False) -> AIMessage
```

Invoke with cache creation input.

### ``test\_unicode\_tool\_call\_integration [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/\#langchain_tests.integration_tests.ChatModelIntegrationTests.test_unicode_tool_call_integration "Copy anchor link to this section for reference")

```
test_unicode_tool_call_integration(
    model: BaseChatModel,
    *,
    tool_choice: str | None = None,
    force_tool_call: bool = True,
) -> None
```

Generic integration test for Unicode characters in tool calls.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `model` | The chat model to test<br>**TYPE:**`BaseChatModel` |
| `tool_choice` | Tool choice parameter to pass to `bind_tools()`<br>(provider-specific)<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `force_tool_call` | Whether to force a tool call<br>(use `tool_choice=True` if None)<br>**TYPE:**`bool`**DEFAULT:**`True` |

Tests that Unicode characters in tool call arguments are preserved correctly,
not escaped as `\\uXXXX` sequences.

Back to top