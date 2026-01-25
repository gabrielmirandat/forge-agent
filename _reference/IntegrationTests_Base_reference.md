[Skip to content](https://reference.langchain.com/python/langchain_tests/integration_tests/base/#base-tests)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_tests/integration_tests/base.md "Edit this page")

# Base tests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#base-tests "Copy anchor link to this section for reference")

## ``ChatModelTests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests "Copy anchor link to this section for reference")

Bases: `BaseStandardTests`

Base class for chat model tests.

| METHOD | DESCRIPTION |
| --- | --- |
| `model` | Model fixture. |
| `my_adder_tool` | Adder tool fixture. |
| `test_no_overrides_DO_NOT_OVERRIDE` | Test that no standard tests are overridden. |

### ``chat\_model\_class`abstractmethod``property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.chat_model_class "Copy anchor link to this section for reference")

```
chat_model_class: type[BaseChatModel]
```

The chat model class to test, e.g., `ChatParrotLink`.

### ``chat\_model\_params`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.chat_model_params "Copy anchor link to this section for reference")

```
chat_model_params: dict[str, Any]
```

Initialization parameters for the chat model.

### ``standard\_chat\_model\_params`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.standard_chat_model_params "Copy anchor link to this section for reference")

```
standard_chat_model_params: dict[str, Any]
```

Standard chat model parameters.

### ``has\_tool\_calling`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.has_tool_calling "Copy anchor link to this section for reference")

```
has_tool_calling: bool
```

Whether the model supports tool calling.

### ``has\_tool\_choice`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.has_tool_choice "Copy anchor link to this section for reference")

```
has_tool_choice: bool
```

Whether the model supports tool calling.

### ``has\_structured\_output`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.has_structured_output "Copy anchor link to this section for reference")

```
has_structured_output: bool
```

Whether the chat model supports structured output.

### ``structured\_output\_kwargs`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.structured_output_kwargs "Copy anchor link to this section for reference")

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

### ``supports\_json\_mode`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.supports_json_mode "Copy anchor link to this section for reference")

```
supports_json_mode: bool
```

Whether the chat model supports JSON mode.

### ``supports\_image\_inputs`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.supports_image_inputs "Copy anchor link to this section for reference")

```
supports_image_inputs: bool
```

Supports image inputs.

Whether the chat model supports image inputs, defaults to
`False`.

### ``supports\_image\_urls`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.supports_image_urls "Copy anchor link to this section for reference")

```
supports_image_urls: bool
```

Supports image inputs from URLs.

Whether the chat model supports image inputs from URLs, defaults to
`False`.

### ``supports\_pdf\_inputs`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.supports_pdf_inputs "Copy anchor link to this section for reference")

```
supports_pdf_inputs: bool
```

Whether the chat model supports PDF inputs, defaults to `False`.

### ``supports\_audio\_inputs`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.supports_audio_inputs "Copy anchor link to this section for reference")

```
supports_audio_inputs: bool
```

Supports audio inputs.

Whether the chat model supports audio inputs, defaults to `False`.

### ``supports\_video\_inputs`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.supports_video_inputs "Copy anchor link to this section for reference")

```
supports_video_inputs: bool
```

Supports video inputs.

Whether the chat model supports video inputs, defaults to `False`.

No current tests are written for this feature.

### ``returns\_usage\_metadata`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.returns_usage_metadata "Copy anchor link to this section for reference")

```
returns_usage_metadata: bool
```

Returns usage metadata.

Whether the chat model returns usage metadata on invoke and streaming
responses.

### ``supports\_anthropic\_inputs`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.supports_anthropic_inputs "Copy anchor link to this section for reference")

```
supports_anthropic_inputs: bool
```

Whether the chat model supports Anthropic-style inputs.

### ``supports\_image\_tool\_message`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.supports_image_tool_message "Copy anchor link to this section for reference")

```
supports_image_tool_message: bool
```

Supports image `ToolMessage` objects.

Whether the chat model supports `ToolMessage` objects that include image
content.

### ``supports\_pdf\_tool\_message`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.supports_pdf_tool_message "Copy anchor link to this section for reference")

```
supports_pdf_tool_message: bool
```

Supports PDF `ToolMessage` objects.

Whether the chat model supports `ToolMessage` objects that include PDF
content.

### ``enable\_vcr\_tests`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.enable_vcr_tests "Copy anchor link to this section for reference")

```
enable_vcr_tests: bool
```

Whether to enable VCR tests for the chat model.

Warning

See `enable_vcr_tests` dropdown `above <ChatModelTests>` for more
information.

### ``supported\_usage\_metadata\_details`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.supported_usage_metadata_details "Copy anchor link to this section for reference")

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

### ``supports\_model\_override`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.supports_model_override "Copy anchor link to this section for reference")

```
supports_model_override: bool
```

Whether the model supports overriding the model name at runtime.

Defaults to `True`.

If `True`, the model accepts a `model` kwarg in `invoke()`, `stream()`,
etc. that overrides the model specified at initialization.

This enables dynamic model selection without creating new instances.

### ``model\_override\_value`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.model_override_value "Copy anchor link to this section for reference")

```
model_override_value: str | None
```

Alternative model name to use when testing model override.

Should return a valid model name that differs from the default model.
Required if `supports_model_override` is `True`.

### ``model [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.model "Copy anchor link to this section for reference")

```
model(request: Any) -> BaseChatModel
```

Model fixture.

### ``my\_adder\_tool [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.my_adder_tool "Copy anchor link to this section for reference")

```
my_adder_tool() -> BaseTool
```

Adder tool fixture.

### ``test\_no\_overrides\_DO\_NOT\_OVERRIDE [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.chat_models.ChatModelTests.test_no_overrides_DO_NOT_OVERRIDE "Copy anchor link to this section for reference")

```
test_no_overrides_DO_NOT_OVERRIDE() -> None
```

Test that no standard tests are overridden.

## ``EmbeddingsTests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.embeddings.EmbeddingsTests "Copy anchor link to this section for reference")

Bases: `BaseStandardTests`

Embeddings tests base class.

| METHOD | DESCRIPTION |
| --- | --- |
| `model` | Embeddings model fixture. |
| `test_no_overrides_DO_NOT_OVERRIDE` | Test that no standard tests are overridden. |

### ``embeddings\_class`abstractmethod``property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.embeddings.EmbeddingsTests.embeddings_class "Copy anchor link to this section for reference")

```
embeddings_class: type[Embeddings]
```

Embeddings class.

### ``embedding\_model\_params`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.embeddings.EmbeddingsTests.embedding_model_params "Copy anchor link to this section for reference")

```
embedding_model_params: dict[str, Any]
```

Embeddings model parameters.

### ``model [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.embeddings.EmbeddingsTests.model "Copy anchor link to this section for reference")

```
model() -> Embeddings
```

Embeddings model fixture.

### ``test\_no\_overrides\_DO\_NOT\_OVERRIDE [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.embeddings.EmbeddingsTests.test_no_overrides_DO_NOT_OVERRIDE "Copy anchor link to this section for reference")

```
test_no_overrides_DO_NOT_OVERRIDE() -> None
```

Test that no standard tests are overridden.

## ``ToolsTests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.tools.ToolsTests "Copy anchor link to this section for reference")

Bases: `BaseStandardTests`

Base class for testing tools.

This won't show in the documentation, but the docstrings will be inherited by
subclasses.

| METHOD | DESCRIPTION |
| --- | --- |
| `tool` | Tool fixture. |
| `test_no_overrides_DO_NOT_OVERRIDE` | Test that no standard tests are overridden. |

### ``tool\_constructor`abstractmethod``property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.tools.ToolsTests.tool_constructor "Copy anchor link to this section for reference")

```
tool_constructor: type[BaseTool] | BaseTool
```

Returns a class or instance of a tool to be tested.

### ``tool\_constructor\_params`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.tools.ToolsTests.tool_constructor_params "Copy anchor link to this section for reference")

```
tool_constructor_params: dict[str, Any]
```

Returns a dictionary of parameters to pass to the tool constructor.

### ``tool\_invoke\_params\_example`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.tools.ToolsTests.tool_invoke_params_example "Copy anchor link to this section for reference")

```
tool_invoke_params_example: dict[str, Any]
```

Returns a dictionary representing the "args" of an example tool call.

This should NOT be a `ToolCall` dict - it should not have
`{"name", "id", "args"}` keys.

### ``tool [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.tools.ToolsTests.tool "Copy anchor link to this section for reference")

```
tool() -> BaseTool
```

Tool fixture.

### ``test\_no\_overrides\_DO\_NOT\_OVERRIDE [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.unit_tests.tools.ToolsTests.test_no_overrides_DO_NOT_OVERRIDE "Copy anchor link to this section for reference")

```
test_no_overrides_DO_NOT_OVERRIDE() -> None
```

Test that no standard tests are overridden.

## ``BaseStandardTests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.base.BaseStandardTests "Copy anchor link to this section for reference")

Base class for standard tests.

| METHOD | DESCRIPTION |
| --- | --- |
| `test_no_overrides_DO_NOT_OVERRIDE` | Test that no standard tests are overridden. |

### ``test\_no\_overrides\_DO\_NOT\_OVERRIDE [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/base/\#langchain_tests.base.BaseStandardTests.test_no_overrides_DO_NOT_OVERRIDE "Copy anchor link to this section for reference")

```
test_no_overrides_DO_NOT_OVERRIDE() -> None
```

Test that no standard tests are overridden.

Back to top