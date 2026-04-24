[Skip to content](https://reference.langchain.com/python/langchain_core/utils/#langchain_core.utils)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_core/utils.md "Edit this page")

# Utilities

## ``utils [¶](https://reference.langchain.com/python/langchain_core/utils/\#langchain_core.utils "Copy anchor link to this section for reference")

Utility functions for LangChain.

These functions do not depend on any other LangChain module.

## ``function\_calling [¶](https://reference.langchain.com/python/langchain_core/utils/\#langchain_core.utils.function_calling "Copy anchor link to this section for reference")

Methods for creating function specs in the style of OpenAI Functions.

| FUNCTION | DESCRIPTION |
| --- | --- |
| `convert_to_json_schema` | Convert a schema representation to a JSON schema. |
| `convert_to_openai_tool` | Convert a tool-like object to an OpenAI tool schema. |

### ``convert\_to\_json\_schema [¶](https://reference.langchain.com/python/langchain_core/utils/\#langchain_core.utils.function_calling.convert_to_json_schema "Copy anchor link to this section for reference")

```
convert_to_json_schema(
    schema: dict[str, Any] | type[BaseModel] | Callable | BaseTool,
    *,
    strict: bool | None = None,
) -> dict[str, Any]
```

Convert a schema representation to a JSON schema.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `schema` | The schema to convert.<br>**TYPE:**`dict[str, Any] | type[BaseModel] | Callable | BaseTool` |
| `strict` | If `True`, model output is guaranteed to exactly match the JSON Schema<br>provided in the function definition. If `None`, `strict` argument will not<br>be included in function definition.<br>**TYPE:**`bool | None`**DEFAULT:**`None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the input is not a valid OpenAI-format tool. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, Any]` | A JSON schema representation of the input schema. |

### ``convert\_to\_openai\_tool [¶](https://reference.langchain.com/python/langchain_core/utils/\#langchain_core.utils.function_calling.convert_to_openai_tool "Copy anchor link to this section for reference")

```
convert_to_openai_tool(
    tool: Mapping[str, Any] | type[BaseModel] | Callable | BaseTool,
    *,
    strict: bool | None = None,
) -> dict[str, Any]
```

Convert a tool-like object to an OpenAI tool schema.

[OpenAI tool schema reference](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools)

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tool` | Either a dictionary, a `pydantic.BaseModel` class, Python function, or<br>`BaseTool`. If a dictionary is passed in, it is assumed to already be a<br>valid OpenAI function, a JSON schema with top-level `title` key specified,<br>an Anthropic format tool, or an Amazon Bedrock Converse format tool.<br>**TYPE:**`Mapping[str, Any] | type[BaseModel] | Callable | BaseTool` |
| `strict` | If `True`, model output is guaranteed to exactly match the JSON Schema<br>provided in the function definition. If `None`, `strict` argument will not<br>be included in tool definition.<br>**TYPE:**`bool | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `dict[str, Any]` | A dict version of the passed in tool which is compatible with the |
| `dict[str, Any]` | OpenAI tool-calling API. |

Behavior changed in `langchain-core` 0.3.16

`description` and `parameters` keys are now optional. Only `name` is
required and guaranteed to be part of the output.

Behavior changed in `langchain-core` 0.3.44

Return OpenAI Responses API-style tools unchanged. This includes
any dict with `"type"` in `"file_search"`, `"function"`,
`"computer_use_preview"`, `"web_search_preview"`.

Behavior changed in `langchain-core` 0.3.63

Added support for OpenAI's image generation built-in tool.

Back to top