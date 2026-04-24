![Revisit consent button](https://uploads-ssl.webflow.com/65ff950538088944d66126b3/662ef3209b872e92e41212f6_cookieicon.png)

![](https://cdn-cookieyes.com/assets/images/close.svg)

We value your privacy

We use cookies to improve your experience and to understand how our site is used. Some analytics tools may share limited data with our advertising partners. You can opt out at any time.

Do Not Sell or Share My Personal Information

Opt-out Preferences![](https://cdn-cookieyes.com/assets/images/close.svg)

We use cookies to improve your experience and to understand how our site is used. Some analytics tools may share limited data with our advertising partners. You can opt out of this sharing at any time by selecting **“Do Not Sell or Share My Personal Information”** and saving your preferences.

Do Not Sell or Share My Personal Information

CancelSave My Preferences

[Skip to content](https://reference.langchain.com/python/langchain_core/callbacks/#langchain_core.callbacks.base.BaseCallbackHandler)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_core/callbacks.md "Edit this page")

# Callbacks

## ``BaseCallbackHandler [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler "Copy anchor link to this section for reference")

Bases: `LLMManagerMixin`, `ChainManagerMixin`, `ToolManagerMixin`, `RetrieverManagerMixin`, `CallbackManagerMixin`, `RunManagerMixin`

[<code class="doc-symbol doc-symbol-heading doc-symbol-class"></code>            <span class="doc doc-object-name doc-class-name">BaseCallbackHandler</span> (<code>langchain_core.callbacks.base.BaseCallbackHandler</code>)](https://reference.langchain.com/python/langchain_core/callbacks/#langchain_core.callbacks.base.BaseCallbackHandler "<code class=\"doc-symbol doc-symbol-heading doc-symbol-class\"></code>            <span class=\"doc doc-object-name doc-class-name\">BaseCallbackHandler</span> (<code>langchain_core.callbacks.base.BaseCallbackHandler</code>)")

[BaseCallbackHandler](https://reference.langchain.com/python/langchain_core/callbacks/#langchain_core.callbacks.base.BaseCallbackHandler)

LLMManagerMixin

ChainManagerMixin

ToolManagerMixin

RetrieverManagerMixin

CallbackManagerMixin

RunManagerMixin

Base callback handler.

| METHOD | DESCRIPTION |
| --- | --- |
| `on_text` | Run on an arbitrary text. |
| `on_retry` | Run on a retry event. |
| `on_custom_event` | Override to define a handler for a custom event. |
| `on_llm_start` | Run when LLM starts running. |
| `on_chat_model_start` | Run when a chat model starts running. |
| `on_retriever_start` | Run when the `Retriever` starts running. |
| `on_chain_start` | Run when a chain starts running. |
| `on_tool_start` | Run when the tool starts running. |
| `on_retriever_error` | Run when `Retriever` errors. |
| `on_retriever_end` | Run when `Retriever` ends running. |
| `on_tool_end` | Run when the tool ends running. |
| `on_tool_error` | Run when tool errors. |
| `on_chain_end` | Run when chain ends running. |
| `on_chain_error` | Run when chain errors. |
| `on_agent_action` | Run on agent action. |
| `on_agent_finish` | Run on the agent end. |
| `on_llm_new_token` | Run on new output token. Only available when streaming is enabled. |
| `on_llm_end` | Run when LLM ends running. |
| `on_llm_error` | Run when LLM errors. |

### ``raise\_error`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.raise_error "Copy anchor link to this section for reference")

```
raise_error: bool = False
```

Whether to raise an error if an exception occurs.

### ``run\_inline`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.run_inline "Copy anchor link to this section for reference")

```
run_inline: bool = False
```

Whether to run the callback inline.

### ``ignore\_llm`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.ignore_llm "Copy anchor link to this section for reference")

```
ignore_llm: bool
```

Whether to ignore LLM callbacks.

### ``ignore\_retry`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.ignore_retry "Copy anchor link to this section for reference")

```
ignore_retry: bool
```

Whether to ignore retry callbacks.

### ``ignore\_chain`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.ignore_chain "Copy anchor link to this section for reference")

```
ignore_chain: bool
```

Whether to ignore chain callbacks.

### ``ignore\_agent`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.ignore_agent "Copy anchor link to this section for reference")

```
ignore_agent: bool
```

Whether to ignore agent callbacks.

### ``ignore\_retriever`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.ignore_retriever "Copy anchor link to this section for reference")

```
ignore_retriever: bool
```

Whether to ignore retriever callbacks.

### ``ignore\_chat\_model`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.ignore_chat_model "Copy anchor link to this section for reference")

```
ignore_chat_model: bool
```

Whether to ignore chat model callbacks.

### ``ignore\_custom\_event`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.ignore_custom_event "Copy anchor link to this section for reference")

```
ignore_custom_event: bool
```

Ignore custom event.

### ``on\_text [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_text "Copy anchor link to this section for reference")

```
on_text(
    text: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
) -> Any
```

Run on an arbitrary text.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The text.<br>**TYPE:**`str` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retry [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_retry "Copy anchor link to this section for reference")

```
on_retry(
    retry_state: RetryCallState,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run on a retry event.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `retry_state` | The retry state.<br>**TYPE:**`RetryCallState` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_custom\_event [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_custom_event "Copy anchor link to this section for reference")

```
on_custom_event(
    name: str,
    data: Any,
    *,
    run_id: UUID,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Override to define a handler for a custom event.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | The name of the custom event.<br>**TYPE:**`str` |
| `data` | The data for the custom event. Format will match the format specified<br>by the user.<br>**TYPE:**`Any` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID` |
| `tags` | The tags associated with the custom event (includes inherited tags).<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata associated with the custom event (includes inherited<br>metadata).<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |

### ``on\_llm\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_llm_start "Copy anchor link to this section for reference")

```
on_llm_start(
    serialized: dict[str, Any],
    prompts: list[str],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when LLM starts running.

Warning

This method is called for non-chat models (regular LLMs). If you're
implementing a handler for a chat model, you should use
`on_chat_model_start` instead.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized LLM.<br>**TYPE:**`dict[str, Any]` |
| `prompts` | The prompts.<br>**TYPE:**`list[str]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chat\_model\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_chat_model_start "Copy anchor link to this section for reference")

```
on_chat_model_start(
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when a chat model starts running.

Warning

This method is called for chat models. If you're implementing a handler for
a non-chat model, you should use `on_llm_start` instead.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chat model.<br>**TYPE:**`dict[str, Any]` |
| `messages` | The messages.<br>**TYPE:**`list[list[BaseMessage]]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retriever\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_retriever_start "Copy anchor link to this section for reference")

```
on_retriever_start(
    serialized: dict[str, Any],
    query: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when the `Retriever` starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized `Retriever`.<br>**TYPE:**`dict[str, Any]` |
| `query` | The query.<br>**TYPE:**`str` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chain\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_chain_start "Copy anchor link to this section for reference")

```
on_chain_start(
    serialized: dict[str, Any],
    inputs: dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when a chain starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chain.<br>**TYPE:**`dict[str, Any]` |
| `inputs` | The inputs.<br>**TYPE:**`dict[str, Any]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_tool\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_tool_start "Copy anchor link to this section for reference")

```
on_tool_start(
    serialized: dict[str, Any],
    input_str: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when the tool starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chain.<br>**TYPE:**`dict[str, Any]` |
| `input_str` | The input string.<br>**TYPE:**`str` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `inputs` | The inputs.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retriever\_error [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_retriever_error "Copy anchor link to this section for reference")

```
on_retriever_error(
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run when `Retriever` errors.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that occurred.<br>**TYPE:**`BaseException` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retriever\_end [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_retriever_end "Copy anchor link to this section for reference")

```
on_retriever_end(
    documents: Sequence[Document],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run when `Retriever` ends running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents retrieved.<br>**TYPE:**`Sequence[Document]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_tool\_end [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_tool_end "Copy anchor link to this section for reference")

```
on_tool_end(
    output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
) -> Any
```

Run when the tool ends running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `output` | The output of the tool.<br>**TYPE:**`Any` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_tool\_error [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_tool_error "Copy anchor link to this section for reference")

```
on_tool_error(
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run when tool errors.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that occurred.<br>**TYPE:**`BaseException` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chain\_end [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_chain_end "Copy anchor link to this section for reference")

```
on_chain_end(
    outputs: dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run when chain ends running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `outputs` | The outputs of the chain.<br>**TYPE:**`dict[str, Any]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chain\_error [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_chain_error "Copy anchor link to this section for reference")

```
on_chain_error(
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run when chain errors.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that occurred.<br>**TYPE:**`BaseException` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_agent\_action [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_agent_action "Copy anchor link to this section for reference")

```
on_agent_action(
    action: AgentAction,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run on agent action.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `action` | The agent action.<br>**TYPE:**`AgentAction` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_agent\_finish [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_agent_finish "Copy anchor link to this section for reference")

```
on_agent_finish(
    finish: AgentFinish,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run on the agent end.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `finish` | The agent finish.<br>**TYPE:**`AgentFinish` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_llm\_new\_token [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_llm_new_token "Copy anchor link to this section for reference")

```
on_llm_new_token(
    token: str,
    *,
    chunk: GenerationChunk | ChatGenerationChunk | None = None,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any
```

Run on new output token. Only available when streaming is enabled.

For both chat models and non-chat models (legacy LLMs).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `token` | The new token.<br>**TYPE:**`str` |
| `chunk` | The new generated chunk, containing content and other information.<br>**TYPE:**`GenerationChunk | ChatGenerationChunk | None`**DEFAULT:**`None` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_llm\_end [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_llm_end "Copy anchor link to this section for reference")

```
on_llm_end(
    response: LLMResult,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any
```

Run when LLM ends running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `response` | The response which was generated.<br>**TYPE:**`LLMResult` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_llm\_error [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.BaseCallbackHandler.on_llm_error "Copy anchor link to this section for reference")

```
on_llm_error(
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any
```

Run when LLM errors.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that occurred.<br>**TYPE:**`BaseException` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

## ``AsyncCallbackHandler [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler "Copy anchor link to this section for reference")

Bases: `BaseCallbackHandler`

Base async callback handler.

| METHOD | DESCRIPTION |
| --- | --- |
| `on_llm_start` | Run when the model starts running. |
| `on_chat_model_start` | Run when a chat model starts running. |
| `on_llm_new_token` | Run on new output token. Only available when streaming is enabled. |
| `on_llm_end` | Run when the model ends running. |
| `on_llm_error` | Run when LLM errors. |
| `on_chain_start` | Run when a chain starts running. |
| `on_chain_end` | Run when a chain ends running. |
| `on_chain_error` | Run when chain errors. |
| `on_tool_start` | Run when the tool starts running. |
| `on_tool_end` | Run when the tool ends running. |
| `on_tool_error` | Run when tool errors. |
| `on_text` | Run on an arbitrary text. |
| `on_retry` | Run on a retry event. |
| `on_agent_action` | Run on agent action. |
| `on_agent_finish` | Run on the agent end. |
| `on_retriever_start` | Run on the retriever start. |
| `on_retriever_end` | Run on the retriever end. |
| `on_retriever_error` | Run on retriever error. |
| `on_custom_event` | Override to define a handler for custom events. |

### ``raise\_error`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.raise_error "Copy anchor link to this section for reference")

```
raise_error: bool = False
```

Whether to raise an error if an exception occurs.

### ``run\_inline`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.run_inline "Copy anchor link to this section for reference")

```
run_inline: bool = False
```

Whether to run the callback inline.

### ``ignore\_llm`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.ignore_llm "Copy anchor link to this section for reference")

```
ignore_llm: bool
```

Whether to ignore LLM callbacks.

### ``ignore\_retry`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.ignore_retry "Copy anchor link to this section for reference")

```
ignore_retry: bool
```

Whether to ignore retry callbacks.

### ``ignore\_chain`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.ignore_chain "Copy anchor link to this section for reference")

```
ignore_chain: bool
```

Whether to ignore chain callbacks.

### ``ignore\_agent`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.ignore_agent "Copy anchor link to this section for reference")

```
ignore_agent: bool
```

Whether to ignore agent callbacks.

### ``ignore\_retriever`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.ignore_retriever "Copy anchor link to this section for reference")

```
ignore_retriever: bool
```

Whether to ignore retriever callbacks.

### ``ignore\_chat\_model`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.ignore_chat_model "Copy anchor link to this section for reference")

```
ignore_chat_model: bool
```

Whether to ignore chat model callbacks.

### ``ignore\_custom\_event`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.ignore_custom_event "Copy anchor link to this section for reference")

```
ignore_custom_event: bool
```

Ignore custom event.

### ``on\_llm\_start`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_llm_start "Copy anchor link to this section for reference")

```
on_llm_start(
    serialized: dict[str, Any],
    prompts: list[str],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None
```

Run when the model starts running.

Warning

This method is called for non-chat models (regular LLMs). If you're
implementing a handler for a chat model, you should use
`on_chat_model_start` instead.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized LLM.<br>**TYPE:**`dict[str, Any]` |
| `prompts` | The prompts.<br>**TYPE:**`list[str]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chat\_model\_start`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_chat_model_start "Copy anchor link to this section for reference")

```
on_chat_model_start(
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when a chat model starts running.

Warning

This method is called for chat models. If you're implementing a handler for
a non-chat model, you should use `on_llm_start` instead.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chat model.<br>**TYPE:**`dict[str, Any]` |
| `messages` | The messages.<br>**TYPE:**`list[list[BaseMessage]]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_llm\_new\_token`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_llm_new_token "Copy anchor link to this section for reference")

```
on_llm_new_token(
    token: str,
    *,
    chunk: GenerationChunk | ChatGenerationChunk | None = None,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> None
```

Run on new output token. Only available when streaming is enabled.

For both chat models and non-chat models (legacy LLMs).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `token` | The new token.<br>**TYPE:**`str` |
| `chunk` | The new generated chunk, containing content and other information.<br>**TYPE:**`GenerationChunk | ChatGenerationChunk | None`**DEFAULT:**`None` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_llm\_end`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_llm_end "Copy anchor link to this section for reference")

```
on_llm_end(
    response: LLMResult,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> None
```

Run when the model ends running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `response` | The response which was generated.<br>**TYPE:**`LLMResult` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_llm\_error`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_llm_error "Copy anchor link to this section for reference")

```
on_llm_error(
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> None
```

Run when LLM errors.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that occurred.<br>**TYPE:**`BaseException` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>- response (LLMResult): The response which was generated before<br>the error occurred.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chain\_start`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_chain_start "Copy anchor link to this section for reference")

```
on_chain_start(
    serialized: dict[str, Any],
    inputs: dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None
```

Run when a chain starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chain.<br>**TYPE:**`dict[str, Any]` |
| `inputs` | The inputs.<br>**TYPE:**`dict[str, Any]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chain\_end`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_chain_end "Copy anchor link to this section for reference")

```
on_chain_end(
    outputs: dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> None
```

Run when a chain ends running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `outputs` | The outputs of the chain.<br>**TYPE:**`dict[str, Any]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chain\_error`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_chain_error "Copy anchor link to this section for reference")

```
on_chain_error(
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> None
```

Run when chain errors.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that occurred.<br>**TYPE:**`BaseException` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_tool\_start`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_tool_start "Copy anchor link to this section for reference")

```
on_tool_start(
    serialized: dict[str, Any],
    input_str: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None
```

Run when the tool starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized tool.<br>**TYPE:**`dict[str, Any]` |
| `input_str` | The input string.<br>**TYPE:**`str` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `inputs` | The inputs.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_tool\_end`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_tool_end "Copy anchor link to this section for reference")

```
on_tool_end(
    output: Any,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> None
```

Run when the tool ends running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `output` | The output of the tool.<br>**TYPE:**`Any` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_tool\_error`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_tool_error "Copy anchor link to this section for reference")

```
on_tool_error(
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> None
```

Run when tool errors.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that occurred.<br>**TYPE:**`BaseException` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_text`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_text "Copy anchor link to this section for reference")

```
on_text(
    text: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> None
```

Run on an arbitrary text.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The text.<br>**TYPE:**`str` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retry`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_retry "Copy anchor link to this section for reference")

```
on_retry(
    retry_state: RetryCallState,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run on a retry event.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `retry_state` | The retry state.<br>**TYPE:**`RetryCallState` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_agent\_action`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_agent_action "Copy anchor link to this section for reference")

```
on_agent_action(
    action: AgentAction,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> None
```

Run on agent action.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `action` | The agent action.<br>**TYPE:**`AgentAction` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_agent\_finish`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_agent_finish "Copy anchor link to this section for reference")

```
on_agent_finish(
    finish: AgentFinish,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> None
```

Run on the agent end.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `finish` | The agent finish.<br>**TYPE:**`AgentFinish` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retriever\_start`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_retriever_start "Copy anchor link to this section for reference")

```
on_retriever_start(
    serialized: dict[str, Any],
    query: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None
```

Run on the retriever start.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized retriever.<br>**TYPE:**`dict[str, Any]` |
| `query` | The query.<br>**TYPE:**`str` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retriever\_end`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_retriever_end "Copy anchor link to this section for reference")

```
on_retriever_end(
    documents: Sequence[Document],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> None
```

Run on the retriever end.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents retrieved.<br>**TYPE:**`Sequence[Document]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retriever\_error`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_retriever_error "Copy anchor link to this section for reference")

```
on_retriever_error(
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> None
```

Run on retriever error.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that occurred.<br>**TYPE:**`BaseException` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_custom\_event`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.base.AsyncCallbackHandler.on_custom_event "Copy anchor link to this section for reference")

```
on_custom_event(
    name: str,
    data: Any,
    *,
    run_id: UUID,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None
```

Override to define a handler for custom events.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | The name of the custom event.<br>**TYPE:**`str` |
| `data` | The data for the custom event. Format will match<br>the format specified by the user.<br>**TYPE:**`Any` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID` |
| `tags` | The tags associated with the custom event<br>(includes inherited tags).<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata associated with the custom event<br>(includes inherited metadata).<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |

## ``BaseCallbackManager [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager "Copy anchor link to this section for reference")

Bases: `CallbackManagerMixin`

Base callback manager.

| METHOD | DESCRIPTION |
| --- | --- |
| `on_llm_start` | Run when LLM starts running. |
| `on_chat_model_start` | Run when a chat model starts running. |
| `on_retriever_start` | Run when the `Retriever` starts running. |
| `on_chain_start` | Run when a chain starts running. |
| `on_tool_start` | Run when the tool starts running. |
| `__init__` | Initialize callback manager. |
| `copy` | Return a copy of the callback manager. |
| `merge` | Merge the callback manager with another callback manager. |
| `add_handler` | Add a handler to the callback manager. |
| `remove_handler` | Remove a handler from the callback manager. |
| `set_handlers` | Set handlers as the only handlers on the callback manager. |
| `set_handler` | Set handler as the only handler on the callback manager. |
| `add_tags` | Add tags to the callback manager. |
| `remove_tags` | Remove tags from the callback manager. |
| `add_metadata` | Add metadata to the callback manager. |
| `remove_metadata` | Remove metadata from the callback manager. |

### ``is\_async`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.is_async "Copy anchor link to this section for reference")

```
is_async: bool
```

Whether the callback manager is async.

### ``on\_llm\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.on_llm_start "Copy anchor link to this section for reference")

```
on_llm_start(
    serialized: dict[str, Any],
    prompts: list[str],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when LLM starts running.

Warning

This method is called for non-chat models (regular LLMs). If you're
implementing a handler for a chat model, you should use
`on_chat_model_start` instead.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized LLM.<br>**TYPE:**`dict[str, Any]` |
| `prompts` | The prompts.<br>**TYPE:**`list[str]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chat\_model\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.on_chat_model_start "Copy anchor link to this section for reference")

```
on_chat_model_start(
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when a chat model starts running.

Warning

This method is called for chat models. If you're implementing a handler for
a non-chat model, you should use `on_llm_start` instead.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chat model.<br>**TYPE:**`dict[str, Any]` |
| `messages` | The messages.<br>**TYPE:**`list[list[BaseMessage]]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retriever\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.on_retriever_start "Copy anchor link to this section for reference")

```
on_retriever_start(
    serialized: dict[str, Any],
    query: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when the `Retriever` starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized `Retriever`.<br>**TYPE:**`dict[str, Any]` |
| `query` | The query.<br>**TYPE:**`str` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chain\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.on_chain_start "Copy anchor link to this section for reference")

```
on_chain_start(
    serialized: dict[str, Any],
    inputs: dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when a chain starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chain.<br>**TYPE:**`dict[str, Any]` |
| `inputs` | The inputs.<br>**TYPE:**`dict[str, Any]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_tool\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.on_tool_start "Copy anchor link to this section for reference")

```
on_tool_start(
    serialized: dict[str, Any],
    input_str: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when the tool starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chain.<br>**TYPE:**`dict[str, Any]` |
| `input_str` | The input string.<br>**TYPE:**`str` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `inputs` | The inputs.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.__init__ "Copy anchor link to this section for reference")

```
__init__(
    handlers: list[BaseCallbackHandler],
    inheritable_handlers: list[BaseCallbackHandler] | None = None,
    parent_run_id: UUID | None = None,
    *,
    tags: list[str] | None = None,
    inheritable_tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inheritable_metadata: dict[str, Any] | None = None,
) -> None
```

Initialize callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handlers` | The handlers.<br>**TYPE:**`list[BaseCallbackHandler]` |
| `inheritable_handlers` | The inheritable handlers.<br>**TYPE:**`list[BaseCallbackHandler] | None`**DEFAULT:**`None` |
| `parent_run_id` | The parent run ID.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `inheritable_tags` | The inheritable tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `inheritable_metadata` | The inheritable metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |

### ``copy [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.copy "Copy anchor link to this section for reference")

```
copy() -> Self
```

Return a copy of the callback manager.

### ``merge [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.merge "Copy anchor link to this section for reference")

```
merge(other: BaseCallbackManager) -> Self
```

Merge the callback manager with another callback manager.

May be overwritten in subclasses.

Primarily used internally within `merge_configs`.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | The merged callback manager of the same type as the current object. |

Example: Merging two callback managers.

````
```python
from langchain_core.callbacks.manager import (
    CallbackManager,
    trace_as_chain_group,
)
from langchain_core.callbacks.stdout import StdOutCallbackHandler

manager = CallbackManager(handlers=[StdOutCallbackHandler()], tags=["tag2"])
with trace_as_chain_group("My Group Name", tags=["tag1"]) as group_manager:
    merged_manager = group_manager.merge(manager)
    print(merged_manager.handlers)
    # [\
    #    <langchain_core.callbacks.stdout.StdOutCallbackHandler object at ...>,\
    #    <langchain_core.callbacks.streaming_stdout.StreamingStdOutCallbackHandler object at ...>,\
    # ]

    print(merged_manager.tags)
    #    ['tag2', 'tag1']
```
````

### ``add\_handler [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.add_handler "Copy anchor link to this section for reference")

```
add_handler(handler: BaseCallbackHandler, inherit: bool = True) -> None
```

Add a handler to the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handler` | The handler to add.<br>**TYPE:**`BaseCallbackHandler` |
| `inherit` | Whether to inherit the handler.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``remove\_handler [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.remove_handler "Copy anchor link to this section for reference")

```
remove_handler(handler: BaseCallbackHandler) -> None
```

Remove a handler from the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handler` | The handler to remove.<br>**TYPE:**`BaseCallbackHandler` |

### ``set\_handlers [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.set_handlers "Copy anchor link to this section for reference")

```
set_handlers(handlers: list[BaseCallbackHandler], inherit: bool = True) -> None
```

Set handlers as the only handlers on the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handlers` | The handlers to set.<br>**TYPE:**`list[BaseCallbackHandler]` |
| `inherit` | Whether to inherit the handlers.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``set\_handler [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.set_handler "Copy anchor link to this section for reference")

```
set_handler(handler: BaseCallbackHandler, inherit: bool = True) -> None
```

Set handler as the only handler on the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handler` | The handler to set.<br>**TYPE:**`BaseCallbackHandler` |
| `inherit` | Whether to inherit the handler.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``add\_tags [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.add_tags "Copy anchor link to this section for reference")

```
add_tags(tags: list[str], inherit: bool = True) -> None
```

Add tags to the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tags` | The tags to add.<br>**TYPE:**`list[str]` |
| `inherit` | Whether to inherit the tags.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``remove\_tags [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.remove_tags "Copy anchor link to this section for reference")

```
remove_tags(tags: list[str]) -> None
```

Remove tags from the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tags` | The tags to remove.<br>**TYPE:**`list[str]` |

### ``add\_metadata [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.add_metadata "Copy anchor link to this section for reference")

```
add_metadata(metadata: dict[str, Any], inherit: bool = True) -> None
```

Add metadata to the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `metadata` | The metadata to add.<br>**TYPE:**`dict[str, Any]` |
| `inherit` | Whether to inherit the metadata.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``remove\_metadata [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.BaseCallbackManager.remove_metadata "Copy anchor link to this section for reference")

```
remove_metadata(keys: list[str]) -> None
```

Remove metadata from the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `keys` | The keys to remove.<br>**TYPE:**`list[str]` |

## ``CallbackManager [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager "Copy anchor link to this section for reference")

Bases: `BaseCallbackManager`

Callback manager for LangChain.

| METHOD | DESCRIPTION |
| --- | --- |
| `on_llm_start` | Run when LLM starts running. |
| `on_chat_model_start` | Run when chat model starts running. |
| `on_chain_start` | Run when chain starts running. |
| `on_tool_start` | Run when tool starts running. |
| `on_retriever_start` | Run when the retriever starts running. |
| `on_custom_event` | Dispatch an adhoc event to the handlers (async version). |
| `configure` | Configure the callback manager. |
| `__init__` | Initialize callback manager. |
| `copy` | Return a copy of the callback manager. |
| `merge` | Merge the callback manager with another callback manager. |
| `add_handler` | Add a handler to the callback manager. |
| `remove_handler` | Remove a handler from the callback manager. |
| `set_handlers` | Set handlers as the only handlers on the callback manager. |
| `set_handler` | Set handler as the only handler on the callback manager. |
| `add_tags` | Add tags to the callback manager. |
| `remove_tags` | Remove tags from the callback manager. |
| `add_metadata` | Add metadata to the callback manager. |
| `remove_metadata` | Remove metadata from the callback manager. |

### ``is\_async`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.is_async "Copy anchor link to this section for reference")

```
is_async: bool
```

Whether the callback manager is async.

### ``on\_llm\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.on_llm_start "Copy anchor link to this section for reference")

```
on_llm_start(
    serialized: dict[str, Any],
    prompts: list[str],
    run_id: UUID | None = None,
    **kwargs: Any,
) -> list[CallbackManagerForLLMRun]
```

Run when LLM starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized LLM.<br>**TYPE:**`dict[str, Any]` |
| `prompts` | The list of prompts.<br>**TYPE:**`list[str]` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[CallbackManagerForLLMRun]` | A callback manager for each prompt as an LLM run. |

### ``on\_chat\_model\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.on_chat_model_start "Copy anchor link to this section for reference")

```
on_chat_model_start(
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    run_id: UUID | None = None,
    **kwargs: Any,
) -> list[CallbackManagerForLLMRun]
```

Run when chat model starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized LLM.<br>**TYPE:**`dict[str, Any]` |
| `messages` | The list of messages.<br>**TYPE:**`list[list[BaseMessage]]` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[CallbackManagerForLLMRun]` | A callback manager for each list of messages as an LLM run. |

### ``on\_chain\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.on_chain_start "Copy anchor link to this section for reference")

```
on_chain_start(
    serialized: dict[str, Any] | None,
    inputs: dict[str, Any] | Any,
    run_id: UUID | None = None,
    **kwargs: Any,
) -> CallbackManagerForChainRun
```

Run when chain starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chain.<br>**TYPE:**`dict[str, Any] | None` |
| `inputs` | The inputs to the chain.<br>**TYPE:**`dict[str, Any] | Any` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `CallbackManagerForChainRun` | The callback manager for the chain run. |

### ``on\_tool\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.on_tool_start "Copy anchor link to this section for reference")

```
on_tool_start(
    serialized: dict[str, Any] | None,
    input_str: str,
    run_id: UUID | None = None,
    parent_run_id: UUID | None = None,
    inputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CallbackManagerForToolRun
```

Run when tool starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | Serialized representation of the tool.<br>**TYPE:**`dict[str, Any] | None` |
| `input_str` | The input to the tool as a string.<br>Non-string inputs are cast to strings.<br>**TYPE:**`str` |
| `run_id` | ID for the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `inputs` | The original input to the tool if provided.<br>Recommended for usage instead of input\_str when the original<br>input is needed.<br>If provided, the inputs are expected to be formatted as a dict.<br>The keys will correspond to the named-arguments in the tool.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | The keyword arguments to pass to the event handler<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `CallbackManagerForToolRun` | The callback manager for the tool run. |

### ``on\_retriever\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.on_retriever_start "Copy anchor link to this section for reference")

```
on_retriever_start(
    serialized: dict[str, Any] | None,
    query: str,
    run_id: UUID | None = None,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> CallbackManagerForRetrieverRun
```

Run when the retriever starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized retriever.<br>**TYPE:**`dict[str, Any] | None` |
| `query` | The query.<br>**TYPE:**`str` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `CallbackManagerForRetrieverRun` | The callback manager for the retriever run. |

### ``on\_custom\_event [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.on_custom_event "Copy anchor link to this section for reference")

```
on_custom_event(
    name: str, data: Any, run_id: UUID | None = None, **kwargs: Any
) -> None
```

Dispatch an adhoc event to the handlers (async version).

This event should NOT be used in any internal LangChain code. The event
is meant specifically for users of the library to dispatch custom
events that are tailored to their application.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | The name of the adhoc event.<br>**TYPE:**`str` |
| `data` | The data for the adhoc event.<br>**TYPE:**`Any` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If additional keyword arguments are passed. |

### ``configure`classmethod`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.configure "Copy anchor link to this section for reference")

```
configure(
    inheritable_callbacks: Callbacks = None,
    local_callbacks: Callbacks = None,
    verbose: bool = False,
    inheritable_tags: list[str] | None = None,
    local_tags: list[str] | None = None,
    inheritable_metadata: dict[str, Any] | None = None,
    local_metadata: dict[str, Any] | None = None,
) -> CallbackManager
```

Configure the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `inheritable_callbacks` | The inheritable callbacks.<br>**TYPE:**`Callbacks`**DEFAULT:**`None` |
| `local_callbacks` | The local callbacks.<br>**TYPE:**`Callbacks`**DEFAULT:**`None` |
| `verbose` | Whether to enable verbose mode.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `inheritable_tags` | The inheritable tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `local_tags` | The local tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `inheritable_metadata` | The inheritable metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `local_metadata` | The local metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `CallbackManager` | The configured callback manager. |

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.__init__ "Copy anchor link to this section for reference")

```
__init__(
    handlers: list[BaseCallbackHandler],
    inheritable_handlers: list[BaseCallbackHandler] | None = None,
    parent_run_id: UUID | None = None,
    *,
    tags: list[str] | None = None,
    inheritable_tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inheritable_metadata: dict[str, Any] | None = None,
) -> None
```

Initialize callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handlers` | The handlers.<br>**TYPE:**`list[BaseCallbackHandler]` |
| `inheritable_handlers` | The inheritable handlers.<br>**TYPE:**`list[BaseCallbackHandler] | None`**DEFAULT:**`None` |
| `parent_run_id` | The parent run ID.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `inheritable_tags` | The inheritable tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `inheritable_metadata` | The inheritable metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |

### ``copy [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.copy "Copy anchor link to this section for reference")

```
copy() -> Self
```

Return a copy of the callback manager.

### ``merge [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.merge "Copy anchor link to this section for reference")

```
merge(other: BaseCallbackManager) -> Self
```

Merge the callback manager with another callback manager.

May be overwritten in subclasses.

Primarily used internally within `merge_configs`.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | The merged callback manager of the same type as the current object. |

Example: Merging two callback managers.

````
```python
from langchain_core.callbacks.manager import (
    CallbackManager,
    trace_as_chain_group,
)
from langchain_core.callbacks.stdout import StdOutCallbackHandler

manager = CallbackManager(handlers=[StdOutCallbackHandler()], tags=["tag2"])
with trace_as_chain_group("My Group Name", tags=["tag1"]) as group_manager:
    merged_manager = group_manager.merge(manager)
    print(merged_manager.handlers)
    # [\
    #    <langchain_core.callbacks.stdout.StdOutCallbackHandler object at ...>,\
    #    <langchain_core.callbacks.streaming_stdout.StreamingStdOutCallbackHandler object at ...>,\
    # ]

    print(merged_manager.tags)
    #    ['tag2', 'tag1']
```
````

### ``add\_handler [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.add_handler "Copy anchor link to this section for reference")

```
add_handler(handler: BaseCallbackHandler, inherit: bool = True) -> None
```

Add a handler to the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handler` | The handler to add.<br>**TYPE:**`BaseCallbackHandler` |
| `inherit` | Whether to inherit the handler.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``remove\_handler [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.remove_handler "Copy anchor link to this section for reference")

```
remove_handler(handler: BaseCallbackHandler) -> None
```

Remove a handler from the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handler` | The handler to remove.<br>**TYPE:**`BaseCallbackHandler` |

### ``set\_handlers [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.set_handlers "Copy anchor link to this section for reference")

```
set_handlers(handlers: list[BaseCallbackHandler], inherit: bool = True) -> None
```

Set handlers as the only handlers on the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handlers` | The handlers to set.<br>**TYPE:**`list[BaseCallbackHandler]` |
| `inherit` | Whether to inherit the handlers.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``set\_handler [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.set_handler "Copy anchor link to this section for reference")

```
set_handler(handler: BaseCallbackHandler, inherit: bool = True) -> None
```

Set handler as the only handler on the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handler` | The handler to set.<br>**TYPE:**`BaseCallbackHandler` |
| `inherit` | Whether to inherit the handler.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``add\_tags [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.add_tags "Copy anchor link to this section for reference")

```
add_tags(tags: list[str], inherit: bool = True) -> None
```

Add tags to the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tags` | The tags to add.<br>**TYPE:**`list[str]` |
| `inherit` | Whether to inherit the tags.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``remove\_tags [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.remove_tags "Copy anchor link to this section for reference")

```
remove_tags(tags: list[str]) -> None
```

Remove tags from the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tags` | The tags to remove.<br>**TYPE:**`list[str]` |

### ``add\_metadata [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.add_metadata "Copy anchor link to this section for reference")

```
add_metadata(metadata: dict[str, Any], inherit: bool = True) -> None
```

Add metadata to the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `metadata` | The metadata to add.<br>**TYPE:**`dict[str, Any]` |
| `inherit` | Whether to inherit the metadata.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``remove\_metadata [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.CallbackManager.remove_metadata "Copy anchor link to this section for reference")

```
remove_metadata(keys: list[str]) -> None
```

Remove metadata from the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `keys` | The keys to remove.<br>**TYPE:**`list[str]` |

## ``AsyncCallbackManager [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager "Copy anchor link to this section for reference")

Bases: `BaseCallbackManager`

Async callback manager that handles callbacks from LangChain.

| METHOD | DESCRIPTION |
| --- | --- |
| `on_llm_start` | Run when LLM starts running. |
| `on_chat_model_start` | Async run when LLM starts running. |
| `on_chain_start` | Async run when chain starts running. |
| `on_tool_start` | Run when the tool starts running. |
| `on_custom_event` | Dispatch an adhoc event to the handlers (async version). |
| `on_retriever_start` | Run when the retriever starts running. |
| `configure` | Configure the async callback manager. |
| `__init__` | Initialize callback manager. |
| `copy` | Return a copy of the callback manager. |
| `merge` | Merge the callback manager with another callback manager. |
| `add_handler` | Add a handler to the callback manager. |
| `remove_handler` | Remove a handler from the callback manager. |
| `set_handlers` | Set handlers as the only handlers on the callback manager. |
| `set_handler` | Set handler as the only handler on the callback manager. |
| `add_tags` | Add tags to the callback manager. |
| `remove_tags` | Remove tags from the callback manager. |
| `add_metadata` | Add metadata to the callback manager. |
| `remove_metadata` | Remove metadata from the callback manager. |

### ``is\_async`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.is_async "Copy anchor link to this section for reference")

```
is_async: bool
```

Return whether the handler is async.

### ``on\_llm\_start`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.on_llm_start "Copy anchor link to this section for reference")

```
on_llm_start(
    serialized: dict[str, Any],
    prompts: list[str],
    run_id: UUID | None = None,
    **kwargs: Any,
) -> list[AsyncCallbackManagerForLLMRun]
```

Run when LLM starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized LLM.<br>**TYPE:**`dict[str, Any]` |
| `prompts` | The list of prompts.<br>**TYPE:**`list[str]` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[AsyncCallbackManagerForLLMRun]` | The list of async callback managers, one for each LLM Run corresponding to |
| `list[AsyncCallbackManagerForLLMRun]` | each prompt. |

### ``on\_chat\_model\_start`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.on_chat_model_start "Copy anchor link to this section for reference")

```
on_chat_model_start(
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    run_id: UUID | None = None,
    **kwargs: Any,
) -> list[AsyncCallbackManagerForLLMRun]
```

Async run when LLM starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized LLM.<br>**TYPE:**`dict[str, Any]` |
| `messages` | The list of messages.<br>**TYPE:**`list[list[BaseMessage]]` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[AsyncCallbackManagerForLLMRun]` | The list of async callback managers, one for each LLM Run corresponding to |
| `list[AsyncCallbackManagerForLLMRun]` | each inner message list. |

### ``on\_chain\_start`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.on_chain_start "Copy anchor link to this section for reference")

```
on_chain_start(
    serialized: dict[str, Any] | None,
    inputs: dict[str, Any] | Any,
    run_id: UUID | None = None,
    **kwargs: Any,
) -> AsyncCallbackManagerForChainRun
```

Async run when chain starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chain.<br>**TYPE:**`dict[str, Any] | None` |
| `inputs` | The inputs to the chain.<br>**TYPE:**`dict[str, Any] | Any` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `AsyncCallbackManagerForChainRun` | The async callback manager for the chain run. |

### ``on\_tool\_start`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.on_tool_start "Copy anchor link to this section for reference")

```
on_tool_start(
    serialized: dict[str, Any] | None,
    input_str: str,
    run_id: UUID | None = None,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> AsyncCallbackManagerForToolRun
```

Run when the tool starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized tool.<br>**TYPE:**`dict[str, Any] | None` |
| `input_str` | The input to the tool.<br>**TYPE:**`str` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `AsyncCallbackManagerForToolRun` | The async callback manager for the tool run. |

### ``on\_custom\_event`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.on_custom_event "Copy anchor link to this section for reference")

```
on_custom_event(
    name: str, data: Any, run_id: UUID | None = None, **kwargs: Any
) -> None
```

Dispatch an adhoc event to the handlers (async version).

This event should NOT be used in any internal LangChain code. The event
is meant specifically for users of the library to dispatch custom
events that are tailored to their application.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | The name of the adhoc event.<br>**TYPE:**`str` |
| `data` | The data for the adhoc event.<br>**TYPE:**`Any` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If additional keyword arguments are passed. |

### ``on\_retriever\_start`async`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.on_retriever_start "Copy anchor link to this section for reference")

```
on_retriever_start(
    serialized: dict[str, Any] | None,
    query: str,
    run_id: UUID | None = None,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> AsyncCallbackManagerForRetrieverRun
```

Run when the retriever starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized retriever.<br>**TYPE:**`dict[str, Any] | None` |
| `query` | The query.<br>**TYPE:**`str` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `AsyncCallbackManagerForRetrieverRun` | The async callback manager for the retriever run. |

### ``configure`classmethod`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.configure "Copy anchor link to this section for reference")

```
configure(
    inheritable_callbacks: Callbacks = None,
    local_callbacks: Callbacks = None,
    verbose: bool = False,
    inheritable_tags: list[str] | None = None,
    local_tags: list[str] | None = None,
    inheritable_metadata: dict[str, Any] | None = None,
    local_metadata: dict[str, Any] | None = None,
) -> AsyncCallbackManager
```

Configure the async callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `inheritable_callbacks` | The inheritable callbacks.<br>**TYPE:**`Callbacks`**DEFAULT:**`None` |
| `local_callbacks` | The local callbacks.<br>**TYPE:**`Callbacks`**DEFAULT:**`None` |
| `verbose` | Whether to enable verbose mode.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `inheritable_tags` | The inheritable tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `local_tags` | The local tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `inheritable_metadata` | The inheritable metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `local_metadata` | The local metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `AsyncCallbackManager` | The configured async callback manager. |

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.__init__ "Copy anchor link to this section for reference")

```
__init__(
    handlers: list[BaseCallbackHandler],
    inheritable_handlers: list[BaseCallbackHandler] | None = None,
    parent_run_id: UUID | None = None,
    *,
    tags: list[str] | None = None,
    inheritable_tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inheritable_metadata: dict[str, Any] | None = None,
) -> None
```

Initialize callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handlers` | The handlers.<br>**TYPE:**`list[BaseCallbackHandler]` |
| `inheritable_handlers` | The inheritable handlers.<br>**TYPE:**`list[BaseCallbackHandler] | None`**DEFAULT:**`None` |
| `parent_run_id` | The parent run ID.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `inheritable_tags` | The inheritable tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `inheritable_metadata` | The inheritable metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |

### ``copy [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.copy "Copy anchor link to this section for reference")

```
copy() -> Self
```

Return a copy of the callback manager.

### ``merge [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.merge "Copy anchor link to this section for reference")

```
merge(other: BaseCallbackManager) -> Self
```

Merge the callback manager with another callback manager.

May be overwritten in subclasses.

Primarily used internally within `merge_configs`.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | The merged callback manager of the same type as the current object. |

Example: Merging two callback managers.

````
```python
from langchain_core.callbacks.manager import (
    CallbackManager,
    trace_as_chain_group,
)
from langchain_core.callbacks.stdout import StdOutCallbackHandler

manager = CallbackManager(handlers=[StdOutCallbackHandler()], tags=["tag2"])
with trace_as_chain_group("My Group Name", tags=["tag1"]) as group_manager:
    merged_manager = group_manager.merge(manager)
    print(merged_manager.handlers)
    # [\
    #    <langchain_core.callbacks.stdout.StdOutCallbackHandler object at ...>,\
    #    <langchain_core.callbacks.streaming_stdout.StreamingStdOutCallbackHandler object at ...>,\
    # ]

    print(merged_manager.tags)
    #    ['tag2', 'tag1']
```
````

### ``add\_handler [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.add_handler "Copy anchor link to this section for reference")

```
add_handler(handler: BaseCallbackHandler, inherit: bool = True) -> None
```

Add a handler to the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handler` | The handler to add.<br>**TYPE:**`BaseCallbackHandler` |
| `inherit` | Whether to inherit the handler.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``remove\_handler [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.remove_handler "Copy anchor link to this section for reference")

```
remove_handler(handler: BaseCallbackHandler) -> None
```

Remove a handler from the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handler` | The handler to remove.<br>**TYPE:**`BaseCallbackHandler` |

### ``set\_handlers [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.set_handlers "Copy anchor link to this section for reference")

```
set_handlers(handlers: list[BaseCallbackHandler], inherit: bool = True) -> None
```

Set handlers as the only handlers on the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handlers` | The handlers to set.<br>**TYPE:**`list[BaseCallbackHandler]` |
| `inherit` | Whether to inherit the handlers.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``set\_handler [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.set_handler "Copy anchor link to this section for reference")

```
set_handler(handler: BaseCallbackHandler, inherit: bool = True) -> None
```

Set handler as the only handler on the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `handler` | The handler to set.<br>**TYPE:**`BaseCallbackHandler` |
| `inherit` | Whether to inherit the handler.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``add\_tags [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.add_tags "Copy anchor link to this section for reference")

```
add_tags(tags: list[str], inherit: bool = True) -> None
```

Add tags to the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tags` | The tags to add.<br>**TYPE:**`list[str]` |
| `inherit` | Whether to inherit the tags.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``remove\_tags [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.remove_tags "Copy anchor link to this section for reference")

```
remove_tags(tags: list[str]) -> None
```

Remove tags from the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tags` | The tags to remove.<br>**TYPE:**`list[str]` |

### ``add\_metadata [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.add_metadata "Copy anchor link to this section for reference")

```
add_metadata(metadata: dict[str, Any], inherit: bool = True) -> None
```

Add metadata to the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `metadata` | The metadata to add.<br>**TYPE:**`dict[str, Any]` |
| `inherit` | Whether to inherit the metadata.<br>**TYPE:**`bool`**DEFAULT:**`True` |

### ``remove\_metadata [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.manager.AsyncCallbackManager.remove_metadata "Copy anchor link to this section for reference")

```
remove_metadata(keys: list[str]) -> None
```

Remove metadata from the callback manager.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `keys` | The keys to remove.<br>**TYPE:**`list[str]` |

## ``UsageMetadataCallbackHandler [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler "Copy anchor link to this section for reference")

Bases: `BaseCallbackHandler`

Callback Handler that tracks AIMessage.usage\_metadata.

Example

```
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

llm_1 = init_chat_model(model="openai:gpt-4o-mini")
llm_2 = init_chat_model(model="anthropic:claude-3-5-haiku-20241022")

callback = UsageMetadataCallbackHandler()
result_1 = llm_1.invoke("Hello", config={"callbacks": [callback]})
result_2 = llm_2.invoke("Hello", config={"callbacks": [callback]})
callback.usage_metadata
```

```
{'gpt-4o-mini-2024-07-18': {'input_tokens': 8,
  'output_tokens': 10,
  'total_tokens': 18,
  'input_token_details': {'audio': 0, 'cache_read': 0},
  'output_token_details': {'audio': 0, 'reasoning': 0}},
 'claude-3-5-haiku-20241022': {'input_tokens': 8,
  'output_tokens': 21,
  'total_tokens': 29,
  'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}
```

Added in `langchain-core` 0.3.49

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Initialize the UsageMetadataCallbackHandler. |
| `on_llm_end` | Collect token usage. |
| `on_text` | Run on an arbitrary text. |
| `on_retry` | Run on a retry event. |
| `on_custom_event` | Override to define a handler for a custom event. |
| `on_llm_start` | Run when LLM starts running. |
| `on_chat_model_start` | Run when a chat model starts running. |
| `on_retriever_start` | Run when the `Retriever` starts running. |
| `on_chain_start` | Run when a chain starts running. |
| `on_tool_start` | Run when the tool starts running. |
| `on_retriever_error` | Run when `Retriever` errors. |
| `on_retriever_end` | Run when `Retriever` ends running. |
| `on_tool_end` | Run when the tool ends running. |
| `on_tool_error` | Run when tool errors. |
| `on_chain_end` | Run when chain ends running. |
| `on_chain_error` | Run when chain errors. |
| `on_agent_action` | Run on agent action. |
| `on_agent_finish` | Run on the agent end. |
| `on_llm_new_token` | Run on new output token. Only available when streaming is enabled. |
| `on_llm_error` | Run when LLM errors. |

### ``raise\_error`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.raise_error "Copy anchor link to this section for reference")

```
raise_error: bool = False
```

Whether to raise an error if an exception occurs.

### ``run\_inline`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.run_inline "Copy anchor link to this section for reference")

```
run_inline: bool = False
```

Whether to run the callback inline.

### ``ignore\_llm`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.ignore_llm "Copy anchor link to this section for reference")

```
ignore_llm: bool
```

Whether to ignore LLM callbacks.

### ``ignore\_retry`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.ignore_retry "Copy anchor link to this section for reference")

```
ignore_retry: bool
```

Whether to ignore retry callbacks.

### ``ignore\_chain`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.ignore_chain "Copy anchor link to this section for reference")

```
ignore_chain: bool
```

Whether to ignore chain callbacks.

### ``ignore\_agent`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.ignore_agent "Copy anchor link to this section for reference")

```
ignore_agent: bool
```

Whether to ignore agent callbacks.

### ``ignore\_retriever`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.ignore_retriever "Copy anchor link to this section for reference")

```
ignore_retriever: bool
```

Whether to ignore retriever callbacks.

### ``ignore\_chat\_model`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.ignore_chat_model "Copy anchor link to this section for reference")

```
ignore_chat_model: bool
```

Whether to ignore chat model callbacks.

### ``ignore\_custom\_event`property`[¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.ignore_custom_event "Copy anchor link to this section for reference")

```
ignore_custom_event: bool
```

Ignore custom event.

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.__init__ "Copy anchor link to this section for reference")

```
__init__() -> None
```

Initialize the UsageMetadataCallbackHandler.

### ``on\_llm\_end [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_llm_end "Copy anchor link to this section for reference")

```
on_llm_end(response: LLMResult, **kwargs: Any) -> None
```

Collect token usage.

### ``on\_text [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_text "Copy anchor link to this section for reference")

```
on_text(
    text: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
) -> Any
```

Run on an arbitrary text.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The text.<br>**TYPE:**`str` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retry [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_retry "Copy anchor link to this section for reference")

```
on_retry(
    retry_state: RetryCallState,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run on a retry event.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `retry_state` | The retry state.<br>**TYPE:**`RetryCallState` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_custom\_event [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_custom_event "Copy anchor link to this section for reference")

```
on_custom_event(
    name: str,
    data: Any,
    *,
    run_id: UUID,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Override to define a handler for a custom event.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | The name of the custom event.<br>**TYPE:**`str` |
| `data` | The data for the custom event. Format will match the format specified<br>by the user.<br>**TYPE:**`Any` |
| `run_id` | The ID of the run.<br>**TYPE:**`UUID` |
| `tags` | The tags associated with the custom event (includes inherited tags).<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata associated with the custom event (includes inherited<br>metadata).<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |

### ``on\_llm\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_llm_start "Copy anchor link to this section for reference")

```
on_llm_start(
    serialized: dict[str, Any],
    prompts: list[str],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when LLM starts running.

Warning

This method is called for non-chat models (regular LLMs). If you're
implementing a handler for a chat model, you should use
`on_chat_model_start` instead.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized LLM.<br>**TYPE:**`dict[str, Any]` |
| `prompts` | The prompts.<br>**TYPE:**`list[str]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chat\_model\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_chat_model_start "Copy anchor link to this section for reference")

```
on_chat_model_start(
    serialized: dict[str, Any],
    messages: list[list[BaseMessage]],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when a chat model starts running.

Warning

This method is called for chat models. If you're implementing a handler for
a non-chat model, you should use `on_llm_start` instead.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chat model.<br>**TYPE:**`dict[str, Any]` |
| `messages` | The messages.<br>**TYPE:**`list[list[BaseMessage]]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retriever\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_retriever_start "Copy anchor link to this section for reference")

```
on_retriever_start(
    serialized: dict[str, Any],
    query: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when the `Retriever` starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized `Retriever`.<br>**TYPE:**`dict[str, Any]` |
| `query` | The query.<br>**TYPE:**`str` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chain\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_chain_start "Copy anchor link to this section for reference")

```
on_chain_start(
    serialized: dict[str, Any],
    inputs: dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when a chain starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chain.<br>**TYPE:**`dict[str, Any]` |
| `inputs` | The inputs.<br>**TYPE:**`dict[str, Any]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_tool\_start [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_tool_start "Copy anchor link to this section for reference")

```
on_tool_start(
    serialized: dict[str, Any],
    input_str: str,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any
```

Run when the tool starts running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `serialized` | The serialized chain.<br>**TYPE:**`dict[str, Any]` |
| `input_str` | The input string.<br>**TYPE:**`str` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `metadata` | The metadata.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `inputs` | The inputs.<br>**TYPE:**`dict[str, Any] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retriever\_error [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_retriever_error "Copy anchor link to this section for reference")

```
on_retriever_error(
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run when `Retriever` errors.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that occurred.<br>**TYPE:**`BaseException` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_retriever\_end [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_retriever_end "Copy anchor link to this section for reference")

```
on_retriever_end(
    documents: Sequence[Document],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run when `Retriever` ends running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents retrieved.<br>**TYPE:**`Sequence[Document]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_tool\_end [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_tool_end "Copy anchor link to this section for reference")

```
on_tool_end(
    output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
) -> Any
```

Run when the tool ends running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `output` | The output of the tool.<br>**TYPE:**`Any` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_tool\_error [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_tool_error "Copy anchor link to this section for reference")

```
on_tool_error(
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run when tool errors.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that occurred.<br>**TYPE:**`BaseException` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chain\_end [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_chain_end "Copy anchor link to this section for reference")

```
on_chain_end(
    outputs: dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run when chain ends running.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `outputs` | The outputs of the chain.<br>**TYPE:**`dict[str, Any]` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_chain\_error [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_chain_error "Copy anchor link to this section for reference")

```
on_chain_error(
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run when chain errors.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that occurred.<br>**TYPE:**`BaseException` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_agent\_action [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_agent_action "Copy anchor link to this section for reference")

```
on_agent_action(
    action: AgentAction,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run on agent action.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `action` | The agent action.<br>**TYPE:**`AgentAction` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_agent\_finish [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_agent_finish "Copy anchor link to this section for reference")

```
on_agent_finish(
    finish: AgentFinish,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    **kwargs: Any,
) -> Any
```

Run on the agent end.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `finish` | The agent finish.<br>**TYPE:**`AgentFinish` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_llm\_new\_token [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_llm_new_token "Copy anchor link to this section for reference")

```
on_llm_new_token(
    token: str,
    *,
    chunk: GenerationChunk | ChatGenerationChunk | None = None,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any
```

Run on new output token. Only available when streaming is enabled.

For both chat models and non-chat models (legacy LLMs).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `token` | The new token.<br>**TYPE:**`str` |
| `chunk` | The new generated chunk, containing content and other information.<br>**TYPE:**`GenerationChunk | ChatGenerationChunk | None`**DEFAULT:**`None` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

### ``on\_llm\_error [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.UsageMetadataCallbackHandler.on_llm_error "Copy anchor link to this section for reference")

```
on_llm_error(
    error: BaseException,
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any
```

Run when LLM errors.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that occurred.<br>**TYPE:**`BaseException` |
| `run_id` | The ID of the current run.<br>**TYPE:**`UUID` |
| `parent_run_id` | The ID of the parent run.<br>**TYPE:**`UUID | None`**DEFAULT:**`None` |
| `tags` | The tags.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

## ``get\_usage\_metadata\_callback [¶](https://reference.langchain.com/python/langchain_core/callbacks/\#langchain_core.callbacks.usage.get_usage_metadata_callback "Copy anchor link to this section for reference")

```
get_usage_metadata_callback(
    name: str = "usage_metadata_callback",
) -> Generator[UsageMetadataCallbackHandler, None, None]
```

Get usage metadata callback.

Get context manager for tracking usage metadata across chat model calls using
[`AIMessage.usage_metadata`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage.usage_metadata "<code class=\"doc-symbol doc-symbol-heading doc-symbol-attribute\"></code>            <span class=\"doc doc-object-name doc-attribute-name\">usage_metadata</span>     <span class=\"doc doc-labels\">       <small class=\"doc doc-label doc-label-class-attribute\"><code>class-attribute</code></small>       <small class=\"doc doc-label doc-label-instance-attribute\"><code>instance-attribute</code></small>   </span>").

| PARAMETER | DESCRIPTION |
| --- | --- |
| `name` | The name of the context variable.<br>**TYPE:**`str`**DEFAULT:**`'usage_metadata_callback'` |

| YIELDS | DESCRIPTION |
| --- | --- |
| `UsageMetadataCallbackHandler` | The usage metadata callback. |

Example

```
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback

llm_1 = init_chat_model(model="openai:gpt-4o-mini")
llm_2 = init_chat_model(model="anthropic:claude-3-5-haiku-20241022")

with get_usage_metadata_callback() as cb:
    llm_1.invoke("Hello")
    llm_2.invoke("Hello")
    print(cb.usage_metadata)
```

```
{
    "gpt-4o-mini-2024-07-18": {
        "input_tokens": 8,
        "output_tokens": 10,
        "total_tokens": 18,
        "input_token_details": {"audio": 0, "cache_read": 0},
        "output_token_details": {"audio": 0, "reasoning": 0},
    },
    "claude-3-5-haiku-20241022": {
        "input_tokens": 8,
        "output_tokens": 21,
        "total_tokens": 29,
        "input_token_details": {"cache_read": 0, "cache_creation": 0},
    },
}
```

Added in `langchain-core` 0.3.49

Back to top