![Revisit consent button](https://uploads-ssl.webflow.com/65ff950538088944d66126b3/662ef3209b872e92e41212f6_cookieicon.png)

![](https://cdn-cookieyes.com/assets/images/close.svg)

We value your privacy

We use cookies to improve your experience and to understand how our site is used. Some analytics tools may share limited data with our advertising partners. You can opt out at any time.

Do Not Sell or Share My Personal Information

Opt-out Preferences![](https://cdn-cookieyes.com/assets/images/close.svg)

We use cookies to improve your experience and to understand how our site is used. Some analytics tools may share limited data with our advertising partners. You can opt out of this sharing at any time by selecting **“Do Not Sell or Share My Personal Information”** and saving your preferences.

Do Not Sell or Share My Personal Information

CancelSave My Preferences

[Skip to content](https://reference.langchain.com/python/langchain_core/exceptions/#langchain_core.exceptions.OutputParserException)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_core/exceptions.md "Edit this page")

# Exceptions

## ``OutputParserException [¶](https://reference.langchain.com/python/langchain_core/exceptions/\#langchain_core.exceptions.OutputParserException "Copy anchor link to this section for reference")

Bases: `ValueError`, `LangChainException`

Exception that output parsers should raise to signify a parsing error.

This exists to differentiate parsing errors from other code or execution errors
that also may arise inside the output parser.

`OutputParserException` will be available to catch and handle in ways to fix the
parsing error, while other errors will be raised.

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Create an `OutputParserException`. |

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/exceptions/\#langchain_core.exceptions.OutputParserException.__init__ "Copy anchor link to this section for reference")

```
__init__(
    error: Any,
    observation: str | None = None,
    llm_output: str | None = None,
    send_to_llm: bool = False,
)
```

Create an `OutputParserException`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `error` | The error that's being re-raised or an error message.<br>**TYPE:**`Any` |
| `observation` | String explanation of error which can be passed to a model to<br>try and remediate the issue.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `llm_output` | String model output which is error-ing.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `send_to_llm` | Whether to send the observation and llm\_output back to an Agent<br>after an `OutputParserException` has been raised.<br>This gives the underlying model driving the agent the context that the<br>previous output was improperly structured, in the hopes that it will<br>update the output to the correct format.<br>**TYPE:**`bool`**DEFAULT:**`False` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If `send_to_llm` is `True` but either observation or<br>`llm_output` are not provided. |

Back to top