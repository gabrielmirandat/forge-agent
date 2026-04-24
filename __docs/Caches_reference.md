[Skip to content](https://reference.langchain.com/python/langchain_core/caches/#langchain_core.caches)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_core/caches.md "Edit this page")

# Caches

## ``caches [¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches "Copy anchor link to this section for reference")

Optional caching layer for language models.

Distinct from provider-based [prompt caching](https://docs.langchain.com/oss/python/langchain/models#prompt-caching).

Beta feature

This is a beta feature. Please be wary of deploying experimental code to production
unless you've taken appropriate precautions.

A cache is useful for two reasons:

1. It can save you money by reducing the number of API calls you make to the LLM
    provider if you're often requesting the same completion multiple times.
2. It can speed up your application by reducing the number of API calls you make to the
    LLM provider.

### ``InMemoryCache [¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.InMemoryCache "Copy anchor link to this section for reference")

Bases: `BaseCache`

Cache that stores things in memory.

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Initialize with empty cache. |
| `lookup` | Look up based on `prompt` and `llm_string`. |
| `update` | Update cache based on `prompt` and `llm_string`. |
| `clear` | Clear cache. |
| `alookup` | Async look up based on `prompt` and `llm_string`. |
| `aupdate` | Async update cache based on `prompt` and `llm_string`. |
| `aclear` | Async clear cache. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.InMemoryCache.__init__ "Copy anchor link to this section for reference")

```
__init__(*, maxsize: int | None = None) -> None
```

Initialize with empty cache.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `maxsize` | The maximum number of items to store in the cache.<br>If `None`, the cache has no maximum size.<br>If the cache exceeds the maximum size, the oldest items are removed.<br>**TYPE:**`int | None`**DEFAULT:**`None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If `maxsize` is less than or equal to `0`. |

#### ``lookup [¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.InMemoryCache.lookup "Copy anchor link to this section for reference")

```
lookup(prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None
```

Look up based on `prompt` and `llm_string`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompt` | A string representation of the prompt.<br>In the case of a chat model, the prompt is a non-trivial<br>serialization of the prompt into the language model.<br>**TYPE:**`str` |
| `llm_string` | A string representation of the LLM configuration.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RETURN_VAL_TYPE | None` | On a cache miss, return `None`. On a cache hit, return the cached value. |

#### ``update [¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.InMemoryCache.update "Copy anchor link to this section for reference")

```
update(prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None
```

Update cache based on `prompt` and `llm_string`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompt` | A string representation of the prompt.<br>In the case of a chat model, the prompt is a non-trivial<br>serialization of the prompt into the language model.<br>**TYPE:**`str` |
| `llm_string` | A string representation of the LLM configuration.<br>**TYPE:**`str` |
| `return_val` | The value to be cached. The value is a list of `Generation`<br>(or subclasses).<br>**TYPE:**`RETURN_VAL_TYPE` |

#### ``clear [¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.InMemoryCache.clear "Copy anchor link to this section for reference")

```
clear(**kwargs: Any) -> None
```

Clear cache.

#### ``alookup`async`[¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.InMemoryCache.alookup "Copy anchor link to this section for reference")

```
alookup(prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None
```

Async look up based on `prompt` and `llm_string`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompt` | A string representation of the prompt.<br>In the case of a chat model, the prompt is a non-trivial<br>serialization of the prompt into the language model.<br>**TYPE:**`str` |
| `llm_string` | A string representation of the LLM configuration.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RETURN_VAL_TYPE | None` | On a cache miss, return `None`. On a cache hit, return the cached value. |

#### ``aupdate`async`[¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.InMemoryCache.aupdate "Copy anchor link to this section for reference")

```
aupdate(prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None
```

Async update cache based on `prompt` and `llm_string`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompt` | A string representation of the prompt.<br>In the case of a chat model, the prompt is a non-trivial<br>serialization of the prompt into the language model.<br>**TYPE:**`str` |
| `llm_string` | A string representation of the LLM configuration.<br>**TYPE:**`str` |
| `return_val` | The value to be cached. The value is a list of `Generation`<br>(or subclasses).<br>**TYPE:**`RETURN_VAL_TYPE` |

#### ``aclear`async`[¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.InMemoryCache.aclear "Copy anchor link to this section for reference")

```
aclear(**kwargs: Any) -> None
```

Async clear cache.

### ``BaseCache [¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.BaseCache "Copy anchor link to this section for reference")

Bases: `ABC`

Interface for a caching layer for LLMs and Chat models.

The cache interface consists of the following methods:

- lookup: Look up a value based on a prompt and `llm_string`.
- update: Update the cache based on a prompt and `llm_string`.
- clear: Clear the cache.

In addition, the cache interface provides an async version of each method.

The default implementation of the async methods is to run the synchronous
method in an executor. It's recommended to override the async methods
and provide async implementations to avoid unnecessary overhead.

| METHOD | DESCRIPTION |
| --- | --- |
| `lookup` | Look up based on `prompt` and `llm_string`. |
| `update` | Update cache based on `prompt` and `llm_string`. |
| `clear` | Clear cache that can take additional keyword arguments. |
| `alookup` | Async look up based on `prompt` and `llm_string`. |
| `aupdate` | Async update cache based on `prompt` and `llm_string`. |
| `aclear` | Async clear cache that can take additional keyword arguments. |

#### ``lookup`abstractmethod`[¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.BaseCache.lookup "Copy anchor link to this section for reference")

```
lookup(prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None
```

Look up based on `prompt` and `llm_string`.

A cache implementation is expected to generate a key from the 2-tuple
of `prompt` and `llm_string` (e.g., by concatenating them with a delimiter).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompt` | A string representation of the prompt.<br>In the case of a chat model, the prompt is a non-trivial<br>serialization of the prompt into the language model.<br>**TYPE:**`str` |
| `llm_string` | A string representation of the LLM configuration.<br>This is used to capture the invocation parameters of the LLM<br>(e.g., model name, temperature, stop tokens, max tokens, etc.).<br>These invocation parameters are serialized into a string representation.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RETURN_VAL_TYPE | None` | On a cache miss, return `None`. On a cache hit, return the cached value. |
| `RETURN_VAL_TYPE | None` | The cached value is a list of `Generation` (or subclasses). |

#### ``update`abstractmethod`[¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.BaseCache.update "Copy anchor link to this section for reference")

```
update(prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None
```

Update cache based on `prompt` and `llm_string`.

The prompt and llm\_string are used to generate a key for the cache.
The key should match that of the lookup method.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompt` | A string representation of the prompt.<br>In the case of a chat model, the prompt is a non-trivial<br>serialization of the prompt into the language model.<br>**TYPE:**`str` |
| `llm_string` | A string representation of the LLM configuration.<br>This is used to capture the invocation parameters of the LLM<br>(e.g., model name, temperature, stop tokens, max tokens, etc.).<br>These invocation parameters are serialized into a string<br>representation.<br>**TYPE:**`str` |
| `return_val` | The value to be cached. The value is a list of `Generation`<br>(or subclasses).<br>**TYPE:**`RETURN_VAL_TYPE` |

#### ``clear`abstractmethod`[¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.BaseCache.clear "Copy anchor link to this section for reference")

```
clear(**kwargs: Any) -> None
```

Clear cache that can take additional keyword arguments.

#### ``alookup`async`[¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.BaseCache.alookup "Copy anchor link to this section for reference")

```
alookup(prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None
```

Async look up based on `prompt` and `llm_string`.

A cache implementation is expected to generate a key from the 2-tuple
of `prompt` and `llm_string` (e.g., by concatenating them with a delimiter).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompt` | A string representation of the prompt.<br>In the case of a chat model, the prompt is a non-trivial<br>serialization of the prompt into the language model.<br>**TYPE:**`str` |
| `llm_string` | A string representation of the LLM configuration.<br>This is used to capture the invocation parameters of the LLM<br>(e.g., model name, temperature, stop tokens, max tokens, etc.).<br>These invocation parameters are serialized into a string<br>representation.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RETURN_VAL_TYPE | None` | On a cache miss, return `None`. On a cache hit, return the cached value. |
| `RETURN_VAL_TYPE | None` | The cached value is a list of `Generation` (or subclasses). |

#### ``aupdate`async`[¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.BaseCache.aupdate "Copy anchor link to this section for reference")

```
aupdate(prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None
```

Async update cache based on `prompt` and `llm_string`.

The prompt and llm\_string are used to generate a key for the cache.
The key should match that of the look up method.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `prompt` | A string representation of the prompt.<br>In the case of a chat model, the prompt is a non-trivial<br>serialization of the prompt into the language model.<br>**TYPE:**`str` |
| `llm_string` | A string representation of the LLM configuration.<br>This is used to capture the invocation parameters of the LLM<br>(e.g., model name, temperature, stop tokens, max tokens, etc.).<br>These invocation parameters are serialized into a string<br>representation.<br>**TYPE:**`str` |
| `return_val` | The value to be cached. The value is a list of `Generation`<br>(or subclasses).<br>**TYPE:**`RETURN_VAL_TYPE` |

#### ``aclear`async`[¶](https://reference.langchain.com/python/langchain_core/caches/\#langchain_core.caches.BaseCache.aclear "Copy anchor link to this section for reference")

```
aclear(**kwargs: Any) -> None
```

Async clear cache that can take additional keyword arguments.

Back to top