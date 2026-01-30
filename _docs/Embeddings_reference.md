[Skip to content](https://reference.langchain.com/python/langchain/embeddings/#langchain.embeddings)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain/embeddings.md "Edit this page")

# Embeddings

Reference docs

This page contains **reference documentation** for Embeddings. See [the docs](https://docs.langchain.com/oss/python/langchain/retrieval#embedding-models) for conceptual guides, tutorials, and examples on using Embeddings.

## ``embeddings [¶](https://reference.langchain.com/python/langchain/embeddings/\#langchain.embeddings "Copy anchor link to this section for reference")

Embeddings models.

Modules moved

With the release of `langchain 1.0.0`, several embeddings modules were moved to
`langchain-classic`, such as `CacheBackedEmbeddings` and all community
embeddings. See [list](https://github.com/langchain-ai/langchain/blob/bdf1cd383ce36dc18381a3bf3fb0a579337a32b5/libs/langchain/langchain/embeddings/__init__.py)
of moved modules to inform your migration.

### ``init\_embeddings [¶](https://reference.langchain.com/python/langchain/embeddings/\#langchain.embeddings.init_embeddings "Copy anchor link to this section for reference")

```
init_embeddings(
    model: str, *, provider: str | None = None, **kwargs: Any
) -> Embeddings
```

Initialize an embedding model from a model name and optional provider.

Note

Requires the integration package for the chosen model provider to be installed.

See the `model_provider` parameter below for specific package names
(e.g., `pip install langchain-openai`).

Refer to the [provider integration's API reference](https://docs.langchain.com/oss/python/integrations/providers)
for supported model parameters to use as `**kwargs`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `model` | The name of the model, e.g. `'openai:text-embedding-3-small'`.<br>You can also specify model and model provider in a single argument using<br>`'{model_provider}:{model}'` format, e.g. `'openai:text-embedding-3-small'`.<br>**TYPE:**`str` |
| `provider` | The model provider if not specified as part of the model arg<br>(see above).<br>Supported `provider` values and the corresponding integration package<br>are:<br>- `openai` -\> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)<br>- `azure_openai` -\> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)<br>- `bedrock` -\> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)<br>- `cohere` -\> [`langchain-cohere`](https://docs.langchain.com/oss/python/integrations/providers/cohere)<br>- `google_vertexai` -\> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)<br>- `huggingface` -\> [`langchain-huggingface`](https://docs.langchain.com/oss/python/integrations/providers/huggingface)<br>- `mistralai` -\> [`langchain-mistralai`](https://docs.langchain.com/oss/python/integrations/providers/mistralai)<br>- `ollama` -\> [`langchain-ollama`](https://docs.langchain.com/oss/python/integrations/providers/ollama)<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `**kwargs` | Additional model-specific parameters passed to the embedding model.<br>These vary by provider. Refer to the specific model provider's<br>[integration reference](https://reference.langchain.com/python/integrations/)<br>for all available parameters.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Embeddings` | An `Embeddings` instance that can generate embeddings for text. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the model provider is not supported or cannot be determined |
| `ImportError` | If the required provider package is not installed |

Example

```
# pip install langchain langchain-openai

# Using a model string
model = init_embeddings("openai:text-embedding-3-small")
model.embed_query("Hello, world!")

# Using explicit provider
model = init_embeddings(model="text-embedding-3-small", provider="openai")
model.embed_documents(["Hello, world!", "Goodbye, world!"])

# With additional parameters
model = init_embeddings("openai:text-embedding-3-small", api_key="sk-...")
```

Added in `langchain` 0.3.9

### ``Embeddings [¶](https://reference.langchain.com/python/langchain/embeddings/\#langchain.embeddings.Embeddings "Copy anchor link to this section for reference")

Bases: `ABC`

Interface for embedding models.

This is an interface meant for implementing text embedding models.

Text embedding models are used to map text to a vector (a point in n-dimensional
space).

Texts that are similar will usually be mapped to points that are close to each
other in this space. The exact details of what's considered "similar" and how
"distance" is measured in this space are dependent on the specific embedding model.

This abstraction contains a method for embedding a list of documents and a method
for embedding a query text. The embedding of a query text is expected to be a single
vector, while the embedding of a list of documents is expected to be a list of
vectors.

Usually the query embedding is identical to the document embedding, but the
abstraction allows treating them independently.

In addition to the synchronous methods, this interface also provides asynchronous
versions of the methods.

By default, the asynchronous methods are implemented using the synchronous methods;
however, implementations may choose to override the asynchronous methods with
an async native implementation for performance reasons.

#### ``embed\_documents`abstractmethod`[¶](https://reference.langchain.com/python/langchain/embeddings/\#langchain.embeddings.Embeddings.embed_documents "Copy anchor link to this section for reference")

```
embed_documents(texts: list[str]) -> list[list[float]]
```

Embed search docs.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | List of text to embed.<br>**TYPE:**`list[str]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[list[float]]` | List of embeddings. |

#### ``embed\_query`abstractmethod`[¶](https://reference.langchain.com/python/langchain/embeddings/\#langchain.embeddings.Embeddings.embed_query "Copy anchor link to this section for reference")

```
embed_query(text: str) -> list[float]
```

Embed query text.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | Text to embed.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[float]` | Embedding. |

#### ``aembed\_documents`async`[¶](https://reference.langchain.com/python/langchain/embeddings/\#langchain.embeddings.Embeddings.aembed_documents "Copy anchor link to this section for reference")

```
aembed_documents(texts: list[str]) -> list[list[float]]
```

Asynchronous Embed search docs.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | List of text to embed.<br>**TYPE:**`list[str]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[list[float]]` | List of embeddings. |

#### ``aembed\_query`async`[¶](https://reference.langchain.com/python/langchain/embeddings/\#langchain.embeddings.Embeddings.aembed_query "Copy anchor link to this section for reference")

```
aembed_query(text: str) -> list[float]
```

Asynchronous Embed query text.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | Text to embed.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[float]` | Embedding. |

Back to top