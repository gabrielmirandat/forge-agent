[Skip to content](https://reference.langchain.com/python/langchain_core/embeddings/#langchain_core.embeddings.embeddings.Embeddings)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_core/embeddings.md "Edit this page")

# Embeddings

## ``Embeddings [¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.embeddings.Embeddings "Copy anchor link to this section for reference")

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

| METHOD | DESCRIPTION |
| --- | --- |
| `embed_documents` | Embed search docs. |
| `embed_query` | Embed query text. |
| `aembed_documents` | Asynchronous Embed search docs. |
| `aembed_query` | Asynchronous Embed query text. |

### ``embed\_documents`abstractmethod`[¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.embeddings.Embeddings.embed_documents "Copy anchor link to this section for reference")

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

### ``embed\_query`abstractmethod`[¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.embeddings.Embeddings.embed_query "Copy anchor link to this section for reference")

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

### ``aembed\_documents`async`[¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.embeddings.Embeddings.aembed_documents "Copy anchor link to this section for reference")

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

### ``aembed\_query`async`[¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.embeddings.Embeddings.aembed_query "Copy anchor link to this section for reference")

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

## ``DeterministicFakeEmbedding [¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.fake.DeterministicFakeEmbedding "Copy anchor link to this section for reference")

Bases: `Embeddings`, `BaseModel`

Deterministic fake embedding model for unit testing purposes.

This embedding model creates embeddings by sampling from a normal distribution
with a seed based on the hash of the text.

Toy model

Do not use this outside of testing, as it is not a real embedding model.

Instantiate

```
from langchain_core.embeddings import DeterministicFakeEmbedding

embed = DeterministicFakeEmbedding(size=100)
```

Embed single text

```
input_text = "The meaning of life is 42"
vector = embed.embed_query(input_text)
print(vector[:3])
```

```
[-0.700234640213188, -0.581266257710429, -1.1328482266445354]
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
[-0.5670477847544458, -0.31403828652395727, -0.5840547508955257]
```

| METHOD | DESCRIPTION |
| --- | --- |
| `embed_documents` | Embed search docs. |
| `embed_query` | Embed query text. |
| `aembed_documents` | Asynchronous Embed search docs. |
| `aembed_query` | Asynchronous Embed query text. |

### ``size`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.fake.DeterministicFakeEmbedding.size "Copy anchor link to this section for reference")

```
size: int
```

The size of the embedding vector.

### ``embed\_documents [¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.fake.DeterministicFakeEmbedding.embed_documents "Copy anchor link to this section for reference")

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

### ``embed\_query [¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.fake.DeterministicFakeEmbedding.embed_query "Copy anchor link to this section for reference")

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

### ``aembed\_documents`async`[¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.fake.DeterministicFakeEmbedding.aembed_documents "Copy anchor link to this section for reference")

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

### ``aembed\_query`async`[¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.fake.DeterministicFakeEmbedding.aembed_query "Copy anchor link to this section for reference")

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

## ``FakeEmbeddings [¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.fake.FakeEmbeddings "Copy anchor link to this section for reference")

Bases: `Embeddings`, `BaseModel`

Fake embedding model for unit testing purposes.

This embedding model creates embeddings by sampling from a normal distribution.

Toy model

Do not use this outside of testing, as it is not a real embedding model.

Instantiate

```
from langchain_core.embeddings import FakeEmbeddings

embed = FakeEmbeddings(size=100)
```

Embed single text

```
input_text = "The meaning of life is 42"
vector = embed.embed_query(input_text)
print(vector[:3])
```

```
[-0.700234640213188, -0.581266257710429, -1.1328482266445354]
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
[-0.5670477847544458, -0.31403828652395727, -0.5840547508955257]
```

| METHOD | DESCRIPTION |
| --- | --- |
| `embed_documents` | Embed search docs. |
| `embed_query` | Embed query text. |
| `aembed_documents` | Asynchronous Embed search docs. |
| `aembed_query` | Asynchronous Embed query text. |

### ``size`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.fake.FakeEmbeddings.size "Copy anchor link to this section for reference")

```
size: int
```

The size of the embedding vector.

### ``embed\_documents [¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.fake.FakeEmbeddings.embed_documents "Copy anchor link to this section for reference")

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

### ``embed\_query [¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.fake.FakeEmbeddings.embed_query "Copy anchor link to this section for reference")

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

### ``aembed\_documents`async`[¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.fake.FakeEmbeddings.aembed_documents "Copy anchor link to this section for reference")

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

### ``aembed\_query`async`[¶](https://reference.langchain.com/python/langchain_core/embeddings/\#langchain_core.embeddings.fake.FakeEmbeddings.aembed_query "Copy anchor link to this section for reference")

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