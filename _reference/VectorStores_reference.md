[Skip to content](https://reference.langchain.com/python/langchain_core/vectorstores/#langchain_core.vectorstores.base.VectorStore)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_core/vectorstores.md "Edit this page")

# Vector stores

## ``VectorStore [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore "Copy anchor link to this section for reference")

Bases: `ABC`

Interface for vector store.

| METHOD | DESCRIPTION |
| --- | --- |
| `add_texts` | Run more texts through the embeddings and add to the `VectorStore`. |
| `delete` | Delete by vector ID or other criteria. |
| `get_by_ids` | Get documents by their IDs. |
| `aget_by_ids` | Async get documents by their IDs. |
| `adelete` | Async delete by vector ID or other criteria. |
| `aadd_texts` | Async run more texts through the embeddings and add to the `VectorStore`. |
| `add_documents` | Add or update documents in the `VectorStore`. |
| `aadd_documents` | Async run more documents through the embeddings and add to the `VectorStore`. |
| `search` | Return docs most similar to query using a specified search type. |
| `asearch` | Async return docs most similar to query using a specified search type. |
| `similarity_search` | Return docs most similar to query. |
| `similarity_search_with_score` | Run similarity search with distance. |
| `asimilarity_search_with_score` | Async run similarity search with distance. |
| `similarity_search_with_relevance_scores` | Return docs and relevance scores in the range `[0, 1]`. |
| `asimilarity_search_with_relevance_scores` | Async return docs and relevance scores in the range `[0, 1]`. |
| `asimilarity_search` | Async return docs most similar to query. |
| `similarity_search_by_vector` | Return docs most similar to embedding vector. |
| `asimilarity_search_by_vector` | Async return docs most similar to embedding vector. |
| `max_marginal_relevance_search` | Return docs selected using the maximal marginal relevance. |
| `amax_marginal_relevance_search` | Async return docs selected using the maximal marginal relevance. |
| `max_marginal_relevance_search_by_vector` | Return docs selected using the maximal marginal relevance. |
| `amax_marginal_relevance_search_by_vector` | Async return docs selected using the maximal marginal relevance. |
| `from_documents` | Return `VectorStore` initialized from documents and embeddings. |
| `afrom_documents` | Async return `VectorStore` initialized from documents and embeddings. |
| `from_texts` | Return `VectorStore` initialized from texts and embeddings. |
| `afrom_texts` | Async return `VectorStore` initialized from texts and embeddings. |
| `as_retriever` | Return `VectorStoreRetriever` initialized from this `VectorStore`. |

### ``embeddings`property`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.embeddings "Copy anchor link to this section for reference")

```
embeddings: Embeddings | None
```

Access the query embedding object if available.

### ``add\_texts [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.add_texts "Copy anchor link to this section for reference")

```
add_texts(
    texts: Iterable[str],
    metadatas: list[dict] | None = None,
    *,
    ids: list[str] | None = None,
    **kwargs: Any,
) -> list[str]
```

Run more texts through the embeddings and add to the `VectorStore`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | Iterable of strings to add to the `VectorStore`.<br>**TYPE:**`Iterable[str]` |
| `metadatas` | Optional list of metadatas associated with the texts.<br>**TYPE:**`list[dict] | None`**DEFAULT:**`None` |
| `ids` | Optional list of IDs associated with the texts.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | `VectorStore` specific parameters.<br>One of the kwargs should be `ids` which is a list of ids<br>associated with the texts.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | List of IDs from adding the texts into the `VectorStore`. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the number of metadatas does not match the number of texts. |
| `ValueError` | If the number of IDs does not match the number of texts. |

### ``delete [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.delete "Copy anchor link to this section for reference")

```
delete(ids: list[str] | None = None, **kwargs: Any) -> bool | None
```

Delete by vector ID or other criteria.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `ids` | List of IDs to delete. If `None`, delete all.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Other keyword arguments that subclasses might use.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `bool | None` | `True` if deletion is successful, `False` otherwise, `None` if not<br>implemented. |

### ``get\_by\_ids [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.get_by_ids "Copy anchor link to this section for reference")

```
get_by_ids(ids: Sequence[str]) -> list[Document]
```

Get documents by their IDs.

The returned documents are expected to have the ID field set to the ID of the
document in the vector store.

Fewer documents may be returned than requested if some IDs are not found or
if there are duplicated IDs.

Users should not assume that the order of the returned documents matches
the order of the input IDs. Instead, users should rely on the ID field of the
returned documents.

This method should **NOT** raise exceptions if no documents are found for
some IDs.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `ids` | List of IDs to retrieve.<br>**TYPE:**`Sequence[str]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects. |

### ``aget\_by\_ids`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.aget_by_ids "Copy anchor link to this section for reference")

```
aget_by_ids(ids: Sequence[str]) -> list[Document]
```

Async get documents by their IDs.

The returned documents are expected to have the ID field set to the ID of the
document in the vector store.

Fewer documents may be returned than requested if some IDs are not found or
if there are duplicated IDs.

Users should not assume that the order of the returned documents matches
the order of the input IDs. Instead, users should rely on the ID field of the
returned documents.

This method should **NOT** raise exceptions if no documents are found for
some IDs.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `ids` | List of IDs to retrieve.<br>**TYPE:**`Sequence[str]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects. |

### ``adelete`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.adelete "Copy anchor link to this section for reference")

```
adelete(ids: list[str] | None = None, **kwargs: Any) -> bool | None
```

Async delete by vector ID or other criteria.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `ids` | List of IDs to delete. If `None`, delete all.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Other keyword arguments that subclasses might use.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `bool | None` | `True` if deletion is successful, `False` otherwise, `None` if not<br>implemented. |

### ``aadd\_texts`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.aadd_texts "Copy anchor link to this section for reference")

```
aadd_texts(
    texts: Iterable[str],
    metadatas: list[dict] | None = None,
    *,
    ids: list[str] | None = None,
    **kwargs: Any,
) -> list[str]
```

Async run more texts through the embeddings and add to the `VectorStore`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | Iterable of strings to add to the `VectorStore`.<br>**TYPE:**`Iterable[str]` |
| `metadatas` | Optional list of metadatas associated with the texts.<br>**TYPE:**`list[dict] | None`**DEFAULT:**`None` |
| `ids` | Optional list<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | `VectorStore` specific parameters.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | List of IDs from adding the texts into the `VectorStore`. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the number of metadatas does not match the number of texts. |
| `ValueError` | If the number of IDs does not match the number of texts. |

### ``add\_documents [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.add_documents "Copy anchor link to this section for reference")

```
add_documents(documents: list[Document], **kwargs: Any) -> list[str]
```

Add or update documents in the `VectorStore`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | Documents to add to the `VectorStore`.<br>**TYPE:**`list[Document]` |
| `**kwargs` | Additional keyword arguments.<br>If kwargs contains IDs and documents contain ids, the IDs in the kwargs<br>will receive precedence.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | List of IDs of the added texts. |

### ``aadd\_documents`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.aadd_documents "Copy anchor link to this section for reference")

```
aadd_documents(documents: list[Document], **kwargs: Any) -> list[str]
```

Async run more documents through the embeddings and add to the `VectorStore`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | Documents to add to the `VectorStore`.<br>**TYPE:**`list[Document]` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | List of IDs of the added texts. |

### ``search [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.search "Copy anchor link to this section for reference")

```
search(query: str, search_type: str, **kwargs: Any) -> list[Document]
```

Return docs most similar to query using a specified search type.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Input text.<br>**TYPE:**`str` |
| `search_type` | Type of search to perform.<br>Can be `'similarity'`, `'mmr'`, or `'similarity_score_threshold'`.<br>**TYPE:**`str` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects most similar to the query. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If `search_type` is not one of `'similarity'`,<br>`'mmr'`, or `'similarity_score_threshold'`. |

### ``asearch`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.asearch "Copy anchor link to this section for reference")

```
asearch(query: str, search_type: str, **kwargs: Any) -> list[Document]
```

Async return docs most similar to query using a specified search type.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Input text.<br>**TYPE:**`str` |
| `search_type` | Type of search to perform.<br>Can be `'similarity'`, `'mmr'`, or `'similarity_score_threshold'`.<br>**TYPE:**`str` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects most similar to the query. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If `search_type` is not one of `'similarity'`,<br>`'mmr'`, or `'similarity_score_threshold'`. |

### ``similarity\_search`abstractmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.similarity_search "Copy anchor link to this section for reference")

```
similarity_search(query: str, k: int = 4, **kwargs: Any) -> list[Document]
```

Return docs most similar to query.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Input text.<br>**TYPE:**`str` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects most similar to the query. |

### ``similarity\_search\_with\_score [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.similarity_search_with_score "Copy anchor link to this section for reference")

```
similarity_search_with_score(*args: Any, **kwargs: Any) -> list[tuple[Document, float]]
```

Run similarity search with distance.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `*args` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`()` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[tuple[Document, float]]` | List of tuples of `(doc, similarity_score)`. |

### ``asimilarity\_search\_with\_score`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.asimilarity_search_with_score "Copy anchor link to this section for reference")

```
asimilarity_search_with_score(
    *args: Any, **kwargs: Any
) -> list[tuple[Document, float]]
```

Async run similarity search with distance.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `*args` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`()` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[tuple[Document, float]]` | List of tuples of `(doc, similarity_score)`. |

### ``similarity\_search\_with\_relevance\_scores [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.similarity_search_with_relevance_scores "Copy anchor link to this section for reference")

```
similarity_search_with_relevance_scores(
    query: str, k: int = 4, **kwargs: Any
) -> list[tuple[Document, float]]
```

Return docs and relevance scores in the range `[0, 1]`.

`0` is dissimilar, `1` is most similar.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Input text.<br>**TYPE:**`str` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `**kwargs` | Kwargs to be passed to similarity search.<br>Should include `score_threshold`, an optional floating point value<br>between `0` to `1` to filter the resulting set of retrieved docs.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[tuple[Document, float]]` | List of tuples of `(doc, similarity_score)`. |

### ``asimilarity\_search\_with\_relevance\_scores`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.asimilarity_search_with_relevance_scores "Copy anchor link to this section for reference")

```
asimilarity_search_with_relevance_scores(
    query: str, k: int = 4, **kwargs: Any
) -> list[tuple[Document, float]]
```

Async return docs and relevance scores in the range `[0, 1]`.

`0` is dissimilar, `1` is most similar.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Input text.<br>**TYPE:**`str` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `**kwargs` | Kwargs to be passed to similarity search.<br>Should include `score_threshold`, an optional floating point value<br>between `0` to `1` to filter the resulting set of retrieved docs.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[tuple[Document, float]]` | List of tuples of `(doc, similarity_score)` |

### ``asimilarity\_search`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.asimilarity_search "Copy anchor link to this section for reference")

```
asimilarity_search(query: str, k: int = 4, **kwargs: Any) -> list[Document]
```

Async return docs most similar to query.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Input text.<br>**TYPE:**`str` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects most similar to the query. |

### ``similarity\_search\_by\_vector [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.similarity_search_by_vector "Copy anchor link to this section for reference")

```
similarity_search_by_vector(
    embedding: list[float], k: int = 4, **kwargs: Any
) -> list[Document]
```

Return docs most similar to embedding vector.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `embedding` | Embedding to look up documents similar to.<br>**TYPE:**`list[float]` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects most similar to the query vector. |

### ``asimilarity\_search\_by\_vector`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.asimilarity_search_by_vector "Copy anchor link to this section for reference")

```
asimilarity_search_by_vector(
    embedding: list[float], k: int = 4, **kwargs: Any
) -> list[Document]
```

Async return docs most similar to embedding vector.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `embedding` | Embedding to look up documents similar to.<br>**TYPE:**`list[float]` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects most similar to the query vector. |

### ``max\_marginal\_relevance\_search [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.max_marginal_relevance_search "Copy anchor link to this section for reference")

```
max_marginal_relevance_search(
    query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any
) -> list[Document]
```

Return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity
among selected documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Text to look up documents similar to.<br>**TYPE:**`str` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `fetch_k` | Number of `Document` objects to fetch to pass to MMR algorithm.<br>**TYPE:**`int`**DEFAULT:**`20` |
| `lambda_mult` | Number between `0` and `1` that determines the degree<br>of diversity among the results with `0` corresponding<br>to maximum diversity and `1` to minimum diversity.<br>**TYPE:**`float`**DEFAULT:**`0.5` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects selected by maximal marginal relevance. |

### ``amax\_marginal\_relevance\_search`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.amax_marginal_relevance_search "Copy anchor link to this section for reference")

```
amax_marginal_relevance_search(
    query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any
) -> list[Document]
```

Async return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity
among selected documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Text to look up documents similar to.<br>**TYPE:**`str` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `fetch_k` | Number of `Document` objects to fetch to pass to MMR algorithm.<br>**TYPE:**`int`**DEFAULT:**`20` |
| `lambda_mult` | Number between `0` and `1` that determines the degree<br>of diversity among the results with `0` corresponding<br>to maximum diversity and `1` to minimum diversity.<br>**TYPE:**`float`**DEFAULT:**`0.5` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects selected by maximal marginal relevance. |

### ``max\_marginal\_relevance\_search\_by\_vector [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.max_marginal_relevance_search_by_vector "Copy anchor link to this section for reference")

```
max_marginal_relevance_search_by_vector(
    embedding: list[float],
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    **kwargs: Any,
) -> list[Document]
```

Return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity
among selected documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `embedding` | Embedding to look up documents similar to.<br>**TYPE:**`list[float]` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `fetch_k` | Number of `Document` objects to fetch to pass to MMR algorithm.<br>**TYPE:**`int`**DEFAULT:**`20` |
| `lambda_mult` | Number between `0` and `1` that determines the degree<br>of diversity among the results with `0` corresponding<br>to maximum diversity and `1` to minimum diversity.<br>**TYPE:**`float`**DEFAULT:**`0.5` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects selected by maximal marginal relevance. |

### ``amax\_marginal\_relevance\_search\_by\_vector`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.amax_marginal_relevance_search_by_vector "Copy anchor link to this section for reference")

```
amax_marginal_relevance_search_by_vector(
    embedding: list[float],
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    **kwargs: Any,
) -> list[Document]
```

Async return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity
among selected documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `embedding` | Embedding to look up documents similar to.<br>**TYPE:**`list[float]` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `fetch_k` | Number of `Document` objects to fetch to pass to MMR algorithm.<br>**TYPE:**`int`**DEFAULT:**`20` |
| `lambda_mult` | Number between `0` and `1` that determines the degree<br>of diversity among the results with `0` corresponding<br>to maximum diversity and `1` to minimum diversity.<br>**TYPE:**`float`**DEFAULT:**`0.5` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects selected by maximal marginal relevance. |

### ``from\_documents`classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.from_documents "Copy anchor link to this section for reference")

```
from_documents(documents: list[Document], embedding: Embeddings, **kwargs: Any) -> Self
```

Return `VectorStore` initialized from documents and embeddings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | List of `Document` objects to add to the `VectorStore`.<br>**TYPE:**`list[Document]` |
| `embedding` | Embedding function to use.<br>**TYPE:**`Embeddings` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | `VectorStore` initialized from documents and embeddings. |

### ``afrom\_documents`async``classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.afrom_documents "Copy anchor link to this section for reference")

```
afrom_documents(
    documents: list[Document], embedding: Embeddings, **kwargs: Any
) -> Self
```

Async return `VectorStore` initialized from documents and embeddings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | List of `Document` objects to add to the `VectorStore`.<br>**TYPE:**`list[Document]` |
| `embedding` | Embedding function to use.<br>**TYPE:**`Embeddings` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | `VectorStore` initialized from documents and embeddings. |

### ``from\_texts`abstractmethod``classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.from_texts "Copy anchor link to this section for reference")

```
from_texts(
    texts: list[str],
    embedding: Embeddings,
    metadatas: list[dict] | None = None,
    *,
    ids: list[str] | None = None,
    **kwargs: Any,
) -> VST
```

Return `VectorStore` initialized from texts and embeddings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | Texts to add to the `VectorStore`.<br>**TYPE:**`list[str]` |
| `embedding` | Embedding function to use.<br>**TYPE:**`Embeddings` |
| `metadatas` | Optional list of metadatas associated with the texts.<br>**TYPE:**`list[dict] | None`**DEFAULT:**`None` |
| `ids` | Optional list of IDs associated with the texts.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `VST` | `VectorStore` initialized from texts and embeddings. |

### ``afrom\_texts`async``classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.afrom_texts "Copy anchor link to this section for reference")

```
afrom_texts(
    texts: list[str],
    embedding: Embeddings,
    metadatas: list[dict] | None = None,
    *,
    ids: list[str] | None = None,
    **kwargs: Any,
) -> Self
```

Async return `VectorStore` initialized from texts and embeddings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | Texts to add to the `VectorStore`.<br>**TYPE:**`list[str]` |
| `embedding` | Embedding function to use.<br>**TYPE:**`Embeddings` |
| `metadatas` | Optional list of metadatas associated with the texts.<br>**TYPE:**`list[dict] | None`**DEFAULT:**`None` |
| `ids` | Optional list of IDs associated with the texts.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | `VectorStore` initialized from texts and embeddings. |

### ``as\_retriever [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStore.as_retriever "Copy anchor link to this section for reference")

```
as_retriever(**kwargs: Any) -> VectorStoreRetriever
```

Return `VectorStoreRetriever` initialized from this `VectorStore`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | Keyword arguments to pass to the search function.<br>Can include:<br>- `search_type`: Defines the type of search that the Retriever should<br>perform. Can be `'similarity'` (default), `'mmr'`, or<br>`'similarity_score_threshold'`.<br>- `search_kwargs`: Keyword arguments to pass to the search function.<br>  <br>Can include things like:<br>  - `k`: Amount of documents to return (Default: `4`)<br>  - `score_threshold`: Minimum relevance threshold<br>     for `similarity_score_threshold`<br>  - `fetch_k`: Amount of documents to pass to MMR algorithm<br>     (Default: `20`)<br>  - `lambda_mult`: Diversity of results returned by MMR;<br>     `1` for minimum diversity and 0 for maximum. (Default: `0.5`)<br>  - `filter`: Filter by document metadata<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `VectorStoreRetriever` | Retriever class for `VectorStore`. |

Examples:

```
# Retrieve more documents with higher diversity
# Useful if your dataset has many similar documents
docsearch.as_retriever(
    search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25}
)

# Fetch more documents for the MMR algorithm to consider
# But only return the top 5
docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50})

# Only retrieve documents that have a relevance score
# Above a certain threshold
docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8},
)

# Only get the single most similar document from the dataset
docsearch.as_retriever(search_kwargs={"k": 1})

# Use a filter to only retrieve documents from a specific paper
docsearch.as_retriever(
    search_kwargs={"filter": {"paper_title": "GPT-4 Technical Report"}}
)
```

## ``VectorStoreRetriever [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever "Copy anchor link to this section for reference")

Bases: `BaseRetriever`

Base Retriever class for VectorStore.

| METHOD | DESCRIPTION |
| --- | --- |
| `validate_search_type` | Validate search type. |
| `add_documents` | Add documents to the `VectorStore`. |
| `aadd_documents` | Async add documents to the `VectorStore`. |
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
| `invoke` | Invoke the retriever to get relevant documents. |
| `ainvoke` | Asynchronously invoke the retriever to get relevant documents. |
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

### ``vectorstore`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.vectorstore "Copy anchor link to this section for reference")

```
vectorstore: VectorStore
```

VectorStore to use for retrieval.

### ``search\_type`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.search_type "Copy anchor link to this section for reference")

```
search_type: str = 'similarity'
```

Type of search to perform.

### ``search\_kwargs`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.search_kwargs "Copy anchor link to this section for reference")

```
search_kwargs: dict = Field(default_factory=dict)
```

Keyword arguments to pass to the search function.

### ``name`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.name "Copy anchor link to this section for reference")

```
name: str | None = None
```

The name of the `Runnable`. Used for debugging and tracing.

### ``InputType`property`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.InputType "Copy anchor link to this section for reference")

```
InputType: type[Input]
```

Input type.

The type of input this `Runnable` accepts specified as a type annotation.

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the input type cannot be inferred. |

### ``OutputType`property`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.OutputType "Copy anchor link to this section for reference")

```
OutputType: type[Output]
```

Output Type.

The type of output this `Runnable` produces specified as a type annotation.

| RAISES | DESCRIPTION |
| --- | --- |
| `TypeError` | If the output type cannot be inferred. |

### ``input\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.input_schema "Copy anchor link to this section for reference")

```
input_schema: type[BaseModel]
```

The type of input this `Runnable` accepts specified as a Pydantic model.

### ``output\_schema`property`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.output_schema "Copy anchor link to this section for reference")

```
output_schema: type[BaseModel]
```

Output schema.

The type of output this `Runnable` produces specified as a Pydantic model.

### ``config\_specs`property`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.config_specs "Copy anchor link to this section for reference")

```
config_specs: list[ConfigurableFieldSpec]
```

List configurable fields for this `Runnable`.

### ``lc\_secrets`property`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.lc_secrets "Copy anchor link to this section for reference")

```
lc_secrets: dict[str, str]
```

A map of constructor argument names to secret ids.

For example, `{"openai_api_key": "OPENAI_API_KEY"}`

### ``lc\_attributes`property`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.lc_attributes "Copy anchor link to this section for reference")

```
lc_attributes: dict
```

List of attribute names that should be included in the serialized kwargs.

These attributes must be accepted by the constructor.

Default is an empty dictionary.

### ``tags`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.tags "Copy anchor link to this section for reference")

```
tags: list[str] | None = None
```

Optional list of tags associated with the retriever.

These tags will be associated with each call to this retriever,
and passed as arguments to the handlers defined in `callbacks`.

You can use these to eg identify a specific instance of a retriever with its
use case.

### ``metadata`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.metadata "Copy anchor link to this section for reference")

```
metadata: dict[str, Any] | None = None
```

Optional metadata associated with the retriever.

This metadata will be associated with each call to this retriever,
and passed as arguments to the handlers defined in `callbacks`.

You can use these to eg identify a specific instance of a retriever with its
use case.

### ``validate\_search\_type`classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.validate_search_type "Copy anchor link to this section for reference")

```
validate_search_type(values: dict) -> Any
```

Validate search type.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `values` | Values to validate.<br>**TYPE:**`dict` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Any` | Validated values. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If `search_type` is not one of the allowed search types. |
| `ValueError` | If `score_threshold` is not specified with a float value(`0~1`) |

### ``add\_documents [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.add_documents "Copy anchor link to this section for reference")

```
add_documents(documents: list[Document], **kwargs: Any) -> list[str]
```

Add documents to the `VectorStore`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | Documents to add to the `VectorStore`.<br>**TYPE:**`list[Document]` |
| `**kwargs` | Other keyword arguments that subclasses might use.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | List of IDs of the added texts. |

### ``aadd\_documents`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.aadd_documents "Copy anchor link to this section for reference")

```
aadd_documents(documents: list[Document], **kwargs: Any) -> list[str]
```

Async add documents to the `VectorStore`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | Documents to add to the `VectorStore`.<br>**TYPE:**`list[Document]` |
| `**kwargs` | Other keyword arguments that subclasses might use.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | List of IDs of the added texts. |

### ``get\_name [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.get_name "Copy anchor link to this section for reference")

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

### ``get\_input\_schema [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.get_input_schema "Copy anchor link to this section for reference")

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

### ``get\_input\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.get_input_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_output\_schema [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.get_output_schema "Copy anchor link to this section for reference")

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

### ``get\_output\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.get_output_jsonschema "Copy anchor link to this section for reference")

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

### ``config\_schema [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.config_schema "Copy anchor link to this section for reference")

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

### ``get\_config\_jsonschema [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.get_config_jsonschema "Copy anchor link to this section for reference")

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

### ``get\_graph [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.get_graph "Copy anchor link to this section for reference")

```
get_graph(config: RunnableConfig | None = None) -> Graph
```

Return a graph representation of this `Runnable`.

### ``get\_prompts [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.get_prompts "Copy anchor link to this section for reference")

```
get_prompts(config: RunnableConfig | None = None) -> list[BasePromptTemplate]
```

Return a list of prompts used by this `Runnable`.

### ``\_\_or\_\_ [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.__or__ "Copy anchor link to this section for reference")

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

### ``\_\_ror\_\_ [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.__ror__ "Copy anchor link to this section for reference")

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

### ``pipe [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.pipe "Copy anchor link to this section for reference")

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

### ``pick [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.pick "Copy anchor link to this section for reference")

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

### ``assign [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.assign "Copy anchor link to this section for reference")

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

### ``invoke [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.invoke "Copy anchor link to this section for reference")

```
invoke(
    input: str, config: RunnableConfig | None = None, **kwargs: Any
) -> list[Document]
```

Invoke the retriever to get relevant documents.

Main entry point for synchronous retriever invocations.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The query string.<br>**TYPE:**`str` |
| `config` | Configuration for the retriever.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional arguments to pass to the retriever.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of relevant documents. |

Examples:

```
retriever.invoke("query")
```

### ``ainvoke`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.ainvoke "Copy anchor link to this section for reference")

```
ainvoke(
    input: str, config: RunnableConfig | None = None, **kwargs: Any
) -> list[Document]
```

Asynchronously invoke the retriever to get relevant documents.

Main entry point for asynchronous retriever invocations.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `input` | The query string.<br>**TYPE:**`str` |
| `config` | Configuration for the retriever.<br>**TYPE:**`RunnableConfig | None`**DEFAULT:**`None` |
| `**kwargs` | Additional arguments to pass to the retriever.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of relevant documents. |

Examples:

```
await retriever.ainvoke("query")
```

### ``batch [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.batch "Copy anchor link to this section for reference")

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

### ``batch\_as\_completed [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.batch_as_completed "Copy anchor link to this section for reference")

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

### ``abatch`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.abatch "Copy anchor link to this section for reference")

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

### ``abatch\_as\_completed`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.abatch_as_completed "Copy anchor link to this section for reference")

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

### ``stream [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.stream "Copy anchor link to this section for reference")

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

### ``astream`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.astream "Copy anchor link to this section for reference")

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

### ``astream\_log`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.astream_log "Copy anchor link to this section for reference")

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

### ``astream\_events`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.astream_events "Copy anchor link to this section for reference")

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

### ``transform [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.transform "Copy anchor link to this section for reference")

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

### ``atransform`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.atransform "Copy anchor link to this section for reference")

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

### ``bind [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.bind "Copy anchor link to this section for reference")

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

### ``with\_config [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.with_config "Copy anchor link to this section for reference")

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

### ``with\_listeners [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.with_listeners "Copy anchor link to this section for reference")

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

### ``with\_alisteners [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.with_alisteners "Copy anchor link to this section for reference")

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

### ``with\_types [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.with_types "Copy anchor link to this section for reference")

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

### ``with\_retry [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.with_retry "Copy anchor link to this section for reference")

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

### ``map [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.map "Copy anchor link to this section for reference")

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

### ``with\_fallbacks [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.with_fallbacks "Copy anchor link to this section for reference")

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

### ``as\_tool [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.as_tool "Copy anchor link to this section for reference")

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

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.__init__ "Copy anchor link to this section for reference")

```
__init__(*args: Any, **kwargs: Any) -> None
```

### ``is\_lc\_serializable`classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.is_lc_serializable "Copy anchor link to this section for reference")

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

### ``get\_lc\_namespace`classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.get_lc_namespace "Copy anchor link to this section for reference")

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

### ``lc\_id`classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.lc_id "Copy anchor link to this section for reference")

```
lc_id() -> list[str]
```

Return a unique identifier for this class for serialization purposes.

The unique identifier is a list of strings that describes the path
to the object.

For example, for the class `langchain.llms.openai.OpenAI`, the id is
`["langchain", "llms", "openai", "OpenAI"]`.

### ``to\_json [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.to_json "Copy anchor link to this section for reference")

```
to_json() -> SerializedConstructor | SerializedNotImplemented
```

Serialize the `Runnable` to JSON.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedConstructor | SerializedNotImplemented` | A JSON-serializable representation of the `Runnable`. |

### ``to\_json\_not\_implemented [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.to_json_not_implemented "Copy anchor link to this section for reference")

```
to_json_not_implemented() -> SerializedNotImplemented
```

Serialize a "not implemented" object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedNotImplemented` | `SerializedNotImplemented`. |

### ``configurable\_fields [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.configurable_fields "Copy anchor link to this section for reference")

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

### ``configurable\_alternatives [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.base.VectorStoreRetriever.configurable_alternatives "Copy anchor link to this section for reference")

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

## ``InMemoryVectorStore [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore "Copy anchor link to this section for reference")

Bases: `VectorStore`

In-memory vector store implementation.

Uses a dictionary, and computes cosine similarity for search using numpy.

Setup

Install `langchain-core`.

```
pip install -U langchain-core
```

Key init args — indexing params:
embedding\_function: Embeddings
Embedding function to use.

Instantiate

```
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vector_store = InMemoryVectorStore(OpenAIEmbeddings())
```

Add Documents

```
from langchain_core.documents import Document

document_1 = Document(id="1", page_content="foo", metadata={"baz": "bar"})
document_2 = Document(id="2", page_content="thud", metadata={"bar": "baz"})
document_3 = Document(id="3", page_content="i will be deleted :(")

documents = [document_1, document_2, document_3]
vector_store.add_documents(documents=documents)
```

Inspect documents

```
top_n = 10
for index, (id, doc) in enumerate(vector_store.store.items()):
    if index < top_n:
        # docs have keys 'id', 'vector', 'text', 'metadata'
        print(f"{id}: {doc['text']}")
    else:
        break
```

Delete Documents

```
vector_store.delete(ids=["3"])
```

Search

```
results = vector_store.similarity_search(query="thud", k=1)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")
```

```
* thud [{'bar': 'baz'}]
```

Search with filter

```
def _filter_function(doc: Document) -> bool:
    return doc.metadata.get("bar") == "baz"

results = vector_store.similarity_search(
    query="thud", k=1, filter=_filter_function
)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")
```

```
* thud [{'bar': 'baz'}]
```

Search with score

```
results = vector_store.similarity_search_with_score(query="qux", k=1)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
```

```
* [SIM=0.832268] foo [{'baz': 'bar'}]
```

Async

```
# add documents
# await vector_store.aadd_documents(documents=documents)

# delete documents
# await vector_store.adelete(ids=["3"])

# search
# results = vector_store.asimilarity_search(query="thud", k=1)

# search with score
results = await vector_store.asimilarity_search_with_score(query="qux", k=1)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
```

```
* [SIM=0.832268] foo [{'baz': 'bar'}]
```

Use as Retriever

```
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
)
retriever.invoke("thud")
```

```
[Document(id='2', metadata={'bar': 'baz'}, page_content='thud')]
```

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Initialize with the given embedding function. |
| `delete` | Delete by vector ID or other criteria. |
| `adelete` | Async delete by vector ID or other criteria. |
| `add_documents` | Add or update documents in the `VectorStore`. |
| `aadd_documents` | Async run more documents through the embeddings and add to the `VectorStore`. |
| `get_by_ids` | Get documents by their ids. |
| `aget_by_ids` | Async get documents by their ids. |
| `similarity_search_with_score_by_vector` | Search for the most similar documents to the given embedding. |
| `similarity_search_with_score` | Run similarity search with distance. |
| `asimilarity_search_with_score` | Async run similarity search with distance. |
| `similarity_search_by_vector` | Return docs most similar to embedding vector. |
| `asimilarity_search_by_vector` | Async return docs most similar to embedding vector. |
| `similarity_search` | Return docs most similar to query. |
| `asimilarity_search` | Async return docs most similar to query. |
| `max_marginal_relevance_search_by_vector` | Return docs selected using the maximal marginal relevance. |
| `max_marginal_relevance_search` | Return docs selected using the maximal marginal relevance. |
| `amax_marginal_relevance_search` | Async return docs selected using the maximal marginal relevance. |
| `from_texts` | Return `VectorStore` initialized from texts and embeddings. |
| `afrom_texts` | Async return `VectorStore` initialized from texts and embeddings. |
| `load` | Load a vector store from a file. |
| `dump` | Dump the vector store to a file. |
| `add_texts` | Run more texts through the embeddings and add to the `VectorStore`. |
| `aadd_texts` | Async run more texts through the embeddings and add to the `VectorStore`. |
| `search` | Return docs most similar to query using a specified search type. |
| `asearch` | Async return docs most similar to query using a specified search type. |
| `similarity_search_with_relevance_scores` | Return docs and relevance scores in the range `[0, 1]`. |
| `asimilarity_search_with_relevance_scores` | Async return docs and relevance scores in the range `[0, 1]`. |
| `amax_marginal_relevance_search_by_vector` | Async return docs selected using the maximal marginal relevance. |
| `from_documents` | Return `VectorStore` initialized from documents and embeddings. |
| `afrom_documents` | Async return `VectorStore` initialized from documents and embeddings. |
| `as_retriever` | Return `VectorStoreRetriever` initialized from this `VectorStore`. |

### ``embeddings`property`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.embeddings "Copy anchor link to this section for reference")

```
embeddings: Embeddings
```

Access the query embedding object if available.

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.__init__ "Copy anchor link to this section for reference")

```
__init__(embedding: Embeddings) -> None
```

Initialize with the given embedding function.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `embedding` | embedding function to use.<br>**TYPE:**`Embeddings` |

### ``delete [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.delete "Copy anchor link to this section for reference")

```
delete(ids: Sequence[str] | None = None, **kwargs: Any) -> None
```

Delete by vector ID or other criteria.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `ids` | List of IDs to delete. If `None`, delete all.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Other keyword arguments that subclasses might use.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `bool | None` | `True` if deletion is successful, `False` otherwise, `None` if not<br>implemented. |

### ``adelete`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.adelete "Copy anchor link to this section for reference")

```
adelete(ids: Sequence[str] | None = None, **kwargs: Any) -> None
```

Async delete by vector ID or other criteria.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `ids` | List of IDs to delete. If `None`, delete all.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Other keyword arguments that subclasses might use.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `bool | None` | `True` if deletion is successful, `False` otherwise, `None` if not<br>implemented. |

### ``add\_documents [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.add_documents "Copy anchor link to this section for reference")

```
add_documents(
    documents: list[Document], ids: list[str] | None = None, **kwargs: Any
) -> list[str]
```

Add or update documents in the `VectorStore`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | Documents to add to the `VectorStore`.<br>**TYPE:**`list[Document]` |
| `**kwargs` | Additional keyword arguments.<br>If kwargs contains IDs and documents contain ids, the IDs in the kwargs<br>will receive precedence.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | List of IDs of the added texts. |

### ``aadd\_documents`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.aadd_documents "Copy anchor link to this section for reference")

```
aadd_documents(
    documents: list[Document], ids: list[str] | None = None, **kwargs: Any
) -> list[str]
```

Async run more documents through the embeddings and add to the `VectorStore`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | Documents to add to the `VectorStore`.<br>**TYPE:**`list[Document]` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | List of IDs of the added texts. |

### ``get\_by\_ids [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.get_by_ids "Copy anchor link to this section for reference")

```
get_by_ids(ids: Sequence[str]) -> list[Document]
```

Get documents by their ids.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `ids` | The IDs of the documents to get.<br>**TYPE:**`Sequence[str]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

### ``aget\_by\_ids`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.aget_by_ids "Copy anchor link to this section for reference")

```
aget_by_ids(ids: Sequence[str]) -> list[Document]
```

Async get documents by their ids.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `ids` | The IDs of the documents to get.<br>**TYPE:**`Sequence[str]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

### ``similarity\_search\_with\_score\_by\_vector [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.similarity_search_with_score_by_vector "Copy anchor link to this section for reference")

```
similarity_search_with_score_by_vector(
    embedding: list[float],
    k: int = 4,
    filter: Callable[[Document], bool] | None = None,
    **_kwargs: Any,
) -> list[tuple[Document, float]]
```

Search for the most similar documents to the given embedding.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `embedding` | The embedding to search for.<br>**TYPE:**`list[float]` |
| `k` | The number of documents to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `filter` | A function to filter the documents.<br>**TYPE:**`Callable[[Document], bool] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[tuple[Document, float]]` | A list of tuples of Document objects and their similarity scores. |

### ``similarity\_search\_with\_score [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.similarity_search_with_score "Copy anchor link to this section for reference")

```
similarity_search_with_score(
    query: str, k: int = 4, **kwargs: Any
) -> list[tuple[Document, float]]
```

Run similarity search with distance.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `*args` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`()` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[tuple[Document, float]]` | List of tuples of `(doc, similarity_score)`. |

### ``asimilarity\_search\_with\_score`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.asimilarity_search_with_score "Copy anchor link to this section for reference")

```
asimilarity_search_with_score(
    query: str, k: int = 4, **kwargs: Any
) -> list[tuple[Document, float]]
```

Async run similarity search with distance.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `*args` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`()` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[tuple[Document, float]]` | List of tuples of `(doc, similarity_score)`. |

### ``similarity\_search\_by\_vector [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.similarity_search_by_vector "Copy anchor link to this section for reference")

```
similarity_search_by_vector(
    embedding: list[float], k: int = 4, **kwargs: Any
) -> list[Document]
```

Return docs most similar to embedding vector.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `embedding` | Embedding to look up documents similar to.<br>**TYPE:**`list[float]` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects most similar to the query vector. |

### ``asimilarity\_search\_by\_vector`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.asimilarity_search_by_vector "Copy anchor link to this section for reference")

```
asimilarity_search_by_vector(
    embedding: list[float], k: int = 4, **kwargs: Any
) -> list[Document]
```

Async return docs most similar to embedding vector.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `embedding` | Embedding to look up documents similar to.<br>**TYPE:**`list[float]` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects most similar to the query vector. |

### ``similarity\_search [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.similarity_search "Copy anchor link to this section for reference")

```
similarity_search(query: str, k: int = 4, **kwargs: Any) -> list[Document]
```

Return docs most similar to query.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Input text.<br>**TYPE:**`str` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects most similar to the query. |

### ``asimilarity\_search`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.asimilarity_search "Copy anchor link to this section for reference")

```
asimilarity_search(query: str, k: int = 4, **kwargs: Any) -> list[Document]
```

Async return docs most similar to query.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Input text.<br>**TYPE:**`str` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects most similar to the query. |

### ``max\_marginal\_relevance\_search\_by\_vector [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.max_marginal_relevance_search_by_vector "Copy anchor link to this section for reference")

```
max_marginal_relevance_search_by_vector(
    embedding: list[float],
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    *,
    filter: Callable[[Document], bool] | None = None,
    **kwargs: Any,
) -> list[Document]
```

Return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity
among selected documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `embedding` | Embedding to look up documents similar to.<br>**TYPE:**`list[float]` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `fetch_k` | Number of `Document` objects to fetch to pass to MMR algorithm.<br>**TYPE:**`int`**DEFAULT:**`20` |
| `lambda_mult` | Number between `0` and `1` that determines the degree<br>of diversity among the results with `0` corresponding<br>to maximum diversity and `1` to minimum diversity.<br>**TYPE:**`float`**DEFAULT:**`0.5` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects selected by maximal marginal relevance. |

### ``max\_marginal\_relevance\_search [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.max_marginal_relevance_search "Copy anchor link to this section for reference")

```
max_marginal_relevance_search(
    query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any
) -> list[Document]
```

Return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity
among selected documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Text to look up documents similar to.<br>**TYPE:**`str` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `fetch_k` | Number of `Document` objects to fetch to pass to MMR algorithm.<br>**TYPE:**`int`**DEFAULT:**`20` |
| `lambda_mult` | Number between `0` and `1` that determines the degree<br>of diversity among the results with `0` corresponding<br>to maximum diversity and `1` to minimum diversity.<br>**TYPE:**`float`**DEFAULT:**`0.5` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects selected by maximal marginal relevance. |

### ``amax\_marginal\_relevance\_search`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.amax_marginal_relevance_search "Copy anchor link to this section for reference")

```
amax_marginal_relevance_search(
    query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any
) -> list[Document]
```

Async return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity
among selected documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Text to look up documents similar to.<br>**TYPE:**`str` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `fetch_k` | Number of `Document` objects to fetch to pass to MMR algorithm.<br>**TYPE:**`int`**DEFAULT:**`20` |
| `lambda_mult` | Number between `0` and `1` that determines the degree<br>of diversity among the results with `0` corresponding<br>to maximum diversity and `1` to minimum diversity.<br>**TYPE:**`float`**DEFAULT:**`0.5` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects selected by maximal marginal relevance. |

### ``from\_texts`classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.from_texts "Copy anchor link to this section for reference")

```
from_texts(
    texts: list[str],
    embedding: Embeddings,
    metadatas: list[dict] | None = None,
    **kwargs: Any,
) -> InMemoryVectorStore
```

Return `VectorStore` initialized from texts and embeddings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | Texts to add to the `VectorStore`.<br>**TYPE:**`list[str]` |
| `embedding` | Embedding function to use.<br>**TYPE:**`Embeddings` |
| `metadatas` | Optional list of metadatas associated with the texts.<br>**TYPE:**`list[dict] | None`**DEFAULT:**`None` |
| `ids` | Optional list of IDs associated with the texts.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `VST` | `VectorStore` initialized from texts and embeddings. |

### ``afrom\_texts`async``classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.afrom_texts "Copy anchor link to this section for reference")

```
afrom_texts(
    texts: list[str],
    embedding: Embeddings,
    metadatas: list[dict] | None = None,
    **kwargs: Any,
) -> InMemoryVectorStore
```

Async return `VectorStore` initialized from texts and embeddings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | Texts to add to the `VectorStore`.<br>**TYPE:**`list[str]` |
| `embedding` | Embedding function to use.<br>**TYPE:**`Embeddings` |
| `metadatas` | Optional list of metadatas associated with the texts.<br>**TYPE:**`list[dict] | None`**DEFAULT:**`None` |
| `ids` | Optional list of IDs associated with the texts.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | `VectorStore` initialized from texts and embeddings. |

### ``load`classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.load "Copy anchor link to this section for reference")

```
load(path: str, embedding: Embeddings, **kwargs: Any) -> InMemoryVectorStore
```

Load a vector store from a file.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `path` | The path to load the vector store from.<br>**TYPE:**`str` |
| `embedding` | The embedding to use.<br>**TYPE:**`Embeddings` |
| `**kwargs` | Additional arguments to pass to the constructor.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `InMemoryVectorStore` | A VectorStore object. |

### ``dump [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.dump "Copy anchor link to this section for reference")

```
dump(path: str) -> None
```

Dump the vector store to a file.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `path` | The path to dump the vector store to.<br>**TYPE:**`str` |

### ``add\_texts [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.add_texts "Copy anchor link to this section for reference")

```
add_texts(
    texts: Iterable[str],
    metadatas: list[dict] | None = None,
    *,
    ids: list[str] | None = None,
    **kwargs: Any,
) -> list[str]
```

Run more texts through the embeddings and add to the `VectorStore`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | Iterable of strings to add to the `VectorStore`.<br>**TYPE:**`Iterable[str]` |
| `metadatas` | Optional list of metadatas associated with the texts.<br>**TYPE:**`list[dict] | None`**DEFAULT:**`None` |
| `ids` | Optional list of IDs associated with the texts.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | `VectorStore` specific parameters.<br>One of the kwargs should be `ids` which is a list of ids<br>associated with the texts.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | List of IDs from adding the texts into the `VectorStore`. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the number of metadatas does not match the number of texts. |
| `ValueError` | If the number of IDs does not match the number of texts. |

### ``aadd\_texts`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.aadd_texts "Copy anchor link to this section for reference")

```
aadd_texts(
    texts: Iterable[str],
    metadatas: list[dict] | None = None,
    *,
    ids: list[str] | None = None,
    **kwargs: Any,
) -> list[str]
```

Async run more texts through the embeddings and add to the `VectorStore`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | Iterable of strings to add to the `VectorStore`.<br>**TYPE:**`Iterable[str]` |
| `metadatas` | Optional list of metadatas associated with the texts.<br>**TYPE:**`list[dict] | None`**DEFAULT:**`None` |
| `ids` | Optional list<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `**kwargs` | `VectorStore` specific parameters.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | List of IDs from adding the texts into the `VectorStore`. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the number of metadatas does not match the number of texts. |
| `ValueError` | If the number of IDs does not match the number of texts. |

### ``search [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.search "Copy anchor link to this section for reference")

```
search(query: str, search_type: str, **kwargs: Any) -> list[Document]
```

Return docs most similar to query using a specified search type.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Input text.<br>**TYPE:**`str` |
| `search_type` | Type of search to perform.<br>Can be `'similarity'`, `'mmr'`, or `'similarity_score_threshold'`.<br>**TYPE:**`str` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects most similar to the query. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If `search_type` is not one of `'similarity'`,<br>`'mmr'`, or `'similarity_score_threshold'`. |

### ``asearch`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.asearch "Copy anchor link to this section for reference")

```
asearch(query: str, search_type: str, **kwargs: Any) -> list[Document]
```

Async return docs most similar to query using a specified search type.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Input text.<br>**TYPE:**`str` |
| `search_type` | Type of search to perform.<br>Can be `'similarity'`, `'mmr'`, or `'similarity_score_threshold'`.<br>**TYPE:**`str` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects most similar to the query. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If `search_type` is not one of `'similarity'`,<br>`'mmr'`, or `'similarity_score_threshold'`. |

### ``similarity\_search\_with\_relevance\_scores [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.similarity_search_with_relevance_scores "Copy anchor link to this section for reference")

```
similarity_search_with_relevance_scores(
    query: str, k: int = 4, **kwargs: Any
) -> list[tuple[Document, float]]
```

Return docs and relevance scores in the range `[0, 1]`.

`0` is dissimilar, `1` is most similar.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Input text.<br>**TYPE:**`str` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `**kwargs` | Kwargs to be passed to similarity search.<br>Should include `score_threshold`, an optional floating point value<br>between `0` to `1` to filter the resulting set of retrieved docs.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[tuple[Document, float]]` | List of tuples of `(doc, similarity_score)`. |

### ``asimilarity\_search\_with\_relevance\_scores`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.asimilarity_search_with_relevance_scores "Copy anchor link to this section for reference")

```
asimilarity_search_with_relevance_scores(
    query: str, k: int = 4, **kwargs: Any
) -> list[tuple[Document, float]]
```

Async return docs and relevance scores in the range `[0, 1]`.

`0` is dissimilar, `1` is most similar.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `query` | Input text.<br>**TYPE:**`str` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `**kwargs` | Kwargs to be passed to similarity search.<br>Should include `score_threshold`, an optional floating point value<br>between `0` to `1` to filter the resulting set of retrieved docs.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[tuple[Document, float]]` | List of tuples of `(doc, similarity_score)` |

### ``amax\_marginal\_relevance\_search\_by\_vector`async`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.amax_marginal_relevance_search_by_vector "Copy anchor link to this section for reference")

```
amax_marginal_relevance_search_by_vector(
    embedding: list[float],
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    **kwargs: Any,
) -> list[Document]
```

Async return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity
among selected documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `embedding` | Embedding to look up documents similar to.<br>**TYPE:**`list[float]` |
| `k` | Number of `Document` objects to return.<br>**TYPE:**`int`**DEFAULT:**`4` |
| `fetch_k` | Number of `Document` objects to fetch to pass to MMR algorithm.<br>**TYPE:**`int`**DEFAULT:**`20` |
| `lambda_mult` | Number between `0` and `1` that determines the degree<br>of diversity among the results with `0` corresponding<br>to maximum diversity and `1` to minimum diversity.<br>**TYPE:**`float`**DEFAULT:**`0.5` |
| `**kwargs` | Arguments to pass to the search method.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects selected by maximal marginal relevance. |

### ``from\_documents`classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.from_documents "Copy anchor link to this section for reference")

```
from_documents(documents: list[Document], embedding: Embeddings, **kwargs: Any) -> Self
```

Return `VectorStore` initialized from documents and embeddings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | List of `Document` objects to add to the `VectorStore`.<br>**TYPE:**`list[Document]` |
| `embedding` | Embedding function to use.<br>**TYPE:**`Embeddings` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | `VectorStore` initialized from documents and embeddings. |

### ``afrom\_documents`async``classmethod`[¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.afrom_documents "Copy anchor link to this section for reference")

```
afrom_documents(
    documents: list[Document], embedding: Embeddings, **kwargs: Any
) -> Self
```

Async return `VectorStore` initialized from documents and embeddings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | List of `Document` objects to add to the `VectorStore`.<br>**TYPE:**`list[Document]` |
| `embedding` | Embedding function to use.<br>**TYPE:**`Embeddings` |
| `**kwargs` | Additional keyword arguments.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | `VectorStore` initialized from documents and embeddings. |

### ``as\_retriever [¶](https://reference.langchain.com/python/langchain_core/vectorstores/\#langchain_core.vectorstores.in_memory.InMemoryVectorStore.as_retriever "Copy anchor link to this section for reference")

```
as_retriever(**kwargs: Any) -> VectorStoreRetriever
```

Return `VectorStoreRetriever` initialized from this `VectorStore`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `**kwargs` | Keyword arguments to pass to the search function.<br>Can include:<br>- `search_type`: Defines the type of search that the Retriever should<br>perform. Can be `'similarity'` (default), `'mmr'`, or<br>`'similarity_score_threshold'`.<br>- `search_kwargs`: Keyword arguments to pass to the search function.<br>  <br>Can include things like:<br>  - `k`: Amount of documents to return (Default: `4`)<br>  - `score_threshold`: Minimum relevance threshold<br>     for `similarity_score_threshold`<br>  - `fetch_k`: Amount of documents to pass to MMR algorithm<br>     (Default: `20`)<br>  - `lambda_mult`: Diversity of results returned by MMR;<br>     `1` for minimum diversity and 0 for maximum. (Default: `0.5`)<br>  - `filter`: Filter by document metadata<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `VectorStoreRetriever` | Retriever class for `VectorStore`. |

Examples:

```
# Retrieve more documents with higher diversity
# Useful if your dataset has many similar documents
docsearch.as_retriever(
    search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25}
)

# Fetch more documents for the MMR algorithm to consider
# But only return the top 5
docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50})

# Only retrieve documents that have a relevance score
# Above a certain threshold
docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8},
)

# Only get the single most similar document from the dataset
docsearch.as_retriever(search_kwargs={"k": 1})

# Use a filter to only retrieve documents from a specific paper
docsearch.as_retriever(
    search_kwargs={"filter": {"paper_title": "GPT-4 Technical Report"}}
)
```

Back to top