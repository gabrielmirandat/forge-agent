[Skip to content](https://reference.langchain.com/python/langchain_core/document_loaders/#langchain_core.document_loaders)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_core/document_loaders.md "Edit this page")

# Document loaders

## ``document\_loaders [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders "Copy anchor link to this section for reference")

Document loaders.

### ``BaseLoader [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.BaseLoader "Copy anchor link to this section for reference")

Bases: `ABC`

Interface for Document Loader.

Implementations should implement the lazy-loading method using generators
to avoid loading all documents into memory at once.

`load` is provided just for user convenience and should not be overridden.

| METHOD | DESCRIPTION |
| --- | --- |
| `load` | Load data into `Document` objects. |
| `aload` | Load data into `Document` objects. |
| `load_and_split` | Load `Document` and split into chunks. Chunks are returned as `Document`. |
| `lazy_load` | A lazy loader for `Document`. |
| `alazy_load` | A lazy loader for `Document`. |

#### ``load [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.BaseLoader.load "Copy anchor link to this section for reference")

```
load() -> list[Document]
```

Load data into `Document` objects.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | The documents. |

#### ``aload`async`[¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.BaseLoader.aload "Copy anchor link to this section for reference")

```
aload() -> list[Document]
```

Load data into `Document` objects.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | The documents. |

#### ``load\_and\_split [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.BaseLoader.load_and_split "Copy anchor link to this section for reference")

```
load_and_split(text_splitter: TextSplitter | None = None) -> list[Document]
```

Load `Document` and split into chunks. Chunks are returned as `Document`.

Danger

Do not override this method. It should be considered to be deprecated!

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text_splitter` | `TextSplitter` instance to use for splitting documents.<br>Defaults to `RecursiveCharacterTextSplitter`.<br>**TYPE:**`TextSplitter | None`**DEFAULT:**`None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If `langchain-text-splitters` is not installed<br>and no `text_splitter` is provided. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document`. |

#### ``lazy\_load [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.BaseLoader.lazy_load "Copy anchor link to this section for reference")

```
lazy_load() -> Iterator[Document]
```

A lazy loader for `Document`.

| YIELDS | DESCRIPTION |
| --- | --- |
| `Document` | The `Document` objects. |

#### ``alazy\_load`async`[¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.BaseLoader.alazy_load "Copy anchor link to this section for reference")

```
alazy_load() -> AsyncIterator[Document]
```

A lazy loader for `Document`.

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[Document]` | The `Document` objects. |

### ``BaseBlobParser [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.BaseBlobParser "Copy anchor link to this section for reference")

Bases: `ABC`

Abstract interface for blob parsers.

A blob parser provides a way to parse raw data stored in a blob into one
or more `Document` objects.

The parser can be composed with blob loaders, making it easy to reuse
a parser independent of how the blob was originally loaded.

| METHOD | DESCRIPTION |
| --- | --- |
| `lazy_parse` | Lazy parsing interface. |
| `parse` | Eagerly parse the blob into a `Document` or list of `Document` objects. |

#### ``lazy\_parse`abstractmethod`[¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.BaseBlobParser.lazy_parse "Copy anchor link to this section for reference")

```
lazy_parse(blob: Blob) -> Iterator[Document]
```

Lazy parsing interface.

Subclasses are required to implement this method.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `blob` | `Blob` instance<br>**TYPE:**`Blob` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Iterator[Document]` | Generator of `Document` objects |

#### ``parse [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.BaseBlobParser.parse "Copy anchor link to this section for reference")

```
parse(blob: Blob) -> list[Document]
```

Eagerly parse the blob into a `Document` or list of `Document` objects.

This is a convenience method for interactive development environment.

Production applications should favor the `lazy_parse` method instead.

Subclasses should generally not over-ride this parse method.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `blob` | `Blob` instance<br>**TYPE:**`Blob` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects |

### ``BlobLoader [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.BlobLoader "Copy anchor link to this section for reference")

Bases: `ABC`

Abstract interface for blob loaders implementation.

Implementer should be able to load raw content from a storage system according
to some criteria and return the raw content lazily as a stream of blobs.

| METHOD | DESCRIPTION |
| --- | --- |
| `yield_blobs` | A lazy loader for raw data represented by LangChain's `Blob` object. |

#### ``yield\_blobs`abstractmethod`[¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.BlobLoader.yield_blobs "Copy anchor link to this section for reference")

```
yield_blobs() -> Iterable[Blob]
```

A lazy loader for raw data represented by LangChain's `Blob` object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `Iterable[Blob]` | A generator over blobs |

### ``LangSmithLoader [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.LangSmithLoader "Copy anchor link to this section for reference")

Bases: `BaseLoader`

Load LangSmith Dataset examples as `Document` objects.

Loads the example inputs as the `Document` page content and places the entire
example into the `Document` metadata. This allows you to easily create few-shot
example retrievers from the loaded documents.

Lazy loading example

```
from langchain_core.document_loaders import LangSmithLoader

loader = LangSmithLoader(dataset_id="...", limit=100)
docs = []
for doc in loader.lazy_load():
    docs.append(doc)
```

```
# -> [Document("...", metadata={"inputs": {...}, "outputs": {...}, ...}), ...]
```

| METHOD | DESCRIPTION |
| --- | --- |
| `load` | Load data into `Document` objects. |
| `aload` | Load data into `Document` objects. |
| `load_and_split` | Load `Document` and split into chunks. Chunks are returned as `Document`. |
| `alazy_load` | A lazy loader for `Document`. |
| `__init__` | Create a LangSmith loader. |
| `lazy_load` | A lazy loader for `Document`. |

#### ``load [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.LangSmithLoader.load "Copy anchor link to this section for reference")

```
load() -> list[Document]
```

Load data into `Document` objects.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | The documents. |

#### ``aload`async`[¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.LangSmithLoader.aload "Copy anchor link to this section for reference")

```
aload() -> list[Document]
```

Load data into `Document` objects.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | The documents. |

#### ``load\_and\_split [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.LangSmithLoader.load_and_split "Copy anchor link to this section for reference")

```
load_and_split(text_splitter: TextSplitter | None = None) -> list[Document]
```

Load `Document` and split into chunks. Chunks are returned as `Document`.

Danger

Do not override this method. It should be considered to be deprecated!

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text_splitter` | `TextSplitter` instance to use for splitting documents.<br>Defaults to `RecursiveCharacterTextSplitter`.<br>**TYPE:**`TextSplitter | None`**DEFAULT:**`None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If `langchain-text-splitters` is not installed<br>and no `text_splitter` is provided. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document`. |

#### ``alazy\_load`async`[¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.LangSmithLoader.alazy_load "Copy anchor link to this section for reference")

```
alazy_load() -> AsyncIterator[Document]
```

A lazy loader for `Document`.

| YIELDS | DESCRIPTION |
| --- | --- |
| `AsyncIterator[Document]` | The `Document` objects. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.LangSmithLoader.__init__ "Copy anchor link to this section for reference")

```
__init__(
    *,
    dataset_id: UUID | str | None = None,
    dataset_name: str | None = None,
    example_ids: Sequence[UUID | str] | None = None,
    as_of: datetime | str | None = None,
    splits: Sequence[str] | None = None,
    inline_s3_urls: bool = True,
    offset: int = 0,
    limit: int | None = None,
    metadata: dict | None = None,
    filter: str | None = None,
    content_key: str = "",
    format_content: Callable[..., str] | None = None,
    client: Client | None = None,
    **client_kwargs: Any,
) -> None
```

Create a LangSmith loader.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `dataset_id` | The ID of the dataset to filter by.<br>**TYPE:**`UUID | str | None`**DEFAULT:**`None` |
| `dataset_name` | The name of the dataset to filter by.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `content_key` | The inputs key to set as Document page content. `'.'` characters<br>are interpreted as nested keys. E.g. `content_key="first.second"` will<br>result in<br>`Document(page_content=format_content(example.inputs["first"]["second"]))`<br>**TYPE:**`str`**DEFAULT:**`''` |
| `format_content` | Function for converting the content extracted from the example<br>inputs into a string. Defaults to JSON-encoding the contents.<br>**TYPE:**`Callable[..., str] | None`**DEFAULT:**`None` |
| `example_ids` | The IDs of the examples to filter by.<br>**TYPE:**`Sequence[UUID | str] | None`**DEFAULT:**`None` |
| `as_of` | The dataset version tag or timestamp to retrieve the examples as of.<br>Response examples will only be those that were present at the time of<br>the tagged (or timestamped) version.<br>**TYPE:**`datetime | str | None`**DEFAULT:**`None` |
| `splits` | A list of dataset splits, which are<br>divisions of your dataset such as `train`, `test`, or `validation`.<br>Returns examples only from the specified splits.<br>**TYPE:**`Sequence[str] | None`**DEFAULT:**`None` |
| `inline_s3_urls` | Whether to inline S3 URLs.<br>**TYPE:**`bool`**DEFAULT:**`True` |
| `offset` | The offset to start from.<br>**TYPE:**`int`**DEFAULT:**`0` |
| `limit` | The maximum number of examples to return.<br>**TYPE:**`int | None`**DEFAULT:**`None` |
| `metadata` | Metadata to filter by.<br>**TYPE:**`dict | None`**DEFAULT:**`None` |
| `filter` | A structured filter string to apply to the examples.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `client` | LangSmith Client. If not provided will be initialized from below args.<br>**TYPE:**`Client | None`**DEFAULT:**`None` |
| `client_kwargs` | Keyword args to pass to LangSmith client init. Should only be<br>specified if `client` isn't.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If both `client` and `client_kwargs` are provided. |

#### ``lazy\_load [¶](https://reference.langchain.com/python/langchain_core/document_loaders/\#langchain_core.document_loaders.LangSmithLoader.lazy_load "Copy anchor link to this section for reference")

```
lazy_load() -> Iterator[Document]
```

A lazy loader for `Document`.

| YIELDS | DESCRIPTION |
| --- | --- |
| `Document` | The `Document` objects. |

Back to top