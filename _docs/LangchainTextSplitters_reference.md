[Skip to content](https://reference.langchain.com/python/langchain_text_splitters/#langchain-text-splitters)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_text_splitters/index.md "Edit this page")

# `langchain-text-splitters` [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain-text-splitters "Copy anchor link to this section for reference")

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-text-splitters?label=%20)](https://pypi.org/project/langchain-text-splitters/#history)[![PyPI - License](https://img.shields.io/pypi/l/langchain-text-splitters)](https://opensource.org/licenses/MIT)[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-text-splitters)](https://pypistats.org/packages/langchain-text-splitters)

Reference documentation for the [`langchain-text-splitters`](https://pypi.org/project/langchain-text-splitters/) package.

## ``langchain\_text\_splitters [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters "Copy anchor link to this section for reference")

Text Splitters are classes for splitting text.

Note

`MarkdownHeaderTextSplitter` and `HTMLHeaderTextSplitter` do not derive from
`TextSplitter`.

| FUNCTION | DESCRIPTION |
| --- | --- |
| `split_text_on_tokens` | Split incoming text and return chunks using tokenizer. |

### ``Language [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.Language "Copy anchor link to this section for reference")

Bases: `str`, `Enum`

Enum of the programming languages.

### ``TextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TextSplitter "Copy anchor link to this section for reference")

Bases: `BaseDocumentTransformer`, `ABC`

Interface for splitting text into chunks.

| METHOD | DESCRIPTION |
| --- | --- |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `__init__` | Create a new TextSplitter. |
| `split_text` | Split text into multiple components. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_documents` | Split documents. |
| `from_huggingface_tokenizer` | Text splitter that uses Hugging Face tokenizer to count length. |
| `from_tiktoken_encoder` | Text splitter that uses `tiktoken` encoder to count length. |
| `transform_documents` | Transform sequence of documents by splitting them. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TextSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    length_function: Callable[[str], int] = len,
    keep_separator: bool | Literal["start", "end"] = False,
    add_start_index: bool = False,
    strip_whitespace: bool = True,
) -> None
```

Create a new TextSplitter.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `chunk_size` | Maximum size of chunks to return<br>**TYPE:**`int`**DEFAULT:**`4000` |
| `chunk_overlap` | Overlap in characters between chunks<br>**TYPE:**`int`**DEFAULT:**`200` |
| `length_function` | Function that measures the length of given chunks<br>**TYPE:**`Callable[[str], int]`**DEFAULT:**`len` |
| `keep_separator` | Whether to keep the separator and where to place it<br>in each corresponding chunk `(True='start')`<br>**TYPE:**`bool | Literal['start', 'end']`**DEFAULT:**`False` |
| `add_start_index` | If `True`, includes chunk's start index in metadata<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `strip_whitespace` | If `True`, strips whitespace from the start and end of<br>every document<br>**TYPE:**`bool`**DEFAULT:**`True` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If `chunk_size` is less than or equal to 0 |
| `ValueError` | If `chunk_overlap` is less than 0 |
| `ValueError` | If `chunk_overlap` is greater than chunk\_size |

#### ``split\_text`abstractmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[str]
```

Split text into multiple components.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The text to split.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of text chunks. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TextSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TextSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents to split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split documents. |

#### ``from\_huggingface\_tokenizer`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TextSplitter.from_huggingface_tokenizer "Copy anchor link to this section for reference")

```
from_huggingface_tokenizer(
    tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter
```

Text splitter that uses Hugging Face tokenizer to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tokenizer` | The Hugging Face tokenizer to use.<br>**TYPE:**`PreTrainedTokenizerBase` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `TextSplitter` | An instance of `TextSplitter` using the Hugging Face tokenizer for length |
| `TextSplitter` | calculation. |

#### ``from\_tiktoken\_encoder`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TextSplitter.from_tiktoken_encoder "Copy anchor link to this section for reference")

```
from_tiktoken_encoder(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self
```

Text splitter that uses `tiktoken` encoder to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | An instance of `TextSplitter` using tiktoken for length calculation. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TextSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The sequence of documents to split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A list of split documents. |

### ``Tokenizer`dataclass`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.Tokenizer "Copy anchor link to this section for reference")

Tokenizer data class.

#### ``chunk\_overlap`instance-attribute`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.Tokenizer.chunk_overlap "Copy anchor link to this section for reference")

```
chunk_overlap: int
```

Overlap in tokens between chunks

#### ``tokens\_per\_chunk`instance-attribute`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.Tokenizer.tokens_per_chunk "Copy anchor link to this section for reference")

```
tokens_per_chunk: int
```

Maximum number of tokens per chunk

#### ``decode`instance-attribute`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.Tokenizer.decode "Copy anchor link to this section for reference")

```
decode: Callable[[list[int]], str]
```

Function to decode a list of token IDs to a string

#### ``encode`instance-attribute`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.Tokenizer.encode "Copy anchor link to this section for reference")

```
encode: Callable[[str], list[int]]
```

Function to encode a string to a list of token IDs

### ``TokenTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TokenTextSplitter "Copy anchor link to this section for reference")

Bases: `TextSplitter`

Splitting text to tokens using model tokenizer.

| METHOD | DESCRIPTION |
| --- | --- |
| `transform_documents` | Transform sequence of documents by splitting them. |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_documents` | Split documents. |
| `from_huggingface_tokenizer` | Text splitter that uses Hugging Face tokenizer to count length. |
| `from_tiktoken_encoder` | Text splitter that uses `tiktoken` encoder to count length. |
| `__init__` | Create a new `TextSplitter`. |
| `split_text` | Splits the input text into smaller chunks based on tokenization. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TokenTextSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The sequence of documents to split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A list of split documents. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TokenTextSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TokenTextSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TokenTextSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents to split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split documents. |

#### ``from\_huggingface\_tokenizer`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TokenTextSplitter.from_huggingface_tokenizer "Copy anchor link to this section for reference")

```
from_huggingface_tokenizer(
    tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter
```

Text splitter that uses Hugging Face tokenizer to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tokenizer` | The Hugging Face tokenizer to use.<br>**TYPE:**`PreTrainedTokenizerBase` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `TextSplitter` | An instance of `TextSplitter` using the Hugging Face tokenizer for length |
| `TextSplitter` | calculation. |

#### ``from\_tiktoken\_encoder`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TokenTextSplitter.from_tiktoken_encoder "Copy anchor link to this section for reference")

```
from_tiktoken_encoder(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self
```

Text splitter that uses `tiktoken` encoder to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | An instance of `TextSplitter` using tiktoken for length calculation. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TokenTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> None
```

Create a new `TextSplitter`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.TokenTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[str]
```

Splits the input text into smaller chunks based on tokenization.

This method uses a custom tokenizer configuration to encode the input text
into tokens, processes the tokens in chunks of a specified size with overlap,
and decodes them back into text chunks. The splitting is performed using the
`split_text_on_tokens` function.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The input text to be split into smaller chunks.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of text chunks, where each chunk is derived from a portion<br>of the input text based on the tokenization and chunking rules. |

### ``CharacterTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.CharacterTextSplitter "Copy anchor link to this section for reference")

Bases: `TextSplitter`

Splitting text that looks at characters.

| METHOD | DESCRIPTION |
| --- | --- |
| `transform_documents` | Transform sequence of documents by splitting them. |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_documents` | Split documents. |
| `from_huggingface_tokenizer` | Text splitter that uses Hugging Face tokenizer to count length. |
| `from_tiktoken_encoder` | Text splitter that uses `tiktoken` encoder to count length. |
| `__init__` | Create a new TextSplitter. |
| `split_text` | Split into chunks without re-inserting lookaround separators. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.CharacterTextSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The sequence of documents to split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A list of split documents. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.CharacterTextSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.CharacterTextSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.CharacterTextSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents to split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split documents. |

#### ``from\_huggingface\_tokenizer`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.CharacterTextSplitter.from_huggingface_tokenizer "Copy anchor link to this section for reference")

```
from_huggingface_tokenizer(
    tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter
```

Text splitter that uses Hugging Face tokenizer to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tokenizer` | The Hugging Face tokenizer to use.<br>**TYPE:**`PreTrainedTokenizerBase` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `TextSplitter` | An instance of `TextSplitter` using the Hugging Face tokenizer for length |
| `TextSplitter` | calculation. |

#### ``from\_tiktoken\_encoder`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.CharacterTextSplitter.from_tiktoken_encoder "Copy anchor link to this section for reference")

```
from_tiktoken_encoder(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self
```

Text splitter that uses `tiktoken` encoder to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | An instance of `TextSplitter` using tiktoken for length calculation. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.CharacterTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    separator: str = "\n\n", is_separator_regex: bool = False, **kwargs: Any
) -> None
```

Create a new TextSplitter.

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.CharacterTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[str]
```

Split into chunks without re-inserting lookaround separators.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The text to split.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of text chunks. |

### ``RecursiveCharacterTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveCharacterTextSplitter "Copy anchor link to this section for reference")

Bases: `TextSplitter`

Splitting text by recursively look at characters.

Recursively tries to split by different characters to find one
that works.

| METHOD | DESCRIPTION |
| --- | --- |
| `transform_documents` | Transform sequence of documents by splitting them. |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_documents` | Split documents. |
| `from_huggingface_tokenizer` | Text splitter that uses Hugging Face tokenizer to count length. |
| `from_tiktoken_encoder` | Text splitter that uses `tiktoken` encoder to count length. |
| `__init__` | Create a new TextSplitter. |
| `split_text` | Split the input text into smaller chunks based on predefined separators. |
| `from_language` | Return an instance of this class based on a specific language. |
| `get_separators_for_language` | Retrieve a list of separators specific to the given language. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveCharacterTextSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The sequence of documents to split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A list of split documents. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveCharacterTextSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveCharacterTextSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveCharacterTextSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents to split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split documents. |

#### ``from\_huggingface\_tokenizer`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveCharacterTextSplitter.from_huggingface_tokenizer "Copy anchor link to this section for reference")

```
from_huggingface_tokenizer(
    tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter
```

Text splitter that uses Hugging Face tokenizer to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tokenizer` | The Hugging Face tokenizer to use.<br>**TYPE:**`PreTrainedTokenizerBase` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `TextSplitter` | An instance of `TextSplitter` using the Hugging Face tokenizer for length |
| `TextSplitter` | calculation. |

#### ``from\_tiktoken\_encoder`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveCharacterTextSplitter.from_tiktoken_encoder "Copy anchor link to this section for reference")

```
from_tiktoken_encoder(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self
```

Text splitter that uses `tiktoken` encoder to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | An instance of `TextSplitter` using tiktoken for length calculation. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveCharacterTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    separators: list[str] | None = None,
    keep_separator: bool | Literal["start", "end"] = True,
    is_separator_regex: bool = False,
    **kwargs: Any,
) -> None
```

Create a new TextSplitter.

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveCharacterTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[str]
```

Split the input text into smaller chunks based on predefined separators.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The input text to be split.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of text chunks obtained after splitting. |

#### ``from\_language`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveCharacterTextSplitter.from_language "Copy anchor link to this section for reference")

```
from_language(language: Language, **kwargs: Any) -> RecursiveCharacterTextSplitter
```

Return an instance of this class based on a specific language.

This method initializes the text splitter with language-specific separators.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `language` | The language to configure the text splitter for.<br>**TYPE:**`Language` |
| `**kwargs` | Additional keyword arguments to customize the splitter.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RecursiveCharacterTextSplitter` | An instance of the text splitter configured for the specified language. |

#### ``get\_separators\_for\_language`staticmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveCharacterTextSplitter.get_separators_for_language "Copy anchor link to this section for reference")

```
get_separators_for_language(language: Language) -> list[str]
```

Retrieve a list of separators specific to the given language.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `language` | The language for which to get the separators.<br>**TYPE:**`Language` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of separators appropriate for the specified language. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the language is not implemented or supported. |

### ``ElementType [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.ElementType "Copy anchor link to this section for reference")

Bases: `TypedDict`

Element type as typed dict.

### ``HTMLHeaderTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLHeaderTextSplitter "Copy anchor link to this section for reference")

Split HTML content into structured Documents based on specified headers.

Splits HTML content by detecting specified header tags and creating hierarchical
`Document` objects that reflect the semantic structure of the original content. For
each identified section, the splitter associates the extracted text with metadata
corresponding to the encountered headers.

If no specified headers are found, the entire content is returned as a single
`Document`. This allows for flexible handling of HTML input, ensuring that
information is organized according to its semantic headers.

The splitter provides the option to return each HTML element as a separate
`Document` or aggregate them into semantically meaningful chunks. It also
gracefully handles multiple levels of nested headers, creating a rich,
hierarchical representation of the content.

Example

```
from langchain_text_splitters.html_header_text_splitter import (
    HTMLHeaderTextSplitter,
)

# Define headers for splitting on h1 and h2 tags.
headers_to_split_on = [("h1", "Main Topic"), ("h2", "Sub Topic")]

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_element=False
)

html_content = """
<html>
    <body>
        <h1>Introduction</h1>
        <p>Welcome to the introduction section.</p>
        <h2>Background</h2>
        <p>Some background details here.</p>
        <h1>Conclusion</h1>
        <p>Final thoughts.</p>
    </body>
</html>
"""

documents = splitter.split_text(html_content)

# 'documents' now contains Document objects reflecting the hierarchy:
# - Document with metadata={"Main Topic": "Introduction"} and
#   content="Introduction"
# - Document with metadata={"Main Topic": "Introduction"} and
#   content="Welcome to the introduction section."
# - Document with metadata={"Main Topic": "Introduction",
#   "Sub Topic": "Background"} and content="Background"
# - Document with metadata={"Main Topic": "Introduction",
#   "Sub Topic": "Background"} and content="Some background details here."
# - Document with metadata={"Main Topic": "Conclusion"} and
#   content="Conclusion"
# - Document with metadata={"Main Topic": "Conclusion"} and
#   content="Final thoughts."
```

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Initialize with headers to split on. |
| `split_text` | Split the given text into a list of `Document` objects. |
| `split_text_from_url` | Fetch text content from a URL and split it into documents. |
| `split_text_from_file` | Split HTML content from a file into a list of `Document` objects. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLHeaderTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    headers_to_split_on: list[tuple[str, str]], return_each_element: bool = False
) -> None
```

Initialize with headers to split on.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `headers_to_split_on` | A list of `(header_tag,<br>header_name)` pairs representing the headers that define splitting<br>boundaries. For example, `[("h1", "Header 1"), ("h2", "Header 2")]`<br>will split content by `h1` and `h2` tags, assigning their textual<br>content to the `Document` metadata.<br>**TYPE:**`list[tuple[str, str]]` |
| `return_each_element` | If `True`, every HTML element encountered<br>(including headers, paragraphs, etc.) is returned as a separate<br>`Document`. If `False`, content under the same header hierarchy is<br>aggregated into fewer `Document` objects.<br>**TYPE:**`bool`**DEFAULT:**`False` |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLHeaderTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[Document]
```

Split the given text into a list of `Document` objects.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The HTML text to split.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split Document objects. Each `Document` contains<br>`page_content` holding the extracted text and `metadata` that maps<br>the header hierarchy to their corresponding titles. |

#### ``split\_text\_from\_url [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLHeaderTextSplitter.split_text_from_url "Copy anchor link to this section for reference")

```
split_text_from_url(url: str, timeout: int = 10, **kwargs: Any) -> list[Document]
```

Fetch text content from a URL and split it into documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `url` | The URL to fetch content from.<br>**TYPE:**`str` |
| `timeout` | Timeout for the request.<br>**TYPE:**`int`**DEFAULT:**`10` |
| `**kwargs` | Additional keyword arguments for the request.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split Document objects. Each `Document` contains<br>`page_content` holding the extracted text and `metadata` that maps<br>the header hierarchy to their corresponding titles. |

| RAISES | DESCRIPTION |
| --- | --- |
| `RequestException` | If the HTTP request fails. |

#### ``split\_text\_from\_file [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLHeaderTextSplitter.split_text_from_file "Copy anchor link to this section for reference")

```
split_text_from_file(file: str | IO[str]) -> list[Document]
```

Split HTML content from a file into a list of `Document` objects.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `file` | A file path or a file-like object containing HTML content.<br>**TYPE:**`str | IO[str]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split Document objects. Each `Document` contains<br>`page_content` holding the extracted text and `metadata` that maps<br>the header hierarchy to their corresponding titles. |

### ``HTMLSectionSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSectionSplitter "Copy anchor link to this section for reference")

Splitting HTML files based on specified tag and font sizes.

Requires `lxml` package.

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Create a new `HTMLSectionSplitter`. |
| `split_documents` | Split documents. |
| `split_text` | Split HTML text string. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_html_by_headers` | Split an HTML document into sections based on specified header tags. |
| `convert_possible_tags_to_header` | Convert specific HTML tags to headers using an XSLT transformation. |
| `split_text_from_file` | Split HTML content from a file into a list of `Document` objects. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSectionSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(headers_to_split_on: list[tuple[str, str]], **kwargs: Any) -> None
```

Create a new `HTMLSectionSplitter`.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `headers_to_split_on` | list of tuples of headers we want to track mapped to<br>(arbitrary) keys for metadata. Allowed header values: `h1`, `h2`, `h3`,<br>`h4`, `h5`, `h6` e.g. `[("h1", "Header 1"), ("h2", "Header 2"]`.<br>**TYPE:**`list[tuple[str, str]]` |
| `**kwargs` | Additional optional arguments for customizations.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSectionSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | Iterable of `Document` objects to be split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split `Document` objects. |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSectionSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[Document]
```

Split HTML text string.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | HTML text<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split `Document` objects. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSectionSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_html\_by\_headers [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSectionSplitter.split_html_by_headers "Copy anchor link to this section for reference")

```
split_html_by_headers(html_doc: str) -> list[dict[str, str | None]]
```

Split an HTML document into sections based on specified header tags.

This method uses BeautifulSoup to parse the HTML content and divides it into
sections based on headers defined in `headers_to_split_on`. Each section
contains the header text, content under the header, and the tag name.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `html_doc` | The HTML document to be split into sections.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[dict[str, str | None]]` | A list of dictionaries representing sections. Each dictionary contains:<br>- `'header'`: The header text or a default title for the first section.<br>- `'content'`: The content under the header.<br>- `'tag_name'`: The name of the header tag (e.g., `h1`, `h2`). |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If BeautifulSoup is not installed. |

#### ``convert\_possible\_tags\_to\_header [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSectionSplitter.convert_possible_tags_to_header "Copy anchor link to this section for reference")

```
convert_possible_tags_to_header(html_content: str) -> str
```

Convert specific HTML tags to headers using an XSLT transformation.

This method uses an XSLT file to transform the HTML content, converting
certain tags into headers for easier parsing. If no XSLT path is provided,
the HTML content is returned unchanged.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `html_content` | The HTML content to be transformed.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `str` | The transformed HTML content as a string. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the `lxml` library is not installed. |

#### ``split\_text\_from\_file [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSectionSplitter.split_text_from_file "Copy anchor link to this section for reference")

```
split_text_from_file(file: StringIO) -> list[Document]
```

Split HTML content from a file into a list of `Document` objects.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `file` | A file path or a file-like object containing HTML content.<br>**TYPE:**`StringIO` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split Document objects. |

### ``HTMLSemanticPreservingSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSemanticPreservingSplitter "Copy anchor link to this section for reference")

Bases: `BaseDocumentTransformer`

Split HTML content preserving semantic structure.

Splits HTML content by headers into generalized chunks, preserving semantic
structure. If chunks exceed the maximum chunk size, it uses
RecursiveCharacterTextSplitter for further splitting.

The splitter preserves full HTML elements and converts links to Markdown-like links.
It can also preserve images, videos, and audio elements by converting them into
Markdown format. Note that some chunks may exceed the maximum size to maintain
semantic integrity.

Added in `langchain-text-splitters` 0.3.5

Example

````
from langchain_text_splitters.html import HTMLSemanticPreservingSplitter

def custom_iframe_extractor(iframe_tag):
    ```
    Custom handler function to extract the 'src' attribute from an <iframe> tag.
    Converts the iframe to a Markdown-like link: [iframe:<src>](src).

    Args:
        iframe_tag (bs4.element.Tag): The <iframe> tag to be processed.

    Returns:
        str: A formatted string representing the iframe in Markdown-like format.
    ```
    iframe_src = iframe_tag.get('src', '')
    return f"[iframe:{iframe_src}]({iframe_src})"

text_splitter = HTMLSemanticPreservingSplitter(
    headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")],
    max_chunk_size=500,
    preserve_links=True,
    preserve_images=True,
    custom_handlers={"iframe": custom_iframe_extractor}
)
````

| METHOD | DESCRIPTION |
| --- | --- |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `__init__` | Initialize splitter. |
| `split_text` | Splits the provided HTML text into smaller chunks based on the configuration. |
| `transform_documents` | Transform sequence of documents by splitting them. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSemanticPreservingSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSemanticPreservingSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    headers_to_split_on: list[tuple[str, str]],
    *,
    max_chunk_size: int = 1000,
    chunk_overlap: int = 0,
    separators: list[str] | None = None,
    elements_to_preserve: list[str] | None = None,
    preserve_links: bool = False,
    preserve_images: bool = False,
    preserve_videos: bool = False,
    preserve_audio: bool = False,
    custom_handlers: dict[str, Callable[[Tag], str]] | None = None,
    stopword_removal: bool = False,
    stopword_lang: str = "english",
    normalize_text: bool = False,
    external_metadata: dict[str, str] | None = None,
    allowlist_tags: list[str] | None = None,
    denylist_tags: list[str] | None = None,
    preserve_parent_metadata: bool = False,
    keep_separator: bool | Literal["start", "end"] = True,
) -> None
```

Initialize splitter.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `headers_to_split_on` | HTML headers (e.g., `h1`, `h2`)<br>that define content sections.<br>**TYPE:**`list[tuple[str, str]]` |
| `max_chunk_size` | Maximum size for each chunk, with allowance for<br>exceeding this limit to preserve semantics.<br>**TYPE:**`int`**DEFAULT:**`1000` |
| `chunk_overlap` | Number of characters to overlap between chunks to ensure<br>contextual continuity.<br>**TYPE:**`int`**DEFAULT:**`0` |
| `separators` | Delimiters used by `RecursiveCharacterTextSplitter` for<br>further splitting.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `elements_to_preserve` | HTML tags (e.g., `table`, `ul`) to remain<br>intact during splitting.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `preserve_links` | Converts `a` tags to Markdown links (`[text](url)`).<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `preserve_images` | Converts `img` tags to Markdown images (`![alt](src)`).<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `preserve_videos` | Converts `video` tags to Markdown<br>video links (`![video](src)`).<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `preserve_audio` | Converts `audio` tags to Markdown<br>audio links (`![audio](src)`).<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `custom_handlers` | Optional custom handlers for<br>specific HTML tags, allowing tailored extraction or processing.<br>**TYPE:**`dict[str, Callable[[Tag], str]] | None`**DEFAULT:**`None` |
| `stopword_removal` | Optionally remove stopwords from the text.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `stopword_lang` | The language of stopwords to remove.<br>**TYPE:**`str`**DEFAULT:**`'english'` |
| `normalize_text` | Optionally normalize text<br>(e.g., lowercasing, removing punctuation).<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `external_metadata` | Additional metadata to attach to<br>the Document objects.<br>**TYPE:**`dict[str, str] | None`**DEFAULT:**`None` |
| `allowlist_tags` | Only these tags will be retained in<br>the HTML.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `denylist_tags` | These tags will be removed from the HTML.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `preserve_parent_metadata` | Whether to pass through parent document<br>metadata to split documents when calling<br>`transform_documents/atransform_documents()`.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `keep_separator` | Whether separators<br>should be at the beginning of a chunk, at the end, or not at all.<br>**TYPE:**`bool | Literal['start', 'end']`**DEFAULT:**`True` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If BeautifulSoup or NLTK (when stopword removal is enabled)<br>is not installed. |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSemanticPreservingSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[Document]
```

Splits the provided HTML text into smaller chunks based on the configuration.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The HTML content to be split.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects containing the split content. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HTMLSemanticPreservingSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> list[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A sequence of split `Document` objects. |

### ``RecursiveJsonSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveJsonSplitter "Copy anchor link to this section for reference")

Splits JSON data into smaller, structured chunks while preserving hierarchy.

This class provides methods to split JSON data into smaller dictionaries or
JSON-formatted strings based on configurable maximum and minimum chunk sizes.
It supports nested JSON structures, optionally converts lists into dictionaries
for better chunking, and allows the creation of document objects for further use.

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Initialize the chunk size configuration for text processing. |
| `split_json` | Splits JSON into a list of JSON chunks. |
| `split_text` | Splits JSON into a list of JSON formatted strings. |
| `create_documents` | Create a list of `Document` objects from a list of json objects (`dict`). |

#### ``max\_chunk\_size`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveJsonSplitter.max_chunk_size "Copy anchor link to this section for reference")

```
max_chunk_size: int = max_chunk_size
```

The maximum size for each chunk.

#### ``min\_chunk\_size`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveJsonSplitter.min_chunk_size "Copy anchor link to this section for reference")

```
min_chunk_size: int = (
    min_chunk_size if min_chunk_size is not None else max(max_chunk_size - 200, 50)
)
```

The minimum size for each chunk, derived from `max_chunk_size` if not
explicitly provided.

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveJsonSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(max_chunk_size: int = 2000, min_chunk_size: int | None = None) -> None
```

Initialize the chunk size configuration for text processing.

This constructor sets up the maximum and minimum chunk sizes, ensuring that
the `min_chunk_size` defaults to a value slightly smaller than the
`max_chunk_size` if not explicitly provided.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `max_chunk_size` | The maximum size for a chunk.<br>**TYPE:**`int`**DEFAULT:**`2000` |
| `min_chunk_size` | The minimum size for a chunk. If `None`,<br>defaults to the maximum chunk size minus 200, with a lower bound of 50.<br>**TYPE:**`int | None`**DEFAULT:**`None` |

#### ``split\_json [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveJsonSplitter.split_json "Copy anchor link to this section for reference")

```
split_json(
    json_data: dict[str, Any], convert_lists: bool = False
) -> list[dict[str, Any]]
```

Splits JSON into a list of JSON chunks.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `json_data` | The JSON data to be split.<br>**TYPE:**`dict[str, Any]` |
| `convert_lists` | Whether to convert lists in the JSON to dictionaries<br>before splitting.<br>**TYPE:**`bool`**DEFAULT:**`False` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[dict[str, Any]]` | A list of JSON chunks. |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveJsonSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(
    json_data: dict[str, Any], convert_lists: bool = False, ensure_ascii: bool = True
) -> list[str]
```

Splits JSON into a list of JSON formatted strings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `json_data` | The JSON data to be split.<br>**TYPE:**`dict[str, Any]` |
| `convert_lists` | Whether to convert lists in the JSON to dictionaries<br>before splitting.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `ensure_ascii` | Whether to ensure ASCII encoding in the JSON strings.<br>**TYPE:**`bool`**DEFAULT:**`True` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of JSON formatted strings. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.RecursiveJsonSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[dict[str, Any]],
    convert_lists: bool = False,
    ensure_ascii: bool = True,
    metadatas: list[dict[Any, Any]] | None = None,
) -> list[Document]
```

Create a list of `Document` objects from a list of json objects (`dict`).

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of JSON data to be split and converted into documents.<br>**TYPE:**`list[dict[str, Any]]` |
| `convert_lists` | Whether to convert lists to dictionaries before splitting.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `ensure_ascii` | Whether to ensure ASCII encoding in the JSON strings.<br>**TYPE:**`bool`**DEFAULT:**`True` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

### ``JSFrameworkTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.JSFrameworkTextSplitter "Copy anchor link to this section for reference")

Bases: `RecursiveCharacterTextSplitter`

Text splitter that handles React (JSX), Vue, and Svelte code.

This splitter extends RecursiveCharacterTextSplitter to handle
React (JSX), Vue, and Svelte code by:

1. Detecting and extracting custom component tags from the text
2. Using those tags as additional separators along with standard JS syntax

The splitter combines:

- Custom component tags as separators (e.g. <Component, <div)
- JavaScript syntax elements (function, const, if, etc)
- Standard text splitting on newlines

This allows chunks to break at natural boundaries in
React, Vue, and Svelte component code.

| METHOD | DESCRIPTION |
| --- | --- |
| `transform_documents` | Transform sequence of documents by splitting them. |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_documents` | Split documents. |
| `from_huggingface_tokenizer` | Text splitter that uses Hugging Face tokenizer to count length. |
| `from_tiktoken_encoder` | Text splitter that uses `tiktoken` encoder to count length. |
| `from_language` | Return an instance of this class based on a specific language. |
| `get_separators_for_language` | Retrieve a list of separators specific to the given language. |
| `__init__` | Initialize the JS Framework text splitter. |
| `split_text` | Split text into chunks. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.JSFrameworkTextSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The sequence of documents to split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A list of split documents. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.JSFrameworkTextSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.JSFrameworkTextSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.JSFrameworkTextSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents to split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split documents. |

#### ``from\_huggingface\_tokenizer`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.JSFrameworkTextSplitter.from_huggingface_tokenizer "Copy anchor link to this section for reference")

```
from_huggingface_tokenizer(
    tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter
```

Text splitter that uses Hugging Face tokenizer to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tokenizer` | The Hugging Face tokenizer to use.<br>**TYPE:**`PreTrainedTokenizerBase` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `TextSplitter` | An instance of `TextSplitter` using the Hugging Face tokenizer for length |
| `TextSplitter` | calculation. |

#### ``from\_tiktoken\_encoder`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.JSFrameworkTextSplitter.from_tiktoken_encoder "Copy anchor link to this section for reference")

```
from_tiktoken_encoder(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self
```

Text splitter that uses `tiktoken` encoder to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | An instance of `TextSplitter` using tiktoken for length calculation. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``from\_language`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.JSFrameworkTextSplitter.from_language "Copy anchor link to this section for reference")

```
from_language(language: Language, **kwargs: Any) -> RecursiveCharacterTextSplitter
```

Return an instance of this class based on a specific language.

This method initializes the text splitter with language-specific separators.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `language` | The language to configure the text splitter for.<br>**TYPE:**`Language` |
| `**kwargs` | Additional keyword arguments to customize the splitter.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RecursiveCharacterTextSplitter` | An instance of the text splitter configured for the specified language. |

#### ``get\_separators\_for\_language`staticmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.JSFrameworkTextSplitter.get_separators_for_language "Copy anchor link to this section for reference")

```
get_separators_for_language(language: Language) -> list[str]
```

Retrieve a list of separators specific to the given language.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `language` | The language for which to get the separators.<br>**TYPE:**`Language` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of separators appropriate for the specified language. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the language is not implemented or supported. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.JSFrameworkTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    separators: list[str] | None = None,
    chunk_size: int = 2000,
    chunk_overlap: int = 0,
    **kwargs: Any,
) -> None
```

Initialize the JS Framework text splitter.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `separators` | Optional list of custom separator strings to use<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `chunk_size` | Maximum size of chunks to return<br>**TYPE:**`int`**DEFAULT:**`2000` |
| `chunk_overlap` | Overlap in characters between chunks<br>**TYPE:**`int`**DEFAULT:**`0` |
| `**kwargs` | Additional arguments to pass to parent class<br>**TYPE:**`Any`**DEFAULT:**`{}` |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.JSFrameworkTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[str]
```

Split text into chunks.

This method splits the text into chunks by:

- Extracting unique opening component tags using regex
- Creating separators list with extracted tags and JS separators
- Splitting the text using the separators by calling the parent class method

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | String containing code to split<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | List of text chunks split on component and JS boundaries |

### ``KonlpyTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.KonlpyTextSplitter "Copy anchor link to this section for reference")

Bases: `TextSplitter`

Splitting text using Konlpy package.

It is good for splitting Korean text.

| METHOD | DESCRIPTION |
| --- | --- |
| `transform_documents` | Transform sequence of documents by splitting them. |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_documents` | Split documents. |
| `from_huggingface_tokenizer` | Text splitter that uses Hugging Face tokenizer to count length. |
| `from_tiktoken_encoder` | Text splitter that uses `tiktoken` encoder to count length. |
| `__init__` | Initialize the Konlpy text splitter. |
| `split_text` | Split text into multiple components. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.KonlpyTextSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The sequence of documents to split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A list of split documents. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.KonlpyTextSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.KonlpyTextSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.KonlpyTextSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents to split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split documents. |

#### ``from\_huggingface\_tokenizer`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.KonlpyTextSplitter.from_huggingface_tokenizer "Copy anchor link to this section for reference")

```
from_huggingface_tokenizer(
    tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter
```

Text splitter that uses Hugging Face tokenizer to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tokenizer` | The Hugging Face tokenizer to use.<br>**TYPE:**`PreTrainedTokenizerBase` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `TextSplitter` | An instance of `TextSplitter` using the Hugging Face tokenizer for length |
| `TextSplitter` | calculation. |

#### ``from\_tiktoken\_encoder`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.KonlpyTextSplitter.from_tiktoken_encoder "Copy anchor link to this section for reference")

```
from_tiktoken_encoder(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self
```

Text splitter that uses `tiktoken` encoder to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | An instance of `TextSplitter` using tiktoken for length calculation. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.KonlpyTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(separator: str = '\n\n', **kwargs: Any) -> None
```

Initialize the Konlpy text splitter.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `separator` | The separator to use when combining splits.<br>**TYPE:**`str`**DEFAULT:**`'\n\n'` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If Konlpy is not installed. |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.KonlpyTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[str]
```

Split text into multiple components.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The text to split.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of text chunks. |

### ``LatexTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.LatexTextSplitter "Copy anchor link to this section for reference")

Bases: `RecursiveCharacterTextSplitter`

Attempts to split the text along Latex-formatted layout elements.

| METHOD | DESCRIPTION |
| --- | --- |
| `transform_documents` | Transform sequence of documents by splitting them. |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `split_text` | Split the input text into smaller chunks based on predefined separators. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_documents` | Split documents. |
| `from_huggingface_tokenizer` | Text splitter that uses Hugging Face tokenizer to count length. |
| `from_tiktoken_encoder` | Text splitter that uses `tiktoken` encoder to count length. |
| `from_language` | Return an instance of this class based on a specific language. |
| `get_separators_for_language` | Retrieve a list of separators specific to the given language. |
| `__init__` | Initialize a LatexTextSplitter. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.LatexTextSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The sequence of documents to split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A list of split documents. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.LatexTextSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.LatexTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[str]
```

Split the input text into smaller chunks based on predefined separators.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The input text to be split.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of text chunks obtained after splitting. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.LatexTextSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.LatexTextSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents to split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split documents. |

#### ``from\_huggingface\_tokenizer`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.LatexTextSplitter.from_huggingface_tokenizer "Copy anchor link to this section for reference")

```
from_huggingface_tokenizer(
    tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter
```

Text splitter that uses Hugging Face tokenizer to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tokenizer` | The Hugging Face tokenizer to use.<br>**TYPE:**`PreTrainedTokenizerBase` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `TextSplitter` | An instance of `TextSplitter` using the Hugging Face tokenizer for length |
| `TextSplitter` | calculation. |

#### ``from\_tiktoken\_encoder`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.LatexTextSplitter.from_tiktoken_encoder "Copy anchor link to this section for reference")

```
from_tiktoken_encoder(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self
```

Text splitter that uses `tiktoken` encoder to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | An instance of `TextSplitter` using tiktoken for length calculation. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``from\_language`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.LatexTextSplitter.from_language "Copy anchor link to this section for reference")

```
from_language(language: Language, **kwargs: Any) -> RecursiveCharacterTextSplitter
```

Return an instance of this class based on a specific language.

This method initializes the text splitter with language-specific separators.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `language` | The language to configure the text splitter for.<br>**TYPE:**`Language` |
| `**kwargs` | Additional keyword arguments to customize the splitter.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RecursiveCharacterTextSplitter` | An instance of the text splitter configured for the specified language. |

#### ``get\_separators\_for\_language`staticmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.LatexTextSplitter.get_separators_for_language "Copy anchor link to this section for reference")

```
get_separators_for_language(language: Language) -> list[str]
```

Retrieve a list of separators specific to the given language.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `language` | The language for which to get the separators.<br>**TYPE:**`Language` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of separators appropriate for the specified language. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the language is not implemented or supported. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.LatexTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(**kwargs: Any) -> None
```

Initialize a LatexTextSplitter.

### ``ExperimentalMarkdownSyntaxTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.ExperimentalMarkdownSyntaxTextSplitter "Copy anchor link to this section for reference")

An experimental text splitter for handling Markdown syntax.

This splitter aims to retain the exact whitespace of the original text while
extracting structured metadata, such as headers. It is a re-implementation of the
MarkdownHeaderTextSplitter with notable changes to the approach and
additional features.

Key Features:

- Retains the original whitespace and formatting of the Markdown text.
- Extracts headers, code blocks, and horizontal rules as metadata.
- Splits out code blocks and includes the language in the "Code" metadata key.
- Splits text on horizontal rules (`---`) as well.
- Defaults to sensible splitting behavior, which can be overridden using the
`headers_to_split_on` parameter.

Example:

```
headers_to_split_on = [\
    ("#", "Header 1"),\
    ("##", "Header 2"),\
]
splitter = ExperimentalMarkdownSyntaxTextSplitter(
    headers_to_split_on=headers_to_split_on
)
chunks = splitter.split(text)
for chunk in chunks:
    print(chunk)
```

This class is currently experimental and subject to change based on feedback and
further development.

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Initialize the text splitter with header splitting and formatting options. |
| `split_text` | Split the input text into structured chunks. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.ExperimentalMarkdownSyntaxTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    headers_to_split_on: list[tuple[str, str]] | None = None,
    return_each_line: bool = False,
    strip_headers: bool = True,
) -> None
```

Initialize the text splitter with header splitting and formatting options.

This constructor sets up the required configuration for splitting text into
chunks based on specified headers and formatting preferences.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `headers_to_split_on` | A list of tuples, where each tuple contains a header tag (e.g., "h1")<br>and its corresponding metadata key. If `None`, default headers are used.<br>**TYPE:**`Union[list[tuple[str, str]], None]`**DEFAULT:**`None` |
| `return_each_line` | Whether to return each line as an individual chunk.<br>Defaults to `False`, which aggregates lines into larger chunks.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `strip_headers` | Whether to exclude headers from the resulting chunks.<br>**TYPE:**`bool`**DEFAULT:**`True` |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.ExperimentalMarkdownSyntaxTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[Document]
```

Split the input text into structured chunks.

This method processes the input text line by line, identifying and handling
specific patterns such as headers, code blocks, and horizontal rules to
split it into structured chunks based on headers, code blocks, and
horizontal rules.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The input text to be split into chunks.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects representing the structured |
| `list[Document]` | chunks of the input text. If `return_each_line` is enabled, each line |
| `list[Document]` | is returned as a separate `Document`. |

### ``HeaderType [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.HeaderType "Copy anchor link to this section for reference")

Bases: `TypedDict`

Header type as typed dict.

### ``LineType [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.LineType "Copy anchor link to this section for reference")

Bases: `TypedDict`

Line type as typed dict.

### ``MarkdownHeaderTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownHeaderTextSplitter "Copy anchor link to this section for reference")

Splitting markdown files based on specified headers.

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Create a new MarkdownHeaderTextSplitter. |
| `aggregate_lines_to_chunks` | Combine lines with common metadata into chunks. |
| `split_text` | Split markdown file. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownHeaderTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    headers_to_split_on: list[tuple[str, str]],
    return_each_line: bool = False,
    strip_headers: bool = True,
    custom_header_patterns: dict[str, int] | None = None,
) -> None
```

Create a new MarkdownHeaderTextSplitter.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `headers_to_split_on` | Headers we want to track<br>**TYPE:**`list[tuple[str, str]]` |
| `return_each_line` | Return each line w/ associated headers<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `strip_headers` | Strip split headers from the content of the chunk<br>**TYPE:**`bool`**DEFAULT:**`True` |
| `custom_header_patterns` | Optional dict mapping header patterns to their<br>levels. For example: {" **": 1, "\***": 2} to treat **Header** as<br>level 1 and **_Header_** as level 2 headers.<br>**TYPE:**`dict[str, int] | None`**DEFAULT:**`None` |

#### ``aggregate\_lines\_to\_chunks [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownHeaderTextSplitter.aggregate_lines_to_chunks "Copy anchor link to this section for reference")

```
aggregate_lines_to_chunks(lines: list[LineType]) -> list[Document]
```

Combine lines with common metadata into chunks.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `lines` | Line of text / associated header metadata<br>**TYPE:**`list[LineType]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of Documents with common metadata aggregated. |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownHeaderTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[Document]
```

Split markdown file.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | Markdown file<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | List of `Document` objects. |

### ``MarkdownTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownTextSplitter "Copy anchor link to this section for reference")

Bases: `RecursiveCharacterTextSplitter`

Attempts to split the text along Markdown-formatted headings.

| METHOD | DESCRIPTION |
| --- | --- |
| `transform_documents` | Transform sequence of documents by splitting them. |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `split_text` | Split the input text into smaller chunks based on predefined separators. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_documents` | Split documents. |
| `from_huggingface_tokenizer` | Text splitter that uses Hugging Face tokenizer to count length. |
| `from_tiktoken_encoder` | Text splitter that uses `tiktoken` encoder to count length. |
| `from_language` | Return an instance of this class based on a specific language. |
| `get_separators_for_language` | Retrieve a list of separators specific to the given language. |
| `__init__` | Initialize a MarkdownTextSplitter. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownTextSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The sequence of documents to split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A list of split documents. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownTextSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[str]
```

Split the input text into smaller chunks based on predefined separators.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The input text to be split.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of text chunks obtained after splitting. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownTextSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownTextSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents to split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split documents. |

#### ``from\_huggingface\_tokenizer`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownTextSplitter.from_huggingface_tokenizer "Copy anchor link to this section for reference")

```
from_huggingface_tokenizer(
    tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter
```

Text splitter that uses Hugging Face tokenizer to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tokenizer` | The Hugging Face tokenizer to use.<br>**TYPE:**`PreTrainedTokenizerBase` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `TextSplitter` | An instance of `TextSplitter` using the Hugging Face tokenizer for length |
| `TextSplitter` | calculation. |

#### ``from\_tiktoken\_encoder`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownTextSplitter.from_tiktoken_encoder "Copy anchor link to this section for reference")

```
from_tiktoken_encoder(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self
```

Text splitter that uses `tiktoken` encoder to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | An instance of `TextSplitter` using tiktoken for length calculation. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``from\_language`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownTextSplitter.from_language "Copy anchor link to this section for reference")

```
from_language(language: Language, **kwargs: Any) -> RecursiveCharacterTextSplitter
```

Return an instance of this class based on a specific language.

This method initializes the text splitter with language-specific separators.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `language` | The language to configure the text splitter for.<br>**TYPE:**`Language` |
| `**kwargs` | Additional keyword arguments to customize the splitter.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RecursiveCharacterTextSplitter` | An instance of the text splitter configured for the specified language. |

#### ``get\_separators\_for\_language`staticmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownTextSplitter.get_separators_for_language "Copy anchor link to this section for reference")

```
get_separators_for_language(language: Language) -> list[str]
```

Retrieve a list of separators specific to the given language.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `language` | The language for which to get the separators.<br>**TYPE:**`Language` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of separators appropriate for the specified language. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the language is not implemented or supported. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.MarkdownTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(**kwargs: Any) -> None
```

Initialize a MarkdownTextSplitter.

### ``NLTKTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.NLTKTextSplitter "Copy anchor link to this section for reference")

Bases: `TextSplitter`

Splitting text using NLTK package.

| METHOD | DESCRIPTION |
| --- | --- |
| `transform_documents` | Transform sequence of documents by splitting them. |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_documents` | Split documents. |
| `from_huggingface_tokenizer` | Text splitter that uses Hugging Face tokenizer to count length. |
| `from_tiktoken_encoder` | Text splitter that uses `tiktoken` encoder to count length. |
| `__init__` | Initialize the NLTK splitter. |
| `split_text` | Split text into multiple components. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.NLTKTextSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The sequence of documents to split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A list of split documents. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.NLTKTextSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.NLTKTextSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.NLTKTextSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents to split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split documents. |

#### ``from\_huggingface\_tokenizer`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.NLTKTextSplitter.from_huggingface_tokenizer "Copy anchor link to this section for reference")

```
from_huggingface_tokenizer(
    tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter
```

Text splitter that uses Hugging Face tokenizer to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tokenizer` | The Hugging Face tokenizer to use.<br>**TYPE:**`PreTrainedTokenizerBase` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `TextSplitter` | An instance of `TextSplitter` using the Hugging Face tokenizer for length |
| `TextSplitter` | calculation. |

#### ``from\_tiktoken\_encoder`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.NLTKTextSplitter.from_tiktoken_encoder "Copy anchor link to this section for reference")

```
from_tiktoken_encoder(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self
```

Text splitter that uses `tiktoken` encoder to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | An instance of `TextSplitter` using tiktoken for length calculation. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.NLTKTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    separator: str = "\n\n",
    language: str = "english",
    *,
    use_span_tokenize: bool = False,
    **kwargs: Any,
) -> None
```

Initialize the NLTK splitter.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `separator` | The separator to use when combining splits.<br>**TYPE:**`str`**DEFAULT:**`'\n\n'` |
| `language` | The language to use.<br>**TYPE:**`str`**DEFAULT:**`'english'` |
| `use_span_tokenize` | Whether to use `span_tokenize` instead of<br>`sent_tokenize`.<br>**TYPE:**`bool`**DEFAULT:**`False` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If NLTK is not installed. |
| `ValueError` | If `use_span_tokenize` is `True` and separator is not `''`. |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.NLTKTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[str]
```

Split text into multiple components.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The text to split.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of text chunks. |

### ``PythonCodeTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.PythonCodeTextSplitter "Copy anchor link to this section for reference")

Bases: `RecursiveCharacterTextSplitter`

Attempts to split the text along Python syntax.

| METHOD | DESCRIPTION |
| --- | --- |
| `transform_documents` | Transform sequence of documents by splitting them. |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `split_text` | Split the input text into smaller chunks based on predefined separators. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_documents` | Split documents. |
| `from_huggingface_tokenizer` | Text splitter that uses Hugging Face tokenizer to count length. |
| `from_tiktoken_encoder` | Text splitter that uses `tiktoken` encoder to count length. |
| `from_language` | Return an instance of this class based on a specific language. |
| `get_separators_for_language` | Retrieve a list of separators specific to the given language. |
| `__init__` | Initialize a PythonCodeTextSplitter. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.PythonCodeTextSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The sequence of documents to split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A list of split documents. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.PythonCodeTextSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.PythonCodeTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[str]
```

Split the input text into smaller chunks based on predefined separators.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The input text to be split.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of text chunks obtained after splitting. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.PythonCodeTextSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.PythonCodeTextSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents to split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split documents. |

#### ``from\_huggingface\_tokenizer`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.PythonCodeTextSplitter.from_huggingface_tokenizer "Copy anchor link to this section for reference")

```
from_huggingface_tokenizer(
    tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter
```

Text splitter that uses Hugging Face tokenizer to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tokenizer` | The Hugging Face tokenizer to use.<br>**TYPE:**`PreTrainedTokenizerBase` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `TextSplitter` | An instance of `TextSplitter` using the Hugging Face tokenizer for length |
| `TextSplitter` | calculation. |

#### ``from\_tiktoken\_encoder`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.PythonCodeTextSplitter.from_tiktoken_encoder "Copy anchor link to this section for reference")

```
from_tiktoken_encoder(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self
```

Text splitter that uses `tiktoken` encoder to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | An instance of `TextSplitter` using tiktoken for length calculation. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``from\_language`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.PythonCodeTextSplitter.from_language "Copy anchor link to this section for reference")

```
from_language(language: Language, **kwargs: Any) -> RecursiveCharacterTextSplitter
```

Return an instance of this class based on a specific language.

This method initializes the text splitter with language-specific separators.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `language` | The language to configure the text splitter for.<br>**TYPE:**`Language` |
| `**kwargs` | Additional keyword arguments to customize the splitter.<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `RecursiveCharacterTextSplitter` | An instance of the text splitter configured for the specified language. |

#### ``get\_separators\_for\_language`staticmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.PythonCodeTextSplitter.get_separators_for_language "Copy anchor link to this section for reference")

```
get_separators_for_language(language: Language) -> list[str]
```

Retrieve a list of separators specific to the given language.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `language` | The language for which to get the separators.<br>**TYPE:**`Language` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of separators appropriate for the specified language. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the language is not implemented or supported. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.PythonCodeTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(**kwargs: Any) -> None
```

Initialize a PythonCodeTextSplitter.

### ``SentenceTransformersTokenTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SentenceTransformersTokenTextSplitter "Copy anchor link to this section for reference")

Bases: `TextSplitter`

Splitting text to tokens using sentence model tokenizer.

| METHOD | DESCRIPTION |
| --- | --- |
| `transform_documents` | Transform sequence of documents by splitting them. |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_documents` | Split documents. |
| `from_huggingface_tokenizer` | Text splitter that uses Hugging Face tokenizer to count length. |
| `from_tiktoken_encoder` | Text splitter that uses `tiktoken` encoder to count length. |
| `__init__` | Create a new TextSplitter. |
| `split_text` | Splits the input text into smaller components by splitting text on tokens. |
| `count_tokens` | Counts the number of tokens in the given text. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SentenceTransformersTokenTextSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The sequence of documents to split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A list of split documents. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SentenceTransformersTokenTextSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SentenceTransformersTokenTextSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SentenceTransformersTokenTextSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents to split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split documents. |

#### ``from\_huggingface\_tokenizer`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SentenceTransformersTokenTextSplitter.from_huggingface_tokenizer "Copy anchor link to this section for reference")

```
from_huggingface_tokenizer(
    tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter
```

Text splitter that uses Hugging Face tokenizer to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tokenizer` | The Hugging Face tokenizer to use.<br>**TYPE:**`PreTrainedTokenizerBase` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `TextSplitter` | An instance of `TextSplitter` using the Hugging Face tokenizer for length |
| `TextSplitter` | calculation. |

#### ``from\_tiktoken\_encoder`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SentenceTransformersTokenTextSplitter.from_tiktoken_encoder "Copy anchor link to this section for reference")

```
from_tiktoken_encoder(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self
```

Text splitter that uses `tiktoken` encoder to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | An instance of `TextSplitter` using tiktoken for length calculation. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SentenceTransformersTokenTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    chunk_overlap: int = 50,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    tokens_per_chunk: int | None = None,
    **kwargs: Any,
) -> None
```

Create a new TextSplitter.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `chunk_overlap` | The number of tokens to overlap between chunks.<br>**TYPE:**`int`**DEFAULT:**`50` |
| `model_name` | The name of the sentence transformer model to use.<br>**TYPE:**`str`**DEFAULT:**`'sentence-transformers/all-mpnet-base-v2'` |
| `tokens_per_chunk` | The number of tokens per chunk. If `None`, uses the<br>maximum tokens allowed by the model.<br>**TYPE:**`int | None`**DEFAULT:**`None` |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the sentence\_transformers package is not installed. |

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SentenceTransformersTokenTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[str]
```

Splits the input text into smaller components by splitting text on tokens.

This method encodes the input text using a private `_encode` method, then
strips the start and stop token IDs from the encoded result. It returns the
processed segments as a list of strings.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The input text to be split.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of string components derived from the input text after encoding and |
| `list[str]` | processing. |

#### ``count\_tokens [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SentenceTransformersTokenTextSplitter.count_tokens "Copy anchor link to this section for reference")

```
count_tokens(*, text: str) -> int
```

Counts the number of tokens in the given text.

This method encodes the input text using a private `_encode` method and
calculates the total number of tokens in the encoded result.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The input text for which the token count is calculated.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `int` | The number of tokens in the encoded text.<br>**TYPE:**`int` |

### ``SpacyTextSplitter [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SpacyTextSplitter "Copy anchor link to this section for reference")

Bases: `TextSplitter`

Splitting text using Spacy package.

Per default, Spacy's `en_core_web_sm` model is used and
its default max\_length is 1000000 (it is the length of maximum character
this model takes which can be increased for large files). For a faster, but
potentially less accurate splitting, you can use `pipeline='sentencizer'`.

| METHOD | DESCRIPTION |
| --- | --- |
| `transform_documents` | Transform sequence of documents by splitting them. |
| `atransform_documents` | Asynchronously transform a list of documents. |
| `create_documents` | Create a list of `Document` objects from a list of texts. |
| `split_documents` | Split documents. |
| `from_huggingface_tokenizer` | Text splitter that uses Hugging Face tokenizer to count length. |
| `from_tiktoken_encoder` | Text splitter that uses `tiktoken` encoder to count length. |
| `__init__` | Initialize the spacy text splitter. |
| `split_text` | Split text into multiple components. |

#### ``transform\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SpacyTextSplitter.transform_documents "Copy anchor link to this section for reference")

```
transform_documents(documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]
```

Transform sequence of documents by splitting them.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The sequence of documents to split.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A list of split documents. |

#### ``atransform\_documents`async`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SpacyTextSplitter.atransform_documents "Copy anchor link to this section for reference")

```
atransform_documents(
    documents: Sequence[Document], **kwargs: Any
) -> Sequence[Document]
```

Asynchronously transform a list of documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | A sequence of `Document` objects to be transformed.<br>**TYPE:**`Sequence[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Sequence[Document]` | A sequence of transformed `Document` objects. |

#### ``create\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SpacyTextSplitter.create_documents "Copy anchor link to this section for reference")

```
create_documents(
    texts: list[str], metadatas: list[dict[Any, Any]] | None = None
) -> list[Document]
```

Create a list of `Document` objects from a list of texts.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `texts` | A list of texts to be split and converted into documents.<br>**TYPE:**`list[str]` |
| `metadatas` | Optional list of metadata to associate with each document.<br>**TYPE:**`list[dict[Any, Any]] | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of `Document` objects. |

#### ``split\_documents [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SpacyTextSplitter.split_documents "Copy anchor link to this section for reference")

```
split_documents(documents: Iterable[Document]) -> list[Document]
```

Split documents.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `documents` | The documents to split.<br>**TYPE:**`Iterable[Document]` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[Document]` | A list of split documents. |

#### ``from\_huggingface\_tokenizer`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SpacyTextSplitter.from_huggingface_tokenizer "Copy anchor link to this section for reference")

```
from_huggingface_tokenizer(
    tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> TextSplitter
```

Text splitter that uses Hugging Face tokenizer to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `tokenizer` | The Hugging Face tokenizer to use.<br>**TYPE:**`PreTrainedTokenizerBase` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `TextSplitter` | An instance of `TextSplitter` using the Hugging Face tokenizer for length |
| `TextSplitter` | calculation. |

#### ``from\_tiktoken\_encoder`classmethod`[¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SpacyTextSplitter.from_tiktoken_encoder "Copy anchor link to this section for reference")

```
from_tiktoken_encoder(
    encoding_name: str = "gpt2",
    model_name: str | None = None,
    allowed_special: Literal["all"] | Set[str] = set(),
    disallowed_special: Literal["all"] | Collection[str] = "all",
    **kwargs: Any,
) -> Self
```

Text splitter that uses `tiktoken` encoder to count length.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `encoding_name` | The name of the tiktoken encoding to use.<br>**TYPE:**`str`**DEFAULT:**`'gpt2'` |
| `model_name` | The name of the model to use. If provided, this will<br>override the `encoding_name`.<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `allowed_special` | Special tokens that are allowed during encoding.<br>**TYPE:**`Literal['all'] | Set[str]`**DEFAULT:**`set()` |
| `disallowed_special` | Special tokens that are disallowed during encoding.<br>**TYPE:**`Literal['all'] | Collection[str]`**DEFAULT:**`'all'` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Self` | An instance of `TextSplitter` using tiktoken for length calculation. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ImportError` | If the tiktoken package is not installed. |

#### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SpacyTextSplitter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    separator: str = "\n\n",
    pipeline: str = "en_core_web_sm",
    max_length: int = 1000000,
    *,
    strip_whitespace: bool = True,
    **kwargs: Any,
) -> None
```

Initialize the spacy text splitter.

#### ``split\_text [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.SpacyTextSplitter.split_text "Copy anchor link to this section for reference")

```
split_text(text: str) -> list[str]
```

Split text into multiple components.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The text to split.<br>**TYPE:**`str` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of text chunks. |

### ``split\_text\_on\_tokens [¶](https://reference.langchain.com/python/langchain_text_splitters/\#langchain_text_splitters.split_text_on_tokens "Copy anchor link to this section for reference")

```
split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> list[str]
```

Split incoming text and return chunks using tokenizer.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The input text to be split.<br>**TYPE:**`str` |
| `tokenizer` | The tokenizer to use for splitting.<br>**TYPE:**`Tokenizer` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | A list of text chunks. |

Back to top