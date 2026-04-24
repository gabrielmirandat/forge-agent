![Revisit consent button](https://uploads-ssl.webflow.com/65ff950538088944d66126b3/662ef3209b872e92e41212f6_cookieicon.png)

![](https://cdn-cookieyes.com/assets/images/close.svg)

We value your privacy

We use cookies to improve your experience and to understand how our site is used. Some analytics tools may share limited data with our advertising partners. You can opt out at any time.

Do Not Sell or Share My Personal Information

Opt-out Preferences![](https://cdn-cookieyes.com/assets/images/close.svg)

We use cookies to improve your experience and to understand how our site is used. Some analytics tools may share limited data with our advertising partners. You can opt out of this sharing at any time by selecting **“Do Not Sell or Share My Personal Information”** and saving your preferences.

Do Not Sell or Share My Personal Information

CancelSave My Preferences

[Skip to content](https://reference.langchain.com/python/langchain_core/documents/#langchain_core.documents)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_core/documents.md "Edit this page")

# Documents

## ``documents [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents "Copy anchor link to this section for reference")

Documents module for data retrieval and processing workflows.

This module provides core abstractions for handling data in retrieval-augmented
generation (RAG) pipelines, vector stores, and document processing workflows.

Documents vs. message content

This module is distinct from `langchain_core.messages.content`, which provides
multimodal content blocks for **LLM chat I/O** (text, images, audio, etc. within
messages).

**Key distinction:**

- **Documents** (this module): For **data retrieval and processing workflows**
  - Vector stores, retrievers, RAG pipelines
  - Text chunking, embedding, and semantic search
  - Example: Chunks of a PDF stored in a vector database
- **Content Blocks** (`messages.content`): For **LLM conversational I/O**
  - Multimodal message content sent to/from models
  - Tool calls, reasoning, citations within chat
  - Example: An image sent to a vision model in a chat message (via
     [`ImageContentBlock`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ImageContentBlock "<code class=\"doc-symbol doc-symbol-heading doc-symbol-class\"></code>            <span class=\"doc doc-object-name doc-class-name\">ImageContentBlock</span>"))

While both can represent similar data types (text, files), they serve different
architectural purposes in LangChain applications.

## ``Document [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document "Copy anchor link to this section for reference")

Bases: `BaseMedia`

Class for storing a piece of text and associated metadata.

Note

`Document` is for **retrieval workflows**, not chat I/O. For sending text
to an LLM in a conversation, use message types from `langchain.messages`.

Example

```
from langchain_core.documents import Document

document = Document(
    page_content="Hello, world!", metadata={"source": "https://example.com"}
)
```

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | Pass page\_content in as positional or named arg. |
| `is_lc_serializable` | Return `True` as this class is serializable. |
| `get_lc_namespace` | Get the namespace of the LangChain object. |
| `__str__` | Override `__str__` to restrict it to page\_content and metadata. |
| `lc_id` | Return a unique identifier for this class for serialization purposes. |
| `to_json` | Serialize the object to JSON. |
| `to_json_not_implemented` | Serialize a "not implemented" object. |

### ``page\_content`instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document.page_content "Copy anchor link to this section for reference")

```
page_content: str
```

String text.

### ``lc\_secrets`property`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document.lc_secrets "Copy anchor link to this section for reference")

```
lc_secrets: dict[str, str]
```

A map of constructor argument names to secret ids.

For example, `{"openai_api_key": "OPENAI_API_KEY"}`

### ``lc\_attributes`property`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document.lc_attributes "Copy anchor link to this section for reference")

```
lc_attributes: dict
```

List of attribute names that should be included in the serialized kwargs.

These attributes must be accepted by the constructor.

Default is an empty dictionary.

### ``id`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document.id "Copy anchor link to this section for reference")

```
id: str | None = Field(default=None, coerce_numbers_to_str=True)
```

An optional identifier for the document.

Ideally this should be unique across the document collection and formatted
as a UUID, but this will not be enforced.

### ``metadata`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document.metadata "Copy anchor link to this section for reference")

```
metadata: dict = Field(default_factory=dict)
```

Arbitrary metadata associated with the content.

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document.__init__ "Copy anchor link to this section for reference")

```
__init__(page_content: str, **kwargs: Any) -> None
```

Pass page\_content in as positional or named arg.

### ``is\_lc\_serializable`classmethod`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document.is_lc_serializable "Copy anchor link to this section for reference")

```
is_lc_serializable() -> bool
```

Return `True` as this class is serializable.

### ``get\_lc\_namespace`classmethod`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document.get_lc_namespace "Copy anchor link to this section for reference")

```
get_lc_namespace() -> list[str]
```

Get the namespace of the LangChain object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `list[str]` | \["langchain", "schema", "document"\] |

### ``\_\_str\_\_ [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document.__str__ "Copy anchor link to this section for reference")

```
__str__() -> str
```

Override `__str__` to restrict it to page\_content and metadata.

| RETURNS | DESCRIPTION |
| --- | --- |
| `str` | A string representation of the `Document`. |

### ``lc\_id`classmethod`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document.lc_id "Copy anchor link to this section for reference")

```
lc_id() -> list[str]
```

Return a unique identifier for this class for serialization purposes.

The unique identifier is a list of strings that describes the path
to the object.

For example, for the class `langchain.llms.openai.OpenAI`, the id is
`["langchain", "llms", "openai", "OpenAI"]`.

### ``to\_json [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document.to_json "Copy anchor link to this section for reference")

```
to_json() -> SerializedConstructor | SerializedNotImplemented
```

Serialize the object to JSON.

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the class has deprecated attributes. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedConstructor | SerializedNotImplemented` | A JSON serializable object or a `SerializedNotImplemented` object. |

### ``to\_json\_not\_implemented [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Document.to_json_not_implemented "Copy anchor link to this section for reference")

```
to_json_not_implemented() -> SerializedNotImplemented
```

Serialize a "not implemented" object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedNotImplemented` | `SerializedNotImplemented`. |

## ``Blob [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob "Copy anchor link to this section for reference")

Bases: `BaseMedia`

Raw data abstraction for document loading and file processing.

Represents raw bytes or text, either in-memory or by file reference. Used
primarily by document loaders to decouple data loading from parsing.

Inspired by [Mozilla's `Blob`](https://developer.mozilla.org/en-US/docs/Web/API/Blob)

Initialize a blob from in-memory data

```
from langchain_core.documents import Blob

blob = Blob.from_data("Hello, world!")

# Read the blob as a string
print(blob.as_string())

# Read the blob as bytes
print(blob.as_bytes())

# Read the blob as a byte stream
with blob.as_bytes_io() as f:
    print(f.read())
```

Load from memory and specify MIME type and metadata

```
from langchain_core.documents import Blob

blob = Blob.from_data(
    data="Hello, world!",
    mime_type="text/plain",
    metadata={"source": "https://example.com"},
)
```

Load the blob from a file

```
from langchain_core.documents import Blob

blob = Blob.from_path("path/to/file.txt")

# Read the blob as a string
print(blob.as_string())

# Read the blob as bytes
print(blob.as_bytes())

# Read the blob as a byte stream
with blob.as_bytes_io() as f:
    print(f.read())
```

| METHOD | DESCRIPTION |
| --- | --- |
| `check_blob_is_valid` | Verify that either data or path is provided. |
| `as_string` | Read data as a string. |
| `as_bytes` | Read data as bytes. |
| `as_bytes_io` | Read data as a byte stream. |
| `from_path` | Load the blob from a path like object. |
| `from_data` | Initialize the `Blob` from in-memory data. |
| `__repr__` | Return the blob representation. |
| `__init__` |  |
| `is_lc_serializable` | Is this class serializable? |
| `get_lc_namespace` | Get the namespace of the LangChain object. |
| `lc_id` | Return a unique identifier for this class for serialization purposes. |
| `to_json` | Serialize the object to JSON. |
| `to_json_not_implemented` | Serialize a "not implemented" object. |

### ``data`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.data "Copy anchor link to this section for reference")

```
data: bytes | str | None = None
```

Raw data associated with the `Blob`.

### ``mimetype`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.mimetype "Copy anchor link to this section for reference")

```
mimetype: str | None = None
```

MIME type, not to be confused with a file extension.

### ``encoding`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.encoding "Copy anchor link to this section for reference")

```
encoding: str = 'utf-8'
```

Encoding to use if decoding the bytes into a string.

Uses `utf-8` as default encoding if decoding to string.

### ``path`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.path "Copy anchor link to this section for reference")

```
path: PathLike | None = None
```

Location where the original content was found.

### ``source`property`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.source "Copy anchor link to this section for reference")

```
source: str | None
```

The source location of the blob as string if known otherwise none.

If a path is associated with the `Blob`, it will default to the path location.

Unless explicitly set via a metadata field called `'source'`, in which
case that value will be used instead.

### ``lc\_secrets`property`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.lc_secrets "Copy anchor link to this section for reference")

```
lc_secrets: dict[str, str]
```

A map of constructor argument names to secret ids.

For example, `{"openai_api_key": "OPENAI_API_KEY"}`

### ``lc\_attributes`property`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.lc_attributes "Copy anchor link to this section for reference")

```
lc_attributes: dict
```

List of attribute names that should be included in the serialized kwargs.

These attributes must be accepted by the constructor.

Default is an empty dictionary.

### ``id`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.id "Copy anchor link to this section for reference")

```
id: str | None = Field(default=None, coerce_numbers_to_str=True)
```

An optional identifier for the document.

Ideally this should be unique across the document collection and formatted
as a UUID, but this will not be enforced.

### ``metadata`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.metadata "Copy anchor link to this section for reference")

```
metadata: dict = Field(default_factory=dict)
```

Arbitrary metadata associated with the content.

### ``check\_blob\_is\_valid`classmethod`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.check_blob_is_valid "Copy anchor link to this section for reference")

```
check_blob_is_valid(values: dict[str, Any]) -> Any
```

Verify that either data or path is provided.

### ``as\_string [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.as_string "Copy anchor link to this section for reference")

```
as_string() -> str
```

Read data as a string.

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the blob cannot be represented as a string. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `str` | The data as a string. |

### ``as\_bytes [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.as_bytes "Copy anchor link to this section for reference")

```
as_bytes() -> bytes
```

Read data as bytes.

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the blob cannot be represented as bytes. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `bytes` | The data as bytes. |

### ``as\_bytes\_io [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.as_bytes_io "Copy anchor link to this section for reference")

```
as_bytes_io() -> Generator[BytesIO | BufferedReader, None, None]
```

Read data as a byte stream.

| RAISES | DESCRIPTION |
| --- | --- |
| `NotImplementedError` | If the blob cannot be represented as a byte stream. |

| YIELDS | DESCRIPTION |
| --- | --- |
| `BytesIO | BufferedReader` | The data as a byte stream. |

### ``from\_path`classmethod`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.from_path "Copy anchor link to this section for reference")

```
from_path(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    mime_type: str | None = None,
    guess_type: bool = True,
    metadata: dict | None = None,
) -> Blob
```

Load the blob from a path like object.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `path` | Path-like object to file to be read<br>**TYPE:**`PathLike` |
| `encoding` | Encoding to use if decoding the bytes into a string<br>**TYPE:**`str`**DEFAULT:**`'utf-8'` |
| `mime_type` | If provided, will be set as the MIME type of the data<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `guess_type` | If `True`, the MIME type will be guessed from the file<br>extension, if a MIME type was not provided<br>**TYPE:**`bool`**DEFAULT:**`True` |
| `metadata` | Metadata to associate with the `Blob`<br>**TYPE:**`dict | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Blob` | `Blob` instance |

### ``from\_data`classmethod`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.from_data "Copy anchor link to this section for reference")

```
from_data(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    mime_type: str | None = None,
    path: str | None = None,
    metadata: dict | None = None,
) -> Blob
```

Initialize the `Blob` from in-memory data.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `data` | The in-memory data associated with the `Blob`<br>**TYPE:**`str | bytes` |
| `encoding` | Encoding to use if decoding the bytes into a string<br>**TYPE:**`str`**DEFAULT:**`'utf-8'` |
| `mime_type` | If provided, will be set as the MIME type of the data<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `path` | If provided, will be set as the source from which the data came<br>**TYPE:**`str | None`**DEFAULT:**`None` |
| `metadata` | Metadata to associate with the `Blob`<br>**TYPE:**`dict | None`**DEFAULT:**`None` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Blob` | `Blob` instance |

### ``\_\_repr\_\_ [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.__repr__ "Copy anchor link to this section for reference")

```
__repr__() -> str
```

Return the blob representation.

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.__init__ "Copy anchor link to this section for reference")

```
__init__(*args: Any, **kwargs: Any) -> None
```

### ``is\_lc\_serializable`classmethod`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.is_lc_serializable "Copy anchor link to this section for reference")

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

### ``get\_lc\_namespace`classmethod`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.get_lc_namespace "Copy anchor link to this section for reference")

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

### ``lc\_id`classmethod`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.lc_id "Copy anchor link to this section for reference")

```
lc_id() -> list[str]
```

Return a unique identifier for this class for serialization purposes.

The unique identifier is a list of strings that describes the path
to the object.

For example, for the class `langchain.llms.openai.OpenAI`, the id is
`["langchain", "llms", "openai", "OpenAI"]`.

### ``to\_json [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.to_json "Copy anchor link to this section for reference")

```
to_json() -> SerializedConstructor | SerializedNotImplemented
```

Serialize the object to JSON.

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the class has deprecated attributes. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedConstructor | SerializedNotImplemented` | A JSON serializable object or a `SerializedNotImplemented` object. |

### ``to\_json\_not\_implemented [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.Blob.to_json_not_implemented "Copy anchor link to this section for reference")

```
to_json_not_implemented() -> SerializedNotImplemented
```

Serialize a "not implemented" object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedNotImplemented` | `SerializedNotImplemented`. |

## ``BaseMedia [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.BaseMedia "Copy anchor link to this section for reference")

Bases: `Serializable`

Base class for content used in retrieval and data processing workflows.

Provides common fields for content that needs to be stored, indexed, or searched.

Note

For multimodal content in **chat messages** (images, audio sent to/from LLMs),
use `langchain.messages` content blocks instead.

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` |  |
| `is_lc_serializable` | Is this class serializable? |
| `get_lc_namespace` | Get the namespace of the LangChain object. |
| `lc_id` | Return a unique identifier for this class for serialization purposes. |
| `to_json` | Serialize the object to JSON. |
| `to_json_not_implemented` | Serialize a "not implemented" object. |

### ``id`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.BaseMedia.id "Copy anchor link to this section for reference")

```
id: str | None = Field(default=None, coerce_numbers_to_str=True)
```

An optional identifier for the document.

Ideally this should be unique across the document collection and formatted
as a UUID, but this will not be enforced.

### ``metadata`class-attribute``instance-attribute`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.BaseMedia.metadata "Copy anchor link to this section for reference")

```
metadata: dict = Field(default_factory=dict)
```

Arbitrary metadata associated with the content.

### ``lc\_secrets`property`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.BaseMedia.lc_secrets "Copy anchor link to this section for reference")

```
lc_secrets: dict[str, str]
```

A map of constructor argument names to secret ids.

For example, `{"openai_api_key": "OPENAI_API_KEY"}`

### ``lc\_attributes`property`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.BaseMedia.lc_attributes "Copy anchor link to this section for reference")

```
lc_attributes: dict
```

List of attribute names that should be included in the serialized kwargs.

These attributes must be accepted by the constructor.

Default is an empty dictionary.

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.BaseMedia.__init__ "Copy anchor link to this section for reference")

```
__init__(*args: Any, **kwargs: Any) -> None
```

### ``is\_lc\_serializable`classmethod`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.BaseMedia.is_lc_serializable "Copy anchor link to this section for reference")

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

### ``get\_lc\_namespace`classmethod`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.BaseMedia.get_lc_namespace "Copy anchor link to this section for reference")

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

### ``lc\_id`classmethod`[¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.BaseMedia.lc_id "Copy anchor link to this section for reference")

```
lc_id() -> list[str]
```

Return a unique identifier for this class for serialization purposes.

The unique identifier is a list of strings that describes the path
to the object.

For example, for the class `langchain.llms.openai.OpenAI`, the id is
`["langchain", "llms", "openai", "OpenAI"]`.

### ``to\_json [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.BaseMedia.to_json "Copy anchor link to this section for reference")

```
to_json() -> SerializedConstructor | SerializedNotImplemented
```

Serialize the object to JSON.

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If the class has deprecated attributes. |

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedConstructor | SerializedNotImplemented` | A JSON serializable object or a `SerializedNotImplemented` object. |

### ``to\_json\_not\_implemented [¶](https://reference.langchain.com/python/langchain_core/documents/\#langchain_core.documents.base.BaseMedia.to_json_not_implemented "Copy anchor link to this section for reference")

```
to_json_not_implemented() -> SerializedNotImplemented
```

Serialize a "not implemented" object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedNotImplemented` | `SerializedNotImplemented`. |

Back to top