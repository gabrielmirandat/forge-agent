[Skip to content](https://reference.langchain.com/python/langchain_core/load/#langchain_core.load.dump.dumpd)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_core/load.md "Edit this page")

# Serialization

## ``dumpd [¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.dump.dumpd "Copy anchor link to this section for reference")

```
dumpd(obj: Any) -> Any
```

Return a dict representation of an object.

Note

Plain dicts containing an `'lc'` key are automatically escaped to prevent
confusion with LC serialization format. The escape marker is removed during
deserialization.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `obj` | The object to dump.<br>**TYPE:**`Any` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Any` | Dictionary that can be serialized to json using `json.dumps`. |

## ``dumps [¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.dump.dumps "Copy anchor link to this section for reference")

```
dumps(obj: Any, *, pretty: bool = False, **kwargs: Any) -> str
```

Return a JSON string representation of an object.

Note

Plain dicts containing an `'lc'` key are automatically escaped to prevent
confusion with LC serialization format. The escape marker is removed during
deserialization.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `obj` | The object to dump.<br>**TYPE:**`Any` |
| `pretty` | Whether to pretty print the json.<br>If `True`, the json will be indented by either 2 spaces or the amount<br>provided in the `indent` kwarg.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `**kwargs` | Additional arguments to pass to `json.dumps`<br>**TYPE:**`Any`**DEFAULT:**`{}` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `str` | A JSON string representation of the object. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If `default` is passed as a kwarg. |

## ``load [¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.load.load "Copy anchor link to this section for reference")

```
load(
    obj: Any,
    *,
    allowed_objects: Iterable[AllowedObject] | Literal["all", "core"] = "core",
    secrets_map: dict[str, str] | None = None,
    valid_namespaces: list[str] | None = None,
    secrets_from_env: bool = False,
    additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]] | None = None,
    ignore_unserializable_fields: bool = False,
    init_validator: InitValidator | None = default_init_validator,
) -> Any
```

Revive a LangChain class from a JSON object.

Use this if you already have a parsed JSON object, eg. from `json.load` or
`orjson.loads`.

Only classes in the allowlist can be instantiated. The default allowlist includes
core LangChain types (messages, prompts, documents, etc.). See
`langchain_core.load.mapping` for the full list.

Beta feature

This is a beta feature. Please be wary of deploying experimental code to
production unless you've taken appropriate precautions.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `obj` | The object to load.<br>**TYPE:**`Any` |
| `allowed_objects` | Allowlist of classes that can be deserialized.<br>- `'core'` (default): Allow classes defined in the serialization mappings<br>for `langchain_core`.<br>- `'all'`: Allow classes defined in the serialization mappings.<br>  <br>This includes core LangChain types (messages, prompts, documents, etc.)<br>and trusted partner integrations. See `langchain_core.load.mapping` for<br>the full list.<br>  <br>- Explicit list of classes: Only those specific classes are allowed.<br>  <br>- `[]`: Disallow all deserialization (will raise on any object).<br>**TYPE:**`Iterable[AllowedObject] | Literal['all', 'core']`**DEFAULT:**`'core'` |
| `secrets_map` | A map of secrets to load.<br>If a secret is not found in the map, it will be loaded from the environment<br>if `secrets_from_env` is `True`.<br>**TYPE:**`dict[str, str] | None`**DEFAULT:**`None` |
| `valid_namespaces` | Additional namespaces (modules) to allow during<br>deserialization, beyond the default trusted namespaces.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `secrets_from_env` | Whether to load secrets from the environment.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `additional_import_mappings` | A dictionary of additional namespace mappings.<br>You can use this to override default mappings or add new mappings.<br>When `allowed_objects` is `None` (using defaults), paths from these<br>mappings are also added to the allowed class paths.<br>**TYPE:**`dict[tuple[str, ...], tuple[str, ...]] | None`**DEFAULT:**`None` |
| `ignore_unserializable_fields` | Whether to ignore unserializable fields.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `init_validator` | Optional callable to validate kwargs before instantiation.<br>If provided, this function is called with `(class_path, kwargs)` where<br>`class_path` is the class path tuple and `kwargs` is the kwargs dict.<br>The validator should raise an exception if the object should not be<br>deserialized, otherwise return `None`.<br>Defaults to `default_init_validator` which blocks jinja2 templates.<br>**TYPE:**`InitValidator | None`**DEFAULT:**`default_init_validator` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Any` | Revived LangChain objects. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If an object's class path is not in the `allowed_objects` allowlist. |

Example

```
from langchain_core.load import load, dumpd
from langchain_core.messages import AIMessage

msg = AIMessage(content="Hello")
data = dumpd(msg)

# Deserialize using default allowlist
loaded = load(data)

# Or with explicit allowlist
loaded = load(data, allowed_objects=[AIMessage])

# Or extend defaults with additional mappings
loaded = load(
    data,
    additional_import_mappings={
        ("my_pkg", "MyClass"): ("my_pkg", "module", "MyClass"),
    },
)
```

## ``loads [¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.load.loads "Copy anchor link to this section for reference")

```
loads(
    text: str,
    *,
    allowed_objects: Iterable[AllowedObject] | Literal["all", "core"] = "core",
    secrets_map: dict[str, str] | None = None,
    valid_namespaces: list[str] | None = None,
    secrets_from_env: bool = False,
    additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]] | None = None,
    ignore_unserializable_fields: bool = False,
    init_validator: InitValidator | None = default_init_validator,
) -> Any
```

Revive a LangChain class from a JSON string.

Equivalent to `load(json.loads(text))`.

Only classes in the allowlist can be instantiated. The default allowlist includes
core LangChain types (messages, prompts, documents, etc.). See
`langchain_core.load.mapping` for the full list.

Beta feature

This is a beta feature. Please be wary of deploying experimental code to
production unless you've taken appropriate precautions.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `text` | The string to load.<br>**TYPE:**`str` |
| `allowed_objects` | Allowlist of classes that can be deserialized.<br>- `'core'` (default): Allow classes defined in the serialization mappings<br>for `langchain_core`.<br>- `'all'`: Allow classes defined in the serialization mappings.<br>  <br>This includes core LangChain types (messages, prompts, documents, etc.)<br>and trusted partner integrations. See `langchain_core.load.mapping` for<br>the full list.<br>  <br>- Explicit list of classes: Only those specific classes are allowed.<br>  <br>- `[]`: Disallow all deserialization (will raise on any object).<br>**TYPE:**`Iterable[AllowedObject] | Literal['all', 'core']`**DEFAULT:**`'core'` |
| `secrets_map` | A map of secrets to load.<br>If a secret is not found in the map, it will be loaded from the environment<br>if `secrets_from_env` is `True`.<br>**TYPE:**`dict[str, str] | None`**DEFAULT:**`None` |
| `valid_namespaces` | Additional namespaces (modules) to allow during<br>deserialization, beyond the default trusted namespaces.<br>**TYPE:**`list[str] | None`**DEFAULT:**`None` |
| `secrets_from_env` | Whether to load secrets from the environment.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `additional_import_mappings` | A dictionary of additional namespace mappings.<br>You can use this to override default mappings or add new mappings.<br>When `allowed_objects` is `None` (using defaults), paths from these<br>mappings are also added to the allowed class paths.<br>**TYPE:**`dict[tuple[str, ...], tuple[str, ...]] | None`**DEFAULT:**`None` |
| `ignore_unserializable_fields` | Whether to ignore unserializable fields.<br>**TYPE:**`bool`**DEFAULT:**`False` |
| `init_validator` | Optional callable to validate kwargs before instantiation.<br>If provided, this function is called with `(class_path, kwargs)` where<br>`class_path` is the class path tuple and `kwargs` is the kwargs dict.<br>The validator should raise an exception if the object should not be<br>deserialized, otherwise return `None`.<br>Defaults to `default_init_validator` which blocks jinja2 templates.<br>**TYPE:**`InitValidator | None`**DEFAULT:**`default_init_validator` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `Any` | Revived LangChain objects. |

| RAISES | DESCRIPTION |
| --- | --- |
| `ValueError` | If an object's class path is not in the `allowed_objects` allowlist. |

## ``Serializable [¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.serializable.Serializable "Copy anchor link to this section for reference")

Bases: `BaseModel`, `ABC`

Serializable base class.

This class is used to serialize objects to JSON.

It relies on the following methods and properties:

- [`is_lc_serializable`](https://reference.langchain.com/python/langchain_core/load/#langchain_core.load.serializable.Serializable.is_lc_serializable "<code class=\"doc-symbol doc-symbol-heading doc-symbol-method\"></code>            <span class=\"doc doc-object-name doc-function-name\">is_lc_serializable</span>     <span class=\"doc doc-labels\">       <small class=\"doc doc-label doc-label-classmethod\"><code>classmethod</code></small>   </span>"): Is this class serializable?

By design, even if a class inherits from `Serializable`, it is not serializable
by default. This is to prevent accidental serialization of objects that should
not be serialized.
\- [`get_lc_namespace`](https://reference.langchain.com/python/langchain_core/load/#langchain_core.load.serializable.Serializable.get_lc_namespace "<code class=\"doc-symbol doc-symbol-heading doc-symbol-method\"></code>            <span class=\"doc doc-object-name doc-function-name\">get_lc_namespace</span>     <span class=\"doc doc-labels\">       <small class=\"doc doc-label doc-label-classmethod\"><code>classmethod</code></small>   </span>"): Get the namespace of the LangChain object.

During deserialization, this namespace is used to identify
the correct class to instantiate.

Please see the `Reviver` class in `langchain_core.load.load` for more details.
During deserialization an additional mapping is handle classes that have moved
or been renamed across package versions.

- [`lc_secrets`](https://reference.langchain.com/python/langchain_core/load/#langchain_core.load.serializable.Serializable.lc_secrets "<code class=\"doc-symbol doc-symbol-heading doc-symbol-attribute\"></code>            <span class=\"doc doc-object-name doc-attribute-name\">lc_secrets</span>     <span class=\"doc doc-labels\">       <small class=\"doc doc-label doc-label-property\"><code>property</code></small>   </span>"): A map of constructor argument names to secret ids.

- [`lc_attributes`](https://reference.langchain.com/python/langchain_core/load/#langchain_core.load.serializable.Serializable.lc_attributes "<code class=\"doc-symbol doc-symbol-heading doc-symbol-attribute\"></code>            <span class=\"doc doc-object-name doc-attribute-name\">lc_attributes</span>     <span class=\"doc doc-labels\">       <small class=\"doc doc-label doc-label-property\"><code>property</code></small>   </span>"): List of additional attribute names that should be included
as part of the serialized representation.

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` |  |
| `is_lc_serializable` | Is this class serializable? |
| `get_lc_namespace` | Get the namespace of the LangChain object. |
| `lc_id` | Return a unique identifier for this class for serialization purposes. |
| `to_json` | Serialize the object to JSON. |
| `to_json_not_implemented` | Serialize a "not implemented" object. |

### ``lc\_secrets`property`[¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.serializable.Serializable.lc_secrets "Copy anchor link to this section for reference")

```
lc_secrets: dict[str, str]
```

A map of constructor argument names to secret ids.

For example, `{"openai_api_key": "OPENAI_API_KEY"}`

### ``lc\_attributes`property`[¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.serializable.Serializable.lc_attributes "Copy anchor link to this section for reference")

```
lc_attributes: dict
```

List of attribute names that should be included in the serialized kwargs.

These attributes must be accepted by the constructor.

Default is an empty dictionary.

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.serializable.Serializable.__init__ "Copy anchor link to this section for reference")

```
__init__(*args: Any, **kwargs: Any) -> None
```

### ``is\_lc\_serializable`classmethod`[¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.serializable.Serializable.is_lc_serializable "Copy anchor link to this section for reference")

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

### ``get\_lc\_namespace`classmethod`[¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.serializable.Serializable.get_lc_namespace "Copy anchor link to this section for reference")

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

### ``lc\_id`classmethod`[¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.serializable.Serializable.lc_id "Copy anchor link to this section for reference")

```
lc_id() -> list[str]
```

Return a unique identifier for this class for serialization purposes.

The unique identifier is a list of strings that describes the path
to the object.

For example, for the class `langchain.llms.openai.OpenAI`, the id is
`["langchain", "llms", "openai", "OpenAI"]`.

### ``to\_json [¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.serializable.Serializable.to_json "Copy anchor link to this section for reference")

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

### ``to\_json\_not\_implemented [¶](https://reference.langchain.com/python/langchain_core/load/\#langchain_core.load.serializable.Serializable.to_json_not_implemented "Copy anchor link to this section for reference")

```
to_json_not_implemented() -> SerializedNotImplemented
```

Serialize a "not implemented" object.

| RETURNS | DESCRIPTION |
| --- | --- |
| `SerializedNotImplemented` | `SerializedNotImplemented`. |

Back to top