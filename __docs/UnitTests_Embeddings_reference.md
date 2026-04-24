[Skip to content](https://reference.langchain.com/python/langchain_tests/unit_tests/embeddings/#embeddings-unit-tests)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_tests/unit_tests/embeddings.md "Edit this page")

# Embeddings unit tests [¶](https://reference.langchain.com/python/langchain_tests/unit_tests/embeddings/\#embeddings-unit-tests "Copy anchor link to this section for reference")

## ``EmbeddingsUnitTests [¶](https://reference.langchain.com/python/langchain_tests/unit_tests/embeddings/\#langchain_tests.unit_tests.EmbeddingsUnitTests "Copy anchor link to this section for reference")

Bases: `EmbeddingsTests`

Base class for embeddings unit tests.

Test subclasses must implement the `embeddings_class` property to specify the
embeddings model to be tested. You can also override the
`embedding_model_params` property to specify initialization parameters.

```
from typing import Type

from langchain_tests.unit_tests import EmbeddingsUnitTests
from my_package.embeddings import MyEmbeddingsModel

class TestMyEmbeddingsModelUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[MyEmbeddingsModel]:
        # Return the embeddings model class to test here
        return MyEmbeddingsModel

    @property
    def embedding_model_params(self) -> dict:
        # Return initialization parameters for the model.
        return {"model": "model-001"}
```

Note

API references for individual test methods include troubleshooting tips.

Testing initialization from environment variables
Overriding the `init_from_env_params` property will enable additional tests
for initialization from environment variables. See below for details.

````
??? note "`init_from_env_params`"

    This property is used in unit tests to test initialization from
    environment variables. It should return a tuple of three dictionaries
    that specify the environment variables, additional initialization args,
    and expected instance attributes to check.

    Defaults to empty dicts. If not overridden, the test is skipped.

    ```python
    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return (
            {
                "MY_API_KEY": "api_key",
            },
            {
                "model": "model-001",
            },
            {
                "my_api_key": "api_key",
            },
        )
    ```
````

| METHOD | DESCRIPTION |
| --- | --- |
| `test_no_overrides_DO_NOT_OVERRIDE` | Test that no standard tests are overridden. |
| `model` | Embeddings model fixture. |
| `test_init` | Test model initialization. |
| `test_init_from_env` | Test initialization from environment variables. |

### ``embeddings\_class`abstractmethod``property`[¶](https://reference.langchain.com/python/langchain_tests/unit_tests/embeddings/\#langchain_tests.unit_tests.EmbeddingsUnitTests.embeddings_class "Copy anchor link to this section for reference")

```
embeddings_class: type[Embeddings]
```

Embeddings class.

### ``embedding\_model\_params`property`[¶](https://reference.langchain.com/python/langchain_tests/unit_tests/embeddings/\#langchain_tests.unit_tests.EmbeddingsUnitTests.embedding_model_params "Copy anchor link to this section for reference")

```
embedding_model_params: dict[str, Any]
```

Embeddings model parameters.

### ``init\_from\_env\_params`property`[¶](https://reference.langchain.com/python/langchain_tests/unit_tests/embeddings/\#langchain_tests.unit_tests.EmbeddingsUnitTests.init_from_env_params "Copy anchor link to this section for reference")

```
init_from_env_params: tuple[dict[str, str], dict[str, Any], dict[str, Any]]
```

Init from env params.

This property is used in unit tests to test initialization from environment
variables. It should return a tuple of three dictionaries that specify the
environment variables, additional initialization args, and expected instance
attributes to check.

### ``test\_no\_overrides\_DO\_NOT\_OVERRIDE [¶](https://reference.langchain.com/python/langchain_tests/unit_tests/embeddings/\#langchain_tests.unit_tests.EmbeddingsUnitTests.test_no_overrides_DO_NOT_OVERRIDE "Copy anchor link to this section for reference")

```
test_no_overrides_DO_NOT_OVERRIDE() -> None
```

Test that no standard tests are overridden.

### ``model [¶](https://reference.langchain.com/python/langchain_tests/unit_tests/embeddings/\#langchain_tests.unit_tests.EmbeddingsUnitTests.model "Copy anchor link to this section for reference")

```
model() -> Embeddings
```

Embeddings model fixture.

### ``test\_init [¶](https://reference.langchain.com/python/langchain_tests/unit_tests/embeddings/\#langchain_tests.unit_tests.EmbeddingsUnitTests.test_init "Copy anchor link to this section for reference")

```
test_init() -> None
```

Test model initialization.

Troubleshooting

If this test fails, ensure that `embedding_model_params` is specified
and the model can be initialized from those params.

### ``test\_init\_from\_env [¶](https://reference.langchain.com/python/langchain_tests/unit_tests/embeddings/\#langchain_tests.unit_tests.EmbeddingsUnitTests.test_init_from_env "Copy anchor link to this section for reference")

```
test_init_from_env() -> None
```

Test initialization from environment variables.

Relies on the `init_from_env_params` property.
Test is skipped if that property is not set.

Troubleshooting

If this test fails, ensure that `init_from_env_params` is specified
correctly and that model parameters are properly set from environment
variables during initialization.

Back to top