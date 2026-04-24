[Skip to content](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/#retriever-integration-tests)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_tests/integration_tests/retrievers.md "Edit this page")

# Retriever integration tests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/\#retriever-integration-tests "Copy anchor link to this section for reference")

## ``RetrieversIntegrationTests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/\#langchain_tests.integration_tests.RetrieversIntegrationTests "Copy anchor link to this section for reference")

Bases: `BaseStandardTests`

Base class for retrievers integration tests.

| METHOD | DESCRIPTION |
| --- | --- |
| `test_no_overrides_DO_NOT_OVERRIDE` | Test that no standard tests are overridden. |
| `retriever` | Return retriever fixture. |
| `test_k_constructor_param` | Test the number of results constructor parameter. |
| `test_invoke_with_k_kwarg` | Test the number of results parameter in `invoke`. |
| `test_invoke_returns_documents` | Test invoke returns documents. |
| `test_ainvoke_returns_documents` | Test ainvoke returns documents. |

### ``retriever\_constructor`abstractmethod``property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/\#langchain_tests.integration_tests.RetrieversIntegrationTests.retriever_constructor "Copy anchor link to this section for reference")

```
retriever_constructor: type[BaseRetriever]
```

A `BaseRetriever` subclass to be tested.

### ``retriever\_constructor\_params`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/\#langchain_tests.integration_tests.RetrieversIntegrationTests.retriever_constructor_params "Copy anchor link to this section for reference")

```
retriever_constructor_params: dict[str, Any]
```

Returns a dictionary of parameters to pass to the retriever constructor.

### ``retriever\_query\_example`abstractmethod``property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/\#langchain_tests.integration_tests.RetrieversIntegrationTests.retriever_query_example "Copy anchor link to this section for reference")

```
retriever_query_example: str
```

Returns a str representing the `query` of an example retriever call.

### ``num\_results\_arg\_name`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/\#langchain_tests.integration_tests.RetrieversIntegrationTests.num_results_arg_name "Copy anchor link to this section for reference")

```
num_results_arg_name: str
```

Returns the name of the parameter for the number of results returned.

Usually something like `k` or `top_k`.

### ``test\_no\_overrides\_DO\_NOT\_OVERRIDE [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/\#langchain_tests.integration_tests.RetrieversIntegrationTests.test_no_overrides_DO_NOT_OVERRIDE "Copy anchor link to this section for reference")

```
test_no_overrides_DO_NOT_OVERRIDE() -> None
```

Test that no standard tests are overridden.

### ``retriever [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/\#langchain_tests.integration_tests.RetrieversIntegrationTests.retriever "Copy anchor link to this section for reference")

```
retriever() -> BaseRetriever
```

Return retriever fixture.

### ``test\_k\_constructor\_param [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/\#langchain_tests.integration_tests.RetrieversIntegrationTests.test_k_constructor_param "Copy anchor link to this section for reference")

```
test_k_constructor_param() -> None
```

Test the number of results constructor parameter.

Test that the retriever constructor accepts a parameter representing
the number of documents to return.

By default, the parameter tested is named `k`, but it can be overridden by
setting the `num_results_arg_name` property.

Note

If the retriever doesn't support configuring the number of results returned
via the constructor, this test can be skipped using a pytest `xfail` on
the test class:

```
@pytest.mark.xfail(
    reason="This retriever doesn't support setting "
    "the number of results via the constructor."
)
def test_k_constructor_param(self) -> None:
    raise NotImplementedError
```

Troubleshooting

If this test fails, the retriever constructor does not accept a number
of results parameter, or the retriever does not return the correct number
of documents ( of the one set in `num_results_arg_name`) when it is
set.

For example, a retriever like...

```
MyRetriever(k=3).invoke("query")
```

...should return 3 documents when invoked with a query.

### ``test\_invoke\_with\_k\_kwarg [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/\#langchain_tests.integration_tests.RetrieversIntegrationTests.test_invoke_with_k_kwarg "Copy anchor link to this section for reference")

```
test_invoke_with_k_kwarg(retriever: BaseRetriever) -> None
```

Test the number of results parameter in `invoke`.

Test that the invoke method accepts a parameter representing
the number of documents to return.

By default, the parameter is named, but it can be overridden by
setting the `num_results_arg_name` property.

Note

If the retriever doesn't support configuring the number of results returned
via the invoke method, this test can be skipped using a pytest `xfail` on
the test class:

```
@pytest.mark.xfail(
    reason="This retriever doesn't support setting "
    "the number of results in the invoke method."
)
def test_invoke_with_k_kwarg(self) -> None:
    raise NotImplementedError
```

Troubleshooting

If this test fails, the retriever's invoke method does not accept a number
of results parameter, or the retriever does not return the correct number
of documents (`k` of the one set in `num_results_arg_name`) when it is
set.

For example, a retriever like...

```
MyRetriever().invoke("query", k=3)
```

...should return 3 documents when invoked with a query.

### ``test\_invoke\_returns\_documents [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/\#langchain_tests.integration_tests.RetrieversIntegrationTests.test_invoke_returns_documents "Copy anchor link to this section for reference")

```
test_invoke_returns_documents(retriever: BaseRetriever) -> None
```

Test invoke returns documents.

If invoked with the example params, the retriever should return a list of
Documents.

Troubleshooting

If this test fails, the retriever's invoke method does not return a list of
`Document` objects. Please confirm that your
`_get_relevant_documents` method returns a list of `Document` objects.

### ``test\_ainvoke\_returns\_documents`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/retrievers/\#langchain_tests.integration_tests.RetrieversIntegrationTests.test_ainvoke_returns_documents "Copy anchor link to this section for reference")

```
test_ainvoke_returns_documents(retriever: BaseRetriever) -> None
```

Test ainvoke returns documents.

If `ainvoke`'d with the example params, the retriever should return a list of
`Document` objects.

See `test_invoke_returns_documents` for more information on
troubleshooting.

Back to top