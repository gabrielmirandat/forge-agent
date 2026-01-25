[Skip to content](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/#vector-store-integration-tests)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_tests/integration_tests/vectorstores.md "Edit this page")

# Vector store integration tests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#vector-store-integration-tests "Copy anchor link to this section for reference")

## ``VectorStoreIntegrationTests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests "Copy anchor link to this section for reference")

Bases: `BaseStandardTests`

Base class for vector store integration tests.

Implementers should subclass this test suite and provide a fixture
that returns an empty vector store for each test.

The fixture should use the `get_embeddings` method to get a pre-defined
embeddings model that should be used for this test suite.

Here is a template:

```
from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_parrot_link.vectorstores import ParrotVectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

class TestParrotVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore."""
        store = ParrotVectorStore(self.get_embeddings())
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            pass
```

In the fixture, before the `yield` we instantiate an empty vector store. In the
`finally` block, we call whatever logic is necessary to bring the vector store
to a clean state.

```
from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_chroma import Chroma

class TestChromaStandard(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty VectorStore for unit tests."""
        store = Chroma(embedding_function=self.get_embeddings())
        try:
            yield store
        finally:
            store.delete_collection()
            pass
```

Note that by default we enable both sync and async tests. To disable either,
override the `has_sync` or `has_async` properties to `False` in the
subclass. For example:

```
class TestParrotVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        ...

    @property
    def has_async(self) -> bool:
        return False
```

Note

API references for individual test methods include troubleshooting tips.

| METHOD | DESCRIPTION |
| --- | --- |
| `test_no_overrides_DO_NOT_OVERRIDE` | Test that no standard tests are overridden. |
| `vectorstore` | Get the `VectorStore` class to test. |
| `get_embeddings` | Get embeddings. |
| `test_vectorstore_is_empty` | Test that the `VectorStore` is empty. |
| `test_add_documents` | Test adding documents into the `VectorStore`. |
| `test_vectorstore_still_empty` | Test that the `VectorStore` is still empty. |
| `test_deleting_documents` | Test deleting documents from the `VectorStore`. |
| `test_deleting_bulk_documents` | Test that we can delete several documents at once. |
| `test_delete_missing_content` | Deleting missing content should not raise an exception. |
| `test_add_documents_with_ids_is_idempotent` | Adding by ID should be idempotent. |
| `test_add_documents_by_id_with_mutation` | Test that we can overwrite by ID using `add_documents`. |
| `test_get_by_ids` | Test get by IDs. |
| `test_get_by_ids_missing` | Test get by IDs with missing IDs. |
| `test_add_documents_documents` | Run `add_documents` tests. |
| `test_add_documents_with_existing_ids` | Test that `add_documents` with existing IDs is idempotent. |
| `test_vectorstore_is_empty_async` | Test that the `VectorStore` is empty. |
| `test_add_documents_async` | Test adding documents into the `VectorStore`. |
| `test_vectorstore_still_empty_async` | Test that the `VectorStore` is still empty. |
| `test_deleting_documents_async` | Test deleting documents from the `VectorStore`. |
| `test_deleting_bulk_documents_async` | Test that we can delete several documents at once. |
| `test_delete_missing_content_async` | Deleting missing content should not raise an exception. |
| `test_add_documents_with_ids_is_idempotent_async` | Adding by ID should be idempotent. |
| `test_add_documents_by_id_with_mutation_async` | Test that we can overwrite by ID using `add_documents`. |
| `test_get_by_ids_async` | Test get by IDs. |
| `test_get_by_ids_missing_async` | Test get by IDs with missing IDs. |
| `test_add_documents_documents_async` | Run `add_documents` tests. |
| `test_add_documents_with_existing_ids_async` | Test that `add_documents` with existing IDs is idempotent. |

### ``has\_sync`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.has_sync "Copy anchor link to this section for reference")

```
has_sync: bool
```

Configurable property to enable or disable sync tests.

### ``has\_async`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.has_async "Copy anchor link to this section for reference")

```
has_async: bool
```

Configurable property to enable or disable async tests.

### ``has\_get\_by\_ids`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.has_get_by_ids "Copy anchor link to this section for reference")

```
has_get_by_ids: bool
```

Whether the `VectorStore` supports `get_by_ids`.

### ``test\_no\_overrides\_DO\_NOT\_OVERRIDE [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_no_overrides_DO_NOT_OVERRIDE "Copy anchor link to this section for reference")

```
test_no_overrides_DO_NOT_OVERRIDE() -> None
```

Test that no standard tests are overridden.

### ``vectorstore`abstractmethod`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.vectorstore "Copy anchor link to this section for reference")

```
vectorstore() -> VectorStore
```

Get the `VectorStore` class to test.

The returned `VectorStore` should be empty.

### ``get\_embeddings`staticmethod`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.get_embeddings "Copy anchor link to this section for reference")

```
get_embeddings() -> Embeddings
```

Get embeddings.

A pre-defined embeddings model that should be used for this test.

This currently uses `DeterministicFakeEmbedding` from `langchain-core`,
which uses numpy to generate random numbers based on a hash of the input text.

The resulting embeddings are not meaningful, but they are deterministic.

### ``test\_vectorstore\_is\_empty [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_vectorstore_is_empty "Copy anchor link to this section for reference")

```
test_vectorstore_is_empty(vectorstore: VectorStore) -> None
```

Test that the `VectorStore` is empty.

Troubleshooting

If this test fails, check that the test class (i.e., sub class of
`VectorStoreIntegrationTests`) initializes an empty vector store in the
`vectorestore` fixture.

### ``test\_add\_documents [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_add_documents "Copy anchor link to this section for reference")

```
test_add_documents(vectorstore: VectorStore) -> None
```

Test adding documents into the `VectorStore`.

Troubleshooting

If this test fails, check that:

1. We correctly initialize an empty vector store in the `vectorestore`
    fixture.
2. Calling `similarity_search` for the top `k` similar documents does
    not threshold by score.
3. We do not mutate the original document object when adding it to the
    vector store (e.g., by adding an ID).

### ``test\_vectorstore\_still\_empty [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_vectorstore_still_empty "Copy anchor link to this section for reference")

```
test_vectorstore_still_empty(vectorstore: VectorStore) -> None
```

Test that the `VectorStore` is still empty.

This test should follow a test that adds documents.

This just verifies that the fixture is set up properly to be empty
after each test.

Troubleshooting

If this test fails, check that the test class (i.e., sub class of
`VectorStoreIntegrationTests`) correctly clears the vector store in the
`finally` block.

### ``test\_deleting\_documents [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_deleting_documents "Copy anchor link to this section for reference")

```
test_deleting_documents(vectorstore: VectorStore) -> None
```

Test deleting documents from the `VectorStore`.

Troubleshooting

If this test fails, check that `add_documents` preserves identifiers
passed in through `ids`, and that `delete` correctly removes
documents.

### ``test\_deleting\_bulk\_documents [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_deleting_bulk_documents "Copy anchor link to this section for reference")

```
test_deleting_bulk_documents(vectorstore: VectorStore) -> None
```

Test that we can delete several documents at once.

Troubleshooting

If this test fails, check that `delete` correctly removes multiple
documents when given a list of IDs.

### ``test\_delete\_missing\_content [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_delete_missing_content "Copy anchor link to this section for reference")

```
test_delete_missing_content(vectorstore: VectorStore) -> None
```

Deleting missing content should not raise an exception.

Troubleshooting

If this test fails, check that `delete` does not raise an exception
when deleting IDs that do not exist.

### ``test\_add\_documents\_with\_ids\_is\_idempotent [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_add_documents_with_ids_is_idempotent "Copy anchor link to this section for reference")

```
test_add_documents_with_ids_is_idempotent(vectorstore: VectorStore) -> None
```

Adding by ID should be idempotent.

Troubleshooting

If this test fails, check that adding the same document twice with the
same IDs has the same effect as adding it once (i.e., it does not
duplicate the documents).

### ``test\_add\_documents\_by\_id\_with\_mutation [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_add_documents_by_id_with_mutation "Copy anchor link to this section for reference")

```
test_add_documents_by_id_with_mutation(vectorstore: VectorStore) -> None
```

Test that we can overwrite by ID using `add_documents`.

Troubleshooting

If this test fails, check that when `add_documents` is called with an
ID that already exists in the vector store, the content is updated
rather than duplicated.

### ``test\_get\_by\_ids [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_get_by_ids "Copy anchor link to this section for reference")

```
test_get_by_ids(vectorstore: VectorStore) -> None
```

Test get by IDs.

This test requires that `get_by_ids` be implemented on the vector store.

Troubleshooting

If this test fails, check that `get_by_ids` is implemented and returns
documents in the same order as the IDs passed in.

Note

`get_by_ids` was added to the `VectorStore` interface in
`langchain-core` version 0.2.11. If difficult to implement, this
test can be skipped by setting the `has_get_by_ids` property to
`False`.

```
@property
def has_get_by_ids(self) -> bool:
    return False
```

### ``test\_get\_by\_ids\_missing [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_get_by_ids_missing "Copy anchor link to this section for reference")

```
test_get_by_ids_missing(vectorstore: VectorStore) -> None
```

Test get by IDs with missing IDs.

Troubleshooting

If this test fails, check that `get_by_ids` is implemented and does not
raise an exception when given IDs that do not exist.

Note

`get_by_ids` was added to the `VectorStore` interface in
`langchain-core` version 0.2.11. If difficult to implement, this
test can be skipped by setting the `has_get_by_ids` property to
`False`.

```
@property
def has_get_by_ids(self) -> bool:
    return False
```

### ``test\_add\_documents\_documents [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_add_documents_documents "Copy anchor link to this section for reference")

```
test_add_documents_documents(vectorstore: VectorStore) -> None
```

Run `add_documents` tests.

Troubleshooting

If this test fails, check that `get_by_ids` is implemented and returns
documents in the same order as the IDs passed in.

Check also that `add_documents` will correctly generate string IDs if
none are provided.

Note

`get_by_ids` was added to the `VectorStore` interface in
`langchain-core` version 0.2.11. If difficult to implement, this
test can be skipped by setting the `has_get_by_ids` property to
`False`.

```
@property
def has_get_by_ids(self) -> bool:
    return False
```

### ``test\_add\_documents\_with\_existing\_ids [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_add_documents_with_existing_ids "Copy anchor link to this section for reference")

```
test_add_documents_with_existing_ids(vectorstore: VectorStore) -> None
```

Test that `add_documents` with existing IDs is idempotent.

Troubleshooting

If this test fails, check that `get_by_ids` is implemented and returns
documents in the same order as the IDs passed in.

This test also verifies that:

1. IDs specified in the `Document.id` field are assigned when adding
    documents.
2. If some documents include IDs and others don't string IDs are generated
    for the latter.

Note

`get_by_ids` was added to the `VectorStore` interface in
`langchain-core` version 0.2.11. If difficult to implement, this
test can be skipped by setting the `has_get_by_ids` property to
`False`.

```
@property
def has_get_by_ids(self) -> bool:
    return False
```

### ``test\_vectorstore\_is\_empty\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_vectorstore_is_empty_async "Copy anchor link to this section for reference")

```
test_vectorstore_is_empty_async(vectorstore: VectorStore) -> None
```

Test that the `VectorStore` is empty.

Troubleshooting

If this test fails, check that the test class (i.e., sub class of
`VectorStoreIntegrationTests`) initializes an empty vector store in the
`vectorestore` fixture.

### ``test\_add\_documents\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_add_documents_async "Copy anchor link to this section for reference")

```
test_add_documents_async(vectorstore: VectorStore) -> None
```

Test adding documents into the `VectorStore`.

Troubleshooting

If this test fails, check that:

1. We correctly initialize an empty vector store in the `vectorestore`
    fixture.
2. Calling `.asimilarity_search` for the top `k` similar documents does
    not threshold by score.
3. We do not mutate the original document object when adding it to the
    vector store (e.g., by adding an ID).

### ``test\_vectorstore\_still\_empty\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_vectorstore_still_empty_async "Copy anchor link to this section for reference")

```
test_vectorstore_still_empty_async(vectorstore: VectorStore) -> None
```

Test that the `VectorStore` is still empty.

This test should follow a test that adds documents.

This just verifies that the fixture is set up properly to be empty
after each test.

Troubleshooting

If this test fails, check that the test class (i.e., sub class of
`VectorStoreIntegrationTests`) correctly clears the vector store in the
`finally` block.

### ``test\_deleting\_documents\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_deleting_documents_async "Copy anchor link to this section for reference")

```
test_deleting_documents_async(vectorstore: VectorStore) -> None
```

Test deleting documents from the `VectorStore`.

Troubleshooting

If this test fails, check that `aadd_documents` preserves identifiers
passed in through `ids`, and that `delete` correctly removes
documents.

### ``test\_deleting\_bulk\_documents\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_deleting_bulk_documents_async "Copy anchor link to this section for reference")

```
test_deleting_bulk_documents_async(vectorstore: VectorStore) -> None
```

Test that we can delete several documents at once.

Troubleshooting

If this test fails, check that `adelete` correctly removes multiple
documents when given a list of IDs.

### ``test\_delete\_missing\_content\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_delete_missing_content_async "Copy anchor link to this section for reference")

```
test_delete_missing_content_async(vectorstore: VectorStore) -> None
```

Deleting missing content should not raise an exception.

Troubleshooting

If this test fails, check that `adelete` does not raise an exception
when deleting IDs that do not exist.

### ``test\_add\_documents\_with\_ids\_is\_idempotent\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_add_documents_with_ids_is_idempotent_async "Copy anchor link to this section for reference")

```
test_add_documents_with_ids_is_idempotent_async(vectorstore: VectorStore) -> None
```

Adding by ID should be idempotent.

Troubleshooting

If this test fails, check that adding the same document twice with the
same IDs has the same effect as adding it once (i.e., it does not
duplicate the documents).

### ``test\_add\_documents\_by\_id\_with\_mutation\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_add_documents_by_id_with_mutation_async "Copy anchor link to this section for reference")

```
test_add_documents_by_id_with_mutation_async(vectorstore: VectorStore) -> None
```

Test that we can overwrite by ID using `add_documents`.

Troubleshooting

If this test fails, check that when `aadd_documents` is called with an
ID that already exists in the vector store, the content is updated
rather than duplicated.

### ``test\_get\_by\_ids\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_get_by_ids_async "Copy anchor link to this section for reference")

```
test_get_by_ids_async(vectorstore: VectorStore) -> None
```

Test get by IDs.

This test requires that `get_by_ids` be implemented on the vector store.

Troubleshooting

If this test fails, check that `get_by_ids` is implemented and returns
documents in the same order as the IDs passed in.

Note

`get_by_ids` was added to the `VectorStore` interface in
`langchain-core` version 0.2.11. If difficult to implement, this
test can be skipped by setting the `has_get_by_ids` property to
`False`.

```
@property
def has_get_by_ids(self) -> bool:
    return False
```

### ``test\_get\_by\_ids\_missing\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_get_by_ids_missing_async "Copy anchor link to this section for reference")

```
test_get_by_ids_missing_async(vectorstore: VectorStore) -> None
```

Test get by IDs with missing IDs.

Troubleshooting

If this test fails, check that `get_by_ids` is implemented and does not
raise an exception when given IDs that do not exist.

Note

`get_by_ids` was added to the `VectorStore` interface in
`langchain-core` version 0.2.11. If difficult to implement, this
test can be skipped by setting the `has_get_by_ids` property to
`False`.

```
@property
def has_get_by_ids(self) -> bool:
    return False
```

### ``test\_add\_documents\_documents\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_add_documents_documents_async "Copy anchor link to this section for reference")

```
test_add_documents_documents_async(vectorstore: VectorStore) -> None
```

Run `add_documents` tests.

Troubleshooting

If this test fails, check that `get_by_ids` is implemented and returns
documents in the same order as the IDs passed in.

Check also that `aadd_documents` will correctly generate string IDs if
none are provided.

Note

`get_by_ids` was added to the `VectorStore` interface in
`langchain-core` version 0.2.11. If difficult to implement, this
test can be skipped by setting the `has_get_by_ids` property to
`False`.

```
@property
def has_get_by_ids(self) -> bool:
    return False
```

### ``test\_add\_documents\_with\_existing\_ids\_async`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/vectorstores/\#langchain_tests.integration_tests.VectorStoreIntegrationTests.test_add_documents_with_existing_ids_async "Copy anchor link to this section for reference")

```
test_add_documents_with_existing_ids_async(vectorstore: VectorStore) -> None
```

Test that `add_documents` with existing IDs is idempotent.

Troubleshooting

If this test fails, check that `get_by_ids` is implemented and returns
documents in the same order as the IDs passed in.

This test also verifies that:

1. IDs specified in the `Document.id` field are assigned when adding
    documents.
2. If some documents include IDs and others don't string IDs are generated
    for the latter.

Note

`get_by_ids` was added to the `VectorStore` interface in
`langchain-core` version 0.2.11. If difficult to implement, this
test can be skipped by setting the `has_get_by_ids` property to
`False`.

```
@property
def has_get_by_ids(self) -> bool:
    return False
```

Back to top