[Skip to content](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/#key-value-store-integration-tests)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_tests/integration_tests/kv_stores.md "Edit this page")

# Key-value store integration tests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#key-value-store-integration-tests "Copy anchor link to this section for reference")

## ``BaseStoreSyncTests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests "Copy anchor link to this section for reference")

Bases: `BaseStandardTests`, `Generic[V]`

Test suite for checking the key-value API of a `BaseStore`.

This test suite verifies the basic key-value API of a `BaseStore`.

The test suite is designed for synchronous key-value stores.

Implementers should subclass this test suite and provide a fixture
that returns an empty key-value store for each test.

| METHOD | DESCRIPTION |
| --- | --- |
| `test_no_overrides_DO_NOT_OVERRIDE` | Test that no standard tests are overridden. |
| `kv_store` | Get the key-value store class to test. |
| `three_values` | Three example values that will be used in the tests. |
| `test_three_values` | Test that the fixture provides three values. |
| `test_kv_store_is_empty` | Test that the key-value store is empty. |
| `test_set_and_get_values` | Test setting and getting values in the key-value store. |
| `test_store_still_empty` | Test that the store is still empty. |
| `test_delete_values` | Test deleting values from the key-value store. |
| `test_delete_bulk_values` | Test that we can delete several values at once. |
| `test_delete_missing_keys` | Deleting missing keys should not raise an exception. |
| `test_set_values_is_idempotent` | Setting values by key should be idempotent. |
| `test_get_can_get_same_value` | Test that the same value can be retrieved multiple times. |
| `test_overwrite_values_by_key` | Test that we can overwrite values by key using mset. |
| `test_yield_keys` | Test that we can yield keys from the store. |

### ``test\_no\_overrides\_DO\_NOT\_OVERRIDE [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.test_no_overrides_DO_NOT_OVERRIDE "Copy anchor link to this section for reference")

```
test_no_overrides_DO_NOT_OVERRIDE() -> None
```

Test that no standard tests are overridden.

### ``kv\_store`abstractmethod`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.kv_store "Copy anchor link to this section for reference")

```
kv_store() -> BaseStore[str, V]
```

Get the key-value store class to test.

The returned key-value store should be EMPTY.

### ``three\_values`abstractmethod`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.three_values "Copy anchor link to this section for reference")

```
three_values() -> tuple[V, V, V]
```

Three example values that will be used in the tests.

### ``test\_three\_values [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.test_three_values "Copy anchor link to this section for reference")

```
test_three_values(three_values: tuple[V, V, V]) -> None
```

Test that the fixture provides three values.

### ``test\_kv\_store\_is\_empty [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.test_kv_store_is_empty "Copy anchor link to this section for reference")

```
test_kv_store_is_empty(kv_store: BaseStore[str, V]) -> None
```

Test that the key-value store is empty.

### ``test\_set\_and\_get\_values [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.test_set_and_get_values "Copy anchor link to this section for reference")

```
test_set_and_get_values(
    kv_store: BaseStore[str, V], three_values: tuple[V, V, V]
) -> None
```

Test setting and getting values in the key-value store.

### ``test\_store\_still\_empty [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.test_store_still_empty "Copy anchor link to this section for reference")

```
test_store_still_empty(kv_store: BaseStore[str, V]) -> None
```

Test that the store is still empty.

This test should follow a test that sets values.

This just verifies that the fixture is set up properly to be empty
after each test.

### ``test\_delete\_values [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.test_delete_values "Copy anchor link to this section for reference")

```
test_delete_values(
    kv_store: BaseStore[str, V], three_values: tuple[V, V, V]
) -> None
```

Test deleting values from the key-value store.

### ``test\_delete\_bulk\_values [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.test_delete_bulk_values "Copy anchor link to this section for reference")

```
test_delete_bulk_values(
    kv_store: BaseStore[str, V], three_values: tuple[V, V, V]
) -> None
```

Test that we can delete several values at once.

### ``test\_delete\_missing\_keys [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.test_delete_missing_keys "Copy anchor link to this section for reference")

```
test_delete_missing_keys(kv_store: BaseStore[str, V]) -> None
```

Deleting missing keys should not raise an exception.

### ``test\_set\_values\_is\_idempotent [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.test_set_values_is_idempotent "Copy anchor link to this section for reference")

```
test_set_values_is_idempotent(
    kv_store: BaseStore[str, V], three_values: tuple[V, V, V]
) -> None
```

Setting values by key should be idempotent.

### ``test\_get\_can\_get\_same\_value [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.test_get_can_get_same_value "Copy anchor link to this section for reference")

```
test_get_can_get_same_value(
    kv_store: BaseStore[str, V], three_values: tuple[V, V, V]
) -> None
```

Test that the same value can be retrieved multiple times.

### ``test\_overwrite\_values\_by\_key [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.test_overwrite_values_by_key "Copy anchor link to this section for reference")

```
test_overwrite_values_by_key(
    kv_store: BaseStore[str, V], three_values: tuple[V, V, V]
) -> None
```

Test that we can overwrite values by key using mset.

### ``test\_yield\_keys [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreSyncTests.test_yield_keys "Copy anchor link to this section for reference")

```
test_yield_keys(kv_store: BaseStore[str, V], three_values: tuple[V, V, V]) -> None
```

Test that we can yield keys from the store.

## ``BaseStoreAsyncTests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests "Copy anchor link to this section for reference")

Bases: `BaseStandardTests`, `Generic[V]`

Test suite for checking the key-value API of a `BaseStore`.

This test suite verifies the basic key-value API of a `BaseStore`.

The test suite is designed for synchronous key-value stores.

Implementers should subclass this test suite and provide a fixture
that returns an empty key-value store for each test.

| METHOD | DESCRIPTION |
| --- | --- |
| `test_no_overrides_DO_NOT_OVERRIDE` | Test that no standard tests are overridden. |
| `kv_store` | Get the key-value store class to test. |
| `three_values` | Three example values that will be used in the tests. |
| `test_three_values` | Test that the fixture provides three values. |
| `test_kv_store_is_empty` | Test that the key-value store is empty. |
| `test_set_and_get_values` | Test setting and getting values in the key-value store. |
| `test_store_still_empty` | Test that the store is still empty. |
| `test_delete_values` | Test deleting values from the key-value store. |
| `test_delete_bulk_values` | Test that we can delete several values at once. |
| `test_delete_missing_keys` | Deleting missing keys should not raise an exception. |
| `test_set_values_is_idempotent` | Setting values by key should be idempotent. |
| `test_get_can_get_same_value` | Test that the same value can be retrieved multiple times. |
| `test_overwrite_values_by_key` | Test that we can overwrite values by key using mset. |
| `test_yield_keys` | Test that we can yield keys from the store. |

### ``test\_no\_overrides\_DO\_NOT\_OVERRIDE [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.test_no_overrides_DO_NOT_OVERRIDE "Copy anchor link to this section for reference")

```
test_no_overrides_DO_NOT_OVERRIDE() -> None
```

Test that no standard tests are overridden.

### ``kv\_store`abstractmethod``async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.kv_store "Copy anchor link to this section for reference")

```
kv_store() -> BaseStore[str, V]
```

Get the key-value store class to test.

The returned key-value store should be EMPTY.

### ``three\_values`abstractmethod`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.three_values "Copy anchor link to this section for reference")

```
three_values() -> tuple[V, V, V]
```

Three example values that will be used in the tests.

### ``test\_three\_values`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.test_three_values "Copy anchor link to this section for reference")

```
test_three_values(three_values: tuple[V, V, V]) -> None
```

Test that the fixture provides three values.

### ``test\_kv\_store\_is\_empty`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.test_kv_store_is_empty "Copy anchor link to this section for reference")

```
test_kv_store_is_empty(kv_store: BaseStore[str, V]) -> None
```

Test that the key-value store is empty.

### ``test\_set\_and\_get\_values`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.test_set_and_get_values "Copy anchor link to this section for reference")

```
test_set_and_get_values(
    kv_store: BaseStore[str, V], three_values: tuple[V, V, V]
) -> None
```

Test setting and getting values in the key-value store.

### ``test\_store\_still\_empty`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.test_store_still_empty "Copy anchor link to this section for reference")

```
test_store_still_empty(kv_store: BaseStore[str, V]) -> None
```

Test that the store is still empty.

This test should follow a test that sets values.

This just verifies that the fixture is set up properly to be empty
after each test.

### ``test\_delete\_values`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.test_delete_values "Copy anchor link to this section for reference")

```
test_delete_values(
    kv_store: BaseStore[str, V], three_values: tuple[V, V, V]
) -> None
```

Test deleting values from the key-value store.

### ``test\_delete\_bulk\_values`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.test_delete_bulk_values "Copy anchor link to this section for reference")

```
test_delete_bulk_values(
    kv_store: BaseStore[str, V], three_values: tuple[V, V, V]
) -> None
```

Test that we can delete several values at once.

### ``test\_delete\_missing\_keys`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.test_delete_missing_keys "Copy anchor link to this section for reference")

```
test_delete_missing_keys(kv_store: BaseStore[str, V]) -> None
```

Deleting missing keys should not raise an exception.

### ``test\_set\_values\_is\_idempotent`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.test_set_values_is_idempotent "Copy anchor link to this section for reference")

```
test_set_values_is_idempotent(
    kv_store: BaseStore[str, V], three_values: tuple[V, V, V]
) -> None
```

Setting values by key should be idempotent.

### ``test\_get\_can\_get\_same\_value`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.test_get_can_get_same_value "Copy anchor link to this section for reference")

```
test_get_can_get_same_value(
    kv_store: BaseStore[str, V], three_values: tuple[V, V, V]
) -> None
```

Test that the same value can be retrieved multiple times.

### ``test\_overwrite\_values\_by\_key`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.test_overwrite_values_by_key "Copy anchor link to this section for reference")

```
test_overwrite_values_by_key(
    kv_store: BaseStore[str, V], three_values: tuple[V, V, V]
) -> None
```

Test that we can overwrite values by key using mset.

### ``test\_yield\_keys`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/kv_stores/\#langchain_tests.integration_tests.BaseStoreAsyncTests.test_yield_keys "Copy anchor link to this section for reference")

```
test_yield_keys(kv_store: BaseStore[str, V], three_values: tuple[V, V, V]) -> None
```

Test that we can yield keys from the store.

Back to top