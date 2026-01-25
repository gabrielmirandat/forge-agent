[Skip to content](https://reference.langchain.com/python/langchain_tests/integration_tests/tools/#tool-integration-tests)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_tests/integration_tests/tools.md "Edit this page")

# Tool integration tests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/tools/\#tool-integration-tests "Copy anchor link to this section for reference")

## ``ToolsIntegrationTests [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/tools/\#langchain_tests.integration_tests.ToolsIntegrationTests "Copy anchor link to this section for reference")

Bases: `ToolsTests`

Base class for tools integration tests.

| METHOD | DESCRIPTION |
| --- | --- |
| `test_no_overrides_DO_NOT_OVERRIDE` | Test that no standard tests are overridden. |
| `tool` | Tool fixture. |
| `test_invoke_matches_output_schema` | Test invoke matches output schema. |
| `test_async_invoke_matches_output_schema` | Test async invoke matches output schema. |
| `test_invoke_no_tool_call` | Test invoke without `ToolCall`. |
| `test_async_invoke_no_tool_call` | Test async invoke without `ToolCall`. |

### ``tool\_constructor`abstractmethod``property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/tools/\#langchain_tests.integration_tests.ToolsIntegrationTests.tool_constructor "Copy anchor link to this section for reference")

```
tool_constructor: type[BaseTool] | BaseTool
```

Returns a class or instance of a tool to be tested.

### ``tool\_constructor\_params`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/tools/\#langchain_tests.integration_tests.ToolsIntegrationTests.tool_constructor_params "Copy anchor link to this section for reference")

```
tool_constructor_params: dict[str, Any]
```

Returns a dictionary of parameters to pass to the tool constructor.

### ``tool\_invoke\_params\_example`property`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/tools/\#langchain_tests.integration_tests.ToolsIntegrationTests.tool_invoke_params_example "Copy anchor link to this section for reference")

```
tool_invoke_params_example: dict[str, Any]
```

Returns a dictionary representing the "args" of an example tool call.

This should NOT be a `ToolCall` dict - it should not have
`{"name", "id", "args"}` keys.

### ``test\_no\_overrides\_DO\_NOT\_OVERRIDE [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/tools/\#langchain_tests.integration_tests.ToolsIntegrationTests.test_no_overrides_DO_NOT_OVERRIDE "Copy anchor link to this section for reference")

```
test_no_overrides_DO_NOT_OVERRIDE() -> None
```

Test that no standard tests are overridden.

### ``tool [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/tools/\#langchain_tests.integration_tests.ToolsIntegrationTests.tool "Copy anchor link to this section for reference")

```
tool() -> BaseTool
```

Tool fixture.

### ``test\_invoke\_matches\_output\_schema [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/tools/\#langchain_tests.integration_tests.ToolsIntegrationTests.test_invoke_matches_output_schema "Copy anchor link to this section for reference")

```
test_invoke_matches_output_schema(tool: BaseTool) -> None
```

Test invoke matches output schema.

If invoked with a `ToolCall`, the tool should return a valid `ToolMessage`
content.

If you have followed the [custom tool guide](https://docs.langchain.com/oss/python/contributing/implement-langchain#tools),
this test should always pass because `ToolCall` inputs are handled by the
`langchain_core.tools.BaseTool` class.

If you have not followed this guide, you should ensure that your tool's
`invoke` method returns a valid ToolMessage content when it receives
a `dict` representing a `ToolCall` as input (as opposed to distinct args).

### ``test\_async\_invoke\_matches\_output\_schema`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/tools/\#langchain_tests.integration_tests.ToolsIntegrationTests.test_async_invoke_matches_output_schema "Copy anchor link to this section for reference")

```
test_async_invoke_matches_output_schema(tool: BaseTool) -> None
```

Test async invoke matches output schema.

If ainvoked with a `ToolCall`, the tool should return a valid `ToolMessage`
content.

For debugging tips, see `test_invoke_matches_output_schema`.

### ``test\_invoke\_no\_tool\_call [¶](https://reference.langchain.com/python/langchain_tests/integration_tests/tools/\#langchain_tests.integration_tests.ToolsIntegrationTests.test_invoke_no_tool_call "Copy anchor link to this section for reference")

```
test_invoke_no_tool_call(tool: BaseTool) -> None
```

Test invoke without `ToolCall`.

If invoked without a `ToolCall`, the tool can return anything
but it shouldn't throw an error.

If this test fails, your tool may not be handling the input you defined
in `tool_invoke_params_example` correctly, and it's throwing an error.

This test doesn't have any checks. It's just to ensure that the tool
doesn't throw an error when invoked with a `dict` of `**kwargs`.

### ``test\_async\_invoke\_no\_tool\_call`async`[¶](https://reference.langchain.com/python/langchain_tests/integration_tests/tools/\#langchain_tests.integration_tests.ToolsIntegrationTests.test_async_invoke_no_tool_call "Copy anchor link to this section for reference")

```
test_async_invoke_no_tool_call(tool: BaseTool) -> None
```

Test async invoke without `ToolCall`.

If ainvoked without a `ToolCall`, the tool can return anything
but it shouldn't throw an error.

For debugging tips, see `test_invoke_no_tool_call`.

Back to top