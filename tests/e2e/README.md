# E2E Test Suite

End-to-end tests that validate the complete system flow: Planner → Executor → Tools → Storage → API → Observability.

## Structure

```
tests/e2e/
├── scenarios/
│   ├── filesystem/      # Filesystem tool tests
│   ├── system/          # System tool tests
│   ├── shell/           # Shell tool tests
│   ├── git/             # Git tool tests
│   ├── github/          # GitHub tool tests
│   ├── multi_tool/      # Multi-tool workflow tests
│   ├── failure_visibility/  # Failure handling tests
│   └── observability/   # Observability tests
├── runner.py            # Test runner and setup
├── assertions.py        # Assertion helpers
└── README.md           # This file
```

## Running Tests

### Run all E2E tests

```bash
pytest tests/e2e/ -v
```

### Run specific scenario

```bash
pytest tests/e2e/scenarios/filesystem/ -v
```

### Run with browser visible (for debugging)

By default, tests run in headless mode. To see the browser:

```bash
# Via pytest option (recomendado)
pytest tests/e2e/ -v --headless=false

# Via environment variable
E2E_HEADLESS=false pytest tests/e2e/ -v
```

**Note**: E2E tests use ONLY the browser - they do NOT call the API directly. The full flow is:
1. Browser: Create run via UI form
2. Browser: Verify plan/execution appear in UI
3. Browser: Navigate to runs list, verify run appears
4. Browser: Click run, verify details
5. Storage: Verify database directly (persistence, status, data)

### Run with backend/frontend management

The test runner automatically starts/stops both backend and frontend. You don't need to start them manually.

## Test Scenarios

### Filesystem Tests

- **test_list_repos.py**: List repositories in ~/repos
- **test_path_validation.py**: Validate path access denial

### System Tests

- **test_system_info.py**: Retrieve system information
- **test_system_tool_no_side_effects.py**: Verify system tool has no side effects

### Shell Tests

- **test_shell_execution.py**: Execute commands in allowed directories
- **test_forbidden_command_fails.py**: Verify forbidden commands are rejected

### Multi-Tool Tests

- **test_multi_tool_workflow.py**: Combine multiple tools in one plan
- **test_analyze_and_propose_workflow.py**: Analyze and propose changes

### Observability Tests

- **test_observability.py**: Validate logs, metrics, and correlation IDs

### Failure Visibility Tests

- **test_failure_visibility.py**: Validate error handling and failure visibility

## Test Requirements

1. **Backend must be startable** (uvicorn available)
2. **LLM must be available** (Ollama running with qwen2.5-coder:7b)
3. **Database will be cleaned** (forge_agent.db will be removed)
4. **Logs will be cleaned** (workspace/logs will be cleared)

## Test Principles

1. **Real execution**: Tests use real tools, not mocks
2. **Full flow**: Tests cover Planner → Executor → Tools → Storage
3. **Observability**: Tests validate logs, metrics, correlation IDs
4. **Failure handling**: Tests validate error visibility and persistence
5. **No silent failures**: All failures must be explicit and traceable

## Adding New Tests

1. Create test file in appropriate scenario directory
2. Use `E2ETestRunner` for setup/teardown
3. Use `E2EAssertions` for assertions
4. Follow existing test patterns
5. Document test purpose in docstring

## Example Test

```python
@pytest.mark.asyncio
async def test_example():
    """Test description."""
    async with E2ETestRunner() as runner:
        assertions = E2EAssertions()
        try:
            await assertions.assert_health()
            run_response = await assertions.create_run("Goal here")
            await assertions.assert_run_success(run_response)
        finally:
            await assertions.close()
```
