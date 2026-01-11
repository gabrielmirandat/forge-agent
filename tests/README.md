# Test Suite Overview

This directory contains comprehensive test suites for the forge-agent system.

## Test Suites

### 1. Smoke Tests (`tests/smoke/`)

**Purpose**: Minimal tests that verify the system is alive and basic paths work.

**Characteristics**:
- Fast (<30 seconds)
- Simple (no deep logic, no tool-specific knowledge)
- Agent-agnostic goals
- Catastrophic failure detection only
- CI-friendly

**Tests**:
- API health check
- Metrics endpoint
- Basic planning (agent-agnostic)
- Basic execution (agent-agnostic)

**Run**:
```bash
pytest tests/smoke/ -v
```


### 2. E2E Tests (`tests/e2e/`)

**Purpose**: Full end-to-end tests that validate complete system flow.

**Characteristics**:
- Real execution (no mocks)
- Full flow: Planner → Executor → Tools → Storage → API → Observability
- Comprehensive coverage (~25-30 scenarios)
- Validates observability, failure handling, multi-tool workflows

**Test Categories**:
- **Filesystem**: List repos, path validation
- **System**: System info, no side effects
- **Shell**: Command execution, forbidden commands
- **Git**: (Placeholder for future Git tests)
- **GitHub**: (Placeholder for future GitHub tests)
- **Multi-tool**: Combined workflows, analyze and propose
- **Observability**: Logs, metrics, correlation IDs
- **Failure Visibility**: Error handling, partial execution

**Run**:
```bash
# All E2E tests
pytest tests/e2e/ -v

# Specific scenario
pytest tests/e2e/scenarios/filesystem/ -v
```

## Test Infrastructure

### E2ETestRunner

Automatically manages backend/frontend lifecycle:
- Starts backend before tests
- Stops backend after tests
- Cleans database and logs
- Context manager for easy use

### E2EAssertions

Helper class with assertion methods:
- `assert_health()`: Backend health check
- `assert_run_success()`: Validate run succeeded
- `assert_run_failure()`: Validate run failed as expected
- `assert_run_persisted()`: Validate run is in storage
- `assert_metrics_incremented()`: Validate metrics
- `assert_logs_have_correlation_ids()`: Validate observability

## Requirements

### For Smoke Tests

- Backend running on `http://localhost:8000` (must be started manually)
- LLM available (Ollama with qwen2.5-coder:7b)

### For E2E Tests

- Backend startable (uvicorn available)
- LLM available (Ollama with qwen2.5-coder:7b)
- Database will be cleaned automatically
- Logs will be cleaned automatically

## Test Principles

1. **Real execution**: Tests use real tools, not mocks
2. **Full flow**: Tests cover complete system flow
3. **Observability**: Tests validate logs, metrics, correlation IDs
4. **Failure handling**: Tests validate error visibility
5. **No silent failures**: All failures must be explicit

## Running All Tests

```bash
# Smoke tests only
pytest tests/smoke/ -v

# E2E tests only
pytest tests/e2e/ -v

# All tests
pytest tests/ -v
```

## Adding New Tests

1. **Smoke tests**: Add to `tests/smoke/` - keep them fast, simple, and agent-agnostic
   - ❌ Do NOT test specific tools
   - ✅ Do test planner + executor wiring
   - ✅ Use agent-agnostic goals
2. **E2E tests**: Add to appropriate scenario directory in `tests/e2e/scenarios/`
3. Use `E2ETestRunner` for E2E tests (automatic backend management)
4. Use `E2EAssertions` for assertions
5. Follow existing test patterns
6. Document test purpose in docstring

## Test Coverage

### Current Coverage

- ✅ API health and metrics
- ✅ Basic planning and execution
- ✅ Filesystem operations (list, path validation)
- ✅ System tool (info, status, no side effects)
- ✅ Shell tool (execution, forbidden commands)
- ✅ Multi-tool workflows
- ✅ Observability (logs, metrics, correlation IDs)
- ✅ Failure visibility

### Future Coverage

- ⏳ Git operations (when Git tool is fully implemented)
- ⏳ GitHub operations (when GitHub tool is fully implemented)
- ⏳ Frontend E2E tests (when needed)
- ⏳ HITL approval flow tests
