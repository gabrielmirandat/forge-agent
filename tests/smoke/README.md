# Smoke Tests

**Purpose**: Minimal tests that verify the system is alive and basic paths work.

## Principles

Smoke tests are:
- **Fast**: Run in <30 seconds
- **Simple**: No deep logic, no tool-specific knowledge
- **Portable**: No hardcoded paths, CI-friendly
- **Catastrophic failure detection**: Catch only major issues

## What Smoke Tests Do

✅ Validate:
- Backend starts and responds to `/health`
- Planner can generate a plan
- Executor can execute (or handle gracefully)
- Metrics endpoint responds
- Basic API wiring works

❌ Do NOT validate:
- Specific tools (that's E2E territory)
- Complex workflows
- Tool-specific behavior
- Detailed observability

## Running

```bash
# Run all smoke tests
pytest tests/smoke/ -v

# Run specific test
pytest tests/smoke/test_health.py -v
```

## Requirements

- Backend must be running on `http://localhost:8000`
- LLM must be available (Ollama with qwen2.5-coder:7b)

## Golden Rule

**If the test knows the name of a tool, it's NOT a smoke test.**

Smoke tests validate:
- ✅ Health endpoints
- ✅ Planner responds
- ✅ Executor doesn't explode

E2E tests validate:
- ✅ Tools (filesystem, shell, system, git, github)
- ✅ Observability
- ✅ Real failures
- ✅ Multi-tool workflows

## See Also

- `tests/e2e/`: Full E2E test suite
