# Forge Agent

A self-hosted, local-first autonomous code agent, inspired by Claude Code, but running entirely on your own machine with full control and privacy.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Core Principles](#core-principles)
- [Architecture](#architecture)
- [Tools](#tools)
- [Frontend](#frontend)
- [Testing](#testing)
- [Model Selection](#model-selection)
- [Development](#development)

## Quick Start

### Prerequisites

- Python 3.12+
- `python3-venv` package (for virtual environment)
- Docker and Docker Compose (for LLM inference with Ollama)
- NVIDIA GPU with 12GB+ VRAM (recommended for local LLM)

### Install python3-venv (Ubuntu/Debian)

```bash
sudo apt install python3.12-venv
```

### Setup

```bash
# Create and configure virtual environment
./setup_venv.sh

# Activate virtual environment
source .venv/bin/activate
```

### Start Services

```bash
# Start Ollama (for LLM inference)
cd decisions/model-selection
docker-compose up -d ollama

# Pull the selected model
docker exec phase1-ollama ollama pull qwen2.5-coder:7b

# Start backend API
cd ../..
source .venv/bin/activate
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Start frontend (in another terminal)
cd frontend
npm install
npm run dev
```

### Run Tests

```bash
# Unit tests (fast, no infrastructure)
pytest tests/unit/ -v

# Integration tests (requires API/LLM)
pytest tests/integration/ -v

# E2E tests (comprehensive, browser-based)
pytest tests/e2e/ -v

# All tests
pytest tests/ -v
```

## Project Structure

```
forge-agent/
├── agent/                  # Core agent implementation
│   ├── config/            # Configuration management
│   ├── llm/               # LLM providers (Ollama, LocalAI)
│   ├── runtime/           # Planner and Executor
│   ├── observability/     # Logging, metrics, tracing
│   ├── storage/           # Persistence layer
│   └── tools/             # Tool implementations
├── api/                   # FastAPI REST API
│   ├── routes/            # API endpoints
│   ├── models/            # Request/response models
│   └── schemas/           # API schemas
├── config/                # Configuration files
│   ├── agent.yaml         # Main agent configuration
│   └── example.yaml       # Example configuration
├── frontend/              # React frontend application
├── tests/                 # Test suites
│   ├── unit/              # Unit tests (mocked, no infrastructure)
│   ├── integration/       # Integration tests (real infrastructure)
│   │   └── smoke/         # Smoke tests (minimal, fast)
│   └── e2e/               # End-to-end tests (comprehensive, browser-based)
├── decisions/             # Architecture decisions and research
│   └── model-selection/   # Model evaluation and selection
└── .venv/                 # Virtual environment
```

## Setup

### Option 1: Using setup script (recommended)

```bash
./setup_venv.sh
source .venv/bin/activate
```

### Option 2: Manual setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
source .venv/bin/activate
pytest tests/unit/ -v
```

## Core Principles

- **LLM = Reasoning Engine**: The LLM only proposes plans, never executes
- **Agent = Control Layer**: Deterministic execution logic
- **Tools = Execution**: Only the Executor invokes tools
- **Privacy First**: Everything runs locally, no cloud dependencies
- **Explicit Contracts**: Each tool has documented contracts and security boundaries
- **Fail Fast**: Tools reject invalid operations with explicit errors
- **No Silent Fallbacks**: Tools never silently fallback to other tools

## Architecture

### Components

1. **Planner**: Generates execution plans from user goals using LLM
2. **Executor**: Executes plans step-by-step using tools
3. **Tools**: Structured interfaces for filesystem, system, shell, git, GitHub operations
4. **Storage**: Persists runs, plans, and execution results
5. **API**: REST API for frontend and external integrations
6. **Frontend**: React web interface for interacting with the agent
7. **Observability**: Logging, metrics, and distributed tracing

### Execution Flow

1. User submits goal via API or frontend
2. Planner generates structured plan using LLM
3. Plan is validated against tool contracts
4. Executor executes plan step-by-step
5. Each step invokes appropriate tool
6. Results are persisted and returned
7. Observability data (logs, metrics, traces) is collected

## Tools

The agent provides several tools for different operations:

### Filesystem Tool

**Purpose**: Structured file I/O with path validation

**Operations**:
- `read_file`: Read file contents
- `write_file`: Write file contents
- `list_directory`: List directory contents
- `create_directory`: Create directories
- `delete_file`: Delete files
- `delete_directory`: Delete directories

**Security**: Path validation against allowed paths, no execution semantics

### System Tool

**Purpose**: Side-effect-free system introspection

**Operations**:
- `get_info`: Get system information (OS, Python version, etc.)
- `get_status`: Get agent status and configuration

**Security**: Read-only, no side effects, no file I/O, no execution

### Shell Tool

**Purpose**: Command execution with validation

**Operations**:
- `execute_command`: Execute shell commands

**Security**: Command validation, restricted commands, timeout enforcement

### Git Tool

**Purpose**: Structured Git operations

**Operations**:
- `create_branch`: Create Git branches
- `commit`: Create Git commits
- `get_status`: Get Git repository status

**Security**: Structured API only, no arbitrary commands

### GitHub Tool

**Purpose**: GitHub API operations

**Operations**:
- `create_pull_request`: Create pull requests
- `get_repository_info`: Get repository information

**Security**: Authentication required, structured API only

### Tool Contracts and Priority Rules

**Core Principle**: Prefer structured tools over execution tools. Use execution tools only when execution semantics are required.

**Priority Rules**:
1. **Prefer structured tools** (filesystem, system, git) over execution tools (shell)
2. **Use shell tool ONLY** when execution semantics are required
3. **Never use shell** for operations that structured tools can handle

**Tool Overlaps**:

| Operation | Preferred Tool | Why |
|-----------|---------------|-----|
| Read a file | `filesystem` | Structured, validated, fast |
| List directory | `filesystem` | Structured, validated, fast |
| Execute a script | `shell` | Execution semantics required |
| Get system info | `system` | Side-effect-free, structured |
| Git operations | `git` | Structured API, auditable |
| GitHub API | `github` | Structured API, authenticated |

**Contract Enforcement**:
- Tools reject operations that violate their contract
- Validation: Path validation, command validation, operation validation
- Explicit errors: Failures are explicit and structured, not silent
- Security invariants: Each tool enforces security boundaries

## Frontend

A simple, operator-focused UI for the Forge Agent API.

### Principles

- **Frontend is dumb** - No reasoning, no retries, no decision-making
- **API is single source of truth** - Frontend displays exactly what the API returns
- **Readability > cleverness** - This is an operator UI
- **Every step must be inspectable** - Every failure must be visible

### Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000` and proxies API requests to `http://localhost:8000`.

### Structure

```
frontend/
├── src/
│   ├── api/
│   │   └── client.ts          # Thin API client (no retries, no transformation)
│   ├── pages/
│   │   ├── RunPage.tsx        # Create new run
│   │   ├── RunsListPage.tsx  # Browse historical runs
│   │   └── RunDetailPage.tsx # Inspect single run
│   ├── components/
│   │   ├── PlanViewer.tsx     # Display plan steps
│   │   ├── ExecutionViewer.tsx # Display execution results
│   │   ├── DiagnosticsViewer.tsx # Display planner diagnostics
│   │   └── JsonBlock.tsx      # Pretty-print JSON with copy
│   ├── types/
│   │   └── api.ts             # TypeScript types (mirror API schemas)
│   ├── App.tsx                # Main app with routing
│   └── main.tsx               # Entry point
```

### Features

- **Create Run**: Submit goals and view plan + execution results
- **Browse Runs**: List historical runs with pagination
- **Inspect Run**: View complete run details including plan, diagnostics, and execution
- **No Auto-Refresh**: Manual refresh only
- **No Polling**: No background updates
- **Explicit Failures**: All failures shown clearly, not hidden

### Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Notes

- Frontend never makes decisions - it only displays what the API returns
- Execution failures are shown as normal results (not UI errors)
- Empty plans are clearly labeled
- All data is inspectable via JSON blocks

## Testing

### Test Suites

#### 1. Unit Tests (`tests/unit/`)

**Purpose**: Fast, isolated tests with mocked dependencies. Maximum code coverage.

**Characteristics**:
- No real infrastructure (mocked LLM, storage, tools)
- Fast execution (<5 seconds for full suite)
- High code coverage
- Test individual components in isolation

**Test Coverage**:
- Configuration loading
- Runtime schema validation
- Planner with mocked LLM
- Executor with mocked tools
- Tool registry and tools (filesystem, system)
- Storage models
- Observability (context, logger)
- LLM providers (with mocked HTTP)
- API endpoints (with mocked dependencies)

**Run**:
```bash
pytest tests/unit/ -v
```

#### 2. Integration Tests (`tests/integration/`)

**Purpose**: Tests that use real infrastructure (API, database, LLM).

**Characteristics**:
- Real HTTP requests to API
- Real database operations
- Real LLM calls (Ollama)
- Validates integration between components

**Test Categories**:
- **Smoke Tests** (`tests/integration/smoke/`): Minimal tests that verify the system is alive
  - Fast (<30 seconds)
  - Simple (no deep logic, no tool-specific knowledge)
  - Agent-agnostic goals
  - Catastrophic failure detection only
  - CI-friendly

**What Smoke Tests Do**:
- ✅ Validate: Backend starts and responds to `/health`, Planner can generate a plan, Executor can execute (or handle gracefully), Metrics endpoint responds, Basic API wiring works
- ❌ Do NOT validate: Specific tools (that's E2E territory), Complex workflows, Tool-specific behavior, Detailed observability

**Golden Rule**: **If the test knows the name of a tool, it's NOT a smoke test.**

**Run**:
```bash
# All integration tests
pytest tests/integration/ -v

# Smoke tests only
pytest tests/integration/smoke/ -v
```

#### 3. E2E Tests (`tests/e2e/`)

**Purpose**: Full end-to-end tests that validate complete system flow.

**Characteristics**:
- Real execution (no mocks)
- Full flow: Planner → Executor → Tools → Storage → API → Observability
- Comprehensive coverage (~25-30 scenarios)
- Validates observability, failure handling, multi-tool workflows
- Browser-based (uses Playwright)

**Test Categories**:
- **Filesystem**: List repos, path validation
- **System**: System info, no side effects
- **Shell**: Command execution, forbidden commands
- **Multi-tool**: Combined workflows, analyze and propose
- **Observability**: Logs, metrics, correlation IDs
- **Failure Visibility**: Error handling, partial execution

**Structure**:
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
└── README.md           # E2E test documentation
```

**Run**:
```bash
# All E2E tests
pytest tests/e2e/ -v

# With visible browser (for debugging)
pytest tests/e2e/ -v --headless=false
# Or via environment variable
E2E_HEADLESS=false pytest tests/e2e/ -v

# Specific scenario
pytest tests/e2e/scenarios/filesystem/ -v
```

**Note**: E2E tests use ONLY the browser - they do NOT call the API directly. The full flow is:
1. Browser: Create run via UI form
2. Browser: Verify plan/execution appear in UI
3. Browser: Navigate to runs list, verify run appears
4. Browser: Click run, verify details
5. Storage: Verify database directly (persistence, status, data)

The test runner automatically starts/stops both backend and frontend. You don't need to start them manually.

### Test Infrastructure

#### E2ETestRunner

Automatically manages backend/frontend lifecycle:
- Starts backend before tests
- Stops backend after tests
- Cleans database and logs (once at session start)
- Context manager for easy use

#### E2EAssertions

Helper class with assertion methods:
- `assert_health()`: Backend health check
- `assert_run_success()`: Validate run succeeded
- `assert_run_failure()`: Validate run failed as expected
- `assert_run_persisted()`: Validate run is in storage
- `assert_metrics_incremented()`: Validate metrics
- `assert_logs_have_correlation_ids()`: Validate observability

### Requirements

#### For Unit Tests

- No infrastructure required (all dependencies mocked)
- Fast execution (<5 seconds)

#### For Integration Tests

- Backend running on `http://localhost:8000` (must be started manually)
- LLM available (Ollama with qwen2.5-coder:7b)

#### For E2E Tests

- Backend startable (uvicorn available)
- LLM available (Ollama with qwen2.5-coder:7b)
- Database will be cleaned automatically (once at session start)
- Logs will be cleaned automatically (once at session start)

### Test Principles

1. **Real execution**: Tests use real tools, not mocks (E2E and Integration)
2. **Full flow**: Tests cover complete system flow
3. **Observability**: Tests validate logs, metrics, correlation IDs
4. **Failure handling**: Tests validate error visibility
5. **No silent failures**: All failures must be explicit

### Running All Tests

```bash
# Unit tests only (fast, no infrastructure)
pytest tests/unit/ -v

# Integration tests only (requires API/LLM running)
pytest tests/integration/ -v

# Smoke tests only (minimal integration)
pytest tests/integration/smoke/ -v

# E2E tests only (full browser-based)
pytest tests/e2e/ -v

# All tests
pytest tests/ -v
```

### Adding New Tests

1. **Unit tests**: Add to `tests/unit/` - use mocks, no real infrastructure
   - ✅ Mock all external dependencies (LLM, storage, HTTP)
   - ✅ Test individual components in isolation
   - ✅ Aim for maximum code coverage
2. **Integration tests**: Add to `tests/integration/` - use real infrastructure
   - ✅ Use real API, database, LLM
   - ✅ Test integration between components
   - ✅ Keep smoke tests minimal and fast
3. **E2E tests**: Add to appropriate scenario directory in `tests/e2e/scenarios/`
   - ✅ Use browser automation
   - ✅ Test full user workflows
4. Use `E2ETestRunner` for E2E tests (automatic backend management)
5. Use `E2EAssertions` for assertions
6. Follow existing test patterns
7. Document test purpose in docstring

### Test Coverage

**Current Coverage**:
- ✅ API health and metrics
- ✅ Basic planning and execution
- ✅ Filesystem operations (list, path validation)
- ✅ System tool (info, status, no side effects)
- ✅ Shell tool (execution, forbidden commands)
- ✅ Multi-tool workflows
- ✅ Observability (logs, metrics, correlation IDs)
- ✅ Failure visibility

**Future Coverage**:
- ⏳ Git operations (when Git tool is fully implemented)
- ⏳ GitHub operations (when GitHub tool is fully implemented)
- ⏳ Frontend E2E tests (when needed)
- ⏳ HITL approval flow tests

## Model Selection

The agent uses **qwen2.5-coder:7b** as the LLM for the Planner component. This section documents the evaluation process and rationale.

### Selected Model: qwen2.5-coder:7b

**Rationale**:
1. **Highest JSON validity (96.7%)** - Exceeds the 90% minimum requirement
2. **Best edge case handling** - Only model to achieve 100% on edge case tests
3. **Excellent consistency (88.6%)** - Most reliable across multiple runs
4. **Zero hallucinations in critical tests** - Perfect in most test scenarios
5. **Acceptable inference speed** - 17-44 tokens/s is sufficient for planning tasks

### Evaluation Methodology

Models were evaluated on the following criteria:

1. **JSON Validity** (≥90% required): Percentage of outputs that are valid JSON
2. **Instruction Following** (≥80% required): Compliance with required fields and structure
3. **Task Decomposition** (≥70% required): Logical and executable steps
4. **Tool Hallucination** (≤10% allowed): References to non-existent tools
5. **Consistency** (≥60% required): Similarity across multiple runs
6. **Hardware Fit** (Critical): Must run within 12GB VRAM

### Evaluation Results

| Model | JSON Validity | Instruction Following | Task Decomposition | Tool Hallucination | Consistency | Hardware Fit | Overall Score | Status |
|-------|---------------|----------------------|-------------------|-------------------|-------------|--------------|---------------|--------|
| llama3.1:8b | 80.0% | 80.0% | 80.0% | 0.30% | 71.5% | ✅ | 24/36 | ⚠️ Conditional |
| **qwen2.5-coder:7b** | **96.7%** | **96.7%** | **96.7%** | **0.41%** | **88.6%** | ✅ | **30/36** | ✅ **Selected** |
| deepseek-coder:6.7b | 5.6% | 5.6% | 5.6% | 0.00% | 0.0% | ✅ | 3/36 | ❌ Failed |

### Hardware Performance

| Model | VRAM Usage | RAM Usage | Inference Speed | Load Time |
|-------|------------|-----------|-----------------|-----------|
| llama3.1:8b | ~5-6 GB | ~8-10 GB | 91-95 tokens/s | <5s |
| **qwen2.5-coder:7b** | **~5-6 GB** | **~8-10 GB** | **17-44 tokens/s** | **<5s** |
| deepseek-coder:6.7b | ~4-5 GB | ~7-9 GB | N/A (failed) | <5s |

### Optimal Configuration

```yaml
model: qwen2.5-coder:7b
provider: ollama
temperature: 0.1  # Low temperature for consistency
max_tokens: 2048
top_p: 0.9
```

### Known Limitations

- Slower than llama3.1:8b (but acceptable for planning tasks)
- Minor hallucination in very complex tasks (2.44% in complex multi-step scenarios)

### Mitigation Strategies

- Use low temperature (0.1) for maximum consistency
- Implement JSON validation and retry logic in Planner
- Add edge case handling for complex scenarios
- Monitor hallucination rate in production

### Test Prompts Used

The evaluation used 6 test scenarios:

1. **Simple File Operation**: Basic instruction following and JSON structure
2. **Multi-Tool Workflow**: Task decomposition across multiple tools
3. **Code Repository Analysis**: Reasoning about code structure
4. **Tool Hallucination Test**: Verifying no invented tools
5. **Complex Multi-Step Task**: Handling complex, multi-stage goals
6. **Edge Case Handling**: Invalid tool requests and safety constraints

Each test was run 5 times per model to measure consistency.

### Evaluation Criteria

**Critical Requirements** (all must pass):
- JSON Validity: ≥90%
- Instruction Following: ≥80%
- Task Decomposition: ≥70%
- Tool Hallucination: ≤10%
- Consistency: ≥60%
- Hardware Fit: <11GB VRAM

**Secondary Criteria** (evaluated but not blocking):
- Inference Speed: ≥10 tokens/second
- Code Understanding: Shows understanding of code structure
- Safety Awareness: Recognizes dangerous operations

### Evaluation Hardware

- **GPU**: NVIDIA RTX 5070 (12GB VRAM)
- **RAM**: 32GB
- **OS**: Ubuntu 24.04
- **Inference**: Local only (no cloud, no paid APIs)

### Model Comparison Details

#### llama3.1:8b

**Strengths**:
- Excellent inference speed (91-95 tokens/s)
- Very low hallucination rate (0.30% average)
- Perfect performance on Tests 2, 4, and 5
- Good consistency on most tests

**Weaknesses**:
- JSON validity below minimum (80% vs 90% required)
- Struggles with edge cases (Test 6: 20% valid)
- Some inconsistency in complex tasks

**Best For**: Fast inference scenarios where edge cases are handled separately

#### qwen2.5-coder:7b (Selected)

**Strengths**:
- Highest JSON validity (96.7% average)
- Excellent instruction following (96.7%)
- Perfect edge case handling (Test 6: 100%)
- Best consistency overall (88.6%)
- Zero hallucinations in most tests

**Weaknesses**:
- Slower inference speed (17-44 tokens/s vs 91-95 for llama)
- Slightly higher hallucination rate in complex tasks (2.44% in Test 5)

**Best For**: Production use where reliability and correctness are prioritized

#### deepseek-coder:6.7b

**Strengths**:
- Zero hallucinations across all tests
- Lower VRAM usage (4-5 GB)
- Smallest model size (3.8 GB)

**Weaknesses**:
- Critical failure: Only 5.6% JSON validity
- Failed to generate valid output in most tests
- Zero consistency (all runs different/invalid)
- Not suitable for production use

**Best For**: Not recommended for this use case

### Model Selection Setup

For detailed setup instructions for model evaluation, see `decisions/model-selection/`:

- **QUICK_START.md**: Get started in 5 minutes using Docker
- **SETUP_GUIDE.md**: Complete setup instructions
- **EVALUATION_PLAN.md**: Evaluation methodology
- **EVALUATION_CRITERIA.md**: Success criteria
- **TEST_PROMPTS.md**: Test prompts used
- **COMPARISON_MATRIX.md**: Detailed comparison results
- **RESULTS_TEMPLATE.md**: Results documentation template

### Quick Start for Model Evaluation

Get started with model evaluation in 5 minutes using Docker containers:

```bash
# Step 1: Start Ollama Container
cd decisions/model-selection
docker-compose up -d ollama

# Step 2: Pull a Model
docker exec phase1-ollama ollama pull llama3.1:8b-instruct

# Step 3: Build Evaluator Container
docker-compose --profile evaluator build evaluator
docker-compose --profile evaluator up -d evaluator

# Step 4: Run Your First Test
docker exec phase1-evaluator python evaluate_model.py --model llama3.1:8b-instruct --test 1

# Step 5: Validate Results
docker exec phase1-evaluator python validate_output.py results/llama3.1-8b-instruct/ --test 1
```

### Test Prompts Details

The evaluation used 6 test scenarios, each run 5 times per model:

1. **Simple File Operation**: Basic instruction following and JSON structure
   - Goal: Read a file and create a backup copy
   - Tests: JSON validity, structure compliance, tool accuracy

2. **Multi-Tool Workflow**: Task decomposition across multiple tools
   - Goal: Create branch, modify file, commit, push
   - Tests: Logical ordering, completeness, tool sequencing

3. **Code Repository Analysis**: Reasoning about code structure
   - Goal: Analyze repository and create summary
   - Tests: Understanding of exploration needs, adaptive planning

4. **Tool Hallucination Test**: Verifying no invented tools
   - Goal: Install dependencies and run tests
   - Tests: Only uses available tools, no invented tool names

5. **Complex Multi-Step Task**: Handling complex, multi-stage goals
   - Goal: Refactor module and create PR
   - Tests: Comprehensive planning, logical ordering, completeness

6. **Edge Case Handling**: Invalid tool requests and safety constraints
   - Goal: Destructive operation request
   - Tests: Safety awareness, tool limitation recognition, graceful refusal

Each test was scored on:
- JSON Validity (0-1)
- Structure Compliance (0-1)
- Tool Accuracy (0-1)
- Logical Ordering (0-1)
- Completeness (0-1)
- Consistency (0-1) - Jaccard similarity across 5 runs

## Development

### Development Workflow

1. Activate virtual environment: `source .venv/bin/activate`
2. Make changes
3. Run tests: `pytest tests/ -v`
4. Deactivate when done: `deactivate`

### Requirements

All dependencies are listed in `requirements.txt`:

- `pydantic>=2.5.0` - Schema validation
- `pydantic-settings>=2.1.0` - Configuration management
- `httpx>=0.25.0` - HTTP client for Ollama
- `pyyaml>=6.0` - YAML parsing
- `fastapi>=0.104.0` - API framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `playwright>=1.40.0` - Browser automation for E2E tests
- `pytest>=7.4.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async test support

### Configuration

Main configuration file: `config/agent.yaml`

Key settings:
- LLM provider and model
- Tool configurations
- Security constraints (allowed paths, forbidden commands)
- Observability settings

## License

[Add license information here]
