# Forge Agent

A self-hosted, local-first autonomous code agent, inspired by Claude Code, but running entirely on your own machine with full control and privacy.

## Project Structure

```
forge-agent/
├── agent/                  # Core agent implementation
│   ├── config/            # Configuration management
│   ├── llm/               # LLM providers
│   ├── runtime/           # Planner and Executor
│   └── tools/             # Tool implementations
├── api/                   # API layer (future)
├── config/                # Configuration files
├── phase1-model-research/  # Phase 1: Model evaluation
│   ├── tests/             # Evaluation scripts
│   └── outputs/           # Evaluation results
├── phase2-planner/        # Phase 2: Planner implementation
│   ├── tests/             # Planner tests
│   └── outputs/           # Test outputs
├── phase3-executor/       # Phase 3: Executor (future)
│   ├── tests/             # Executor tests
│   └── outputs/           # Test outputs
└── .venv/                 # Virtual environment
```

## Quick Start

### Setup

```bash
# Install python3-venv if needed
sudo apt install python3.12-venv

# Setup environment
./setup_venv.sh
source .venv/bin/activate
```

### Run Tests

```bash
# Phase 2 tests
python3 phase2-planner/tests/test_phase2.py
```

## Phase Status

- ✅ **Phase 1**: Model Research - Complete (qwen2.5-coder:7b selected)
- ✅ **Phase 2**: Planner Implementation - Complete
- ⏳ **Phase 3**: Executor Implementation - Pending

## Documentation

- [Documentation Index](docs/README.md) - Complete documentation index
- [Setup Guide](docs/SETUP.md) - Development environment setup
- [Project Structure](docs/PROJECT_STRUCTURE.md) - Project structure overview
- [Phase 1](phase1-model-research/README.md) - Model evaluation
- [Phase 2](phase2-planner/README.md) - Planner implementation
- [Tests](tests/README.md) - Test suite overview (smoke + E2E)

## Core Principles

- **LLM = Reasoning Engine**: The LLM only proposes plans, never executes
- **Agent = Control Layer**: Deterministic execution logic
- **Tools = Execution**: Only the Executor invokes tools
- **Privacy First**: Everything runs locally, no cloud dependencies
