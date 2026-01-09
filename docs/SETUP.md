# Setup Guide

## Prerequisites

- Python 3.12+ 
- `python3-venv` package (for virtual environment)

### Install python3-venv (Ubuntu/Debian)

```bash
sudo apt install python3.12-venv
```

## Quick Setup

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

## Verify Installation

Run the test script:

```bash
source .venv/bin/activate
python3 test_phase2.py
```

## Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run Phase 2 tests
python3 test_phase2.py
```

## Development Workflow

1. Activate virtual environment: `source .venv/bin/activate`
2. Make changes
3. Run tests: `python3 test_phase2.py`
4. Deactivate when done: `deactivate`

## Requirements

All dependencies are listed in `requirements.txt`:

- `pydantic>=2.5.0` - Schema validation
- `pydantic-settings>=2.1.0` - Configuration management
- `httpx>=0.25.0` - HTTP client for Ollama
- `pyyaml>=6.0` - YAML parsing
- `fastapi>=0.104.0` - API framework (for future phases)
- `uvicorn[standard]>=0.24.0` - ASGI server (for future phases)

