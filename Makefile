.PHONY: help

# Default target
.DEFAULT_GOAL := help

# Variables
VENV := .venv
PYTHON := python3
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
FRONTEND_DIR := frontend

help: ## Show this help message
	@echo "Forge Agent - Makefile Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

test-smoke: ## Run smoke tests (pytest)
	@echo "Running smoke tests..."
	@$(PYTHON_VENV) -m pytest tests/smoke/ -v

test-e2e: ## Run E2E tests (pytest, headless)
	@echo "Running E2E tests..."
	@$(PYTHON_VENV) -m pytest tests/e2e/ -v -m e2e

test-e2e-visible: ## Run E2E tests with visible browser (debug)
	@echo "Running E2E tests with visible browser..."
	@E2E_HEADLESS=false $(PYTHON_VENV) -m pytest tests/e2e/ -v -s -m e2e

test: ## Run all tests (smoke + E2E)
	@echo "Running all tests..."
	@$(PYTHON_VENV) -m pytest tests/ -v

test-clean: ## Clean test artifacts (database, logs)
	@echo "Cleaning test artifacts..."
	@rm -f forge_agent.db forge_agent.db-journal
	@rm -rf workspace/logs workspace/audit.log
	@echo "✓ Cleanup complete"

# Development helpers
venv: ## Create virtual environment
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
		echo "✓ Virtual environment created"; \
	else \
		echo "Virtual environment already exists"; \
	fi

install: venv ## Install Python dependencies
	@echo "Installing Python dependencies..."
	@$(PIP) install --upgrade pip -q
	@$(PIP) install -r requirements.txt -q
	@echo "✓ Dependencies installed"

install-frontend: ## Install frontend dependencies
	@echo "Installing frontend dependencies..."
	@cd $(FRONTEND_DIR) && npm install
	@echo "✓ Frontend dependencies installed"

# Quick start commands
start-backend: ## Start backend server (development)
	@echo "Starting backend server..."
	@$(VENV)/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

start-frontend: ## Start frontend server (development)
	@echo "Starting frontend server..."
	@cd $(FRONTEND_DIR) && npm run dev

# Health checks
check-ollama: ## Check if Ollama is running
	@if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then \
		echo "✓ Ollama is running"; \
		curl -s http://localhost:11434/api/tags | python3 -m json.tool | grep -E '"name"|"model"' | head -10; \
	else \
		echo "✗ Ollama is not running. Start with: ollama serve"; \
		exit 1; \
	fi

check-model: ## Check if required model is available
	@if curl -s http://localhost:11434/api/tags | grep -q "qwen2.5-coder:7b"; then \
		echo "✓ Model qwen2.5-coder:7b is available"; \
	else \
		echo "✗ Model qwen2.5-coder:7b not found"; \
		echo "  Pull with: ollama pull qwen2.5-coder:7b"; \
		exit 1; \
	fi

# Full setup (one command to rule them all)
setup: install install-frontend ## Full project setup (venv + frontend)
	@echo ""
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Ensure Ollama is running: ollama serve"
	@echo "  2. Ensure model is available: ollama pull qwen2.5-coder:7b"
	@echo "  3. Run smoke tests: make test-smoke"
	@echo "  4. Run E2E tests: make test-e2e"

