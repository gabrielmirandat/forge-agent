# Phase 2: Planner Implementation

**Status**: ✅ Complete  
**Date**: 2026-01-07

## Overview

Phase 2 implements the **Planner component** that converts high-level goals into structured, validated execution plans using the selected LLM (qwen2.5-coder:7b via Ollama).

## Implementation Summary

### 1. Planner Output Schema (`agent/runtime/schema.py`)

**Purpose**: Define strict Pydantic schemas for plan validation.

**Key Components**:
- `ToolName` enum: Allowed tool names (filesystem, git, github, shell, system)
- Operation enums: Allowed operations per tool (e.g., `FilesystemOperation`, `GitOperation`)
- `PlanStep`: Single execution step with:
  - `step_id`: Sequential identifier
  - `tool`: Tool name (validated enum)
  - `operation`: Operation name (validated against tool)
  - `arguments`: Operation parameters (dict)
  - `rationale`: Brief explanation
- `Plan`: Complete plan with:
  - `plan_id`: Unique identifier
  - `objective`: Goal description
  - `steps`: Ordered list of steps
  - `estimated_time_seconds`: Optional estimate
  - `notes`: Optional notes

**Validation Features**:
- ✅ Rejects unknown tools
- ✅ Rejects invalid operations for each tool
- ✅ Validates sequential step IDs
- ✅ Type-safe with Pydantic

**Error Handling**:
- `PlanningError`: Base exception for planning failures
- `InvalidPlanError`: Raised when plan validation fails (includes validation error details)

### 2. Ollama LLM Provider (`agent/llm/ollama.py`)

**Purpose**: Minimal LLM provider for Ollama integration.

**Features**:
- Implements `LLMProvider` interface
- Supports chat-style messages (system + user)
- Configurable temperature (default: 0.1 for qwen2.5-coder:7b)
- Configurable timeout and max tokens
- Error handling for HTTP failures

**Configuration**:
```python
{
    "base_url": "http://localhost:11434",  # Ollama API URL
    "model": "qwen2.5-coder:7b",           # Model name
    "temperature": 0.1,                    # Low for determinism
    "timeout": 300                         # Request timeout
}
```

### 3. Planner Core (`agent/runtime/planner.py`)

**Purpose**: Main planning logic with LLM integration and validation.

**Key Methods**:
- `plan(goal, context, retry=True)`: Generate validated plan
  - Builds system + user prompts
  - Calls LLM via provider
  - Extracts JSON from response
  - Validates against schema
  - Retries once on failure

**Prompt Design**:
- **System Prompt**:
  - Lists all available tools and operations
  - Explicitly forbids inventing tools
  - Requires valid JSON only (no markdown)
  - Includes example output
  - Clear constraints and instructions

- **User Prompt**:
  - Goal description
  - Optional context
  - Clear instruction to output JSON only

**JSON Extraction**:
- Handles markdown code blocks (```json ... ```)
- Falls back to direct JSON object extraction
- Raises clear error if no JSON found

**Retry Logic**:
- Retries LLM call once on network/API errors
- Retries validation once with clearer error message
- Raises `PlanningError` or `InvalidPlanError` after retries

**Error Handling**:
- `PlanningError`: LLM request failures
- `InvalidPlanError`: Validation failures (includes error details)

### 4. Module Exports

**Updated Files**:
- `agent/llm/__init__.py`: Exports `LLMProvider`, `OllamaProvider`
- `agent/runtime/__init__.py`: Exports `Planner`, `Plan`, `PlanStep`, exceptions

## Architecture

```
┌─────────────────┐
│   Planner       │
│                 │
│  - plan()       │──┐
│  - prompts      │  │
│  - validation   │  │
└─────────────────┘  │
         │           │
         │ uses      │
         ▼           │
┌─────────────────┐  │
│  LLMProvider    │  │
│  (interface)    │  │
└─────────────────┘  │
         ▲           │
         │           │
┌─────────────────┐  │
│ OllamaProvider  │──┘
│                 │
│  - chat()       │
│  - HTTP client  │
└─────────────────┘
         │
         │ HTTP
         ▼
┌─────────────────┐
│   Ollama API    │
│  (localhost)    │
└─────────────────┘
```

## Usage Example

```python
from agent.config.loader import ConfigLoader
from agent.llm.ollama import OllamaProvider
from agent.runtime.planner import Planner

# Load config
config_loader = ConfigLoader()
config = config_loader.load()

# Create LLM provider
llm_config = {
    "base_url": config.llm.base_url,
    "model": "qwen2.5-coder:7b",
    "temperature": 0.1,
    "timeout": 300
}
llm = OllamaProvider(llm_config)

# Create planner
planner = Planner(config, llm)

# Generate plan
goal = "Read src/main.py and create a backup at src/main.py.backup"
plan = await planner.plan(goal)

# Plan is validated and ready for execution
print(f"Plan ID: {plan.plan_id}")
print(f"Objective: {plan.objective}")
for step in plan.steps:
    print(f"  Step {step.step_id}: {step.tool.value}.{step.operation}")
```

## Validation Rules

1. **Tool Validation**: Only tools in `ToolName` enum are allowed
2. **Operation Validation**: Operations must match allowed list for each tool
3. **Step ID Validation**: Step IDs must be sequential starting from 1
4. **JSON Format**: Output must be valid JSON (handles markdown code blocks)
5. **Required Fields**: All required fields must be present

## Error Scenarios

1. **Invalid Tool**: `ValueError` - Tool not in allowed list
2. **Invalid Operation**: `ValueError` - Operation not allowed for tool
3. **Invalid JSON**: `InvalidPlanError` - No valid JSON found in LLM output
4. **Validation Failure**: `InvalidPlanError` - Plan doesn't match schema
5. **LLM Failure**: `PlanningError` - LLM request failed after retry

## Next Steps

✅ Phase 2 Complete: Planner implementation done

**Phase 3 (Future)**: Executor implementation
- Consume validated plans
- Execute steps via ToolRegistry
- Handle errors and rollback
- Return execution results

## Files Created/Modified

**Created**:
- `agent/runtime/schema.py` - Plan schemas and validation
- `agent/llm/ollama.py` - Ollama provider implementation

**Modified**:
- `agent/runtime/planner.py` - Full implementation
- `agent/llm/__init__.py` - Added exports
- `agent/runtime/__init__.py` - Added exports

## Testing Notes

- Schema validation tested manually (requires dependencies)
- LLM integration requires running Ollama instance
- Error handling tested via exception types
- Retry logic implemented but not yet tested end-to-end

## Dependencies

All required dependencies are already in `requirements.txt`:
- `pydantic>=2.5.0` - Schema validation
- `httpx>=0.25.0` - HTTP client for Ollama
- `pydantic-settings>=2.1.0` - Configuration management

