# Phase 2 Planner Hardening - Implementation Summary

## Overview

This document summarizes the structural improvements made to the Planner component to enhance robustness, auditability, and determinism. These improvements are **NOT Phase 2.5 or Phase 3** - they are hardening improvements to the existing Phase 2 implementation.

## Improvements Implemented

### 1. ✅ PlannerDiagnostics (Mandatory)

**Location**: `agent/runtime/schema.py`

**Description**: A lightweight diagnostics object that captures execution metadata for debugging and auditing without affecting execution.

**Fields**:
- `model_name`: LLM model name used
- `temperature`: Temperature setting used
- `retries_used`: Number of retries attempted
- `raw_llm_response`: Raw LLM output (preserved for inspection)
- `extracted_json`: Extracted JSON string (None if extraction failed)
- `validation_errors`: Validation errors encountered (None if validation succeeded)

**Usage**: Returned as part of `PlanResult` alongside the plan, or attached to exceptions for debugging.

**Example**:
```python
result = await planner.plan("Read file test.txt")
print(f"Model: {result.diagnostics.model_name}")
print(f"Retries: {result.diagnostics.retries_used}")
print(f"Raw response: {result.diagnostics.raw_llm_response}")
```

---

### 2. ✅ Empty Plan as First-Class Outcome

**Location**: `agent/runtime/schema.py` - `Plan` class

**Description**: Support for valid "empty plan" outcomes where the Planner determines no action is needed.

**Requirements**:
- `steps` must be an empty list
- `objective` is still required
- `notes` MUST explain why no action is taken (validated by schema)

**Validation**: The schema includes a `@field_validator` that enforces `notes` is required when `steps` is empty.

**Example**:
```python
empty_plan = Plan(
    plan_id="no-action-needed",
    objective="Check if file exists",
    steps=[],
    notes="File already exists and is up to date. No action required."
)
```

**Prompt Update**: The system prompt now explicitly mentions empty plans as a valid outcome and provides an example.

---

### 3. ✅ Deterministic Plan ID

**Location**: `agent/runtime/planner.py` - `_generate_plan_id()` method

**Description**: Replace random UUID-only plan IDs with a deterministic strategy based on:
- Normalized objective (lowercase, stripped)
- Normalized context (sorted keys, JSON stringified)
- Timestamp bucket (rounded to nearest minute for stability)

**Benefits**:
- Same goal + context in same minute = same ID (stable across retries)
- Different goals or contexts = different IDs
- Unique enough for tracing and logging

**Implementation**:
```python
def _generate_plan_id(self, goal: str, context: Optional[Dict[str, Any]] = None) -> str:
    normalized_goal = goal.lower().strip()
    normalized_context = json.dumps(context, sort_keys=True) if context else ""
    timestamp_bucket = int(time.time() // 60)
    hash_input = f"{normalized_goal}|{normalized_context}|{timestamp_bucket}"
    plan_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    return f"plan-{plan_hash}"
```

**Note**: The Planner overrides any `plan_id` from the LLM output with the deterministic one.

---

### 4. ✅ Stricter JSON Extraction Rules

**Location**: `agent/runtime/planner.py` - `_extract_json()` method

**Description**: Hardened JSON extraction logic with strict rules:

**Rules**:
- If NO JSON object is found → `JSONExtractionError`
- If MORE THAN ONE JSON object is found → `JSONExtractionError`
- If JSON is wrapped in markdown code blocks → extract it safely
- The raw LLM response must always be preserved in diagnostics

**Implementation Details**:
- First checks for JSON in markdown code blocks (```json ... ```)
- If no code blocks, searches for complete JSON objects by counting braces
- Fails fast with clear error messages if multiple or no JSON objects found
- Does NOT attempt to merge or guess between multiple JSON objects

**Example Errors**:
```python
# No JSON found
JSONExtractionError("No JSON object found in LLM output...")

# Multiple JSON objects
JSONExtractionError("Multiple JSON objects found (2 found)...")
```

---

### 5. ✅ Explicit Failure Modes

**Location**: `agent/runtime/schema.py` - Exception classes

**Description**: Clear distinction between different failure types with specific exception classes.

**Exception Hierarchy**:
```
PlanningError (base)
├── LLMCommunicationError
├── JSONExtractionError
└── InvalidPlanError
```

**Exception Details**:

1. **LLMCommunicationError**: Raised when LLM communication fails (network, timeout, etc.)
   - Includes `diagnostics` attribute with execution metadata

2. **JSONExtractionError**: Raised when JSON extraction fails
   - Includes `raw_response` attribute with the raw LLM output
   - Includes `diagnostics` attribute

3. **InvalidPlanError**: Raised when plan validation fails (schema, tool validation, etc.)
   - Includes `validation_errors` list with detailed error messages
   - Includes `diagnostics` attribute

**Usage**:
```python
try:
    result = await planner.plan("goal")
except LLMCommunicationError as e:
    print(f"LLM failed: {e}")
    print(f"Diagnostics: {e.diagnostics}")
except JSONExtractionError as e:
    print(f"JSON extraction failed: {e}")
    print(f"Raw response: {e.raw_response}")
except InvalidPlanError as e:
    print(f"Validation failed: {e}")
    print(f"Errors: {e.validation_errors}")
```

---

### 6. ✅ PlanResult Return Type

**Location**: `agent/runtime/schema.py` - `PlanResult` class

**Description**: The `Planner.plan()` method now returns `PlanResult` instead of `Plan` directly.

**Structure**:
```python
class PlanResult(BaseModel):
    plan: Plan  # The generated plan (may be empty)
    diagnostics: PlannerDiagnostics  # Execution diagnostics
```

**Benefits**:
- Always provides diagnostics alongside the plan
- Enables auditing and debugging without affecting execution
- Clear separation between plan data and execution metadata

**Usage**:
```python
result = await planner.plan("Read file test.txt")
plan = result.plan  # Access the plan
diagnostics = result.diagnostics  # Access diagnostics
```

---

## Updated Components

### Files Modified

1. **`agent/runtime/schema.py`**:
   - Added `PlannerDiagnostics` class
   - Added `PlanResult` class
   - Updated `Plan` class to support empty plans
   - Added `@field_validator` for empty plan notes
   - Enhanced exception classes with diagnostics support

2. **`agent/runtime/planner.py`**:
   - Added `_generate_plan_id()` method for deterministic IDs
   - Updated `_extract_json()` with stricter rules
   - Updated `plan()` method to return `PlanResult`
   - Integrated diagnostics tracking throughout execution
   - Enhanced error handling with specific exception types

3. **`agent/runtime/__init__.py`**:
   - Exported new classes: `PlanResult`, `PlannerDiagnostics`, `LLMCommunicationError`, `JSONExtractionError`

### Files Created

1. **`phase2-planner/tests/test_phase2_hardening.py`**:
   - Comprehensive test suite for all hardening improvements
   - Validates diagnostics, empty plans, deterministic IDs, JSON extraction, and failure modes

---

## Testing

All improvements have been validated with comprehensive tests:

```bash
source .venv/bin/activate
python3 phase2-planner/tests/test_phase2_hardening.py
```

**Test Results**: ✅ All tests pass

- ✅ PlannerDiagnostics creation and usage
- ✅ Empty plan validation (with and without notes)
- ✅ Deterministic Plan ID generation
- ✅ Stricter JSON extraction (single, multiple, none)
- ✅ Explicit failure modes (all exception types)
- ✅ PlanResult creation and usage

---

## Acceptance Criteria Met

✅ **The Planner always returns either**:
- a valid Plan (via `PlanResult`)
- or a valid empty Plan (via `PlanResult`)
- or a clear, typed error (`LLMCommunicationError`, `JSONExtractionError`, `InvalidPlanError`)

✅ **All Planner decisions are auditable**:
- Diagnostics capture model name, temperature, retries, raw response, extracted JSON, and validation errors
- Diagnostics are always available (in `PlanResult` or attached to exceptions)

✅ **Raw LLM output is always inspectable**:
- Preserved in `PlannerDiagnostics.raw_llm_response`
- Available even when extraction or validation fails

✅ **Phase 2 is stronger, but still Phase 2**:
- No Executor logic added
- No new tools added
- No architecture changes
- Only robustness, auditability, and determinism improvements

---

## Backward Compatibility

**Breaking Changes**:
- `Planner.plan()` now returns `PlanResult` instead of `Plan` directly
  - **Migration**: Access `result.plan` instead of using the return value directly

**Non-Breaking**:
- All existing schema validation rules remain
- All existing tool/operation validation remains
- All existing prompt structure remains

---

## Next Steps

The Planner is now hardened and ready for:
- Integration with Executor (Phase 3)
- Production use with full auditability
- Debugging and troubleshooting with comprehensive diagnostics

---

## Notes

- The deterministic Plan ID uses a 1-minute timestamp bucket for stability across retries
- Empty plans are a first-class outcome, not an error condition
- All exceptions include diagnostics for complete auditability
- JSON extraction is strict: exactly one JSON object is required, no guessing or merging

