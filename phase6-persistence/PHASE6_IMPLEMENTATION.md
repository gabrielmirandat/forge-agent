# Phase 6: Persistence & Run History - Implementation Summary

## Overview

Phase 6 introduces durable storage for plans, executions, and runs without changing the behavior of Planner, Executor, or API semantics. Storage is **passive** - it never affects execution outcomes.

## Core Principles (Enforced)

✅ **Persistence is passive**
- No logic moves into storage
- No decisions are made based on stored data
- Execution behavior MUST NOT change
- Same request → same execution
- Storage never affects outcomes

✅ **Everything must be auditable**
- Every plan, execution, and run is persisted
- Full history is reconstructible
- Complete JSON snapshots stored

✅ **No coupling**
- Planner and Executor remain storage-agnostic
- API orchestrates persistence explicitly
- Storage failures surface explicitly

## Implementation Summary

### 1. ✅ Persistence Abstraction Layer

**Location**: `agent/storage/`

**Storage Interface** (`base.py`):
```python
class Storage(ABC):
    async def save_plan_result(self, plan_result: PlanResult) -> None
    async def save_execution_result(self, execution_result: ExecutionResult) -> None
    async def save_run(
        self,
        plan_result: PlanResult,
        execution_result: ExecutionResult
    ) -> str  # returns run_id

    async def get_run(self, run_id: str) -> RunRecord
    async def list_runs(self, limit: int, offset: int) -> list[RunSummary]
```

**Rules**:
- Storage has no business logic
- Storage never mutates data
- Storage failures surface explicitly
- Exceptions: `StorageError`, `NotFoundError`

---

### 2. ✅ Data Models (Persistence-Level)

**Location**: `agent/storage/models.py`

**RunRecord**:
```python
class RunRecord(BaseModel):
    run_id: str
    plan_id: str
    objective: str
    plan_result: dict  # Full JSON snapshot
    execution_result: dict  # Full JSON snapshot
    created_at: float
```

**RunSummary**:
```python
class RunSummary(BaseModel):
    run_id: str
    plan_id: str
    objective: str
    success: bool
    created_at: float
```

**Rules**:
- Store raw JSON snapshots
- No normalization yet
- No foreign keys beyond IDs
- Separate from runtime schemas

---

### 3. ✅ SQLite Implementation

**Location**: `agent/storage/sqlite.py`

**Features**:
- Single file database (`forge_agent.db`)
- Auto-migration on startup
- JSON stored as TEXT
- No ORM (uses `aiosqlite` directly)

**Tables**:
```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    plan_result TEXT NOT NULL,  -- JSON
    execution_result TEXT NOT NULL,  -- JSON
    created_at REAL NOT NULL
)

CREATE INDEX idx_runs_created_at ON runs(created_at DESC)
```

**Rules**:
- SQLite only
- Auto-migrate on startup
- JSON stored as TEXT
- No ORM

---

### 4. ✅ API Extensions (Read-Only)

**Location**: `api/routes/runs.py`

#### GET /api/v1/runs
**Purpose**: List runs with pagination

**Query Parameters**:
- `limit` (default: 20, max: 100)
- `offset` (default: 0)

**Response**:
```json
{
  "runs": [RunSummary],
  "limit": 20,
  "offset": 0
}
```

#### GET /api/v1/runs/{run_id}
**Purpose**: Get a complete run by ID

**Response**:
```json
{
  "run": RunRecord
}
```

**Rules**:
- No filtering yet
- No deletion
- No update
- No pagination tricks
- Read-only endpoints

---

### 5. ✅ Orchestration Changes

**Location**: `api/routes/run.py`

**Modified /run Endpoint Flow**:
1. `Planner.plan()`
2. `Executor.execute()`
3. `Storage.save_run()` ← NEW
4. Return response

**Rules**:
- Persistence failure → HTTP 500
- Do NOT retry storage
- Do NOT affect execution result
- Storage is passive - failures don't change execution outcome

---

### 6. ✅ Dependency Wiring

**Location**: `api/dependencies.py`

**Storage Dependency**:
```python
def get_storage() -> Storage:
    # Default: SQLiteStorage
    # Singleton
    # Config-driven (future)
```

**Rules**:
- Singleton
- Config-driven (default: SQLite)
- Replaceable in future phases

---

### 7. ✅ Logging & Traceability

**Enhanced Logs**:
- Log `run_id` when run is persisted
- Log persistence success/failure
- Correlate with existing `request_id`
- Full audit trail

**Not Implemented**:
- Metrics
- Tracing systems
- Background writers

---

## Files Created

1. **`agent/storage/base.py`**:
   - Storage abstraction interface
   - Exception classes

2. **`agent/storage/models.py`**:
   - `RunRecord` - Complete run data
   - `RunSummary` - Run summary for listing

3. **`agent/storage/sqlite.py`**:
   - SQLite storage implementation
   - Auto-migration
   - JSON storage

4. **`agent/storage/__init__.py`**:
   - Package exports

5. **`api/routes/runs.py`**:
   - GET /api/v1/runs - List runs
   - GET /api/v1/runs/{run_id} - Get run

6. **`api/schemas/runs.py`**:
   - `RunsListResponse`
   - `RunDetailResponse`

## Files Modified

1. **`api/dependencies.py`**:
   - Added `get_storage()` dependency

2. **`api/routes/run.py`**:
   - Added persistence step after execution
   - Error handling for storage failures

3. **`api/app.py`**:
   - Registered runs router

4. **`requirements.txt`**:
   - Added `aiosqlite>=0.19.0`

---

## Usage Example

### Persist a run (automatic via /run):
```bash
curl -X POST http://localhost:8000/api/v1/run \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Read the file README.md"
  }'
# Run is automatically persisted
```

### List runs:
```bash
curl http://localhost:8000/api/v1/runs?limit=10&offset=0
```

### Get a specific run:
```bash
curl http://localhost:8000/api/v1/runs/{run_id}
```

---

## Acceptance Criteria Met

✅ **Every /run creates a persisted run**
- `/run` endpoint persists after execution
- Run ID generated and logged

✅ **Can list previous runs**
- GET `/api/v1/runs` endpoint implemented
- Pagination support (limit/offset)

✅ **Can retrieve full run by ID**
- GET `/api/v1/runs/{run_id}` endpoint implemented
- Returns complete `RunRecord` with full JSON snapshots

✅ **Planner and Executor unchanged**
- No modifications to Planner
- No modifications to Executor
- Storage is completely separate

✅ **Storage is passive**
- Storage never affects execution behavior
- Same request → same execution (regardless of storage)
- Storage failures don't change outcomes

✅ **Failure modes are explicit**
- `StorageError` for storage failures
- `NotFoundError` for missing runs
- HTTP 500 for persistence failures
- HTTP 404 for not found

---

## Non-Goals (Not Implemented)

❌ Caching
❌ Search / filtering
❌ Deletes
❌ Updates
❌ Background jobs
❌ Streaming
❌ DB migrations framework
❌ ORM
❌ Versioning
❌ Schema evolution logic

---

## Database Schema

**Table: runs**
- `run_id` (TEXT, PRIMARY KEY) - UUID
- `plan_id` (TEXT, NOT NULL) - Plan identifier
- `objective` (TEXT, NOT NULL) - Plan objective
- `plan_result` (TEXT, NOT NULL) - Full PlanResult as JSON
- `execution_result` (TEXT, NOT NULL) - Full ExecutionResult as JSON
- `created_at` (REAL, NOT NULL) - Unix timestamp

**Index**: `idx_runs_created_at` on `created_at DESC` for efficient listing

---

## Notes

- **Passive Storage**: Storage never influences execution - it only records what happened
- **JSON Snapshots**: Full JSON snapshots ensure complete auditability
- **Simple Schema**: No normalization, no foreign keys (beyond IDs)
- **Auto-Migration**: Database tables created automatically on first use
- **Singleton**: Storage instance is singleton, shared across requests
- **Explicit Failures**: All storage failures surface as HTTP errors, never hidden

---

## Next Steps

The persistence layer is complete and ready for:
- Search/filtering (future phase)
- Deletion/archival (future phase)
- Schema evolution (future phase)
- Alternative storage backends (future phase)
- Caching layer (future phase)

Storage remains passive - it never becomes intelligent or affects execution behavior.

