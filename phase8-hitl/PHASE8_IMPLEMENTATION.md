# Phase 8: Human-in-the-Loop (HITL) - Implementation Summary

## Overview

Phase 8 introduces explicit human approval between planning and execution. The system remains deterministic and auditable. Humans approve — they do not edit or reason.

## Core Principles (Enforced)

✅ **Planner still plans**
- No logic moves out of Planner
- No plan mutation
- Planner behavior unchanged

✅ **Executor still executes**
- Executor only runs approved plans
- No awareness of humans
- Executor behavior unchanged

✅ **Humans approve, never edit**
- Humans cannot modify plans
- Humans can only approve or reject
- No plan mutation

✅ **No hidden automation**
- No auto-approve
- No silent execution
- Approval is explicit and persisted

✅ **Backward compatible**
- HITL is optional
- Existing flows continue to work
- Default: HITL disabled (Phase 7 behavior)

## Implementation Summary

### 1. ✅ Approval State Model

**Location**: `agent/storage/models.py`

**ApprovalStatus Enum**:
```python
class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
```

**Extended RunRecord**:
```python
class RunRecord(BaseModel):
    # ... existing fields ...
    approval_status: ApprovalStatus = PENDING
    approval_reason: Optional[str] = None
    approved_at: Optional[float] = None
    approved_by: Optional[str] = None
    execution_result: Optional[dict] = None  # None if not executed
```

**Rules**:
- Default = PENDING
- Immutable once APPROVED
- REJECTED stops execution permanently
- execution_result can be None (not executed yet)

---

### 2. ✅ Configuration

**Location**: `agent/config/loader.py`

**HumanInTheLoopConfig**:
```python
class HumanInTheLoopConfig(BaseModel):
    enabled: bool = False  # Default: disabled
```

**AgentConfig Extended**:
```python
class AgentConfig(BaseModel):
    # ... existing fields ...
    human_in_the_loop: HumanInTheLoopConfig
```

**Rules**:
- Disabled by default
- When disabled → Phase 7 behavior unchanged
- When enabled → HITL workflow active

---

### 3. ✅ Storage Changes

**Location**: `agent/storage/sqlite.py`

**Schema Migration**:
- Added columns: `approval_status`, `approval_reason`, `approved_at`, `approved_by`
- `execution_result` now nullable (TEXT → can be NULL)
- Auto-migration on startup (ALTER TABLE with try/except)

**New Methods**:
- `update_run_approval(run_id, status, approved_by, reason)` - Update approval status
- `update_run_execution(run_id, execution_result)` - Update execution after approval

**Rules**:
- Never delete old data
- Safe migration (try/except for existing columns)
- Backward compatible with existing runs

---

### 4. ✅ Orchestration Changes

**Location**: `api/routes/run.py`

**Modified /run Endpoint Flow**:

**When HITL enabled**:
1. `Planner.plan()` ← Plan only
2. `Storage.save_run(plan_result, None, PENDING)` ← Persist with PENDING
3. Return plan only (no execution)

**When HITL disabled** (Phase 7 behavior):
1. `Planner.plan()`
2. `Executor.execute()`
3. `Storage.save_run(plan_result, execution_result)`
4. Return plan + execution

**Rules**:
- Execution must NOT occur until approval
- Plan is persisted immediately
- Response includes plan only (execution_result = null)

---

### 5. ✅ New API Endpoints

**Location**: `api/routes/approval.py`

#### POST /api/v1/runs/{run_id}/approve
**Purpose**: Approve a planned run and trigger execution

**Request**:
```json
{
  "approved_by": "string",
  "reason": "optional string",
  "execution_policy": { ...optional }
}
```

**Behavior**:
1. Validate run exists and is PENDING
2. Mark as APPROVED
3. Persist approval metadata
4. Trigger execution
5. Persist execution result
6. Return approval status + execution result

**Response**:
```json
{
  "run_id": "string",
  "approval_status": "approved",
  "execution_result": { ...ExecutionResult }
}
```

#### POST /api/v1/runs/{run_id}/reject
**Purpose**: Reject a planned run

**Request**:
```json
{
  "rejected_by": "string",
  "reason": "string"  // Required
}
```

**Behavior**:
1. Validate run exists and is PENDING
2. Mark as REJECTED
3. Persist reason
4. Execution is never triggered

**Response**:
```json
{
  "status": "rejected",
  "run_id": "string"
}
```

**Error Handling**:
- Run not found → HTTP 404
- Run not PENDING → HTTP 409 (Conflict)
- Storage failure → HTTP 500

---

### 6. ✅ Frontend Changes

**Location**: `frontend/src/`

#### ApprovalPanel Component
**Purpose**: UI for approving/rejecting runs

**Features**:
- Shows approval status (PENDING/APPROVED/REJECTED)
- Approve button (requires name/email)
- Reject button (requires name/email + reason)
- Displays approver and timestamp when approved/rejected

#### RunPage Updates
- Shows "Awaiting Approval" banner when execution_result is null
- Displays plan immediately (even without execution)

#### RunDetailPage Updates
- ApprovalPanel component integrated
- Shows approval status in metadata
- Execution section only shown if execution_result exists
- Clear messaging for pending/rejected states

**UX Rules**:
- ✅ No auto-approve
- ✅ No auto-execute
- ✅ Approval requires explicit click
- ✅ Rejection requires reason
- ✅ Status visually obvious

---

### 7. ✅ TypeScript Types

**Location**: `frontend/src/types/api.ts`

**Updated Types**:
- `RunResponse.execution_result` → `ExecutionResult | null`
- `RunRecord.execution_result` → `ExecutionResult | null`
- Added `ApprovalStatus` type
- Added approval fields to `RunRecord`

---

## Files Modified

1. **`agent/storage/models.py`**:
   - Added `ApprovalStatus` enum
   - Extended `RunRecord` with approval fields

2. **`agent/storage/base.py`**:
   - Added `update_run_approval()` method
   - Added `update_run_execution()` method

3. **`agent/storage/sqlite.py`**:
   - Updated schema with approval columns
   - Auto-migration logic
   - Implemented approval update methods

4. **`agent/config/loader.py`**:
   - Added `HumanInTheLoopConfig`
   - Extended `AgentConfig`

5. **`api/routes/run.py`**:
   - Modified to check HITL config
   - Plan-only flow when HITL enabled

6. **`api/routes/approval.py`** (NEW):
   - Approve endpoint
   - Reject endpoint

7. **`api/schemas/run.py`**:
   - `RunResponse.execution_result` → Optional

8. **`api/schemas/approval.py`** (NEW):
   - `ApproveRequest` / `ApproveResponse`
   - `RejectRequest`

9. **`frontend/src/types/api.ts`**:
   - Updated types for HITL

10. **`frontend/src/api/client.ts`**:
    - Added `approveRun()` function
    - Added `rejectRun()` function

11. **`frontend/src/components/ApprovalPanel.tsx`** (NEW):
    - Approval UI component

12. **`frontend/src/pages/RunPage.tsx`**:
    - Shows "Awaiting Approval" banner

13. **`frontend/src/pages/RunDetailPage.tsx`**:
    - Integrated ApprovalPanel
    - Shows approval status
    - Conditional execution display

---

## Usage Example

### Enable HITL in config:
```yaml
agent:
  human_in_the_loop:
    enabled: true
```

### Create a run (HITL enabled):
```bash
curl -X POST http://localhost:8000/api/v1/run \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Read the file README.md"
  }'
# Returns plan only, execution_result = null
```

### Approve a run:
```bash
curl -X POST http://localhost:8000/api/v1/runs/{run_id}/approve \
  -H "Content-Type: application/json" \
  -d '{
    "approved_by": "user@example.com",
    "reason": "Plan looks safe",
    "execution_policy": {
      "max_retries_per_step": 1,
      "rollback_on_failure": false
    }
  }'
# Triggers execution and returns result
```

### Reject a run:
```bash
curl -X POST http://localhost:8000/api/v1/runs/{run_id}/reject \
  -H "Content-Type: application/json" \
  -d '{
    "rejected_by": "user@example.com",
    "reason": "Plan modifies critical files"
  }'
# Marks as rejected, no execution
```

---

## Acceptance Criteria Met

✅ **Can create a run without executing it**
- `/run` endpoint returns plan only when HITL enabled
- execution_result is null

✅ **Plan is visible before execution**
- Plan returned immediately
- Frontend displays plan before approval

✅ **Human can approve → execution happens**
- `/approve` endpoint triggers execution
- Execution result persisted

✅ **Human can reject → execution never happens**
- `/reject` endpoint marks as rejected
- No execution triggered

✅ **Approval is persisted and auditable**
- Approval status, reason, timestamp, approver all stored
- Full audit trail

✅ **No plan mutation**
- Plans are immutable
- Humans cannot edit plans
- Only approve/reject

✅ **No executor changes**
- Executor unchanged
- Executor doesn't know about approvals
- Just executes approved plans

✅ **Backward compatible**
- HITL disabled by default
- Phase 7 behavior preserved
- Existing flows continue to work

---

## Non-Goals (Not Implemented)

❌ Authentication / RBAC
❌ Multiple approvers
❌ Partial approvals
❌ Editing plans
❌ Time-based auto approval
❌ Notifications
❌ Comments threads

---

## Database Schema

**Updated Table: runs**
- `approval_status` (TEXT, DEFAULT 'pending')
- `approval_reason` (TEXT, NULLABLE)
- `approved_at` (REAL, NULLABLE)
- `approved_by` (TEXT, NULLABLE)
- `execution_result` (TEXT, NULLABLE) ← Now nullable

**Migration**: Auto-migrated on startup (safe, backward compatible)

---

## Notes

- **Explicit Control**: Humans have explicit control over execution
- **No Intelligence**: Approval is not intelligent - it's a gate, not reasoning
- **Immutable Plans**: Plans cannot be edited, only approved/rejected
- **Auditable**: Complete approval trail with who, when, why
- **Backward Compatible**: Default behavior unchanged (HITL disabled)
- **Deterministic**: Same plan + same approval → same execution

---

## Next Steps

The HITL system is complete and ready for:
- Authentication/authorization (future phase)
- Multiple approvers (future phase)
- Approval workflows (future phase)
- Notifications (future phase)

HITL remains a control mechanism - it never becomes intelligent or starts making decisions.

