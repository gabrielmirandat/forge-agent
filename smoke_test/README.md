# Smoke Test - Manual Execution Guide

This document provides step-by-step instructions for manually executing the smoke test validation.

## Prerequisites

### Environment Setup

1. **Python Virtual Environment**
   ```bash
   cd /home/gabriel-miranda/repos/forge-agent
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Playwright Browser Installation**
   ```bash
   playwright install chromium
   ```

3. **Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Ollama (for LLM calls)**
   - Ensure Ollama is running: `ollama serve`
   - Ensure model is pulled: `ollama pull qwen2.5-coder:7b`

## Automated Execution

### Run Full Smoke Test

```bash
cd /home/gabriel-miranda/repos/forge-agent
source .venv/bin/activate
python smoke_test/run_smoke_test.py
```

**Expected Output:**
- Backend starts
- Frontend starts
- Browser opens (headful mode)
- API calls succeed
- UI validation passes
- Exit code: 0 (PASS) or 1 (FAIL)

## Manual Execution Steps

### Step 1: Start Backend

**Terminal 1:**
```bash
cd /home/gabriel-miranda/repos/forge-agent
source .venv/bin/activate
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Expected:**
- Server starts without errors
- Output shows: `Uvicorn running on http://0.0.0.0:8000`

**Validation:**
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": <number>
}
```

**Metrics Check:**
```bash
curl http://localhost:8000/metrics
```

**Expected:**
- Prometheus format output
- Contains `api_requests_total`
- Contains `planner_requests_total`

---

### Step 2: Start Frontend

**Terminal 2:**
```bash
cd /home/gabriel-miranda/repos/forge-agent/frontend
npm run dev
```

**Expected:**
- Server starts
- Output shows: `Local: http://localhost:3000`

**Validation:**
- Open browser: http://localhost:3000
- Page loads without errors
- No console errors (check browser DevTools)

---

### Step 3: API Smoke Test (Success Case)

**Terminal 3:**
```bash
curl -X POST http://localhost:8000/api/v1/run \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "List files in the current directory"
  }'
```

**Expected Response:**
```json
{
  "plan_result": {
    "plan": {
      "plan_id": "...",
      "objective": "List files in the current directory",
      "steps": [...]
    },
    "diagnostics": {...}
  },
  "execution_result": {
    "plan_id": "...",
    "objective": "...",
    "steps": [...],
    "success": true,
    "started_at": <timestamp>,
    "finished_at": <timestamp>
  }
}
```

**Validation Points:**
- HTTP 200 status
- `plan_result.plan` exists
- `plan_result.plan.steps` is an array
- `execution_result` exists
- `execution_result.success = true`

**Backend Logs (Terminal 1):**
- Should show structured JSON logs
- Look for: `"event": "api.request.started"`
- Look for: `"event": "planner.plan.started"`
- Look for: `"event": "executor.execution.started"`
- Look for: `"event": "storage.run.saved"`
- All logs should include `request_id` and `run_id`

---

### Step 4: Persistence Check

**List Runs:**
```bash
curl http://localhost:8000/api/v1/runs
```

**Expected Response:**
```json
{
  "runs": [
    {
      "run_id": "...",
      "plan_id": "...",
      "objective": "...",
      "success": true,
      "created_at": <timestamp>
    }
  ],
  "limit": 20,
  "offset": 0
}
```

**Get Specific Run:**
```bash
# Replace {run_id} with actual run_id from list
curl http://localhost:8000/api/v1/runs/{run_id}
```

**Expected Response:**
```json
{
  "run": {
    "run_id": "...",
    "plan_id": "...",
    "objective": "...",
    "plan_result": {...},
    "execution_result": {...},
    "created_at": <timestamp>,
    "approval_status": "pending"
  }
}
```

**Database Inspection:**
```bash
sqlite3 forge_agent.db "SELECT run_id, objective, approval_status, created_at FROM runs ORDER BY created_at DESC LIMIT 5;"
```

**Expected:**
- Run appears in database
- `plan_result` and `execution_result` stored as JSON TEXT

---

### Step 5: UI Validation

**Browser Steps:**

1. **Open Runs List:**
   - Navigate to: http://localhost:3000/runs
   - **Expected:** Table/list of runs visible
   - **Verify:** Run from Step 3 appears in list

2. **Open Run Detail:**
   - Click on a run in the list
   - **Expected:** Navigate to `/runs/{run_id}`
   - **Verify:**
     - Plan section visible
     - Plan steps rendered
     - Execution section visible (if executed)
     - Execution steps rendered
     - Timestamps visible
     - Raw JSON visible (via JsonBlock component)

3. **Verify No Inferred State:**
   - **Check:** UI shows exactly what API returns
   - **Check:** No optimistic UI updates
   - **Check:** No auto-refresh

---

### Step 6: Observability Validation

**Metrics Check:**
```bash
curl http://localhost:8000/metrics | grep -E "(api_requests_total|planner_requests_total|execution_runs_total|execution_steps_total|storage_operations_total)"
```

**Expected Output:**
```
api_requests_total{endpoint="/api/v1/run",method="POST",status="200"} 1.0
planner_requests_total{status="success"} 1.0
execution_runs_total{status="success"} 1.0
execution_steps_total{tool="system",operation="get_status",status="success"} 1.0
storage_operations_total{operation="save_run",status="success"} 1.0
```

**Log Validation:**
- Check Terminal 1 (backend logs)
- **Verify:** All logs are JSON format
- **Verify:** All logs include `request_id`
- **Verify:** Logs after run creation include `run_id`
- **Verify:** Logs include `plan_id`
- **Verify:** Key events logged:
  - `api.request.started`
  - `planner.plan.started`
  - `planner.plan.completed`
  - `executor.execution.started`
  - `executor.step.started`
  - `executor.step.completed`
  - `executor.execution.stopped`
  - `storage.run.saved`

---

### Step 7: Failure Case Test

**Create Failure Run:**
```bash
curl -X POST http://localhost:8000/api/v1/run \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Delete a file that does not exist: /nonexistent/file.txt"
  }'
```

**Expected Response:**
```json
{
  "plan_result": {...},
  "execution_result": {
    "success": false,
    "stopped_at_step": <number>,
    "steps": [
      {
        "step_id": <number>,
        "success": false,
        "error": "..."
      }
    ]
  }
}
```

**Validation Points:**
- HTTP 200 (execution failure is NOT an HTTP error)
- `execution_result.success = false`
- `execution_result.steps[].error` contains error message
- Run is persisted despite failure

**Backend Logs:**
- Should show: `"event": "executor.step.failed"`
- Should show: `"level": "ERROR"`
- Error message should be in log

**UI Validation:**
- Navigate to failure run in browser
- **Verify:** Failure is visible
- **Verify:** Error message displayed
- **Verify:** No inferred success state

---

### Step 8: HITL Flow Test (Optional)

**Enable HITL:**

Create/update `config/agent.yaml`:
```yaml
agent:
  human_in_the_loop:
    enabled: true
```

**Restart Backend** (Terminal 1: Ctrl+C, then restart)

**Create Run (HITL Enabled):**
```bash
curl -X POST http://localhost:8000/api/v1/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "List files"}'
```

**Expected Response:**
```json
{
  "plan_result": {...},
  "execution_result": null
}
```

**Validation:**
- `execution_result = null`
- Backend logs show: `"event": "approval.pending"`

**Approve Run:**
```bash
# Replace {run_id} with actual run_id
curl -X POST http://localhost:8000/api/v1/runs/{run_id}/approve \
  -H "Content-Type: application/json" \
  -d '{
    "approved_by": "pilot@test",
    "reason": "Looks safe"
  }'
```

**Expected Response:**
```json
{
  "run_id": "...",
  "approval_status": "approved",
  "execution_result": {
    "success": true,
    ...
  }
}
```

**Validation:**
- Execution runs after approval
- `execution_result` persisted
- Backend logs show: `"event": "approval.approved"`

**UI Validation:**
- Navigate to approved run
- **Verify:** Approval status visible
- **Verify:** Execution result visible
- **Verify:** Approver and timestamp visible

---

### Step 9: Cleanup

**Stop Services:**

1. **Frontend (Terminal 2):** Ctrl+C
2. **Backend (Terminal 1):** Ctrl+C

**Expected:**
- Both servers stop cleanly
- No hanging processes

---

## Expected Outputs

### Success Indicators

✅ Backend starts without errors
✅ Frontend loads in browser
✅ API calls return HTTP 200
✅ Runs are persisted to database
✅ UI shows plan and execution results
✅ Logs are structured JSON
✅ Metrics increment correctly
✅ Failures are visible everywhere

### Failure Indicators

❌ Backend fails to start
❌ Frontend fails to load
❌ API returns non-200 status
❌ Runs not persisted
❌ UI shows incorrect data
❌ Logs not structured
❌ Metrics not incrementing
❌ Failures hidden or inferred

---

## Common Failure Modes

### Backend Won't Start

**Symptoms:**
- Port 8000 already in use
- Import errors
- Database errors

**Solutions:**
- Check if port 8000 is free: `lsof -i :8000`
- Verify virtual environment: `which python`
- Check database permissions: `ls -la forge_agent.db`

### Frontend Won't Start

**Symptoms:**
- Port 3000 already in use
- `node_modules` missing
- Build errors

**Solutions:**
- Check if port 3000 is free: `lsof -i :3000`
- Install dependencies: `cd frontend && npm install`
- Check Node.js version: `node --version` (should be >= 16)

### API Calls Fail

**Symptoms:**
- Connection refused
- Timeout errors
- 500 errors

**Solutions:**
- Verify backend is running: `curl http://localhost:8000/health`
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Check backend logs for errors

### UI Not Loading

**Symptoms:**
- Blank page
- Console errors
- Network errors

**Solutions:**
- Check frontend is running: `curl http://localhost:3000`
- Check browser console for errors
- Verify API is accessible from browser

### Persistence Issues

**Symptoms:**
- Runs not appearing in list
- Database file not created
- SQLite errors

**Solutions:**
- Check database file exists: `ls -la forge_agent.db`
- Check database permissions
- Verify storage code is working

---

## Pass / Fail Criteria

### PASS Criteria (All Must Be True)

1. ✅ Backend starts successfully
2. ✅ Frontend starts successfully
3. ✅ Health endpoint returns OK
4. ✅ Metrics endpoint returns Prometheus format
5. ✅ API run creation succeeds (HTTP 200)
6. ✅ Run is persisted to database
7. ✅ UI loads and shows runs
8. ✅ UI shows plan and execution details
9. ✅ Logs are structured JSON
10. ✅ Metrics increment correctly
11. ✅ Failures are visible in API response
12. ✅ Failures are visible in UI
13. ✅ HITL flow works (if tested)

### FAIL Criteria (Any One Fails)

1. ❌ Backend fails to start
2. ❌ Frontend fails to start
3. ❌ Health check fails
4. ❌ Metrics endpoint fails
5. ❌ API run creation fails
6. ❌ Run not persisted
7. ❌ UI fails to load
8. ❌ UI shows incorrect data
9. ❌ Logs not structured
10. ❌ Metrics not incrementing
11. ❌ Failures hidden or inferred

---

## Exit Codes

- **0** = All tests passed
- **1** = One or more tests failed

---

## Notes

- **No Mocking**: All tests use real components
- **No Shortcuts**: Full end-to-end validation
- **Visual Validation**: Browser opens in headful mode
- **Deterministic**: Same inputs → same outputs
- **Fast Failure**: Stops on first critical failure

---

**Last Updated**: 2024-12-19

