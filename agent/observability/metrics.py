"""Prometheus metrics.

Metrics are write-only. No reads. No branching based on metrics.
Metrics never affect logic.
"""

from prometheus_client import Counter, Histogram

# Request metrics
api_requests_total = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "method", "status"]
)
api_request_duration_seconds = Histogram(
    "api_request_duration_seconds", "API request duration", ["endpoint"]
)

# Planning metrics
planner_requests_total = Counter(
    "planner_requests_total", "Total planner requests", ["status"]
)
planner_duration_seconds = Histogram("planner_duration_seconds", "Planning duration")
planner_validation_errors_total = Counter(
    "planner_validation_errors_total", "Total planner validation errors"
)

# Execution metrics
execution_runs_total = Counter(
    "execution_runs_total", "Total execution runs", ["status"]
)
execution_duration_seconds = Histogram("execution_duration_seconds", "Execution duration")
execution_steps_total = Counter(
    "execution_steps_total", "Total execution steps", ["tool", "operation", "status"]
)
execution_step_duration_seconds = Histogram(
    "execution_step_duration_seconds", "Execution step duration", ["tool", "operation"]
)

# HITL metrics
approvals_total = Counter("approvals_total", "Total approvals", ["status"])
approval_pending_duration_seconds = Histogram(
    "approval_pending_duration_seconds", "Time from creation to approval"
)

# Storage metrics
storage_operations_total = Counter(
    "storage_operations_total", "Total storage operations", ["operation", "status"]
)
storage_operation_duration_seconds = Histogram(
    "storage_operation_duration_seconds", "Storage operation duration", ["operation"]
)

