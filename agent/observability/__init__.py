"""Observability package."""

from agent.observability.context import (
    clear_context,
    get_request_id,
    set_request_id,
)
from agent.observability.logger import get_logger, log_event
from agent.observability.metrics import (
    api_request_duration_seconds,
    api_requests_total,
    approval_pending_duration_seconds,
    approvals_total,
    execution_duration_seconds,
    execution_runs_total,
    execution_step_duration_seconds,
    execution_steps_total,
    planner_duration_seconds,
    planner_requests_total,
    planner_validation_errors_total,
    storage_operation_duration_seconds,
    storage_operations_total,
)
from agent.observability.tracing import Span, trace_span

__all__ = [
    "get_logger",
    "log_event",
    "get_request_id",
    "set_request_id",
    "clear_context",
    "trace_span",
    "Span",
    "api_requests_total",
    "api_request_duration_seconds",
    "planner_requests_total",
    "planner_duration_seconds",
    "planner_validation_errors_total",
    "execution_runs_total",
    "execution_duration_seconds",
    "execution_steps_total",
    "execution_step_duration_seconds",
    "approvals_total",
    "approval_pending_duration_seconds",
    "storage_operations_total",
    "storage_operation_duration_seconds",
]

