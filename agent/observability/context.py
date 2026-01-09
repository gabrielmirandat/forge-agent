"""Observability context for request/run correlation.

Provides thread-local context for request_id, run_id, plan_id propagation.
"""

import contextvars
from typing import Optional

# Context variables for correlation IDs
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)
run_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("run_id", default=None)
plan_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "plan_id", default=None
)
step_id_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "step_id", default=None
)


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: str) -> None:
    """Set request ID in context."""
    request_id_var.set(request_id)


def get_run_id() -> Optional[str]:
    """Get current run ID from context."""
    return run_id_var.get()


def set_run_id(run_id: str) -> None:
    """Set run ID in context."""
    run_id_var.set(run_id)


def get_plan_id() -> Optional[str]:
    """Get current plan ID from context."""
    return plan_id_var.get()


def set_plan_id(plan_id: str) -> None:
    """Set plan ID in context."""
    plan_id_var.set(plan_id)


def get_step_id() -> Optional[int]:
    """Get current step ID from context."""
    return step_id_var.get()


def set_step_id(step_id: int) -> None:
    """Set step ID in context."""
    step_id_var.set(step_id)


def clear_context() -> None:
    """Clear all context variables."""
    request_id_var.set(None)
    run_id_var.set(None)
    plan_id_var.set(None)
    step_id_var.set(None)

