"""Observability context for request correlation.

Provides thread-local context for request_id propagation.
Matches OpenCode: only request_id for correlation, no plan_id or step_id.
"""

import contextvars
from typing import Optional

# Context variables for correlation IDs
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: str) -> None:
    """Set request ID in context."""
    request_id_var.set(request_id)


def clear_context() -> None:
    """Clear all context variables."""
    request_id_var.set(None)

