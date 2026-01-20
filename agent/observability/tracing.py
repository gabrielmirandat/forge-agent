"""Lightweight execution tracing.

This is logical tracing, not distributed tracing.
Traces are deterministic and stored in memory/logs only.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from agent.observability.context import get_request_id
from agent.observability.logger import get_logger

_tracer_logger = get_logger("tracer", "tracing")


class Span:
    """A trace span."""

    def __init__(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Initialize span.

        Args:
            name: Span name
            parent_span_id: Parent span ID (None for root)
            attributes: Span attributes
        """
        from agent.id import ascending

        # Use message prefix for span IDs (temporary, not persisted)
        self.span_id = ascending("message")[:16]
        self.parent_span_id = parent_span_id
        self.name = name
        self.attributes = attributes or {}
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    def start(self) -> None:
        """Start the span."""
        self.started_at = time.time()

    def finish(self) -> None:
        """Finish the span."""
        self.finished_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for logging."""
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": (
                (self.finished_at - self.started_at) * 1000
                if self.started_at and self.finished_at
                else None
            ),
            "attributes": self.attributes,
            "request_id": get_request_id(),
        }


@contextmanager
def trace_span(
    name: str,
    parent_span_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    log_on_finish: bool = True,
):
    """Context manager for creating a trace span.

    Args:
        name: Span name
        parent_span_id: Parent span ID
        attributes: Span attributes
        log_on_finish: Whether to log span on finish

    Yields:
        Span instance
    """
    span = Span(name, parent_span_id, attributes)
    span.start()

    try:
        yield span
    finally:
        span.finish()
        if log_on_finish:
            _tracer_logger.info(
                f"span.finished",
                extra={
                    "event": "span.finished",
                    "span": span.to_dict(),
                },
            )

