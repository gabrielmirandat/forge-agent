"""Structured JSON logger.

All logs are JSON. One event per log entry.
Never log secrets or raw tool outputs unless explicitly allowed.
"""

import json
import logging
import sys
import time
from typing import Any, Dict, Optional

from agent.observability.context import (
    get_plan_id,
    get_request_id,
    get_run_id,
    get_step_id,
)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base fields (mandatory)
        log_data: Dict[str, Any] = {
            "timestamp": record.created,
            "level": record.levelname,
            "request_id": get_request_id() or "unknown",
            "run_id": get_run_id(),
            "plan_id": get_plan_id(),
            "component": getattr(record, "component", "unknown"),
            "event": getattr(record, "event", record.getMessage()),
        }

        # Optional fields
        step_id = get_step_id()
        if step_id is not None:
            log_data["step_id"] = step_id

        # Add message if different from event
        if record.getMessage() != log_data["event"]:
            log_data["message"] = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            log_data["error"] = self.formatException(record.exc_info)

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "component",
                "event",
            ):
                log_data[key] = value

        return json.dumps(log_data)


def get_logger(name: str, component: str) -> logging.Logger:
    """Get a structured logger for a component.

    Args:
        name: Logger name
        component: Component name (planner, executor, api, storage, frontend)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Add JSON formatter handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    # Add component to logger
    logger.component = component  # type: ignore

    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    level: str = "INFO",
    **kwargs: Any,
) -> None:
    """Log a structured event.

    Args:
        logger: Logger instance
        event: Event name
        level: Log level (INFO, ERROR, WARN)
        **kwargs: Additional event data
    """
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(event, extra={"event": event, **kwargs})

