"""Unit tests for structured logger."""

import json
import logging
import pytest

from agent.observability.context import set_request_id, set_run_id, set_plan_id, clear_context
from agent.observability.logger import JSONFormatter, get_logger


class TestJSONFormatter:
    """Test JSON formatter."""

    def test_format_basic_log(self):
        """Test formatting a basic log record."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.component = "test_component"
        record.event = "test_event"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["component"] == "test_component"
        assert data["event"] == "test_event"
        assert "timestamp" in data

    def test_format_with_context(self):
        """Test formatting with context variables."""
        set_request_id("req-123")
        set_run_id("run-123")
        set_plan_id("plan-123")

        try:
            formatter = JSONFormatter()
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=None,
            )
            record.component = "test_component"
            record.event = "test_event"

            output = formatter.format(record)
            data = json.loads(output)

            assert data["request_id"] == "req-123"
            assert data["run_id"] == "run-123"
            assert data["plan_id"] == "plan-123"
        finally:
            clear_context()

    def test_format_with_exception(self):
        """Test formatting log with exception."""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Test error",
                args=(),
                exc_info=True,
            )
            record.component = "test_component"
            record.event = "test_event"

            output = formatter.format(record)
            data = json.loads(output)

            assert data["level"] == "ERROR"
            assert "error" in data
            assert "ValueError" in data["error"]

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.component = "test_component"
        record.event = "test_event"
        record.custom_field = "custom_value"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["custom_field"] == "custom_value"


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test_component", "test")
        assert logger.name == "test"
        assert isinstance(logger, logging.Logger)

    def test_logger_has_json_formatter(self):
        """Test that logger uses JSON formatter."""
        logger = get_logger("test_component", "test")
        handler = logger.handlers[0] if logger.handlers else None
        if handler:
            assert isinstance(handler.formatter, JSONFormatter)
