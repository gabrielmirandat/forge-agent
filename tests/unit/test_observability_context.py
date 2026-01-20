"""Unit tests for observability context."""

import pytest

from agent.observability.context import (
    clear_context,
    get_request_id,
    set_request_id,
)


class TestContextVariables:
    """Test context variable management."""

    def test_request_id(self):
        """Test request ID context."""
        assert get_request_id() is None
        set_request_id("req-123")
        assert get_request_id() == "req-123"
        clear_context()
        assert get_request_id() is None

    def test_clear_context(self):
        """Test clearing all context."""
        set_request_id("req-123")

        clear_context()

        assert get_request_id() is None

    def test_context_isolation(self):
        """Test that context variables are isolated."""
        # Set values
        set_request_id("req-1")

        # Verify they're set
        assert get_request_id() == "req-1"

        # Clear and verify isolation
        clear_context()
        assert get_request_id() is None
