"""Unit tests for observability context."""

import pytest

from agent.observability.context import (
    clear_context,
    get_plan_id,
    get_request_id,
    get_run_id,
    get_step_id,
    set_plan_id,
    set_request_id,
    set_run_id,
    set_step_id,
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

    def test_run_id(self):
        """Test run ID context."""
        assert get_run_id() is None
        set_run_id("run-123")
        assert get_run_id() == "run-123"
        clear_context()
        assert get_run_id() is None

    def test_plan_id(self):
        """Test plan ID context."""
        assert get_plan_id() is None
        set_plan_id("plan-123")
        assert get_plan_id() == "plan-123"
        clear_context()
        assert get_plan_id() is None

    def test_step_id(self):
        """Test step ID context."""
        assert get_step_id() is None
        set_step_id(1)
        assert get_step_id() == 1
        set_step_id(2)
        assert get_step_id() == 2
        clear_context()
        assert get_step_id() is None

    def test_clear_context(self):
        """Test clearing all context."""
        set_request_id("req-123")
        set_run_id("run-123")
        set_plan_id("plan-123")
        set_step_id(1)

        clear_context()

        assert get_request_id() is None
        assert get_run_id() is None
        assert get_plan_id() is None
        assert get_step_id() is None

    def test_context_isolation(self):
        """Test that context variables are isolated."""
        # Set values
        set_request_id("req-1")
        set_run_id("run-1")
        set_plan_id("plan-1")
        set_step_id(1)

        # Verify they're set
        assert get_request_id() == "req-1"
        assert get_run_id() == "run-1"
        assert get_plan_id() == "plan-1"
        assert get_step_id() == 1

        # Clear and verify isolation
        clear_context()
        assert get_request_id() is None
        assert get_run_id() is None
        assert get_plan_id() is None
        assert get_step_id() is None
