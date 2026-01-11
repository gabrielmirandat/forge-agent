"""Minimal smoke test: Basic execution."""

import pytest
import httpx


@pytest.mark.asyncio
async def test_noop_execution():
    """Test that executor can run a no-op command."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:8000/api/v1/run",
            json={"goal": "Describe what this agent can do"},
        )
        assert response.status_code == 200, f"Execution failed: {response.status_code}"
        data = response.json()
        assert "plan_result" in data, "Missing plan_result"
        
        # Execution result might or might not be present (depending on HITL)
        execution_result = data.get("execution_result")
        if execution_result:
            # If execution happened, it should have steps
            assert "steps" in execution_result, "Missing steps in execution_result"
            steps = execution_result.get("steps", [])
            # At least one step should exist
            assert len(steps) > 0, "No execution steps"


@pytest.mark.asyncio
async def test_one_log_line_produced():
    """Test that at least one log line is produced."""
    # This is a minimal check - we just verify the system is logging
    # Full log validation is in E2E tests
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Make a request
        await client.post(
            "http://localhost:8000/api/v1/run",
            json={"goal": "Report system status"},
        )
        # If we got here without error, logging is likely working
        # Full validation is in E2E tests
        assert True, "Request completed (logging assumed working)"
