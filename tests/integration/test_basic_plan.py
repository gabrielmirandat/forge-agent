"""Integration test: Basic planning."""

import pytest
import httpx


@pytest.mark.asyncio
async def test_trivial_plan():
    """Test that planner can generate a trivial plan."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:8000/api/v1/run",
            json={"goal": "Report system status"},
        )
        assert response.status_code == 200, f"Plan creation failed: {response.status_code}"
        data = response.json()
        assert "plan_result" in data, "Missing plan_result"
        plan_result = data["plan_result"]
        assert "plan" in plan_result, "Missing plan in plan_result"
        plan = plan_result["plan"]
        assert "plan_id" in plan, "Missing plan_id"
        assert "objective" in plan, "Missing objective"
