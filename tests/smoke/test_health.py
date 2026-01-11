"""Minimal smoke test: API health check."""

import pytest
import httpx


@pytest.mark.asyncio
async def test_api_health():
    """Test that API health endpoint responds."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get("http://localhost:8000/health")
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        data = response.json()
        assert data.get("status") == "ok", f"Health status not ok: {data}"


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Test that metrics endpoint responds."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get("http://localhost:8000/metrics", follow_redirects=True)
        assert response.status_code == 200, f"Metrics check failed: {response.status_code}"
        # Just check that it returns something
        assert len(response.text) > 0, "Metrics endpoint returned empty response"
