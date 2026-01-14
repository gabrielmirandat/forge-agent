"""Unit tests for health endpoint."""

import pytest
import time
from fastapi.testclient import TestClient

from api.app import app


class TestHealthEndpoint:
    """Test health endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health endpoint returns OK."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_health_uptime_increases(self, client):
        """Test that uptime increases over time."""
        response1 = client.get("/health")
        time.sleep(0.1)
        response2 = client.get("/health")
        
        uptime1 = response1.json()["uptime_seconds"]
        uptime2 = response2.json()["uptime_seconds"]
        
        assert uptime2 >= uptime1
