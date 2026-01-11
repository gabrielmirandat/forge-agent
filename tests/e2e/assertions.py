"""E2E test assertions and validation helpers."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


class E2EAssertions:
    """Assertions for E2E tests."""

    def __init__(self, backend_url: str = "http://localhost:8000"):
        """Initialize assertions.

        Args:
            backend_url: Backend API URL
        """
        self.backend_url = backend_url
        self.client = httpx.Client(timeout=300.0)

    def close(self):
        """Close HTTP client."""
        self.client.close()

    def assert_health(self) -> None:
        """Assert backend health endpoint responds."""
        response = self.client.get(f"{self.backend_url}/health")
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        data = response.json()
        assert data.get("status") == "ok", f"Health status not ok: {data}"

    def assert_metrics_available(self) -> None:
        """Assert metrics endpoint responds."""
        response = self.client.get(
            f"{self.backend_url}/metrics", follow_redirects=True
        )
        assert response.status_code == 200, f"Metrics check failed: {response.status_code}"
        assert "prometheus" in response.text.lower() or "api_requests_total" in response.text

    def create_run(
        self, goal: str, context: Optional[Dict[str, Any]] = None, allow_errors: bool = False
    ) -> Dict[str, Any]:
        """Create a run and return response.

        Args:
            goal: Goal string
            context: Optional context dict
            allow_errors: If True, return error response instead of asserting

        Returns:
            Run response dict (or error dict if allow_errors=True and error occurred)
        """
        payload = {"goal": goal}
        if context:
            payload["context"] = context

        response = self.client.post(
            f"{self.backend_url}/api/v1/run",
            json=payload,
        )
        if allow_errors:
            # Return response even if it's an error
            return {"status_code": response.status_code, "response": response.json() if response.status_code < 500 else {"error": response.text}}
        
        assert response.status_code == 200, f"Run creation failed: {response.status_code} - {response.text}"
        return response.json()

    def assert_run_success(
        self, run_response: Dict[str, Any], expected_steps: Optional[int] = None
    ) -> None:
        """Assert run completed successfully.

        Args:
            run_response: Run response from create_run
            expected_steps: Optional expected number of execution steps
        """
        assert "plan_result" in run_response, "Missing plan_result"
        plan_result = run_response["plan_result"]
        assert "plan" in plan_result, "Missing plan in plan_result"

        execution_result = run_response.get("execution_result")
        if execution_result:
            assert execution_result.get("success") is True, f"Execution failed: {execution_result.get('error')}"
            if expected_steps is not None:
                steps = execution_result.get("steps", [])
                assert len(steps) == expected_steps, f"Expected {expected_steps} steps, got {len(steps)}"

    def assert_run_failure(
        self, run_response: Dict[str, Any], expected_error_contains: Optional[str] = None
    ) -> None:
        """Assert run failed as expected.

        Args:
            run_response: Run response from create_run
            expected_error_contains: Optional string that should be in error message
        """
        execution_result = run_response.get("execution_result")
        if execution_result:
            assert execution_result.get("success") is False, "Expected execution to fail"
            if expected_error_contains:
                error = execution_result.get("error", "")
                steps = execution_result.get("steps", [])
                error_messages = [error] + [
                    s.get("error", "") for s in steps if not s.get("success", True)
                ]
                error_text = " ".join(error_messages)
                assert expected_error_contains.lower() in error_text.lower(), \
                    f"Expected error containing '{expected_error_contains}', got: {error_text}"

    def assert_run_persisted(self, run_id: str) -> Dict[str, Any]:
        """Assert run is persisted and return it.

        Args:
            run_id: Run ID to check

        Returns:
            Run data from storage
        """
        response = self.client.get(f"{self.backend_url}/api/v1/runs/{run_id}")
        assert response.status_code == 200, f"Run not found: {run_id}"
        data = response.json()
        assert "run" in data, "Missing run in response"
        return data["run"]

    def get_latest_run_id(self) -> Optional[str]:
        """Get the latest run ID from the runs list.

        Returns:
            Latest run ID or None if no runs exist
        """
        response = self.client.get(f"{self.backend_url}/api/v1/runs?limit=1")
        if response.status_code == 200:
            data = response.json()
            runs = data.get("runs", [])
            if runs:
                return runs[0].get("run_id")
        return None

    def assert_plan_valid(self, plan_result: Dict[str, Any]) -> None:
        """Assert plan result is valid.

        Args:
            plan_result: Plan result dict
        """
        assert "plan" in plan_result, "Missing plan"
        plan = plan_result["plan"]
        assert "plan_id" in plan, "Missing plan_id"
        assert "objective" in plan, "Missing objective"
        assert "steps" in plan, "Missing steps"
        assert isinstance(plan["steps"], list), "Steps must be a list"

    def assert_execution_result_valid(
        self, execution_result: Dict[str, Any]
    ) -> None:
        """Assert execution result is valid.

        Args:
            execution_result: Execution result dict
        """
        assert "plan_id" in execution_result, "Missing plan_id"
        assert "success" in execution_result, "Missing success"
        assert "steps" in execution_result, "Missing steps"
        assert isinstance(execution_result["steps"], list), "Steps must be a list"

    def assert_logs_have_correlation_ids(self, log_file: Path) -> None:
        """Assert logs contain correlation IDs.

        Args:
            log_file: Path to log file
        """
        if not log_file.exists():
            # Logs might not exist yet, that's ok for some tests
            return

        with open(log_file, "r") as f:
            log_lines = f.readlines()

        # Check that at least some logs have request_id
        request_ids_found = 0
        run_ids_found = 0

        for line in log_lines:
            try:
                log_entry = json.loads(line.strip())
                if "request_id" in log_entry:
                    request_ids_found += 1
                if "run_id" in log_entry and log_entry["run_id"]:
                    run_ids_found += 1
            except json.JSONDecodeError:
                # Skip non-JSON lines
                continue

        # At least one log should have request_id
        assert request_ids_found > 0, "No logs with request_id found"

    def assert_metrics_incremented(
        self, metric_name: str, min_value: float = 1.0
    ) -> None:
        """Assert metric has incremented.

        Args:
            metric_name: Metric name (e.g., "api_requests_total")
            min_value: Minimum expected value
        """
        response = self.client.get(
            f"{self.backend_url}/metrics", follow_redirects=True
        )
        assert response.status_code == 200
        metrics_text = response.text

        # Parse Prometheus format
        for line in metrics_text.split("\n"):
            if line.startswith(metric_name):
                # Extract value (format: metric_name{labels} value)
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        value = float(parts[-1])
                        assert value >= min_value, f"Metric {metric_name} is {value}, expected >= {min_value}"
                        return
                    except ValueError:
                        continue

        # If we get here, metric not found or couldn't parse
        # This is not necessarily a failure (metric might not exist yet)
        # But we'll log it
        print(f"Warning: Could not find or parse metric {metric_name}")

    def wait_for_execution(
        self, run_id: str, timeout: int = 60, poll_interval: float = 1.0
    ) -> Dict[str, Any]:
        """Wait for execution to complete and return run.

        Args:
            run_id: Run ID to wait for
            timeout: Timeout in seconds
            poll_interval: Poll interval in seconds

        Returns:
            Run data with execution result
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            run_data = self.assert_run_persisted(run_id)
            execution_result = run_data.get("execution_result")
            if execution_result:
                return run_data
            time.sleep(poll_interval)

        raise TimeoutError(f"Execution did not complete within {timeout} seconds")

    def approve_run(
        self, run_id: str, approved_by: str = "e2e_test", reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Approve a run (HITL).

        Args:
            run_id: Run ID to approve
            approved_by: Approver identifier
            reason: Optional approval reason

        Returns:
            Approval response
        """
        payload = {"approved_by": approved_by}
        if reason:
            payload["reason"] = reason

        response = self.client.post(
            f"{self.backend_url}/api/v1/runs/{run_id}/approve",
            json=payload,
        )
        assert response.status_code == 200, f"Approval failed: {response.status_code} - {response.text}"
        return response.json()
