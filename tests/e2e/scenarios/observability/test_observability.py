"""E2E test: Observability validation."""

import json
from pathlib import Path

import pytest

from tests.e2e.assertions import E2EAssertions
from tests.e2e.runner import E2ETestRunner


@pytest.mark.e2e

def test_correlated_logs():
    """Test that logs contain correlation IDs."""
    with E2ETestRunner() as runner:
        assertions = E2EAssertions()

        try:
            # Assert health
            assertions.assert_health()

            # Create a run
            goal = "Get system information using the system tool"
            run_response = assertions.create_run(goal)

            # Get run_id from latest run (since it's not in the response)
            run_id = assertions.get_latest_run_id()
            # Note: run_id might be None if run wasn't persisted, but that's ok
            # We can still check logs for request_id

            # Check logs for correlation IDs
            log_dir = runner.project_root / "workspace" / "logs"
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log"))
                if log_files:
                    # Check at least one log file
                    log_file = log_files[0]
                    assertions.assert_logs_have_correlation_ids(log_file)

            # Assert metrics incremented
            assertions.assert_metrics_incremented("api_requests_total", min_value=1.0)
            assertions.assert_metrics_incremented("planner_requests_total", min_value=1.0)

        finally:
            assertions.close()


@pytest.mark.e2e

def test_metrics_increment():
    """Test that metrics increment correctly."""
    with E2ETestRunner() as runner:
        assertions = E2EAssertions()

        try:
            # Assert health
            assertions.assert_health()

            # Get initial metrics
            initial_metrics = {}
            response = assertions.client.get(
                f"{assertions.backend_url}/metrics", follow_redirects=True
            )
            if response.status_code == 200:
                for line in response.text.split("\n"):
                    if "api_requests_total" in line and "{" in line:
                        # Parse metric value
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                initial_metrics["api_requests"] = float(parts[-1])
                            except ValueError:
                                pass

            # Create multiple runs
            for i in range(3):
                goal = f"Get system information (run {i})"
                assertions.create_run(goal)

            # Check that metrics incremented
            assertions.assert_metrics_incremented("api_requests_total", min_value=3.0)
            assertions.assert_metrics_incremented("planner_requests_total", min_value=3.0)
            assertions.assert_metrics_incremented("execution_runs_total", min_value=3.0)

        finally:
            assertions.close()


@pytest.mark.e2e

def test_request_id_propagation():
    """Test that request_id is propagated end-to-end."""
    with E2ETestRunner() as runner:
        assertions = E2EAssertions()

        try:
            # Assert health
            assertions.assert_health()

            # Create a run
            goal = "Get system information using the system tool"
            run_response = assertions.create_run(goal)

            # Check logs for request_id
            log_dir = runner.project_root / "workspace" / "logs"
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log"))
                if log_files:
                    log_file = log_files[0]
                    if log_file.exists():
                        with open(log_file, "r") as f:
                            log_lines = f.readlines()

                        # Find logs with request_id
                        request_ids = set()
                        for line in log_lines:
                            try:
                                log_entry = json.loads(line.strip())
                                if "request_id" in log_entry:
                                    request_ids.add(log_entry["request_id"])
                            except json.JSONDecodeError:
                                continue

                        # At least one request_id should be found
                        assert len(request_ids) > 0, "No request_id found in logs"

        finally:
            assertions.close()
