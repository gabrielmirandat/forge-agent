"""Observability (logs + metrics) validation checks."""

import httpx
import re

from smoke_test.config import BACKEND_URL


def check_metrics_incremented() -> bool:
    """Check that metrics have been incremented.

    Returns:
        True if key metrics show activity
    """
    try:
        response = httpx.get(f"{BACKEND_URL}/metrics", timeout=5)
        if response.status_code != 200:
            print("[Observability Check] Failed to fetch metrics")
            return False

        content = response.text

        # Check for key metrics with values > 0
        metrics_to_check = [
            "api_requests_total",
            "planner_requests_total",
            "execution_runs_total",
            "execution_steps_total",
            "storage_operations_total",
        ]

        found_metrics = []
        for metric in metrics_to_check:
            # Look for metric lines with values
            pattern = rf"{metric}\{{[^}}]*\}}\s+(\d+(?:\.\d+)?)"
            matches = re.findall(pattern, content)
            if matches:
                value = float(matches[0])
                if value > 0:
                    found_metrics.append((metric, value))

        if len(found_metrics) < 2:  # At least 2 metrics should have activity
            print(
                f"[Observability Check] Insufficient metrics activity: {len(found_metrics)} metrics found"
            )
            return False

        print(f"[Observability Check] ✓ Metrics incremented ({len(found_metrics)} metrics active)")
        for metric, value in found_metrics:
            print(f"  {metric}: {value}")
        return True
    except Exception as e:
        print(f"[Observability Check] Metrics check failed: {e}")
        return False


def validate_logs_have_correlation_ids(log_output: str) -> bool:
    """Validate that logs contain correlation IDs.

    Args:
        log_output: Captured log output

    Returns:
        True if correlation IDs are present
    """
    # Check for request_id
    if "request_id" not in log_output:
        print("[Observability Check] Logs missing request_id")
        return False

    # Check for run_id (should appear after run creation)
    if "run_id" not in log_output:
        print("[Observability Check] Logs missing run_id")
        return False

    # Check for plan_id
    if "plan_id" not in log_output:
        print("[Observability Check] Logs missing plan_id")
        return False

    # Check for key events
    required_events = [
        "api.request.started",
        "planner.plan.started",
        "executor.execution.started",
    ]

    found_events = []
    for event in required_events:
        if event in log_output:
            found_events.append(event)

    if len(found_events) < 2:
        print(
            f"[Observability Check] Insufficient log events: {len(found_events)} found"
        )
        return False

    print(
        f"[Observability Check] ✓ Logs contain correlation IDs and events ({len(found_events)} events found)"
    )
    return True

