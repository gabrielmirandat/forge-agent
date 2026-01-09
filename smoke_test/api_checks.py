"""API-level validation checks."""

import httpx

from smoke_test.config import API_REQUEST_TIMEOUT, BACKEND_URL


def check_health() -> bool:
    """Check /health endpoint.

    Returns:
        True if health check passes
    """
    try:
        response = httpx.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"[API Check] Health check failed: HTTP {response.status_code}")
            return False

        data = response.json()
        if data.get("status") != "ok":
            print(f"[API Check] Health check failed: status is {data.get('status')}")
            return False

        if "uptime_seconds" not in data:
            print("[API Check] Health check failed: missing uptime_seconds")
            return False

        print("[API Check] ✓ Health endpoint OK")
        return True
    except Exception as e:
        print(f"[API Check] Health check failed: {e}")
        return False


def check_metrics() -> bool:
    """Check /metrics endpoint.

    Returns:
        True if metrics endpoint returns Prometheus format
    """
    try:
        response = httpx.get(f"{BACKEND_URL}/metrics", timeout=5, follow_redirects=True)
        if response.status_code != 200:
            print(f"[API Check] Metrics check failed: HTTP {response.status_code}")
            return False

        content = response.text
        # Check for Prometheus format indicators
        if "api_requests_total" not in content:
            print("[API Check] Metrics check failed: missing api_requests_total")
            return False

        print("[API Check] ✓ Metrics endpoint OK")
        return True
    except Exception as e:
        print(f"[API Check] Metrics check failed: {e}")
        return False


def create_run(goal: str) -> tuple[bool, dict | None]:
    """Create a run via API.

    Args:
        goal: Goal string

    Returns:
        Tuple of (success, response_data)
    """
    try:
        response = httpx.post(
            f"{BACKEND_URL}/api/v1/run",
            json={"goal": goal},
            timeout=API_REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            print(f"[API Check] Create run failed: HTTP {response.status_code}")
            print(f"[API Check] Response: {response.text}")
            return False, None

        data = response.json()

        # Validate response structure
        if "plan_result" not in data:
            print("[API Check] Create run failed: missing plan_result")
            return False, None

        if "execution_result" not in data:
            print("[API Check] Create run failed: missing execution_result")
            return False, None

        plan_result = data["plan_result"]
        if "plan" not in plan_result:
            print("[API Check] Create run failed: missing plan in plan_result")
            return False, None

        # execution_result can be null if HITL is enabled
        execution_result = data.get("execution_result")

        run_id = None
        # Try to extract run_id from response headers or response body
        # (Note: API doesn't return run_id in response, we'll get it from DB)

        print(f"[API Check] ✓ Run created successfully")
        if execution_result:
            print(f"[API Check]   Execution success: {execution_result.get('success', False)}")
        else:
            print(f"[API Check]   Execution pending (HITL enabled)")

        return True, data
    except Exception as e:
        print(f"[API Check] Create run failed: {e}")
        return False, None


def list_runs() -> tuple[bool, list | None]:
    """List runs via API.

    Returns:
        Tuple of (success, runs_list)
    """
    try:
        response = httpx.get(f"{BACKEND_URL}/api/v1/runs", timeout=5)
        if response.status_code != 200:
            print(f"[API Check] List runs failed: HTTP {response.status_code}")
            return False, None

        data = response.json()
        if "runs" not in data:
            print("[API Check] List runs failed: missing runs field")
            return False, None

        runs = data["runs"]
        print(f"[API Check] ✓ List runs OK ({len(runs)} runs found)")
        return True, runs
    except Exception as e:
        print(f"[API Check] List runs failed: {e}")
        return False, None


def get_run(run_id: str) -> tuple[bool, dict | None]:
    """Get a specific run via API.

    Args:
        run_id: Run identifier

    Returns:
        Tuple of (success, run_data)
    """
    try:
        response = httpx.get(f"{BACKEND_URL}/api/v1/runs/{run_id}", timeout=5)
        if response.status_code != 200:
            print(f"[API Check] Get run failed: HTTP {response.status_code}")
            return False, None

        data = response.json()
        if "run" not in data:
            print("[API Check] Get run failed: missing run field")
            return False, None

        run = data["run"]
        print(f"[API Check] ✓ Get run OK (run_id: {run_id})")
        return True, run
    except Exception as e:
        print(f"[API Check] Get run failed: {e}")
        return False, None

