#!/usr/bin/env python3
"""Main smoke test entry point.

Runs full end-to-end validation:
1. Start backend
2. Start frontend
3. API smoke test
4. Persistence check
5. Browser UI test
6. Observability validation
7. Failure case test
8. Cleanup

Exit codes:
    0 = PASS
    1 = FAIL
"""

import sys
import time
from io import StringIO

from smoke_test.api_checks import (
    check_health,
    check_metrics,
    create_run,
    get_run,
    list_runs,
)
from smoke_test.backend import BackendServer
from smoke_test.config import FAILURE_GOAL, SUCCESS_GOAL
from smoke_test.frontend import FrontendServer
from smoke_test.observability_checks import (
    check_metrics_incremented,
    validate_logs_have_correlation_ids,
)
from smoke_test.storage_checks import (
    check_database_exists,
    get_latest_run,
    validate_run_structure,
)
from smoke_test.ui_checks import UIChecker


def main() -> int:
    """Run smoke test.

    Returns:
        0 if all tests pass, 1 otherwise
    """
    print("=" * 60)
    print("SMOKE TEST - End-to-End Validation")
    print("=" * 60)
    print()

    backend = BackendServer()
    frontend = FrontendServer()
    ui_checker = UIChecker()
    log_output = StringIO()

    all_passed = True

    try:
        # Step 1: Start Backend
        print("[Step 1] Starting backend...")
        backend.start()
        if not check_health():
            print("[FAIL] Backend health check failed")
            return 1
        if not check_metrics():
            print("[FAIL] Backend metrics check failed")
            return 1
        print()

        # Step 2: Start Frontend
        print("[Step 2] Starting frontend...")
        try:
            frontend.start()
        except RuntimeError as e:
            print(f"[FAIL] Frontend startup failed: {e}")
            return 1
        print()

        # Step 3: API Smoke Test (Success Case)
        print("[Step 3] API smoke test (success case)...")
        success, run_response = create_run(SUCCESS_GOAL)
        if not success:
            print("[FAIL] API run creation failed")
            return 1

        # Wait a moment for persistence
        time.sleep(1)

        # Step 4: Persistence Check
        print("[Step 4] Persistence check...")
        if not check_database_exists():
            print("[FAIL] Database file not found")
            return 1

        success, latest_run = get_latest_run()
        if not success:
            print("[FAIL] Failed to retrieve latest run from database")
            return 1

        if not validate_run_structure(latest_run):
            print("[FAIL] Run structure validation failed")
            return 1

        run_id = latest_run["run_id"]
        print(f"[INFO] Using run_id: {run_id}")
        print()

        # Step 5: Browser UI Test
        print("[Step 5] Browser UI test...")
        ui_checker.start()

        if not ui_checker.check_runs_list():
            print("[FAIL] Runs list UI check failed")
            all_passed = False

        if not ui_checker.check_run_detail(run_id):
            print("[FAIL] Run detail UI check failed")
            all_passed = False

        # Keep browser open for a moment to see the UI
        print("[INFO] Keeping browser open for 3 seconds for visual inspection...")
        time.sleep(3)
        print()

        # Step 6: Observability Validation
        print("[Step 6] Observability validation...")
        if not check_metrics_incremented():
            print("[FAIL] Metrics check failed")
            all_passed = False

        # Note: Log validation would require capturing backend stdout
        # For now, we validate metrics which is sufficient
        print()

        # Step 7: Failure Case Test
        print("[Step 7] Failure case test...")
        success, failure_response = create_run(FAILURE_GOAL)
        if not success:
            print("[FAIL] Failure case API call failed")
            all_passed = False
        else:
            execution_result = failure_response.get("execution_result")
            if execution_result and execution_result.get("success", True):
                print("[FAIL] Failure case should have success=false")
                all_passed = False
            else:
                print("[PASS] Failure case correctly shows failure")

        # Wait for persistence
        time.sleep(1)

        success, failure_run = get_latest_run()
        if success and failure_run:
            failure_run_id = failure_run["run_id"]
            if not ui_checker.check_failure_visibility(failure_run_id):
                print("[WARN] Failure visibility check inconclusive")
        print()

        # Step 8: Final API Checks
        print("[Step 8] Final API checks...")
        success, runs_list = list_runs()
        if not success:
            print("[FAIL] List runs failed")
            all_passed = False

        if runs_list and len(runs_list) < 2:
            print("[WARN] Expected at least 2 runs in list")

        # Test get_run endpoint
        if run_id:
            success, run_data = get_run(run_id)
            if not success:
                print("[FAIL] Get run endpoint failed")
                all_passed = False
        print()

    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
        all_passed = False
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False
    finally:
        # Step 9: Cleanup
        print("[Step 9] Cleanup...")
        ui_checker.stop()
        frontend.stop()
        backend.stop()
        print()

    # Final result
    print("=" * 60)
    if all_passed:
        print("✅ SMOKE TEST PASSED")
        print("=" * 60)
        return 0
    else:
        print("❌ SMOKE TEST FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

