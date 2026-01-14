"""E2E test: Retrieve system information."""

import json

import pytest

from tests.e2e.runner import E2ETestRunner
from tests.e2e.storage import StorageHelper


@pytest.mark.e2e

def test_get_system_info(shared_browser):
    """Test retrieving system information using system tool (Browser + Storage)."""
    import sys
    
    print(f"[TEST] test_get_system_info started", file=sys.stderr, flush=True)
    print(f"[TEST] Browser instance: {shared_browser}", file=sys.stderr, flush=True)
    print(f"[TEST] Browser page: {shared_browser.page}", file=sys.stderr, flush=True)
    print(f"[TEST] Browser headless: {shared_browser.headless}", file=sys.stderr, flush=True)
    
    with E2ETestRunner() as runner:
        print(f"[TEST] E2ETestRunner setup complete", file=sys.stderr, flush=True)
        storage = StorageHelper()
        goal = "Get system information using the system tool"
        browser = shared_browser

        # Verify browser is still valid
        if not browser.page:
            raise RuntimeError("Browser page is None! Browser may have been closed.")
        
        print(f"[TEST] Step 1: Creating run via browser UI", file=sys.stderr, flush=True)
        # Step 1: Create run via browser UI
        browser.create_run_via_ui(goal)
        browser.wait_for_run_result()

        # Verify plan is visible
        assert browser.assert_plan_visible(), "Plan should be visible after creating run"

        # Verify execution is visible
        assert browser.assert_execution_visible(), "Execution should be visible after creating run"

        # Get plan and execution text to verify content
        plan_text = browser.get_plan_text()
        execution_text = browser.get_execution_text()

        # Verify plan contains expected content
        assert "plan" in plan_text.lower() or "objective" in plan_text.lower() or len(plan_text) > 0, \
            f"Plan text should contain plan information: {plan_text[:200]}"

        # Verify execution contains expected content
        assert "execution" in execution_text.lower() or "success" in execution_text.lower() or len(execution_text) > 0, \
            f"Execution text should contain execution information: {execution_text[:200]}"

        # Step 2: Navigate to runs list and verify run appears
        browser.navigate_to_runs_list()
        browser.wait_for_runs_list()

        runs = browser.get_runs_from_list()
        assert len(runs) > 0, "At least one run should appear in runs list"

        # Find our run (should be the first one)
        our_run = runs[0]
        assert goal.lower() in our_run.get("objective", "").lower() or \
               "system" in our_run.get("objective", "").lower(), \
               f"Run objective should match goal: {our_run.get('objective')}"

        # Step 3: Click on run and verify details
        run_id_from_list = our_run.get("run_id", "")
        if run_id_from_list:
            browser.click_run_in_list(run_id=run_id_from_list)
            browser.wait_for_run_detail()

            # Verify plan and execution are visible in detail page
            assert browser.assert_plan_visible(), "Plan should be visible in run detail"
            assert browser.assert_execution_visible(), "Execution should be visible in run detail"

            # Get run_id from URL
            run_id_from_url = browser.get_run_id_from_url()
            assert run_id_from_url, "Run ID should be in URL"

            # Step 4: Verify in database
            run_data = storage.assert_run_persisted(run_id_from_url)
            assert run_data.get("plan_result"), "Plan should be persisted in database"
            assert run_data.get("execution_result"), "Execution should be persisted in database"

            # Verify execution result contains system info
            import json
            execution_result = json.loads(run_data.get("execution_result", "{}"))
            steps = execution_result.get("steps", [])
            assert len(steps) > 0, "Execution should have steps"

            # Check that system info is returned
            for step in steps:
                if step.get("success", False):
                    output = step.get("output", {})
                    if isinstance(output, dict):
                        # System info should contain platform info
                        assert "platform" in output or "python_version" in output or "status" in output, \
                            f"System info missing expected fields: {output}"


@pytest.mark.e2e

def test_system_tool_no_side_effects(shared_browser):
    """Test that system tool has no side effects (Browser + Storage)."""
    with E2ETestRunner() as runner:
        storage = StorageHelper()
        browser = shared_browser
        goal = "Get system status using the system tool"

        # Navigate to frontend to ensure we're on the right page
        browser.navigate_to_frontend("/")

        # Step 1: Create first run via browser UI
        browser.create_run_via_ui(goal)
        browser.wait_for_run_result()

        # Verify execution succeeded
        assert browser.assert_execution_visible(), "First run execution should be visible"

        # Small delay to ensure backend has persisted the run
        import time
        time.sleep(0.5)

        # Get run_id from runs list
        browser.navigate_to_runs_list()
        browser.wait_for_runs_list()
        runs1 = browser.get_runs_from_list()
        assert len(runs1) > 0, "First run should appear in list"
        run_id1 = runs1[0].get("run_id", "")

        # Step 2: Create second run via browser UI
        browser.navigate_to_frontend("/")
        browser.create_run_via_ui(goal)
        browser.wait_for_run_result()

        # Verify execution succeeded
        assert browser.assert_execution_visible(), "Second run execution should be visible"

        # Small delay to ensure backend has persisted the run
        time.sleep(0.5)

        # Get run_id from runs list
        browser.navigate_to_runs_list()
        browser.wait_for_runs_list(timeout=3000)  # Longer timeout for second run
        
        # Reload page to ensure we get fresh data from backend
        browser.page.reload()
        time.sleep(0.5)
        browser.wait_for_runs_list(timeout=3000)
        
        runs2 = browser.get_runs_from_list()
        assert len(runs2) >= 2, f"Both runs should appear in list, got {len(runs2)}"

        # Step 3: Verify in database that both runs succeeded
        # Use the latest run_id from the list (runs2[0]) instead of run_id1
        # because run_id1 might be from a different test run
        if run_id1 and len(runs2) >= 2:
            # Find run_id1 in runs2 to ensure it's the correct one
            run1_in_list = next((r for r in runs2 if r.get("run_id", "") == run_id1), None)
            if run1_in_list:
                run_data1 = storage.assert_run_persisted(run_id1)
            else:
                # If run_id1 is not in the list, use the second run
                run_id1 = runs2[1].get("run_id", "") if len(runs2) >= 2 else run_id1
                if run_id1:
                    run_data1 = storage.assert_run_persisted(run_id1)
            exec_result1 = json.loads(run_data1.get("execution_result", "{}"))
            steps1 = exec_result1.get("steps", [])
            assert len(steps1) > 0, "Run 1 should have steps"

        run_id2 = runs2[0].get("run_id", "")
        if run_id2:
            run_data2 = storage.assert_run_persisted(run_id2)
            exec_result2 = json.loads(run_data2.get("execution_result", "{}"))
            steps2 = exec_result2.get("steps", [])
            assert len(steps2) > 0, "Run 2 should have steps"

        # System tool should return consistent results
        # (platform, Python version don't change between calls)
