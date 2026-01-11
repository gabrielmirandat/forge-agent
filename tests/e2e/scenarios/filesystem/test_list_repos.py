"""E2E test: List repositories in ~/repos."""

import json
import sys
import time

import pytest

from tests.e2e.runner import E2ETestRunner
from tests.e2e.storage import StorageHelper


@pytest.mark.e2e

def test_list_repositories(shared_browser):
    """Test listing repositories in allowed directory (Browser + Storage)."""
    print("[TEST] Starting test_list_repositories", file=sys.stderr, flush=True)
    print(f"[TEST] Browser headless: {shared_browser.headless}", file=sys.stderr, flush=True)
    
    with E2ETestRunner() as runner:
        print("[TEST] E2ETestRunner started, frontend should be ready", file=sys.stderr, flush=True)
        
        # Wait a bit to ensure frontend is fully ready
        time.sleep(1)
        
        storage = StorageHelper()
        goal = "List all repositories in the ~/repos directory using the filesystem tool"
        browser = shared_browser

        # Step 1: Create run via browser UI
        print("[TEST] About to create run via UI", file=sys.stderr, flush=True)
        browser.create_run_via_ui(goal)
        print("[TEST] Run created, waiting for result", file=sys.stderr, flush=True)
        browser.wait_for_run_result()

        # Verify plan and execution are visible
        assert browser.assert_plan_visible(), "Plan should be visible"
        assert browser.assert_execution_visible(), "Execution should be visible"

        # Step 2: Navigate to runs list and verify run appears
        browser.navigate_to_runs_list()
        browser.wait_for_runs_list()

        runs = browser.get_runs_from_list()
        assert len(runs) > 0, "At least one run should appear in runs list"

        # Find our run
        our_run = runs[0]
        assert "repos" in our_run.get("objective", "").lower() or \
                "filesystem" in our_run.get("objective", "").lower(), \
                f"Run objective should match goal: {our_run.get('objective')}"

        # Step 3: Click on run and verify details
        run_id_from_list = our_run.get("run_id", "")
        if run_id_from_list:
            browser.click_run_in_list(run_id=run_id_from_list)
            browser.wait_for_run_detail()

            # Verify execution is visible
            assert browser.assert_execution_visible(), "Execution should be visible in run detail"

            # Get run_id from URL
            run_id_from_url = browser.get_run_id_from_url()
            assert run_id_from_url, "Run ID should be in URL"

            # Step 4: Verify in database
            run_data = storage.assert_run_persisted(run_id_from_url)
            assert run_data.get("plan_result"), "Plan should be persisted"
            assert run_data.get("execution_result"), "Execution should be persisted"

            # Verify execution result contains directory listing
            execution_result = json.loads(run_data.get("execution_result", "{}"))
            steps = execution_result.get("steps", [])
            assert len(steps) > 0, "Execution should have steps"

            # Check that at least one step succeeded
            successful_steps = [s for s in steps if s.get("success", False)]
            assert len(successful_steps) > 0, "No successful steps"

            # Check that output contains directory entries
            for step in successful_steps:
                output = step.get("output", {})
                if isinstance(output, dict):
                    entries = output.get("entries", [])
                    if entries:
                        # Verify entries have expected structure
                        assert isinstance(entries, list), "Entries should be a list"
                        if len(entries) > 0:
                            entry = entries[0]
                            assert "name" in entry, "Entry missing 'name'"
                            assert "path" in entry, "Entry missing 'path'"
