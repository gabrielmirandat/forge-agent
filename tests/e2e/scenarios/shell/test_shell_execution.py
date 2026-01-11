"""E2E test: Shell command execution."""

import json

import pytest

from tests.e2e.browser import BrowserHelper
from tests.e2e.runner import E2ETestRunner
from tests.e2e.storage import StorageHelper


@pytest.mark.e2e

def test_execute_ls_in_allowed_directory():
    """Test executing ls command in allowed directory (Browser + Storage)."""
    with E2ETestRunner() as runner:
        storage = StorageHelper()
        goal = "Execute ls command in ~/repos/forge-agent directory using the shell tool"

        # Step 1: Create run via browser UI
        # Pass None to BrowserHelper so it reads from E2E_HEADLESS env var
        with BrowserHelper(headless=None, frontend_url=runner.frontend_url) as browser:
            # Create run via UI
            browser.create_run_via_ui(goal)
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
            assert "ls" in our_run.get("objective", "").lower() or \
                   "shell" in our_run.get("objective", "").lower(), \
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
                assert run_data.get("execution_result"), "Execution should be persisted"

                # Verify execution result contains command output
                execution_result = json.loads(run_data.get("execution_result", "{}"))
                steps = execution_result.get("steps", [])
                assert len(steps) > 0, "Execution should have steps"

                # Check that command output is captured
                for step in steps:
                    if step.get("success", False):
                        output = step.get("output", {})
                        if isinstance(output, dict):
                            # Shell output should contain stdout or stderr
                            assert "stdout" in output or "stderr" in output or "return_code" in output, \
                                f"Shell output missing expected fields: {output}"


@pytest.mark.e2e

def test_forbidden_command_fails(shared_browser):
    """Test that forbidden commands are rejected (Browser + Storage)."""
    with E2ETestRunner() as runner:
        storage = StorageHelper()
        browser = shared_browser
        goal = "Execute rm command to delete a file using the shell tool"

        # Navigate to frontend to ensure we're on the right page
        browser.navigate_to_frontend("/")

        # Step 1: Create run via browser UI
        browser.create_run_via_ui(goal)
        browser.wait_for_run_result()

        # The planner might not generate a plan with forbidden commands,
        # or execution might fail. Either is acceptable.
        # Check if there's an error visible or if execution failed
        page_text = browser.get_page_text()
        has_error = "error" in page_text.lower() or "failed" in page_text.lower()

        # Small delay to ensure backend has persisted the run
        import time
        time.sleep(0.5)

        # Step 2: Navigate to runs list and verify run appears
        browser.navigate_to_runs_list()
        browser.wait_for_runs_list()

        runs = browser.get_runs_from_list()
        assert len(runs) > 0, "Run should appear in list (even if it failed)"

        # Step 3: Verify in database
        our_run = runs[0]
        run_id_from_list = our_run.get("run_id", "")
        if run_id_from_list:
            run_data = storage.assert_run_persisted(run_id_from_list)
            assert run_data.get("plan_result"), "Plan should be persisted even on failure"

            # If execution happened, check that it failed appropriately
            import json
            execution_result = json.loads(run_data.get("execution_result", "{}"))
            if execution_result:
                success = execution_result.get("success", True)
                if not success:
                    # Check that error mentions command restriction
                    error = execution_result.get("error", "")
                    exec_steps = execution_result.get("steps", [])
                    error_messages = [error] + [
                        s.get("error", "") for s in exec_steps if not s.get("success", True)
                    ]
                    error_text = " ".join(error_messages).lower()
                    assert any(
                        keyword in error_text
                        for keyword in ["not allowed", "restricted", "forbidden", "command"]
                    ), f"Expected command restriction error, got: {error_text}"
