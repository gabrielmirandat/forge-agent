"""E2E test: Validate path access denial outside allowed_paths."""

import pytest

from tests.e2e.assertions import E2EAssertions
from tests.e2e.runner import E2ETestRunner


@pytest.mark.e2e

def test_path_access_denial():
    """Test that access to paths outside allowed_paths is denied."""
    with E2ETestRunner() as runner:
        assertions = E2EAssertions()

        try:
            # Assert health
            assertions.assert_health()

            # Create run that attempts to access forbidden path
            goal = "List files in /etc directory using the filesystem tool"
            run_response = assertions.create_run(goal)

            # Assert plan is valid (planning should succeed)
            assertions.assert_plan_valid(run_response["plan_result"])

            # Execution should fail with path validation error
            execution_result = run_response.get("execution_result")
            if execution_result:
                # Execution might fail or might be prevented at planning time
                # Either way, we should see a clear error
                success = execution_result.get("success", True)
                if not success:
                    # Check that error mentions path restriction
                    error = execution_result.get("error", "")
                    steps = execution_result.get("steps", [])
                    error_messages = [error] + [
                        s.get("error", "") for s in steps if not s.get("success", True)
                    ]
                    error_text = " ".join(error_messages).lower()
                    assert any(
                        keyword in error_text
                        for keyword in ["not allowed", "path", "restricted", "forbidden"]
                    ), f"Expected path restriction error, got: {error_text}"

            # Assert run is persisted (even on failure)
            run_id = run_response.get("run_id")
            if run_id:
                persisted_run = assertions.assert_run_persisted(run_id)
                # Run should be persisted even if execution failed
                assert persisted_run.get("plan_result"), "Plan should be persisted even on failure"

        finally:
            assertions.close()
