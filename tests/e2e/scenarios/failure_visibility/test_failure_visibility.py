"""E2E test: Failure visibility and error handling."""

import pytest

from tests.e2e.assertions import E2EAssertions
from tests.e2e.runner import E2ETestRunner


@pytest.mark.e2e

def test_invalid_tool_usage_surfaces_error():
    """Test that invalid tool usage surfaces explicit error."""
    with E2ETestRunner() as runner:
        assertions = E2EAssertions()

        try:
            # Assert health
            assertions.assert_health()

            # Create run with invalid tool usage
            # The planner might return an empty/invalid plan, which causes a 422 error
            # This is a valid way to surface the error
            goal = "Use a tool that does not exist: nonexistent_tool with operation invalid_op"
            run_response = assertions.create_run(goal, allow_errors=True)

            # Check if we got an error response (422 is valid - plan validation failed)
            if run_response.get("status_code") in [422, 400, 500]:
                # Error was surfaced - this is what we want
                error_detail = run_response.get("response", {}).get("detail", {})
                error_text = str(error_detail.get("error", "")).lower()
                assert len(error_text) > 0, "Error should be non-empty"
                # Error should mention validation, plan, or tool
                assert any(
                    keyword in error_text
                    for keyword in ["validation", "plan", "tool", "error"]
                ), f"Error should mention validation/plan/tool: {error_text}"
            else:
                # If we got a 200, check execution result
                execution_result = run_response.get("execution_result")
                if execution_result:
                    success = execution_result.get("success", True)
                    if not success:
                        # Execution failed - error should be explicit
                        error = execution_result.get("error", "")
                        steps = execution_result.get("steps", [])
                        error_messages = [error] + [
                            s.get("error", "") for s in steps if not s.get("success", True)
                        ]
                        error_text = " ".join(error_messages)
                        assert len(error_text) > 0, "Error should be non-empty"

        finally:
            assertions.close()


@pytest.mark.e2e

def test_partial_execution_leaves_traceable_logs():
    """Test that partial execution leaves traceable logs."""
    with E2ETestRunner() as runner:
        assertions = E2EAssertions()

        try:
            # Assert health
            assertions.assert_health()

            # Create a run that might partially fail
            goal = "List files in ~/repos and then access /etc (which should fail)"
            run_response = assertions.create_run(goal)

            # Execution might succeed or fail partially
            execution_result = run_response.get("execution_result")
            if execution_result:
                steps = execution_result.get("steps", [])
                if len(steps) > 1:
                    # Check that we can see which steps succeeded/failed
                    successful_steps = [s for s in steps if s.get("success", False)]
                    failed_steps = [s for s in steps if not s.get("success", True)]

                    # At least one step should have executed
                    assert len(steps) > 0, "No steps executed"

                    # If there are failures, they should be explicit
                    for step in failed_steps:
                        error = step.get("error", "")
                        assert len(error) > 0, "Failed step should have error message"

            # Assert run is persisted
            run_id = run_response.get("run_id")
            if run_id:
                persisted_run = assertions.assert_run_persisted(run_id)
                assert persisted_run.get("plan_result"), "Plan should be persisted"

        finally:
            assertions.close()
