"""E2E test: Multi-tool workflows."""

import pytest

from tests.e2e.assertions import E2EAssertions
from tests.e2e.runner import E2ETestRunner


@pytest.mark.e2e

def test_system_and_filesystem_combination():
    """Test combining system and filesystem tools in one plan."""
    with E2ETestRunner() as runner:
        assertions = E2EAssertions()

        try:
            # Assert health
            assertions.assert_health()

            # Create run that uses both system and filesystem
            goal = "Get system information and then list files in ~/repos directory"
            run_response = assertions.create_run(goal)

            # Assert plan is valid
            assertions.assert_plan_valid(run_response["plan_result"])

            # Plan should have multiple steps
            plan = run_response["plan_result"]["plan"]
            steps = plan.get("steps", [])
            assert len(steps) >= 2, f"Expected at least 2 steps, got {len(steps)}"

            # Check that different tools are used
            tools_used = {step.get("tool") for step in steps}
            assert len(tools_used) >= 2, f"Expected at least 2 different tools, got {tools_used}"

            # Assert execution succeeded
            assertions.assert_run_success(run_response)

            # Assert execution result has multiple steps
            execution_result = run_response.get("execution_result")
            assert execution_result, "Missing execution_result"
            exec_steps = execution_result.get("steps", [])
            assert len(exec_steps) >= 2, f"Expected at least 2 execution steps, got {len(exec_steps)}"

            # Assert run is persisted
            run_id = run_response.get("run_id")
            if run_id:
                persisted_run = assertions.assert_run_persisted(run_id)
                assert persisted_run.get("execution_result"), "Execution not persisted"

        finally:
            assertions.close()


@pytest.mark.e2e

def test_analyze_and_propose_workflow():
    """Test workflow: analyze repo → propose change (plan only, no execution)."""
    with E2ETestRunner() as runner:
        assertions = E2EAssertions()

        try:
            # Assert health
            assertions.assert_health()

            # Create run that analyzes and proposes changes
            goal = "Analyze the forge-agent repository structure and propose a refactoring plan"
            run_response = assertions.create_run(goal)

            # Assert plan is valid
            assertions.assert_plan_valid(run_response["plan_result"])

            # Plan should exist (even if empty)
            plan = run_response["plan_result"]["plan"]
            assert "plan_id" in plan, "Missing plan_id"
            assert "objective" in plan, "Missing objective"

            # Execution might or might not happen (depending on plan)
            execution_result = run_response.get("execution_result")
            if execution_result:
                # If execution happened, it should be valid
                assertions.assert_execution_result_valid(execution_result)

            # Assert run is persisted
            run_id = run_response.get("run_id")
            if run_id:
                persisted_run = assertions.assert_run_persisted(run_id)
                assert persisted_run.get("plan_result"), "Plan should be persisted"

        finally:
            assertions.close()
