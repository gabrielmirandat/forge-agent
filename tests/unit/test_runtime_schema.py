"""Unit tests for runtime schema."""

import pytest

from agent.runtime.schema import (
    Plan,
    PlanStep,
    ToolName,
    InvalidPlanError,
    ToolNotFoundError,
    OperationNotSupportedError,
    PlanResult,
    ExecutionResult,
    ExecutionStep,
    PlannerDiagnostics,
)


class TestPlan:
    """Test Plan model."""

    def test_create_valid_plan(self):
        """Test creating a valid plan."""
        step = PlanStep(
            step_id=1,
            tool=ToolName.FILESYSTEM,
            operation="read_file",
            arguments={"path": "test.txt"},
            rationale="Read file",
        )
        plan = Plan(
            plan_id="test-plan",
            objective="Test objective",
            steps=[step],
        )
        
        assert plan.plan_id == "test-plan"
        assert plan.objective == "Test objective"
        assert len(plan.steps) == 1
        assert plan.steps[0].step_id == 1

    def test_plan_sequential_step_ids(self):
        """Test plan requires sequential step IDs."""
        steps = [
            PlanStep(step_id=1, tool=ToolName.FILESYSTEM, operation="read_file", arguments={}, rationale=""),
            PlanStep(step_id=2, tool=ToolName.FILESYSTEM, operation="write_file", arguments={}, rationale=""),
        ]
        plan = Plan(plan_id="test", objective="Test", steps=steps)
        assert len(plan.steps) == 2

    def test_plan_non_sequential_step_ids_error(self):
        """Test plan rejects non-sequential step IDs."""
        steps = [
            PlanStep(step_id=1, tool=ToolName.FILESYSTEM, operation="read_file", arguments={}, rationale=""),
            PlanStep(step_id=3, tool=ToolName.FILESYSTEM, operation="write_file", arguments={}, rationale=""),
        ]
        with pytest.raises(ValueError, match="sequential"):
            Plan(plan_id="test", objective="Test", steps=steps)

    def test_plan_empty_steps(self):
        """Test plan with empty steps."""
        plan = Plan(plan_id="test", objective="Test", steps=[])
        assert len(plan.steps) == 0


class TestPlanStep:
    """Test PlanStep model."""

    def test_create_valid_step(self):
        """Test creating a valid step."""
        step = PlanStep(
            step_id=1,
            tool=ToolName.FILESYSTEM,
            operation="read_file",
            arguments={"path": "test.txt"},
            rationale="Read file",
        )
        
        assert step.step_id == 1
        assert step.tool == ToolName.FILESYSTEM
        assert step.operation == "read_file"
        assert step.arguments == {"path": "test.txt"}

    def test_step_invalid_operation(self):
        """Test step with invalid operation."""
        with pytest.raises(ValueError):
            PlanStep(
                step_id=1,
                tool=ToolName.FILESYSTEM,
                operation="invalid_operation",
                arguments={},
                rationale="",
            )


class TestToolName:
    """Test ToolName enum."""

    def test_tool_name_values(self):
        """Test tool name enum values."""
        assert ToolName.FILESYSTEM == "filesystem"
        assert ToolName.SYSTEM == "system"
        assert ToolName.SHELL == "shell"
        assert ToolName.GIT == "git"
        assert ToolName.GITHUB == "github"


class TestErrors:
    """Test error classes."""

    def test_tool_not_found_error(self):
        """Test ToolNotFoundError."""
        error = ToolNotFoundError("nonexistent_tool")
        assert error.tool_name == "nonexistent_tool"
        assert "nonexistent_tool" in str(error)

    def test_operation_not_supported_error(self):
        """Test OperationNotSupportedError."""
        error = OperationNotSupportedError("filesystem", "invalid_op")
        assert error.tool_name == "filesystem"
        assert error.operation == "invalid_op"
        assert "filesystem" in str(error)
        assert "invalid_op" in str(error)

    def test_invalid_plan_error(self):
        """Test InvalidPlanError."""
        error = InvalidPlanError("Test error", validation_errors=["Error 1", "Error 2"])
        assert error.message == "Test error"
        assert len(error.validation_errors) == 2


class TestPlanResult:
    """Test PlanResult model."""

    def test_create_plan_result_success(self):
        """Test creating successful plan result."""
        plan = Plan(plan_id="test", objective="Test", steps=[])
        result = PlanResult(
            plan=plan,
            diagnostics=PlannerDiagnostics(
                model_name="test-model",
                temperature=0.1,
                retries_used=0,
                raw_llm_response="{}",
                extracted_json="{}",
                validation_errors=None,
            ),
        )
        
        assert result.plan == plan
        assert result.error is None

    def test_create_plan_result_error(self):
        """Test creating plan result with error."""
        result = PlanResult(
            plan=None,
            error="Test error",
            diagnostics=PlannerDiagnostics(
                model_name="test-model",
                temperature=0.1,
                retries_used=0,
                raw_llm_response="",
                extracted_json=None,
                validation_errors=["Error"],
            ),
        )
        
        assert result.plan is None
        assert result.error == "Test error"


class TestExecutionResult:
    """Test ExecutionResult model."""

    def test_create_execution_result(self):
        """Test creating execution result."""
        step = ExecutionStep(
            step_id=1,
            tool="filesystem",
            operation="read_file",
            success=True,
            output="content",
            started_at=1.0,
            finished_at=2.0,
        )
        result = ExecutionResult(
            plan_id="test-plan",
            success=True,
            steps=[step],
            started_at=1.0,
            finished_at=2.0,
        )
        
        assert result.plan_id == "test-plan"
        assert result.success is True
        assert len(result.steps) == 1
