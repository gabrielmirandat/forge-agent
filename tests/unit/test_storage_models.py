"""Unit tests for storage models."""

import pytest

from agent.storage.models import ApprovalStatus, RunRecord, RunSummary


class TestApprovalStatus:
    """Test ApprovalStatus enum."""

    def test_approval_status_values(self):
        """Test approval status enum values."""
        assert ApprovalStatus.PENDING == "pending"
        assert ApprovalStatus.APPROVED == "approved"
        assert ApprovalStatus.REJECTED == "rejected"


class TestRunRecord:
    """Test RunRecord model."""

    def test_create_run_record(self):
        """Test creating a run record."""
        record = RunRecord(
            run_id="test-run-123",
            plan_id="plan-123",
            objective="Test objective",
            plan_result={"plan": {"plan_id": "plan-123"}},
            created_at=1234567890.0,
        )

        assert record.run_id == "test-run-123"
        assert record.plan_id == "plan-123"
        assert record.objective == "Test objective"
        assert record.approval_status == ApprovalStatus.PENDING
        assert record.execution_result is None

    def test_run_record_with_execution(self):
        """Test run record with execution result."""
        record = RunRecord(
            run_id="test-run-123",
            plan_id="plan-123",
            objective="Test objective",
            plan_result={"plan": {"plan_id": "plan-123"}},
            execution_result={"success": True, "steps": []},
            created_at=1234567890.0,
        )

        assert record.execution_result is not None
        assert record.execution_result["success"] is True

    def test_run_record_with_approval(self):
        """Test run record with approval."""
        record = RunRecord(
            run_id="test-run-123",
            plan_id="plan-123",
            objective="Test objective",
            plan_result={"plan": {"plan_id": "plan-123"}},
            created_at=1234567890.0,
            approval_status=ApprovalStatus.APPROVED,
            approved_by="test-user",
            approved_at=1234567891.0,
            approval_reason="Looks good",
        )

        assert record.approval_status == ApprovalStatus.APPROVED
        assert record.approved_by == "test-user"
        assert record.approved_at == 1234567891.0
        assert record.approval_reason == "Looks good"


class TestRunSummary:
    """Test RunSummary model."""

    def test_create_run_summary(self):
        """Test creating a run summary."""
        summary = RunSummary(
            run_id="test-run-123",
            plan_id="plan-123",
            objective="Test objective",
            success=True,
            created_at=1234567890.0,
        )

        assert summary.run_id == "test-run-123"
        assert summary.plan_id == "plan-123"
        assert summary.objective == "Test objective"
        assert summary.success is True
        assert summary.created_at == 1234567890.0

    def test_run_summary_failed(self):
        """Test run summary for failed execution."""
        summary = RunSummary(
            run_id="test-run-123",
            plan_id="plan-123",
            objective="Test objective",
            success=False,
            created_at=1234567890.0,
        )

        assert summary.success is False
