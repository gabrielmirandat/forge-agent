"""API schemas for approval endpoints."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ApproveRequest(BaseModel):
    """Request schema for /runs/{run_id}/approve endpoint."""

    approved_by: str = Field(..., min_length=1, description="Who approved (free-form string)")
    reason: Optional[str] = Field(default=None, description="Optional approval reason")
    execution_policy: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional execution policy"
    )


class ApproveResponse(BaseModel):
    """Response schema for /runs/{run_id}/approve endpoint."""

    run_id: str = Field(..., description="Run identifier")
    approval_status: str = Field(..., description="Approval status")
    execution_result: Dict[str, Any] = Field(..., description="Execution result")


class RejectRequest(BaseModel):
    """Request schema for /runs/{run_id}/reject endpoint."""

    rejected_by: str = Field(..., min_length=1, description="Who rejected (free-form string)")
    reason: str = Field(..., min_length=1, description="Rejection reason (required)")

