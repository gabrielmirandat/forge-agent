"""API schemas for full orchestration endpoint."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Request schema for /run endpoint."""

    goal: str = Field(..., min_length=1, description="High-level goal description")
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional context information"
    )
    execution_policy: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional execution policy"
    )


class RunResponse(BaseModel):
    """Response schema for /run endpoint.

    Maps from internal PlanResult and ExecutionResult to API response.
    Execution result may be None if HITL is enabled (pending approval).
    """

    plan_result: Dict[str, Any] = Field(..., description="Planning result")
    execution_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Execution result (None if pending approval)"
    )

