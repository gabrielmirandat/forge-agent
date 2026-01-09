"""API schemas for execution endpoints."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ExecutionPolicyRequest(BaseModel):
    """Execution policy request schema.

    Maps from internal ExecutionPolicy to API request.
    """

    max_retries_per_step: int = Field(
        default=0, ge=0, description="Maximum retries per step (0 = no retries)"
    )
    retry_delay_seconds: float = Field(
        default=0.0, ge=0.0, description="Fixed delay between retries in seconds"
    )
    rollback_on_failure: bool = Field(
        default=False, description="Whether to rollback successful steps on failure"
    )


class ExecuteRequest(BaseModel):
    """Request schema for /execute endpoint."""

    plan: Dict[str, Any] = Field(..., description="Plan to execute")
    execution_policy: Optional[ExecutionPolicyRequest] = Field(
        default=None, description="Optional execution policy"
    )


class ExecuteResponse(BaseModel):
    """Response schema for /execute endpoint.

    Maps from internal ExecutionResult to API response.
    """

    execution_result: Dict[str, Any] = Field(..., description="Execution result")

