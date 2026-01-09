"""API schemas for planning endpoints."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PlanRequest(BaseModel):
    """Request schema for /plan endpoint."""

    goal: str = Field(..., min_length=1, description="High-level goal description")
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional context information"
    )


class PlanResponse(BaseModel):
    """Response schema for /plan endpoint.

    Maps from internal PlanResult to API response.
    """

    plan: Dict[str, Any] = Field(..., description="Generated execution plan")
    diagnostics: Dict[str, Any] = Field(..., description="Planning diagnostics")

