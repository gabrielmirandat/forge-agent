"""API schemas for run history endpoints."""

from typing import List

from pydantic import BaseModel, Field

from agent.storage.models import RunRecord, RunSummary


class RunsListResponse(BaseModel):
    """Response schema for listing runs."""

    runs: List[RunSummary] = Field(..., description="List of run summaries")
    limit: int = Field(..., description="Limit used for pagination")
    offset: int = Field(..., description="Offset used for pagination")


class RunDetailResponse(BaseModel):
    """Response schema for getting a single run."""

    run: RunRecord = Field(..., description="Complete run record")

