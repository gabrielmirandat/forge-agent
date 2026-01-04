"""Pydantic schemas for API requests and responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GoalRequest(BaseModel):
    """Request to execute a goal."""

    goal: str = Field(..., description="High-level goal description")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    repo_path: Optional[str] = Field(default=None, description="Target repository path")


class PlanStep(BaseModel):
    """Single step in an execution plan."""

    tool: str = Field(..., description="Tool name")
    operation: str = Field(..., description="Operation name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")


class PlanResponse(BaseModel):
    """Response containing execution plan."""

    plan_id: str = Field(..., description="Unique plan identifier")
    steps: List[PlanStep] = Field(..., description="List of plan steps")
    estimated_time: Optional[int] = Field(default=None, description="Estimated execution time in seconds")


class ExecutionResult(BaseModel):
    """Result of plan execution."""

    execution_id: str = Field(..., description="Unique execution identifier")
    status: str = Field(..., description="Execution status (running, completed, failed)")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Step execution results")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ToolInfo(BaseModel):
    """Information about an available tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    enabled: bool = Field(..., description="Whether tool is enabled")


class StatusResponse(BaseModel):
    """Agent status response."""

    status: str = Field(..., description="Agent status")
    version: str = Field(..., description="Agent version")
    tools: List[ToolInfo] = Field(..., description="Available tools")

