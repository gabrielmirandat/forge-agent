"""Agent API routes."""

from fastapi import APIRouter, HTTPException

from api.models.schemas import (
    ExecutionResult,
    GoalRequest,
    PlanResponse,
    StatusResponse,
    ToolInfo,
)

router = APIRouter()


@router.post("/goals", response_model=PlanResponse)
async def create_goal(request: GoalRequest):
    """Create a goal and generate execution plan.

    Args:
        request: Goal request with description and context

    Returns:
        Generated execution plan
    """
    # TODO: Implement goal planning
    # - Load agent config
    # - Initialize planner
    # - Generate plan
    # - Return plan response
    raise HTTPException(status_code=501, detail="Goal planning not yet implemented")


@router.post("/plans/{plan_id}/execute", response_model=ExecutionResult)
async def execute_plan(plan_id: str):
    """Execute a plan.

    Args:
        plan_id: Plan identifier

    Returns:
        Execution result
    """
    # TODO: Implement plan execution
    # - Load plan
    # - Initialize executor
    # - Execute plan
    # - Return results
    raise HTTPException(status_code=501, detail="Plan execution not yet implemented")


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get agent status and available tools.

    Returns:
        Agent status information
    """
    # TODO: Implement status endpoint
    # - Load config
    # - Initialize tool registry
    # - Return status
    return StatusResponse(
        status="operational",
        version="0.1.0",
        tools=[
            ToolInfo(name="filesystem", description="Filesystem operations", enabled=True),
            ToolInfo(name="git", description="Git operations", enabled=True),
            ToolInfo(name="github", description="GitHub API operations", enabled=True),
            ToolInfo(name="shell", description="Shell command execution", enabled=True),
            ToolInfo(name="system", description="System information", enabled=True),
        ],
    )

