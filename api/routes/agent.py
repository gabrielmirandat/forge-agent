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

    The Planner uses an LLM as a reasoning engine to propose execution steps.
    The plan is NOT executed here - it must be submitted to the execute endpoint.

    Args:
        request: Goal request with description and context

    Returns:
        Generated execution plan (not yet executed)
    """
    # TODO: Implement goal planning
    # - Load agent config
    # - Initialize planner (with LLM as reasoning engine)
    # - Generate plan (Planner does NOT execute, only proposes steps)
    # - Return plan response
    raise HTTPException(status_code=501, detail="Goal planning not yet implemented")


@router.post("/plans/{plan_id}/execute", response_model=ExecutionResult)
async def execute_plan(plan_id: str):
    """Execute a plan.

    The Executor is the ONLY component that can invoke tools. It executes
    plans deterministically and owns all retries, error handling, and safety checks.

    Args:
        plan_id: Plan identifier

    Returns:
        Execution result with status and outputs
    """
    # TODO: Implement plan execution
    # - Load plan
    # - Initialize executor (ONLY component allowed to invoke tools)
    # - Execute plan deterministically
    # - Handle retries, errors, and safety checks
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

