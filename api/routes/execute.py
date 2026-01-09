"""Execution endpoint - executes a Plan."""

import logging
import uuid
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status

from agent.runtime.executor import Executor
from agent.runtime.schema import (
    ExecutionPolicy,
    ExecutionError,
    Plan,
)
from api.dependencies import get_executor
from api.schemas.execute import ExecuteRequest, ExecuteResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/execute", response_model=ExecuteResponse, status_code=status.HTTP_200_OK)
async def execute(
    request: ExecuteRequest,
    executor: Executor = Depends(get_executor),
) -> ExecuteResponse:
    """Execute a previously generated Plan.

    Executor only. No Planner call here.

    Args:
        request: Execute request with plan and optional execution policy
        executor: Executor instance (injected)

    Returns:
        ExecuteResponse with execution result (always HTTP 200, even on failure)

    Raises:
        HTTPException: If validation fails
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] POST /execute")

    try:
        # Parse Plan from request
        plan = Plan(**request.plan)

        # Parse ExecutionPolicy if provided
        execution_policy = None
        if request.execution_policy:
            execution_policy = ExecutionPolicy(**request.execution_policy.model_dump())

        # Create executor with policy if provided
        if execution_policy:
            from agent.config.loader import AgentConfig
            from agent.tools.base import ToolRegistry
            from api.dependencies import get_config, get_tool_registry

            config = get_config()
            tool_registry = get_tool_registry(config)
            executor = Executor(config, tool_registry, execution_policy)

        # Execute plan
        execution_result = await executor.execute(plan)

        logger.info(
            f"[{request_id}] Execution completed - plan_id: {execution_result.plan_id}, "
            f"success: {execution_result.success}"
        )

        # Always return HTTP 200, even on failure
        # Execution failure is indicated by execution_result.success = false
        return ExecuteResponse(execution_result=execution_result.model_dump())

    except Exception as e:
        # Validation errors (e.g., invalid Plan schema)
        if isinstance(e, (ValueError, TypeError)):
            logger.error(f"[{request_id}] Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"Invalid request: {str(e)}"},
            ) from e

        logger.exception(f"[{request_id}] Unexpected error during execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error"},
        ) from e

