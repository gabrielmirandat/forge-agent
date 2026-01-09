"""Full orchestration endpoint - Planner → Executor."""

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from agent.config.loader import AgentConfig
from agent.observability import (
    api_request_duration_seconds,
    api_requests_total,
    approvals_total,
    approval_pending_duration_seconds,
    get_logger,
    log_event,
    set_request_id,
    set_run_id,
    trace_span,
)
from agent.runtime.executor import Executor
from agent.runtime.planner import Planner
from agent.runtime.schema import (
    ExecutionPolicy,
    InvalidPlanError,
    LLMCommunicationError,
    PlanningError,
)
from agent.storage import ApprovalStatus, Storage, StorageError
from api.dependencies import get_config, get_executor, get_planner, get_storage, get_tool_registry
from api.schemas.run import RunRequest, RunResponse

router = APIRouter()
logger = get_logger("api.run", "api")


@router.post("/run", response_model=RunResponse, status_code=status.HTTP_200_OK)
async def run(
    request: RunRequest,
    planner: Planner = Depends(get_planner),
    storage: Storage = Depends(get_storage),
) -> RunResponse:
    """Full orchestration: Planner → Executor.

    This is the main orchestration endpoint.

    Args:
        request: Run request with goal, context, and optional execution policy
        planner: Planner instance (injected)

    Returns:
        RunResponse with plan result and execution result

    Raises:
        HTTPException: If planning fails
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    start_time = time.time()
    log_event(logger, "api.request.started", endpoint="/run", method="POST", goal=request.goal[:100])

    try:
        with trace_span("run", attributes={"goal": request.goal[:100]}):
            # Step 1: Call Planner
            plan_result = await planner.plan(request.goal, request.context)

        logger.info(
            f"[{request_id}] Planning succeeded - plan_id: {plan_result.plan.plan_id}, "
            f"steps: {len(plan_result.plan.steps)}"
        )

        # Step 2: Check if HITL is enabled
        config = get_config()
        hitl_enabled = config.human_in_the_loop.enabled

        if hitl_enabled:
            # HITL enabled: Plan only, do NOT execute
            # Step 3: Persist run with PENDING approval status
            try:
                run_id = await storage.save_run(
                    plan_result, None, ApprovalStatus.PENDING
                )
                set_run_id(run_id)
                approvals_total.labels(status="pending").inc()

                log_event(
                    logger,
                    "approval.pending",
                    run_id=run_id,
                    plan_id=plan_result.plan.plan_id,
                )
            except StorageError as e:
                duration = time.time() - start_time
                api_requests_total.labels(endpoint="/run", method="POST", status="500").inc()
                api_request_duration_seconds.labels(endpoint="/run").observe(duration)

                log_event(
                    logger,
                    "api.request.failed",
                    level="ERROR",
                    endpoint="/run",
                    method="POST",
                    status=500,
                    duration_ms=duration * 1000,
                    error=str(e),
                )

                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"error": "Failed to persist run"},
                ) from e

            duration = time.time() - start_time
            api_requests_total.labels(endpoint="/run", method="POST", status="200").inc()
            api_request_duration_seconds.labels(endpoint="/run").observe(duration)

            log_event(
                logger,
                "api.request.completed",
                endpoint="/run",
                method="POST",
                status=200,
                duration_ms=duration * 1000,
                run_id=run_id,
                hitl_enabled=True,
            )

            # Return plan only (no execution)
            return RunResponse(
                plan_result={
                    "plan": plan_result.plan.model_dump(),
                    "diagnostics": plan_result.diagnostics.model_dump(),
                },
                execution_result=None,  # No execution yet
            )

        # HITL disabled: Continue with Phase 7 behavior (plan + execute)
        # Step 3: Parse ExecutionPolicy if provided
        execution_policy = None
        if request.execution_policy:
            execution_policy = ExecutionPolicy(**request.execution_policy)

        # Step 4: Execute plan
        tool_registry = get_tool_registry(config)
        executor = Executor(config, tool_registry, execution_policy)

        execution_result = await executor.execute(plan_result.plan)

        logger.info(
            f"[{request_id}] Execution completed - success: {execution_result.success}, "
            f"stopped_at_step: {execution_result.stopped_at_step}"
        )

        # Step 5: Persist run
        try:
            run_id = await storage.save_run(plan_result, execution_result)
            logger.info(f"[{request_id}] Run persisted - run_id: {run_id}")
        except StorageError as e:
            # Persistence failure → HTTP 500
            # Do NOT affect execution result
            logger.error(f"[{request_id}] Failed to persist run: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Failed to persist run"},
            ) from e

        # Always return HTTP 200
        # Execution failure is indicated by execution_result.success = false
        return RunResponse(
            plan_result={
                "plan": plan_result.plan.model_dump(),
                "diagnostics": plan_result.diagnostics.model_dump(),
            },
            execution_result=execution_result.model_dump(),
        )

    except InvalidPlanError as e:
        logger.error(f"[{request_id}] Invalid plan error: {e}")
        error_detail = {"error": str(e)}
        if hasattr(e, "diagnostics") and e.diagnostics:
            error_detail["diagnostics"] = e.diagnostics.model_dump()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_detail,
        ) from e

    except LLMCommunicationError as e:
        logger.error(f"[{request_id}] LLM communication error: {e}")
        error_detail = {"error": str(e)}
        if hasattr(e, "diagnostics") and e.diagnostics:
            error_detail["diagnostics"] = e.diagnostics.model_dump()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=error_detail,
        ) from e

    except PlanningError as e:
        logger.error(f"[{request_id}] Planning error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Planning failed"},
        ) from e

    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error"},
        ) from e

